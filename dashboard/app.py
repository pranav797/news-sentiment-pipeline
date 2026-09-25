"""Streamlit dashboard: news sentiment overlaid on price for a watchlist.

Reads the DuckDB warehouse read-only (a peer of the API), so it runs whether
or not the API process is up. Launch with:

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import WATCHLIST  # noqa: E402
from db import get_readonly_connection  # noqa: E402

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
PRICE_COLOR = "#2563eb"


# --- Data access -------------------------------------------------------------
def _df(sql: str, params: list) -> pd.DataFrame:
    """Run a read-only query and return the result as a DataFrame."""
    connection = get_readonly_connection()
    try:
        return connection.execute(sql, params).df()
    finally:
        connection.close()


@st.cache_data(ttl=300)
def load_aligned(ticker: str, hours: int | None) -> pd.DataFrame:
    sql = """
        SELECT time_bucket, average_score, article_count, positive_share,
               negative_share, open, high, low, close, volume, hourly_return
        FROM aligned
        WHERE ticker = ?
    """
    params: list = [ticker]
    if hours is not None:
        sql += " AND time_bucket >= now() - INTERVAL (?) HOUR"
        params.append(hours)
    sql += " ORDER BY time_bucket"
    return _df(sql, params)


@st.cache_data(ttl=300)
def load_headlines(ticker: str, limit: int = 20) -> pd.DataFrame:
    sql = """
        SELECT n.published_at, n.title, s.sentiment, s.score, s.topic,
               n.source, n.url
        FROM news n
        JOIN sentiment s ON s.headline_id = n.id
        WHERE n.ticker = ?
        ORDER BY n.published_at DESC
        LIMIT ?
    """
    return _df(sql, [ticker, limit])


@st.cache_data(ttl=300)
def load_signals(threshold: float = 0.3) -> pd.DataFrame:
    sql = """
        WITH ranked AS (
            SELECT
                ticker,
                time_bucket,
                average_score,
                LAG(average_score) OVER (
                    PARTITION BY ticker ORDER BY time_bucket
                ) AS previous_score,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker ORDER BY time_bucket DESC
                ) AS recency
            FROM sentiment_hourly
        )
        SELECT
            ticker,
            average_score,
            previous_score,
            average_score - previous_score AS delta
        FROM ranked
        WHERE recency = 1
          AND previous_score IS NOT NULL
          AND ABS(average_score - previous_score) >= ?
        ORDER BY ABS(average_score - previous_score) DESC
    """
    return _df(sql, [threshold])


# --- Chart -------------------------------------------------------------------
def build_price_sentiment_chart(frame: pd.DataFrame, ticker: str) -> go.Figure:
    """Dual-axis chart: price line (left) and average sentiment bars (right)."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    prices = frame.dropna(subset=["close"])
    figure.add_trace(
        go.Scatter(
            x=prices["time_bucket"],
            y=prices["close"],
            name="Price (close)",
            mode="lines",
            line=dict(color=PRICE_COLOR, width=2),
            connectgaps=True,
        ),
        secondary_y=False,
    )

    sentiment = frame.dropna(subset=["average_score"])
    bar_colors = [
        POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
        for value in sentiment["average_score"]
    ]
    figure.add_trace(
        go.Bar(
            x=sentiment["time_bucket"],
            y=sentiment["average_score"],
            name="Avg sentiment",
            marker_color=bar_colors,
            opacity=0.55,
            hovertemplate="%{x}<br>sentiment %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    figure.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1
        ),
        hovermode="x unified",
        bargap=0.45,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(
        title_text="Time (UTC)",
        showgrid=False,
        showline=True,
        linecolor="rgba(148,163,184,0.3)",
    )
    figure.update_yaxes(
        title_text="Price ($)",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="Sentiment",
        range=[-1.05, 1.05],
        tickvals=[-1, -0.5, 0, 0.5, 1],
        secondary_y=True,
        showgrid=False,
        zeroline=True,
        zerolinecolor="rgba(148,163,184,0.35)",
    )
    return figure


# --- UI ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="News Sentiment Pipeline", layout="wide")
    st.title("Financial News Sentiment vs. Price")
    st.caption(
        "Near-real-time headline sentiment aligned with intraday prices. "
        "All times UTC."
    )

    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("Ticker", sorted({t.upper() for t in WATCHLIST}))
        window = st.selectbox(
            "Window",
            options=[24, 48, 72, 168, 0],
            index=1,
            format_func=lambda h: "All" if h == 0 else f"Last {h}h",
        )
        threshold = st.slider("Signal threshold", 0.0, 2.0, 0.3, 0.1)
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    hours = None if window == 0 else window

    try:
        aligned = load_aligned(ticker, hours)
        headlines = load_headlines(ticker)
        signals = load_signals(threshold)
    except FileNotFoundError:
        st.error(
            "The warehouse hasn't been created yet. Run the pipeline first "
            "(`python scheduler.py` or a manual run), then refresh."
        )
        return
    except Exception as error:  # e.g. warehouse briefly locked by a writer
        st.warning(f"Warehouse temporarily unavailable: {error}")
        return

    if aligned.empty:
        st.info(f"No data yet for {ticker}. Let the pipeline run a few cycles.")
        return

    # Summary metrics for the selected ticker.
    scored = aligned.dropna(subset=["average_score"])
    latest_sentiment = float(scored["average_score"].iloc[-1]) if not scored.empty else None
    prev_sentiment = (
        float(scored["average_score"].iloc[-2]) if len(scored) >= 2 else None
    )
    article_total = int(aligned["article_count"].sum())
    latest_close = aligned.dropna(subset=["close"])
    latest_price = float(latest_close["close"].iloc[-1]) if not latest_close.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Latest avg sentiment",
        f"{latest_sentiment:+.2f}" if latest_sentiment is not None else "—",
        delta=(
            f"{latest_sentiment - prev_sentiment:+.2f}"
            if latest_sentiment is not None and prev_sentiment is not None
            else None
        ),
    )
    col2.metric("Articles in window", article_total)
    col3.metric(
        "Latest price",
        f"${latest_price:,.2f}" if latest_price is not None else "—",
    )
    if not signals.empty:
        top = signals.iloc[0]
        col4.metric(
            "Biggest mover (watchlist)",
            top["ticker"],
            delta=f"{top['delta']:+.2f}",
        )
    else:
        col4.metric("Biggest mover (watchlist)", "—")

    # Main chart.
    st.subheader(f"{ticker} — sentiment vs. price")
    st.plotly_chart(
        build_price_sentiment_chart(aligned, ticker), use_container_width=True
    )

    # Watchlist signals — compact table, natural width.
    st.subheader("Sentiment movers")
    st.caption(f"Change vs. prior hour ≥ {threshold:.1f}")
    if signals.empty:
        st.write("No sharp moves in the latest window.")
    else:
        st.dataframe(
            signals.rename(
                columns={
                    "ticker": "Ticker",
                    "average_score": "Now",
                    "previous_score": "Prev",
                    "delta": "Δ",
                }
            ),
            hide_index=True,
            use_container_width=False,
            column_config={
                "Now": st.column_config.NumberColumn(format="%.2f"),
                "Prev": st.column_config.NumberColumn(format="%.2f"),
                "Δ": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

    # Latest headlines — full page width, source dropped to reduce clutter.
    st.subheader(f"Latest {ticker} headlines")
    if headlines.empty:
        st.write("No scored headlines yet.")
    else:
        st.dataframe(
            headlines.drop(columns=["source"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "published_at": st.column_config.DatetimeColumn(
                    "Published (UTC)", format="YYYY-MM-DD HH:mm", width="small"
                ),
                "title": st.column_config.TextColumn("Headline", width="large"),
                "sentiment": st.column_config.TextColumn("Label", width="small"),
                "score": st.column_config.NumberColumn("Score", format="%+.2f", width="small"),
                "topic": st.column_config.TextColumn("Topic", width="medium"),
                "url": st.column_config.LinkColumn(
                    "Link", display_text="open", width="small"
                ),
            },
        )


if __name__ == "__main__":
    main()
