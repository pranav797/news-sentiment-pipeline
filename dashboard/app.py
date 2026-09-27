"""Streamlit dashboard: news sentiment overlaid on price for a watchlist.

Reads the DuckDB warehouse read-only (a peer of the API), so it runs whether
or not the API process is up. Launch with:

    streamlit run dashboard/app.py
"""

import logging
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ENABLE_FINBERT, WATCHLIST  # noqa: E402
from db import get_readonly_connection  # noqa: E402

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
PRICE_COLOR = "#2563eb"

logger = logging.getLogger(__name__)

# "Refresh data" clears a cache shared by every visitor, so rate-limit it
# server-wide to stop one visitor from hammering the warehouse.
REFRESH_COOLDOWN_SECONDS = 30
_refresh_lock = threading.Lock()
_last_refresh = 0.0


def _try_refresh() -> bool:
    """Clear the data cache unless it was cleared within the cooldown."""
    global _last_refresh
    with _refresh_lock:
        now = time.monotonic()
        if now - _last_refresh < REFRESH_COOLDOWN_SECONDS:
            return False
        _last_refresh = now
    st.cache_data.clear()
    return True


def _safe_link(url: object) -> object:
    """Keep only absolute http(s) links; anything else (javascript:, data:)
    is dropped so it can never render as a clickable link."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return url
    return None


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
def load_comparison(ticker: str) -> pd.DataFrame:
    sql = """
        SELECT title, published_at, llm_label, llm_score,
               finbert_label, finbert_score, labels_agree
        FROM sentiment_comparison
        WHERE ticker = ?
        ORDER BY published_at DESC
    """
    return _df(sql, [ticker])


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
        # Per-visitor view setting only. Whether FinBERT runs at all is a
        # server-side deployment setting (ENABLE_FINBERT), never changeable by
        # public visitors.
        show_comparison = st.toggle(
            "LLM vs. FinBERT comparison",
            value=ENABLE_FINBERT,
            disabled=not ENABLE_FINBERT,
            help=(
                "Show how the general LLM and the finance-tuned FinBERT model "
                "score the same headlines."
                if ENABLE_FINBERT
                else "FinBERT scoring is turned off on this deployment to save memory."
            ),
        )
        if st.button("Refresh data"):
            if _try_refresh():
                st.rerun()
            else:
                st.caption("Data was refreshed moments ago — try again shortly.")

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
    except Exception:  # e.g. warehouse briefly locked by a writer
        # Log internally; don't show error text (file paths etc.) to visitors.
        logger.warning("Warehouse unavailable", exc_info=True)
        st.warning("Data is temporarily unavailable. Please refresh in a moment.")
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
        display = headlines.drop(columns=["source"])
        display["url"] = display["url"].map(_safe_link)
        st.dataframe(
            display,
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

    if not show_comparison:
        return

    # Two-model comparison — LLM vs. FinBERT (Phase 8). Loaded resiliently so
    # an older warehouse without the view doesn't blank the page.
    try:
        comparison = load_comparison(ticker)
    except Exception:
        comparison = pd.DataFrame()

    st.subheader("Model comparison — LLM vs. FinBERT")
    if comparison.empty:
        st.write("No FinBERT scores yet. Run the pipeline to populate them.")
    else:
        agreement = float(comparison["labels_agree"].mean())
        disagreements = comparison[~comparison["labels_agree"]]
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Headlines compared", len(comparison))
        mcol2.metric("Label agreement", f"{agreement:.0%}")
        mcol3.metric("Disagreements", len(disagreements))

        st.caption("Headlines where the two models assigned different labels")
        if disagreements.empty:
            st.write("The two models agree on every scored headline.")
        else:
            st.dataframe(
                disagreements[
                    ["title", "llm_label", "llm_score", "finbert_label", "finbert_score"]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "title": st.column_config.TextColumn("Headline", width="large"),
                    "llm_label": st.column_config.TextColumn("LLM", width="small"),
                    "llm_score": st.column_config.NumberColumn(
                        "LLM score", format="%+.2f", width="small"
                    ),
                    "finbert_label": st.column_config.TextColumn(
                        "FinBERT", width="small"
                    ),
                    "finbert_score": st.column_config.NumberColumn(
                        "FinBERT score", format="%+.2f", width="small"
                    ),
                },
            )


if __name__ == "__main__":
    main()
