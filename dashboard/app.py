"""Streamlit dashboard: news sentiment overlaid on price for a watchlist.

Reads the DuckDB warehouse read-only (a peer of the API), so it runs whether
or not the API process is up. Launch with:

    streamlit run dashboard/app.py
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from datetime import time as dtime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make the shared src/ modules importable regardless of the launch directory.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _setting(name: str) -> str | None:
    """Environment variable first, then Streamlit secrets (Community Cloud)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file, e.g. local runs
        return None


# Hosted mode (Streamlit Community Cloud): the pipeline runs in GitHub Actions
# and publishes the warehouse to the repo's `data` branch; this app downloads
# it. Unset locally, where the dashboard reads the local data/ directory.
WAREHOUSE_REPO = _setting("WAREHOUSE_REPO")  # e.g. "owner/repo"
if WAREHOUSE_REPO:
    # If the switch file can't be fetched, never claim LLM scoring is live.
    os.environ.setdefault("LLM_SCORING_DEFAULT", "off")

from config import DATA_DIR, ENABLE_FINBERT, PRICE_INTERVAL, WATCHLIST  # noqa: E402
from db import get_readonly_connection  # noqa: E402
from llm_switch import is_llm_enabled  # noqa: E402

# "Midnight Aurora" palette, matching .streamlit/config.toml.
POSITIVE_COLOR = "#34D399"
NEGATIVE_COLOR = "#FB7185"
PRICE_COLOR = "#22D3EE"

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
@st.cache_data(ttl=600, show_spinner=False)
def sync_warehouse() -> None:
    """Hosted mode: download the latest published warehouse, at most every 10
    minutes. On failure the last good copy is kept."""
    token = _setting("GITHUB_TOKEN")  # read-only token, needed only while private
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("warehouse.duckdb", "llm_scoring.json"):
        url = f"https://raw.githubusercontent.com/{WAREHOUSE_REPO}/data/{name}"
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                body = response.read()
        except OSError:  # URLError/HTTPError
            logger.warning("Could not download %s", name, exc_info=True)
            continue
        tmp = DATA_DIR / f"{name}.download"
        tmp.write_bytes(body)
        os.replace(tmp, DATA_DIR / name)  # atomic: readers never see a partial file


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
               negative_share, scorer, open, high, low, close, volume,
               hourly_return
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
        SELECT published_at, title, label AS sentiment, score, topic,
               scorer, publisher AS source, url
        FROM headline_sentiment
        WHERE ticker = ?
        ORDER BY published_at DESC
        LIMIT ?
    """
    return _df(sql, [ticker, limit])


@st.cache_data(ttl=60)
def load_last_scored() -> object:
    """Timestamp of the most recent LLM-scored headline (None if none yet)."""
    return _df("SELECT max(scored_at) AS last_scored FROM sentiment", [])[
        "last_scored"
    ].iloc[0]


@st.cache_data(ttl=300)
def load_latest_price(ticker: str) -> pd.DataFrame:
    """The most recent price bar for a ticker, regardless of the chart window."""
    return _df(
        "SELECT timestamp, close FROM prices WHERE ticker = ? AND close IS NOT NULL "
        "ORDER BY timestamp DESC LIMIT 1",
        [ticker],
    )


# --- Market hours ------------------------------------------------------------
# NYSE/Nasdaq regular session. Exchange holidays aren't modelled: on a holiday
# the status reads "delayed" rather than "closed", which is still accurate.
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
STALE_AFTER = pd.Timedelta(minutes=90)  # hourly bars; allow one missed bar


def market_is_open(now: datetime) -> bool:
    local = now.astimezone(MARKET_TZ)
    return local.weekday() < 5 and MARKET_OPEN <= local.time() < MARKET_CLOSE


def price_status(latest: pd.DataFrame, now: datetime) -> dict | None:
    """Describe the latest price: its value, when it was set, and whether the
    market is open ("open"), closed ("closed"), or open but without recent
    trades ("delayed", e.g. a holiday)."""
    if latest.empty:
        return None

    bar_start = pd.Timestamp(latest["timestamp"].iloc[0]).tz_convert("UTC")
    try:
        bar_length = pd.Timedelta(PRICE_INTERVAL)
    except ValueError:
        bar_length = pd.Timedelta(0)
    # A bar's close is the price at its end — but the session's last bar is
    # partial (e.g. 15:30-16:00 ET) and an in-progress bar hasn't ended yet,
    # so never report a time after the session close or after now.
    local_start = bar_start.tz_convert(MARKET_TZ)
    session_close = local_start.normalize() + pd.Timedelta(
        hours=MARKET_CLOSE.hour, minutes=MARKET_CLOSE.minute
    )
    as_of = min(
        bar_start + bar_length, session_close.tz_convert("UTC"), pd.Timestamp(now)
    )

    if not market_is_open(now):
        state = "closed"
    elif pd.Timestamp(now) - as_of <= STALE_AFTER:
        state = "open"
    else:
        state = "delayed"
    return {"price": float(latest["close"].iloc[0]), "as_of": as_of, "state": state}


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
                scorer,
                LAG(average_score) OVER (
                    PARTITION BY ticker ORDER BY time_bucket
                ) AS previous_score,
                LAG(scorer) OVER (
                    PARTITION BY ticker ORDER BY time_bucket
                ) AS previous_scorer,
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
          -- Compare like with like: a switch between the LLM and FinBERT is
          -- a change of model, not a change in sentiment.
          AND previous_scorer = scorer
          AND ABS(average_score - previous_score) >= ?
        ORDER BY ABS(average_score - previous_score) DESC
    """
    return _df(sql, [threshold])


@st.cache_data(ttl=300)
def load_mood(hours: int | None) -> pd.DataFrame:
    """Per-ticker sentiment over the window for the market-mood map. Like the
    hourly buckets, a ticker's average uses a single model (the LLM if it scored
    any of its headlines in the window, else FinBERT), never a blend."""
    sql = """
        WITH w AS (
            SELECT ticker, score, scorer FROM headline_sentiment
            WHERE published_at >= now() - INTERVAL (?) HOUR
        ),
        pick AS (
            SELECT ticker, COUNT(*) AS articles,
                   CASE WHEN bool_or(scorer = 'llm') THEN 'llm' ELSE 'finbert' END AS scorer
            FROM w GROUP BY ticker
        )
        SELECT w.ticker, AVG(w.score) AS avg_score, p.articles, p.scorer
        FROM w JOIN pick p ON p.ticker = w.ticker AND p.scorer = w.scorer
        GROUP BY w.ticker, p.articles, p.scorer
    """
    mood = _df(sql, [hours if hours is not None else 24 * 365 * 50])
    return mood[mood["ticker"].isin(WATCHLIST)]


# --- Chart -------------------------------------------------------------------
def build_price_sentiment_chart(
    frame: pd.DataFrame,
    ticker: str,
    status: dict | None = None,
    window_start: pd.Timestamp | None = None,
    now: datetime | None = None,
) -> go.Figure:
    """Dual-axis chart: price line (left) and average sentiment bars (right).

    When the market isn't trading, a dashed "last close" line runs from the
    final trade to now, so the price axis stays meaningful and the gap reads as
    "no trading" rather than missing data.
    """
    figure = make_subplots(specs=[[{"secondary_y": True}]])

    prices = frame.dropna(subset=["close"])
    if not prices.empty:
        figure.add_trace(  # soft glow beneath the price line
            go.Scatter(
                x=prices["time_bucket"],
                y=prices["close"],
                mode="lines",
                line=dict(color=PRICE_COLOR, width=9),
                opacity=0.12,
                hoverinfo="skip",
                showlegend=False,
                connectgaps=True,
            ),
            secondary_y=False,
        )
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

    if status is not None and status["state"] != "open" and now is not None:
        # Continue from the last plotted price point, else the window's start.
        if not prices.empty:
            start = prices["time_bucket"].max()
        elif window_start is not None:
            start = max(window_start, status["as_of"])
        else:
            start = status["as_of"]
        end = pd.Timestamp(now)
        if start < end:
            label = "Last close" if status["state"] == "closed" else "Last trade"
            figure.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[status["price"], status["price"]],
                    name=f"{label} ${status['price']:,.2f}",
                    mode="lines",
                    # Drawn boldly: when it's the only price trace the axis
                    # centres it, where it can sit on the sentiment zero line.
                    line=dict(color=PRICE_COLOR, width=2.5, dash="dash"),
                    opacity=0.9,
                    hovertemplate=(
                        f"{label} ${status['price']:,.2f} "
                        f"({status['as_of']:%a %H:%M} UTC)<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    # One bar trace per scoring model, so FinBERT-fallback hours are visibly
    # distinct (hatched) from LLM-scored hours.
    sentiment = frame.dropna(subset=["average_score"])
    for scorer, model, label, pattern in [
        ("llm", "LLM", "Sentiment (LLM)", ""),
        ("finbert", "FinBERT", "Sentiment (FinBERT)", "/"),
    ]:
        bars = sentiment[sentiment["scorer"] == scorer]
        if bars.empty:
            continue
        figure.add_trace(
            go.Bar(
                x=bars["time_bucket"],
                y=bars["average_score"],
                name=label,
                marker=dict(
                    color=[
                        POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
                        for value in bars["average_score"]
                    ],
                    pattern=dict(shape=pattern, fgcolor="rgba(255,255,255,0.6)"),
                ),
                opacity=0.55,
                hovertemplate=f"%{{x}}<br>sentiment %{{y:.2f}} ({model})<extra></extra>",
            ),
            secondary_y=True,
        )

    figure.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0
        ),
        hovermode="x unified",
        # Each hour belongs to exactly one model's trace, so overlay keeps
        # every bar centred on its timestamp instead of grouping side by side.
        barmode="overlay",
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


def build_mood_map(mood: pd.DataFrame) -> go.Figure:
    """Sector -> ticker treemap: tile size = article count, color = sentiment.

    Built node by node so every tile (including sector and root tiles, whose
    sentiment is the article-weighted mean of their children) has a label.
    """
    data = mood.assign(
        sector=mood["ticker"].map(lambda t: WATCHLIST[t][1]),
        name=mood["ticker"].map(lambda t: WATCHLIST[t][0]),
        weighted=mood["avg_score"] * mood["articles"],
    )
    sectors = data.groupby("sector").agg(articles=("articles", "sum"), weighted=("weighted", "sum"))
    root = "Watchlist"
    ids = [root, *sectors.index, *data["ticker"]]
    parents = ["", *[root] * len(sectors), *data["sector"]]
    values = [int(data["articles"].sum()), *sectors["articles"], *data["articles"]]
    scores = [
        data["weighted"].sum() / data["articles"].sum(),
        *(sectors["weighted"] / sectors["articles"]),
        *data["avg_score"],
    ]
    names = [root, *sectors.index, *data["name"]]

    figure = go.Figure(
        go.Treemap(
            ids=ids,
            labels=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            text=[f"{score:+.2f}" for score in scores],
            customdata=names,
            texttemplate="<b>%{label}</b><br>%{text}",
            hovertemplate=(
                "<b>%{label}</b> · %{customdata}<br>sentiment %{text}"
                "<br>%{value} articles<extra></extra>"
            ),
            marker=dict(
                colors=scores,
                colorscale=[(0, NEGATIVE_COLOR), (0.5, "#334155"), (1, POSITIVE_COLOR)],
                cmin=-1,
                cmax=1,
                cornerradius=6,
                line=dict(color="#0A0E1A", width=2),
                colorbar=dict(title="Sentiment", tickvals=[-1, -0.5, 0, 0.5, 1], thickness=12),
            ),
            root_color="rgba(0,0,0,0)",
        )
    )
    figure.update_layout(
        height=460, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)"
    )
    return figure


# --- UI ----------------------------------------------------------------------
# Static styling for the header only (our own markup, no user data).
HERO_CSS = """
<style>
.hero {
  padding: 1.4rem 1.6rem; margin-bottom: 0.6rem; border-radius: 1rem;
  border: 1px solid #1E2A45;
  background:
    radial-gradient(120% 160% at 0% 0%, rgba(34,211,238,0.20) 0%,
      rgba(167,139,250,0.12) 40%, rgba(10,14,26,0) 72%),
    #0F1628;
}
.hero-title {
  font-family: "Space Grotesk", sans-serif; font-weight: 700;
  font-size: 2.2rem; line-height: 1.15;
  background: linear-gradient(90deg, #67E8F9 0%, #A78BFA 55%, #F0ABFC 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: #94A3B8; margin-top: 0.4rem; font-size: 0.95rem; }
</style>
"""

WINDOWS = {"24h": 24, "48h": 48, "72h": 72, "1W": 168, "All": None}
SCORER_NAMES = {"llm": "LLM", "finbert": "FinBERT"}


def _ticker_label(ticker: str) -> str:
    return f"{ticker} · {WATCHLIST.get(ticker, (ticker, ''))[0]}"


def render_header(status: dict | None) -> None:
    st.markdown(
        HERO_CSS
        + '<div class="hero"><div class="hero-title">Market Sentiment Radar</div>'
        + f'<div class="hero-sub">Near-real-time news sentiment for {len(WATCHLIST)} '
        + "large caps, aligned with intraday prices · All times UTC</div></div>",
        unsafe_allow_html=True,
    )
    # Which model produces live sentiment, so FinBERT-fallback or stale scores
    # are never mistaken for live LLM scores.
    if is_llm_enabled():
        badges = [":green-badge[:material/bolt: LLM scoring live]"]
    elif ENABLE_FINBERT:
        badges = [":orange-badge[:material/pause_circle: LLM paused · FinBERT scores new headlines]"]
    else:
        try:
            last = load_last_scored()
        except Exception:
            last = None
        since = (
            f" · last scored {pd.Timestamp(last):%b %d %H:%M}"
            if last is not None and not pd.isna(last)
            else ""
        )
        badges = [f":gray-badge[:material/pause_circle: Sentiment scoring paused{since}]"]
    if status is not None:
        badges.append(
            {
                "open": ":green-badge[:material/show_chart: Market open]",
                "closed": ":gray-badge[:material/bedtime: Market closed]",
                "delayed": ":orange-badge[:material/schedule: No recent trades]",
            }[status["state"]]
        )
    badges.append(":blue-badge[:material/update: Pipeline runs every 30 min]")
    st.markdown(" ".join(badges))


def main() -> None:
    st.set_page_config(
        page_title="Market Sentiment Radar", page_icon=":material/radar:", layout="wide"
    )
    if WAREHOUSE_REPO:
        sync_warehouse()

    with st.sidebar:
        st.markdown("### :material/radar: Sentiment Radar")
        sectors = sorted({sector for _, sector in WATCHLIST.values()})
        sector = st.pills("Sector", sectors, selection_mode="single")
        tickers = sorted(t for t, (_, s) in WATCHLIST.items() if sector in (None, s))
        ticker = st.selectbox(
            "Ticker",
            tickers,
            index=tickers.index("AAPL") if "AAPL" in tickers else 0,
            format_func=_ticker_label,
        )
        window_label = st.segmented_control("Window", list(WINDOWS), default="48h") or "48h"
        hours = WINDOWS[window_label]
        threshold = st.slider(
            "Mover threshold", 0.0, 2.0, 0.3, 0.1,
            help="Minimum hour-over-hour change in sentiment to count as a sharp move.",
        )
        # Per-visitor view setting only. Whether FinBERT runs at all is a
        # server-side setting (ENABLE_FINBERT) that visitors can never change.
        show_comparison = st.toggle(
            "Show model comparison",
            value=ENABLE_FINBERT,
            disabled=not ENABLE_FINBERT,
            help=(
                "Show a tab comparing how the general LLM and the finance-tuned "
                "FinBERT model scored the same headlines. This only changes your "
                "view; it does not turn LLM scoring on or off."
                if ENABLE_FINBERT
                else "FinBERT scoring is turned off on this deployment to save memory."
            ),
        )
        if st.button("Refresh data", icon=":material/refresh:", use_container_width=True):
            if _try_refresh():
                st.rerun()
            else:
                st.caption("Data was refreshed moments ago — try again shortly.")
        st.caption("Headlines: Yahoo Finance · Prices: yfinance · Sentiment: GPT-4o-mini + FinBERT")

    now = datetime.now(timezone.utc)
    window_start = pd.Timestamp(now) - pd.Timedelta(hours=hours) if hours else None

    try:
        aligned = load_aligned(ticker, hours)
        headlines = load_headlines(ticker)
        signals = load_signals(threshold)
        mood = load_mood(hours)
        status = price_status(load_latest_price(ticker), now)
    except FileNotFoundError:
        render_header(None)
        st.error(
            "The warehouse hasn't been created yet. Run the pipeline first "
            "(`python scheduler.py` or a manual run), then refresh."
        )
        return
    except Exception:  # e.g. warehouse briefly locked by a writer
        # Log internally; don't show error text (file paths etc.) to visitors.
        logger.warning("Warehouse unavailable", exc_info=True)
        render_header(None)
        st.warning("Data is temporarily unavailable. Please refresh in a moment.")
        return

    render_header(status)

    # Summary cards for the selected ticker.
    scored = aligned.dropna(subset=["average_score"])
    latest_sentiment = float(scored["average_score"].iloc[-1]) if not scored.empty else None
    prev_sentiment = float(scored["average_score"].iloc[-2]) if len(scored) >= 2 else None
    returns = aligned["hourly_return"].dropna()

    st.markdown(f"#### {_ticker_label(ticker)}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Sentiment · latest hour",
        f"{latest_sentiment:+.2f}" if latest_sentiment is not None else "—",
        delta=(
            f"{latest_sentiment - prev_sentiment:+.2f}"
            if latest_sentiment is not None and prev_sentiment is not None
            else None
        ),
        border=True,
    )
    col2.metric(f"Articles · {window_label}", int(aligned["article_count"].sum()), border=True)
    # Always the most recent price (not limited to the chart window).
    if status is not None:
        as_of = status["as_of"]
        price_label = {
            "open": f"Price · {as_of:%H:%M} UTC",
            "closed": f"Last close · {as_of:%a %H:%M} UTC",
            "delayed": f"Last trade · {as_of:%a %H:%M} UTC",
        }[status["state"]]
        col3.metric(
            price_label,
            f"${status['price']:,.2f}",
            delta=f"{returns.iloc[-1]:+.2%} last hour" if not returns.empty else None,
            border=True,
        )
    else:
        col3.metric("Price", "—", border=True)
    if not signals.empty:
        top = signals.iloc[0]
        col4.metric(
            "Biggest mover · watchlist", top["ticker"], delta=f"{top['delta']:+.2f}", border=True
        )
    else:
        col4.metric("Biggest mover · watchlist", "—", border=True)

    labels = [
        f":material/show_chart: {ticker} chart",
        ":material/grid_view: Market mood",
        ":material/newspaper: Headlines",
    ]
    if show_comparison:
        labels.append(":material/psychology: Model comparison")
    tabs = st.tabs(labels)

    with tabs[0]:
        if aligned.empty and status is None:
            st.info(f"No data yet for {ticker}. New tickers fill in within about an hour.")
        else:
            st.plotly_chart(
                build_price_sentiment_chart(aligned, ticker, status, window_start, now),
                use_container_width=True,
            )
            st.caption(
                "Bars: average headline sentiment per hour (hatched = scored by FinBERT "
                "while LLM scoring is paused). Line: price."
            )

    with tabs[1]:
        left, right = st.columns([3, 2], gap="large")
        with left:
            if mood.empty:
                st.info("No headlines in this window yet.")
            else:
                st.plotly_chart(build_mood_map(mood), use_container_width=True)
                st.caption(f"Tile size: articles · color: average sentiment over {window_label}")
        with right:
            st.markdown("##### Sentiment movers")
            st.caption(f"Hour-over-hour change ≥ {threshold:.1f}")
            if signals.empty:
                st.write("No sharp moves right now.")
            else:
                st.dataframe(
                    signals.assign(name=signals["ticker"].map(lambda t: WATCHLIST.get(t, (t,))[0]))[
                        ["ticker", "name", "average_score", "previous_score", "delta"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "ticker": "Ticker",
                        "name": "Company",
                        "average_score": st.column_config.NumberColumn("Now", format="%+.2f"),
                        "previous_score": st.column_config.NumberColumn("Prev", format="%+.2f"),
                        "delta": st.column_config.NumberColumn("Δ", format="%+.2f"),
                    },
                )

    with tabs[2]:
        if headlines.empty:
            st.write("No scored headlines yet.")
        else:
            display = headlines.drop(columns=["source"])
            display["scorer"] = display["scorer"].map(SCORER_NAMES)
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
                    "score": st.column_config.ProgressColumn(
                        "Score", format="%+.2f", min_value=-1, max_value=1, width="small"
                    ),
                    "topic": st.column_config.TextColumn("Topic", width="medium"),
                    "scorer": st.column_config.TextColumn("Scored by", width="small"),
                    "url": st.column_config.LinkColumn("Link", display_text="open", width="small"),
                },
            )

    if not show_comparison:
        return

    with tabs[3]:
        st.caption(
            "Headlines scored by both models: the general LLM and the finance-tuned "
            "FinBERT. This set only grows while LLM scoring is switched on."
        )
        # Loaded resiliently so an older warehouse without the view still renders.
        try:
            comparison = load_comparison(ticker)
        except Exception:
            comparison = pd.DataFrame()
        if comparison.empty:
            st.write("No headlines have been scored by both models yet.")
            return
        agreement = float(comparison["labels_agree"].mean())
        disagreements = comparison[~comparison["labels_agree"]]
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Headlines compared", len(comparison), border=True)
        mcol2.metric("Label agreement", f"{agreement:.0%}", border=True)
        mcol3.metric("Disagreements", len(disagreements), border=True)
        st.markdown("##### Where the models disagree")
        if disagreements.empty:
            st.write("The two models agree on every scored headline.")
        else:
            st.dataframe(
                disagreements[["title", "llm_label", "llm_score", "finbert_label", "finbert_score"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "title": st.column_config.TextColumn("Headline", width="large"),
                    "llm_label": st.column_config.TextColumn("LLM", width="small"),
                    "llm_score": st.column_config.NumberColumn("LLM score", format="%+.2f", width="small"),
                    "finbert_label": st.column_config.TextColumn("FinBERT", width="small"),
                    "finbert_score": st.column_config.NumberColumn(
                        "FinBERT score", format="%+.2f", width="small"
                    ),
                },
            )


if __name__ == "__main__":
    main()
