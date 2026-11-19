"""Build the static dashboard for GitHub Pages from the warehouse.

Reuses the Streamlit dashboard's queries and Plotly figures, so the two never
drift: every figure is pre-rendered per time window here, and the page only
picks which one to show. Run after the pipeline:

    python web/build.py [out_dir]     # default: _site
"""

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))
import app  # noqa: E402  (also puts src/ on sys.path)
from config import ENABLE_FINBERT, WATCHLIST  # noqa: E402
from llm_switch import is_llm_enabled  # noqa: E402

# Outside `streamlit run` every cached call logs a harmless "no runtime" warning.
for name in list(logging.root.manager.loggerDict):
    if name.startswith("streamlit"):
        logging.getLogger(name).setLevel(logging.ERROR)

# Dark styling without embedding a full Plotly template (~10 KB) in every figure.
THEME = dict(
    template="none",
    font=dict(family="Inter, sans-serif", color="#E2E8F0"),
    hoverlabel=dict(bgcolor="#121A2E", bordercolor="#1E2A45", font=dict(color="#E2E8F0")),
)


def fig_json(figure) -> dict:
    figure.update_layout(**THEME)
    figure.update_xaxes(automargin=True).update_yaxes(automargin=True)  # the template did this
    return json.loads(figure.to_json())


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def ticker_payload(ticker: str, now: datetime) -> dict:
    aligned = app.load_aligned(ticker, None)
    status = app.price_status(app.load_latest_price(ticker), now)
    windows = {}
    for label, hours in app.WINDOWS.items():
        start = pd.Timestamp(now) - pd.Timedelta(hours=hours) if hours else None
        frame = aligned[aligned["time_bucket"] >= start] if start is not None else aligned
        scores = frame["average_score"].dropna()
        returns = frame["hourly_return"].dropna()
        windows[label] = {
            "sentiment": float(scores.iloc[-1]) if len(scores) else None,
            "prev": float(scores.iloc[-2]) if len(scores) >= 2 else None,
            "articles": int(frame["article_count"].sum()),
            "return": float(returns.iloc[-1]) if len(returns) else None,
            "chart": None
            if frame.empty and status is None
            else fig_json(app.build_price_sentiment_chart(frame, ticker, status, start, now)),
        }

    headlines = app.load_headlines(ticker)
    headlines["url"] = headlines["url"].map(app._safe_link)
    headlines["scorer"] = headlines["scorer"].map(app.SCORER_NAMES)

    comparison = pd.DataFrame()
    if ENABLE_FINBERT:
        try:  # an older warehouse may lack the view
            comparison = app.load_comparison(ticker)
        except Exception:
            pass

    return {
        "status": status and {**status, "as_of": status["as_of"].isoformat()},
        "windows": windows,
        "headlines": records(headlines.drop(columns=["source"])),
        "comparison": records(comparison),
    }


def build(out: Path) -> None:
    now = datetime.now(timezone.utc)
    shutil.rmtree(out, ignore_errors=True)
    (out / "data" / "t").mkdir(parents=True)

    for ticker in WATCHLIST:
        (out / "data" / "t" / f"{ticker}.json").write_text(
            json.dumps(ticker_payload(ticker, now)), encoding="utf-8"
        )

    mood = {}
    for label, hours in app.WINDOWS.items():
        frame = app.load_mood(hours)
        mood[label] = None if frame.empty else fig_json(app.build_mood_map(frame))
    signals = app.load_signals(0.0)  # the page applies the visitor's threshold

    summary = {
        "generated_at": now.isoformat(),
        "llm_live": is_llm_enabled(),
        "finbert": ENABLE_FINBERT,
        "watchlist": WATCHLIST,
        "windows": list(app.WINDOWS),
        "mood": mood,
        "signals": records(signals),
    }
    (out / "data" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{PLOTLY_VERSION}}", plotly.offline.get_plotlyjs_version())
    (out / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site")
