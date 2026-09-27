"""Owner-only on/off switch for LLM (OpenAI) sentiment scoring.

The switch is a small JSON file on the data volume. Only the worker container
mounts that volume read-write; the internet-facing API and dashboard mount it
read-only, so the switch can only be flipped by someone with shell access to
the server. Turning scoring on can carry an expiry, so a forgotten switch
turns itself off.

Usage (on the server, via llm.sh, or directly):
    python src/llm_switch.py on [HOURS]   # default LLM_DEFAULT_ON_HOURS
    python src/llm_switch.py off
    python src/llm_switch.py status

Any unreadable or malformed switch file is treated as OFF, so a fault can
never cause unexpected spending.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

from config import LLM_DEFAULT_ON_HOURS, LLM_SCORING_DEFAULT, LLM_SWITCH_FILE


def _read_state() -> dict | None:
    """Return the stored switch state, None if never set, or {} if unreadable."""
    try:
        return json.loads(LLM_SWITCH_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {}  # unreadable or malformed -> treated as off


def llm_status() -> tuple[bool, datetime | None]:
    """Return (enabled, expires_at). expires_at is None when there's no expiry."""
    state = _read_state()
    if state is None:
        return LLM_SCORING_DEFAULT, None
    if not isinstance(state, dict) or state.get("enabled") is not True:
        return False, None

    until = state.get("until")
    if until is None:
        return True, None
    try:
        expires_at = datetime.fromisoformat(until)
    except (TypeError, ValueError):
        return False, None
    if expires_at.tzinfo is None:
        return False, None
    if datetime.now(timezone.utc) >= expires_at:
        return False, expires_at
    return True, expires_at


def is_llm_enabled() -> bool:
    return llm_status()[0]


def set_llm_enabled(enabled: bool, hours: float | None = None) -> None:
    """Persist the switch atomically. ``hours`` sets an auto-off expiry."""
    state: dict = {"enabled": enabled}
    if enabled and hours is not None:
        state["until"] = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

    LLM_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LLM_SWITCH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, LLM_SWITCH_FILE)  # atomic: readers never see a partial file


def describe() -> str:
    enabled, expires_at = llm_status()
    if not enabled:
        if expires_at is not None:
            return f"LLM scoring is OFF (expired {expires_at:%Y-%m-%d %H:%M} UTC)"
        return "LLM scoring is OFF"
    if expires_at is None:
        return "LLM scoring is ON (no expiry)"
    remaining = expires_at - datetime.now(timezone.utc)
    minutes = int(remaining.total_seconds() // 60)
    return (
        f"LLM scoring is ON until {expires_at:%Y-%m-%d %H:%M} UTC "
        f"({minutes // 60}h {minutes % 60}m left)"
    )


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "on":
        try:
            hours = float(argv[2]) if len(argv) > 2 else LLM_DEFAULT_ON_HOURS
        except ValueError:
            print("HOURS must be a number, e.g. `on 2`")
            return 2
        if hours <= 0:
            print("HOURS must be positive")
            return 2
        set_llm_enabled(True, hours)
    elif command == "off":
        set_llm_enabled(False)
    elif command != "status":
        print("usage: llm_switch.py on [HOURS] | off | status")
        return 2
    print(describe())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
