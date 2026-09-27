#!/usr/bin/env bash
# Owner-only switch for LLM (OpenAI) sentiment scoring on the deployed server.
# Run it over SSH from the project directory:
#
#   sudo ./llm.sh on        # score with the LLM for 4 hours, then auto-off
#   sudo ./llm.sh on 2      # ... for 2 hours
#   sudo ./llm.sh off       # stop immediately (even mid-cycle)
#   sudo ./llm.sh status
#
# The switch lives on the data volume, which only the worker can write; the
# public API and dashboard can read it but never change it.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 on [HOURS] | off | status" >&2
  exit 2
}

switch() {
  docker compose exec -T worker python src/llm_switch.py "$@"
}

case "${1:-status}" in
  on)
    # Only a positive number may follow; anything else is rejected before use.
    if [[ $# -ge 2 && ! "$2" =~ ^[0-9]+([.][0-9]+)?$ ]]; then usage; fi
    switch on ${2:+"$2"}
    # Restarting the worker runs a pipeline cycle immediately, so a demo shows
    # fresh scores within minutes instead of waiting for the next interval.
    echo "Restarting the worker to score new headlines now..."
    docker compose restart worker >/dev/null
    echo "Done. Follow progress with: docker compose logs -f worker"
    ;;
  off)    switch off ;;
  status) switch status ;;
  *)      usage ;;
esac
