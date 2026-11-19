#!/usr/bin/env bash
set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$HARNESS_DIR/.." && pwd)"
MODE="${1:-discuss}"
cd "$PROJECT_DIR"
case "$MODE" in
  discuss)
    echo "[Research Harness] DISCUSS mode: Claude plan permission mode (read-only)."
    exec claude --plugin-dir "$HARNESS_DIR" --permission-mode plan
    ;;
  execute)
    echo "[Research Harness] EXECUTE mode: standard Claude permissions. Get approval for costly jobs."
    exec claude --plugin-dir "$HARNESS_DIR"
    ;;
  *)
    echo "Usage: bash harness/scripts/start.sh [discuss|execute]" >&2
    exit 2
    ;;
esac
