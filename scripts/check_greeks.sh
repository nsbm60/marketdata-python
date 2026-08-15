#!/usr/bin/env bash
# CI hygiene for the greeks package (plan PR0).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== forbid asyncio/threading under greeks/ =="
if grep -RInE '^(from|import) (asyncio|threading)\b' greeks/ --include='*.py'; then
  echo "ERROR: asyncio/threading imports are forbidden under greeks/" >&2
  exit 1
fi
echo "ok"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/venv/bin/python" ]]; then
    PYTHON="$ROOT/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi

echo "== mypy --strict (greeks) =="
"$PYTHON" -m mypy --strict greeks
echo "ok"

echo "== pytest tests/greeks =="
"$PYTHON" -m pytest tests/greeks -q
echo "ok"

echo "All greeks checks passed."
