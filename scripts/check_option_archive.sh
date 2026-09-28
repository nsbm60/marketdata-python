#!/usr/bin/env bash
# CI hygiene for the option_archive package (spec PR0). Mirrors check_greeks.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== forbid asyncio/threading under option_archive/ =="
if grep -RInE '^(from|import) (asyncio|threading)\b' option_archive/ --include='*.py'; then
  echo "ERROR: asyncio/threading imports are forbidden under option_archive/ (process fleet only)" >&2
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

echo "== import smoke: archive entry module =="
# Import the entry point with the venv python so a MISSING RUNTIME DEP fails setup
# here, loudly, rather than at `systemctl start`. Importing does NOT run main() —
# the __name__ == "__main__" guard is False under import.
"$PYTHON" -c "import option_archive.archive; import option_archive.__main__"
echo "ok"

echo "== mypy --strict (option_archive) =="
"$PYTHON" -m mypy --strict option_archive
echo "ok"

echo "== pytest tests/option_archive =="
"$PYTHON" -m pytest tests/option_archive -q
echo "ok"

echo "All option_archive checks passed."
