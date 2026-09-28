#!/usr/bin/env bash
# One-time bootstrap on drogon. Run as root, from a checkout of this repo:
#   git clone <repo> && cd marketdata-python && sudo ./deploy/setup-drogon.sh
#
# Creates everything deploy.sh assumes exists — the mdapps service user, the app /
# venv / config dirs, the venv built with the pinned /usr/bin/python3.14, the
# credentials file (template only; you fill in the real keys), and the systemd
# units + daemon-reload — then does a FIRST install of the code from this checkout
# (so the box is self-sufficient), and as its FINAL act runs the full test suite
# with the venv python, failing loudly if it is not green.
#
# It does NOT enable the timer or start a run. Idempotent — safe to re-run.
# (Ongoing code updates use deploy.sh from a dev checkout; this is the bootstrap.)
set -euo pipefail

SERVICE_USER="${SERVICE_USER:-mdapps}"
PYTHON="${PYTHON:-/usr/bin/python3.14}"   # PINNED interpreter (deadsnakes on drogon)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
APP_DIR=/opt/option_archive/app
VENV=/opt/option_archive/venv

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }
[[ -x "${PYTHON}" ]] || { echo "pinned interpreter ${PYTHON} not found — install it first (deadsnakes)"; exit 1; }

# 1. service user. mdapps likely pre-exists (Scala batch), so --create-home never
#    fires — assert the home dir exists on disk (the queue DB resolves under it).
if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
fi
HOME_DIR="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
[[ -n "${HOME_DIR}" ]] || { echo "cannot determine ${SERVICE_USER} home dir"; exit 1; }
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${HOME_DIR}"

# 2. directories, owned by the service user
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${APP_DIR}" "${VENV}"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 /etc/option_archive

# 3. venv OUTSIDE the app tree, built with the pinned interpreter
"${PYTHON}" -m venv "${VENV}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${VENV}"
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q --upgrade pip

# 4. FIRST code install from this checkout (tests INCLUDED — certification needs
#    them), then editable install with dev extras so deps + pytest/mypy land in the
#    venv. Same mirror-with-guards rule as deploy.sh.
rsync -a --delete \
  --exclude '.git/' --exclude 'venv/' --exclude 'data/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.db' \
  "${REPO_ROOT}/" "${APP_DIR}/"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}[dev]"

# 5. credentials file — TEMPLATE ONLY (never real keys), mode 0600, mdapps-readable
if [[ ! -f /etc/option_archive/env ]]; then
  install -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0600 "${HERE}/env.example" /etc/option_archive/env
  echo ">>> Created /etc/option_archive/env from template — FILL IN the credentials (0600)."
fi

# 6. unit + timer + daemon-reload — NOT enabled, NOT started
install -m 0644 "${HERE}/option-archive.service" /etc/systemd/system/
install -m 0644 "${HERE}/option-archive.timer" /etc/systemd/system/
systemctl daemon-reload

# 7. FINAL ACT: certify on the pinned interpreter (no-asyncio grep + mypy --strict +
#    pytest). Fail loudly and stop if not green — a box that can't run the suite
#    green does not proceed to seed/workers.
echo ">>> Certifying with ${VENV}/bin/python (hygiene + mypy --strict + pytest) ..."
if ! sudo -u "${SERVICE_USER}" env PYTHON="${VENV}/bin/python" bash "${APP_DIR}/scripts/check_option_archive.sh"; then
  echo "!!! CERTIFICATION FAILED on $(hostname) — the suite is not green. Do NOT seed or start workers." >&2
  exit 1
fi

cat <<NEXT
>>> Certification PASSED on $(hostname) (python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')).
>>> Setup complete. Nothing was started. Next:
      1) fill in /etc/option_archive/env  (Massive + Alpaca keys only)
      2) (ongoing updates) ./deploy/deploy.sh from a dev checkout
      3) enable the nightly run:  systemctl enable --now option-archive.timer
         or run one pass now:     systemctl start option-archive   (watch: journalctl -u option-archive -f)
NEXT
