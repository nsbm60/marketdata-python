#!/usr/bin/env bash
# One-time setup on drogon. Run as root (sudo). Idempotent — safe to re-run.
#
# Creates the service user, the app/venv/state/config dirs, the venv with a PINNED
# interpreter, the credentials file (from template), and installs the systemd units.
# It does NOT deploy code (that is deploy.sh from a dev checkout) and does NOT start
# workers (that is manual, after seeding).
set -euo pipefail

SERVICE_USER="${SERVICE_USER:-mdapps}"
PYTHON="${PYTHON:-/usr/bin/python3.14}"   # PINNED interpreter (deadsnakes on drogon)
HERE="$(cd "$(dirname "$0")" && pwd)"

id "${SERVICE_USER}" >/dev/null 2>&1 \
  || useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"

install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" /opt/option_archive/app /opt/option_archive/venv
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 /etc/option_archive

# venv OUTSIDE the app tree, built with the pinned interpreter.
"${PYTHON}" -m venv /opt/option_archive/venv
chown -R "${SERVICE_USER}:${SERVICE_USER}" /opt/option_archive/venv
sudo -u "${SERVICE_USER}" /opt/option_archive/venv/bin/pip install -q --upgrade pip

# Credentials file — created once from the template; fill in real values (0600).
if [[ ! -f /etc/option_archive/env ]]; then
  install -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0600 "${HERE}/env.example" /etc/option_archive/env
  echo "Created /etc/option_archive/env from template — FILL IN the credentials."
fi

install -m 0644 "${HERE}/option_archive-worker@.service" /etc/systemd/system/
install -m 0644 "${HERE}/option_archive-seed.service" /etc/systemd/system/
systemctl daemon-reload

cat <<'NEXT'
Setup complete. Next:
  1) fill in /etc/option_archive/env (credentials, mode 0600)
  2) from a dev checkout:  ./deploy/deploy.sh
  3) seed once:            systemctl start option_archive-seed
                           journalctl -u option_archive-seed -f     # watch, get the final task count
  4) start the fleet:      systemctl enable --now option_archive-worker@{1,2}
                           (2 instances x multipart-16 = 32 connections; the endpoint
                            stalls under load, so start at 2 and let ingest_log decide)
NEXT
