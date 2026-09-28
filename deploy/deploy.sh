#!/usr/bin/env bash
# Deploy option_archive to the worker host. Run from a dev checkout.
#
# Ships the app tree to /opt/option_archive/app via rsync --delete (an exact
# mirror), then reinstalls the package into the venv that lives OUTSIDE that tree.
# Does NOT start or enable workers — that is a deliberate manual step after the
# queue is seeded (see setup-drogon.sh output).
set -euo pipefail

HOST="${OPTION_ARCHIVE_HOST:-drogon}"
APP_DIR=/opt/option_archive/app
VENV=/opt/option_archive/venv
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Restart is OPT-IN: a config-only deploy must not kill in-flight day-downloads
# and re-fetch them at Massive's throttle. Default = ship code, leave workers on
# the old code until their next natural (or your explicit) restart.
DEPLOY_RESTART="${DEPLOY_RESTART:-0}"

echo "Deploying ${REPO_ROOT} -> ${HOST}:${APP_DIR}"

# --delete keeps APP_DIR an exact mirror of the checkout. Everything mutable lives
# OUTSIDE APP_DIR and is untouchable by --delete: the venv, /etc/option_archive/env,
# .git, and the SQLite queue DB (the config's queue_db_path resolves under the
# service user's home — outside the repo per the queue-path ruling). tests/ IS
# shipped: the on-host certification step runs the suite against the pinned 3.14.
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.db' \
  "${REPO_ROOT}/" "${HOST}:${APP_DIR}/"

# Reinstall (editable) so a dependency change (e.g. boto3) is picked up, and reload
# unit definitions (daemon-reload does NOT restart running instances). Restarting
# workers onto the new code is opt-in (DEPLOY_RESTART=1) so a config-only ship does
# not interrupt an in-flight day-download. The pinned interpreter is the venv's
# python, never the system 'python'.
ssh "${HOST}" bash -s <<REMOTE
set -euo pipefail
"${VENV}/bin/pip" install -q -e "${APP_DIR}"
sudo systemctl daemon-reload
if [[ "${DEPLOY_RESTART}" == "1" ]]; then
  sudo systemctl try-restart 'option_archive-worker@*' || true
  echo "restarted running workers onto the new code"
else
  echo "code shipped; workers running OLD code until their next restart (re-run with DEPLOY_RESTART=1 to restart now)"
fi
echo "deployed on \$(hostname); python \$(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
REMOTE

echo "Done. This script does NOT start workers. To bring the fleet up:"
echo "  1) seed:  systemctl start option_archive-seed   (watch: journalctl -u option_archive-seed -f)"
echo "  2) start: systemctl enable --now option_archive-worker@{1,2}   # ONLY after seed finishes and the enqueued= count is checked"
