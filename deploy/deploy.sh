#!/usr/bin/env bash
# Deploy option_archive to the worker host. Run from a dev checkout.
#
# Ships the app tree to /opt/option_archive/app via rsync --delete (an exact
# mirror), then reinstalls the package into the venv that lives OUTSIDE that tree.
# Does NOT start or enable a run — the nightly timer or a manual start does that
# (see setup-drogon.sh output).
set -euo pipefail

HOST="${OPTION_ARCHIVE_HOST:-drogon}"
APP_DIR=/opt/option_archive/app
VENV=/opt/option_archive/venv
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Deploying ${REPO_ROOT} -> ${HOST}:${APP_DIR}"

# --delete keeps APP_DIR an exact mirror of the checkout. Everything mutable lives
# OUTSIDE APP_DIR and is untouchable by --delete: the venv, /etc/option_archive/env,
# and .git. tests/ IS shipped: the on-host certification step runs the suite
# against the pinned 3.14.
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
echo "code shipped; the next timer firing (or 'systemctl start option-archive') runs it"
echo "deployed on \$(hostname); python \$(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
REMOTE

echo "Done. This script does NOT start a run. To run:"
echo "  nightly:  systemctl enable --now option-archive.timer"
echo "  one pass: systemctl start option-archive   (watch: journalctl -u option-archive -f)"
