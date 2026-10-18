#!/usr/bin/env bash
# Deploy the Python services tree to drogon. Run from a dev checkout.
#
# Ships the repo to /opt/marketdata-python/app via rsync --delete (an exact mirror),
# reinstalls the package into the shared venv (outside that tree), and reloads unit
# definitions. All services share this one tree + venv (see deploy/SERVICES.md).
#
# Restarting onto the new code is opt-in and per-service, because a ship should not
# silently bounce a running service:
#   ./deploy/deploy-services.sh                          # ship + reload only
#   ./deploy/deploy-services.sh --restart pubsub-broker  # ...and restart these
#   ./deploy/deploy-services.sh --restart-all            # ...and restart all managed
# The broker is stateless (restart is instant + lossless); restart stateful services
# deliberately.
set -euo pipefail

HOST="${MARKETDATA_HOST:-drogon}"
APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Keep in sync with setup-drogon-services.sh (used to validate --restart names and
# to expand --restart-all).
SERVICES=(pubsub-broker)

RESTART=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) shift; while [[ $# -gt 0 && "$1" != --* ]]; do RESTART+=("$1"); shift; done ;;
    --restart-all) RESTART=("${SERVICES[@]}"); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Validate restart targets are services we manage (a typo must not silently no-op).
for svc in "${RESTART[@]}"; do
  if ! printf '%s\n' "${SERVICES[@]}" | grep -qx "${svc}"; then
    echo "not a managed service: ${svc} (known: ${SERVICES[*]})" >&2; exit 2
  fi
done

echo "Deploying ${REPO_ROOT} -> ${HOST}:${APP_DIR}"
echo "Restart after ship: ${RESTART[*]:-(none)}"

# --delete keeps APP_DIR an exact mirror. Everything mutable lives OUTSIDE APP_DIR and
# is untouchable: the venv, /etc/marketdata-python/*.env, .git. tests/ IS shipped (the
# on-host checks run against the pinned 3.14).
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude 'data/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.db' \
  "${REPO_ROOT}/" "${HOST}:${APP_DIR}/"

RESTART_LIST="${RESTART[*]:-}"
ssh "${HOST}" bash -s <<REMOTE
set -euo pipefail
"${VENV}/bin/pip" install -q -e "${APP_DIR}"
sudo systemctl daemon-reload
for svc in ${RESTART_LIST}; do
  echo "restarting \${svc}"
  sudo systemctl restart "\${svc}"
done
echo "deployed on \$(hostname); python \$(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
REMOTE

if [[ ${#RESTART[@]} -eq 0 ]]; then
  echo "Done. Code shipped + units reloaded; no service restarted."
  echo "Restart onto new code when ready, e.g.: sudo systemctl restart pubsub-broker"
else
  echo "Done. Shipped and restarted: ${RESTART[*]}"
fi
