#!/usr/bin/env bash
# Runs ON drogon, as root, sent by ./deploy-python on the Mac. Not installed: staged under
# ~/deploy-staging and run each time.
#   deploy-python-remote.sh <staging-dir> [--restart <svc>...] [--restart-all]
#
# This is the ONE implementation of the Python deploy mechanics (it subsumes the old
# deploy-services.sh, which ran on drogon from the home checkout — no longer used for deploying).
# From the staged pack it:
#   - unpacks the committed tree into /opt/marketdata-python/app (a FROZEN PLAIN COPY — /opt is
#     never a git repo; the same overlay-copy rule the old deploy-services.sh used: a file deleted
#     from the repo is NOT pruned from /opt, clear a stale path by hand on removal, rare);
#   - reinstalls (editable) into the shared venv so a dependency change is picked up;
#   - refreshes every managed unit file and daemon-reloads;
#   - writes the deploy record (commit + deploy time, shipped from the Mac) next to the app;
#   - restarts ONLY the named services (a ship must not silently bounce a running service).
set -euo pipefail

APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
SERVICE_USER=mdapps
RECORD="${APP_DIR%/*}/deploy-record"   # next to the app: /opt/marketdata-python/deploy-record

# The managed services: the only names --restart accepts, and what --restart-all expands to. Their
# unit files live in the deployed tree under deploy/<svc>.service. (setup-drogon-services.sh carries
# the same list for the one-time bootstrap.)
SERVICES=(pubsub-broker greeks-server)

staging="${1:-}"; shift || true
[[ -n "${staging}" ]] || { echo "usage: $0 <staging-dir> [--restart <svc>...] [--restart-all]" >&2; exit 2; }

RESTART=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) shift; while [[ $# -gt 0 && "$1" != --* ]]; do RESTART+=("$1"); shift; done ;;
    --restart-all) RESTART=("${SERVICES[@]}"); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for svc in "${RESTART[@]}"; do
  if ! printf '%s\n' "${SERVICES[@]}" | grep -qx "${svc}"; then
    echo "not a managed service: ${svc} (known: ${SERVICES[*]})" >&2; exit 2
  fi
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo) — reinstalls units and restarts services"; exit 1; }
[[ -d "${APP_DIR}" ]] || { echo "${APP_DIR} missing — run setup-drogon-services.sh first"; exit 1; }
[[ -f "${staging}/app.tar.gz" ]] || { echo "staged pack missing: ${staging}/app.tar.gz" >&2; exit 1; }
[[ -f "${staging}/deploy-record" ]] || { echo "staged deploy record missing: ${staging}/deploy-record" >&2; exit 1; }

# Unpack the committed tree into /opt. Overlay copy (no prune), frozen plain files (the pack from
# `git archive` carries no .git).
echo "unpacking app.tar.gz -> ${APP_DIR}"
tar -C "${APP_DIR}" -xzf "${staging}/app.tar.gz"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

# Reinstall (editable against /opt) so a dependency change is picked up.
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}"

# Refresh every managed unit from the freshly deployed copy, then reload definitions.
for svc in "${SERVICES[@]}"; do
  install -m 0644 "${APP_DIR}/deploy/${svc}.service" /etc/systemd/system/
done
systemctl daemon-reload

# Deploy record next to the app (commit + deploy time), owned by the service user.
install -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0644 "${staging}/deploy-record" "${RECORD}"
rm -f "${staging}/app.tar.gz" "${staging}/deploy-record"

echo "deployed to ${APP_DIR} on $(hostname); python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
echo "--- ${RECORD} ---"; cat "${RECORD}"

for svc in "${RESTART[@]}"; do
  echo "restarting ${svc}"
  systemctl restart "${svc}"
done
if [[ ${#RESTART[@]} -eq 0 ]]; then
  echo "No service restarted. Restart onto new code when ready, e.g.:"
  echo "  sudo systemctl restart pubsub-broker"
else
  echo "Restarted: ${RESTART[*]}"
fi
