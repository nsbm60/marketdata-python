#!/usr/bin/env bash
# Update the drogon Python services. Run ON drogon, as root, from the git checkout
# in your home dir:
#   cd ~/marketdata-python && git pull && sudo ./deploy/deploy-services.sh --restart pubsub-broker
#
# TWO TREES, deliberately. A service must NOT run from the git repo: the repo is
# modifiable source (git pull, branch switches, half-finished edits), and a live
# service should never execute whatever happens to be in a working tree. So:
#   - ~/marketdata-python        the git checkout — SOURCE, you edit/pull here
#   - /opt/marketdata-python/app a frozen COPY (plain files, no .git) the service RUNS from
# Deploy = copy the checkout into /opt (local cp on drogon — no rsync, no git in /opt),
# reinstall into the shared venv, refresh units, restart named services. /opt only ever
# changes on a deliberate deploy. All services share the one /opt tree + venv (SERVICES.md).
#
# Restart is opt-in and per-service — a ship must not silently bounce a running service:
#   sudo ./deploy/deploy-services.sh                          # copy + reinstall + refresh units
#   sudo ./deploy/deploy-services.sh --restart pubsub-broker  # ...and restart these
#   sudo ./deploy/deploy-services.sh --restart-all            # ...and restart all managed
# The broker is stateless (restart is instant + lossless); restart stateful services
# deliberately.
set -euo pipefail

APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
SERVICE_USER=mdapps

# The git checkout this script is run from (the home development repo). Its working
# tree is what gets copied into /opt — commit/pull before deploying so /opt mirrors a
# known state.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Keep in sync with setup-drogon-services.sh (validates --restart names, expands --restart-all).
SERVICES=(pubsub-broker greeks-server)

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
[[ -d "${REPO_ROOT}/.git" ]] || { echo "${REPO_ROOT} is not a git checkout — run this from your home repo"; exit 1; }
[[ -d "${APP_DIR}" ]] || { echo "${APP_DIR} missing — run setup-drogon-services.sh first"; exit 1; }

DEPLOY_REF="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '(no HEAD)')"

# Copy the checkout into /opt as a FROZEN PLAIN COPY (no .git — /opt is never a repo).
# A local tar-pack-and-unpack is the simplest no-rsync mirror: pack the checkout minus
# the source-only / mutable dirs, unpack into /opt. The venv is a sibling under /opt
# (not under app/) and is never touched. Overlay copy: a file deleted from the repo is
# NOT pruned from /opt — on a removal, clear the stale path by hand (rare).
echo "deploying ${REPO_ROOT} @ ${DEPLOY_REF} -> ${APP_DIR} (local copy)"
tar -C "${REPO_ROOT}" \
    --exclude='./.git' --exclude='./venv' --exclude='./data' \
    --exclude='__pycache__' --exclude='.mypy_cache' --exclude='.pytest_cache' \
    --exclude='./.claude' --exclude='./target' --exclude='*.pyc' \
    -cf - . | tar -C "${APP_DIR}" -xf -
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

# Reinstall (editable against /opt) so a dependency change is picked up.
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}"

# Refresh every managed unit from the deployed copy, then reload definitions.
for svc in "${SERVICES[@]}"; do
  install -m 0644 "${APP_DIR}/deploy/${svc}.service" /etc/systemd/system/
done
systemctl daemon-reload

for svc in "${RESTART[@]}"; do
  echo "restarting ${svc}"
  systemctl restart "${svc}"
done

echo "deployed ${DEPLOY_REF} to ${APP_DIR} on $(hostname); python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
if [[ ${#RESTART[@]} -eq 0 ]]; then
  echo "No service restarted. Restart onto new code when ready:"
  echo "  sudo systemctl restart pubsub-broker"
else
  echo "Restarted: ${RESTART[*]}"
fi
