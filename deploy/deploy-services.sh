#!/usr/bin/env bash
# Update the drogon Python services IN PLACE. Run ON drogon, as root:
#   sudo /opt/marketdata-python/app/deploy/deploy-services.sh --restart pubsub-broker
#
# The app tree is a git checkout (run-from-development-checkout model, like the Scala
# services) — update = `git pull` + reinstall into the shared venv + restart. No rsync,
# no "from a dev checkout". All services share this one tree + venv (see SERVICES.md).
#
# Restarting onto the new code is opt-in and per-service — a ship should not silently
# bounce a running service:
#   sudo ./deploy-services.sh                          # pull + reinstall + refresh units
#   sudo ./deploy-services.sh --restart pubsub-broker  # ...and restart these
#   sudo ./deploy-services.sh --restart-all            # ...and restart all managed
# The broker is stateless (restart is instant + lossless); restart stateful services
# deliberately.
set -euo pipefail

APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
SERVICE_USER=mdapps

# Keep in sync with setup-drogon-services.sh (validates --restart names, expands --restart-all).
SERVICES=(pubsub-broker)

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
[[ -d "${APP_DIR}/.git" ]] || { echo "${APP_DIR} is not a git checkout — run setup-drogon-services.sh first"; exit 1; }

# Pull the checkout as its owner (fast-forward only: fail loudly if it has diverged
# rather than create a merge commit on a deployed box).
sudo -u "${SERVICE_USER}" git -C "${APP_DIR}" pull --ff-only

# Reinstall (editable) so a dependency change is picked up.
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}"

# Refresh every managed unit from the checkout (a pull may have changed a .service
# file; the installed copy under /etc is not touched by git), then reload definitions.
for svc in "${SERVICES[@]}"; do
  install -m 0644 "${APP_DIR}/deploy/${svc}.service" /etc/systemd/system/
done
systemctl daemon-reload

for svc in "${RESTART[@]}"; do
  echo "restarting ${svc}"
  systemctl restart "${svc}"
done

echo "pulled + reinstalled on $(hostname); python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
if [[ ${#RESTART[@]} -eq 0 ]]; then
  echo "No service restarted. Restart onto new code when ready:"
  echo "  sudo systemctl restart pubsub-broker"
else
  echo "Restarted: ${RESTART[*]}"
fi
