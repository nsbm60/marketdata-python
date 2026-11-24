#!/usr/bin/env bash
# Runs ON the target (drogon or balerion), as root, sent by ./deploy-python on the Mac. Not installed:
# staged under ~/deploy-staging and run each time.
#   deploy-python-remote.sh <staging-dir> <host> [--restart <svc>...] [--restart-all]
#
# This is the ONE implementation of the Python deploy mechanics. The host's managed services and pip
# extras come from deploy/hosts.sh (staged beside this script), so the host list lives in one place.
# From the staged pack it:
#   - unpacks the committed tree into /opt/marketdata-python/app (a FROZEN PLAIN COPY — /opt is
#     never a git repo; overlay copy: a file deleted from the repo is NOT pruned from /opt);
#   - reinstalls (editable, with the host's extras) into the shared venv so a dependency change is
#     picked up;
#   - refreshes the unit files for THIS host's services and daemon-reloads (nothing on a host with no
#     services, and it says so);
#   - writes the deploy record (commit + deploy time, shipped from the Mac) next to the app;
#   - restarts ONLY the named services (a ship must not silently bounce a running service).
set -euo pipefail

APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
SERVICE_USER=mdapps
RECORD="${APP_DIR%/*}/deploy-record"   # next to the app: /opt/marketdata-python/deploy-record

staging="${1:-}"; shift || true
host="${1:-}"; shift || true
[[ -n "${staging}" && -n "${host}" ]] || {
  echo "usage: $0 <staging-dir> <host> [--restart <svc>...] [--restart-all]" >&2; exit 2; }

# Per-host services and extras from the single source (staged beside this script).
# shellcheck source=hosts.sh
source "${staging}/hosts.sh"
host_config "${host}" || { echo "unknown host: ${host} (accepted: ${ACCEPTED_HOSTS})" >&2; exit 2; }

# The restart list: explicit names (validated against THIS host's services) or --restart-all (expands
# to the host's services). A host with no services rejects every --restart name and restarts nothing.
RESTART=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart) shift; while [[ $# -gt 0 && "$1" != --* ]]; do RESTART+=("$1"); shift; done ;;
    --restart-all) RESTART=(${HOST_SERVICES[@]+"${HOST_SERVICES[@]}"}); shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
for svc in ${RESTART[@]+"${RESTART[@]}"}; do
  svc_in_host "${svc}" || {
    echo "not a managed service on ${host}: ${svc} (host services: ${HOST_SERVICES[*]:-none})" >&2
    exit 2; }
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo) — reinstalls units and restarts services"; exit 1; }
[[ -d "${APP_DIR}" ]] || {
  echo "${APP_DIR} missing — run the host's setup first (setup-drogon-services.sh / setup-balerion.sh)"; exit 1; }
[[ -f "${staging}/app.tar.gz" ]] || { echo "staged pack missing: ${staging}/app.tar.gz" >&2; exit 1; }
[[ -f "${staging}/deploy-record" ]] || { echo "staged deploy record missing: ${staging}/deploy-record" >&2; exit 1; }

# Unpack the committed tree into /opt. Overlay copy (no prune), frozen plain files (the pack from
# `git archive` carries no .git).
echo "unpacking app.tar.gz -> ${APP_DIR}"
tar -C "${APP_DIR}" -xzf "${staging}/app.tar.gz"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"

# Reinstall (editable against /opt) so a dependency change is picked up, with this host's extras (e.g.
# balerion's `ml` → GPU XGBoost). This also build_ext-compiles the native "Let's Be Rational" C++
# extension (greeks/native/lbr) — the host needs a C++ toolchain:
#   sudo apt-get install -y build-essential   # one-time; provides g++
if [[ -n "${HOST_EXTRAS}" ]]; then
  echo "pip install -e .[${HOST_EXTRAS}]"
  sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}[${HOST_EXTRAS}]"
else
  sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}"
fi

# Refresh the unit files for THIS host's services, then reload definitions. A host with no services
# installs nothing and does no daemon-reload.
if [[ -n "${HOST_SERVICES[*]:-}" ]]; then
  for svc in "${HOST_SERVICES[@]}"; do
    install -m 0644 "${APP_DIR}/deploy/${svc}.service" /etc/systemd/system/
  done
  systemctl daemon-reload
else
  echo "no managed services on ${host} — no unit files installed, no daemon-reload"
fi

# Deploy record next to the app (commit + deploy time), owned by the service user.
install -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0644 "${staging}/deploy-record" "${RECORD}"
rm -f "${staging}/app.tar.gz" "${staging}/deploy-record" "${staging}/hosts.sh"

echo "deployed to ${APP_DIR} on $(hostname); python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')"
echo "--- ${RECORD} ---"; cat "${RECORD}"

for svc in ${RESTART[@]+"${RESTART[@]}"}; do
  echo "restarting ${svc}"
  systemctl restart "${svc}"
done
if [[ ${#RESTART[@]} -eq 0 ]]; then
  echo "No service restarted."
else
  echo "Restarted: ${RESTART[*]}"
fi
