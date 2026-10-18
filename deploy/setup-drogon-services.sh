#!/usr/bin/env bash
# One-time bootstrap for the long-running Python SERVICES on drogon (pub/sub broker
# now; greeks server and optimizer to come). Run as root, from a checkout of this repo:
#   git clone <repo> && cd marketdata-python && sudo ./deploy/setup-drogon-services.sh
#
# These are distinct from option_archive (a batch job at /opt/option_archive). All
# the services are the SAME repo run at different entry points, so they share ONE app
# tree and ONE venv under /opt/marketdata-python; a service is just a systemd unit
# (its own ExecStart / env / RuntimeDirectory). See deploy/SERVICES.md.
#
# Creates the mdapps service user (reused if present), the shared app/venv/config
# dirs, the venv on the pinned /usr/bin/python3.14, installs EVERY managed unit, does
# a FIRST code install, and as its final act certifies each service. It does NOT
# enable or start anything — matching the archive discipline (don't start before you
# look). Idempotent — safe to re-run. Ongoing updates use deploy-services.sh.
set -euo pipefail

SERVICE_USER="${SERVICE_USER:-mdapps}"
PYTHON="${PYTHON:-/usr/bin/python3.14}"   # PINNED interpreter (deadsnakes on drogon)
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
CONFIG_DIR=/etc/marketdata-python

# The managed services. Adding one = drop deploy/<name>.service (+ optional
# deploy/<name>.env.example) and add its <name> here. No new script.
SERVICES=(pubsub-broker)

# Per-service certification: the check that must pass on-host before we call the box
# good. A service with no entry here is installed but not certified (logged).
certify_service() {
  case "$1" in
    pubsub-broker) ( cd "${APP_DIR}" && sudo -u "${SERVICE_USER}" "${VENV}/bin/python" -m pytest tests/service -q ) ;;
    # greeks-server) sudo -u "${SERVICE_USER}" env PYTHON="${VENV}/bin/python" bash "${APP_DIR}/scripts/check_greeks.sh" ;;
    # optimizer)     ... ;;
    *) echo ">>> no certification defined for '$1' — installed but not certified"; return 0 ;;
  esac
}

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }
[[ -x "${PYTHON}" ]] || { echo "pinned interpreter ${PYTHON} not found — install it first (deadsnakes)"; exit 1; }

# 1. service user. mdapps likely pre-exists (Scala batch + option_archive), so this
#    rarely fires — assert the home dir exists on disk regardless.
if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
fi
HOME_DIR="$(getent passwd "${SERVICE_USER}" | cut -d: -f6)"
[[ -n "${HOME_DIR}" ]] || { echo "cannot determine ${SERVICE_USER} home dir"; exit 1; }
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${HOME_DIR}"

# 2. shared directories, owned by the service user
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${APP_DIR}" "${VENV}"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0750 "${CONFIG_DIR}"

# 3. venv OUTSIDE the app tree, built with the pinned interpreter
"${PYTHON}" -m venv "${VENV}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${VENV}"
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q --upgrade pip

# 4. FIRST code install from this checkout (tests INCLUDED — on-host certification
#    runs them), then editable install with dev extras so deps + pytest/mypy land in
#    the venv. Same mirror-with-guards rule as deploy-services.sh.
rsync -a --delete \
  --exclude '.git/' --exclude 'venv/' --exclude 'data/' \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.db' \
  "${REPO_ROOT}/" "${APP_DIR}/"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}[dev]"

# 5. per-service config files — TEMPLATE ONLY (never real keys), 0600, mdapps-readable.
#    Only for services that ship a deploy/<name>.env.example; the broker ships none.
for svc in "${SERVICES[@]}"; do
  tmpl="${HERE}/${svc}.env.example"
  dest="${CONFIG_DIR}/${svc}.env"
  if [[ -f "${tmpl}" && ! -f "${dest}" ]]; then
    install -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0600 "${tmpl}" "${dest}"
    echo ">>> Created ${dest} from template — FILL IN the credentials (0600)."
  fi
done

# 6. install every managed unit + daemon-reload — NOT enabled, NOT started
for svc in "${SERVICES[@]}"; do
  install -m 0644 "${HERE}/${svc}.service" /etc/systemd/system/
done
systemctl daemon-reload

# 7. FINAL ACT: certify each service on the pinned interpreter. Fail loudly and stop
#    if any is not green — a box that can't run a service's suite does not proceed.
for svc in "${SERVICES[@]}"; do
  echo ">>> Certifying ${svc} with ${VENV}/bin/python ..."
  if ! certify_service "${svc}"; then
    echo "!!! CERTIFICATION FAILED for ${svc} on $(hostname). Do NOT enable it." >&2
    exit 1
  fi
done

cat <<NEXT
>>> Certification PASSED on $(hostname) (python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')).
>>> Setup complete. Nothing was started. Next:
      1) fill in any ${CONFIG_DIR}/<service>.env that was templated (the broker needs none)
      2) (ongoing updates) ./deploy/deploy-services.sh from a dev checkout
      3) enable a service on boot + start it now, e.g.:
           systemctl enable --now pubsub-broker     (watch: journalctl -u pubsub-broker -f)
NEXT
