#!/usr/bin/env bash
# One-time bootstrap for balerion, the GPU box used for batch/ML work (vol_dataset builds, model
# training). Run as root, from a checkout of this repo, with the pinned interpreter set:
#   git clone <repo> && cd marketdata-python && sudo PYTHON=/usr/bin/python3.14 ./deploy/setup-balerion.sh
#
# balerion runs NO long-running services (those live on drogon — see deploy/SERVICES.md). It shares
# the same /opt/marketdata-python/{app,venv} layout and the mdapps user, installs the dev + ml pip
# extras (GPU XGBoost), points the app's data/ at the big /data disk, and — as its FINAL ACT —
# certifies that CUDA is actually available to XGBoost and that the vol_dataset tests pass. It does
# NOT install any systemd unit. Idempotent. Ongoing updates use `./deploy-python balerion` on the Mac.
set -euo pipefail

SERVICE_USER="${SERVICE_USER:-mdapps}"
# PINNED interpreter — NO default on balerion. Set PYTHON=/path/to/python3.14 explicitly.
PYTHON="${PYTHON:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
APP_DIR=/opt/marketdata-python/app
VENV=/opt/marketdata-python/venv
CONFIG_DIR=/etc/marketdata-python
DATA_DIR=/data/marketdata-python

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)"; exit 1; }
[[ -n "${PYTHON}" ]] || {
  echo "PYTHON is required (no default on balerion) — set the pinned interpreter, e.g.:"
  echo "  sudo PYTHON=/usr/bin/python3.14 ./deploy/setup-balerion.sh"
  exit 1; }
[[ -x "${PYTHON}" ]] || { echo "PYTHON=${PYTHON} is not executable — point it at the pinned python3.14"; exit 1; }

# C++ toolchain: the greeks native extension (greeks/native/lbr) build_ext-compiles on install.
command -v g++ >/dev/null 2>&1 || {
  echo "C++ toolchain missing (needed to build the greeks native extension on install):"
  echo "  sudo apt-get install -y build-essential"
  exit 1; }

# 1. service user (shared with drogon's mdapps if this box ever had it; usually created fresh here).
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

# 4. first FROZEN PLAIN COPY of the checkout into app/ (no .git; data/ excluded — it becomes a
#    symlink below). Ongoing updates use ./deploy-python balerion (git archive HEAD -> /opt).
tar -C "${REPO_ROOT}" \
    --exclude='./.git' --exclude='./venv' --exclude='./data' \
    --exclude='__pycache__' --exclude='.mypy_cache' --exclude='.pytest_cache' \
    --exclude='./.claude' --exclude='./target' --exclude='*.pyc' \
    -cf - . | tar -C "${APP_DIR}" -xf -
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${APP_DIR}"
# editable install with dev + ml extras (pytest/mypy + GPU XGBoost) into the pinned venv.
sudo -u "${SERVICE_USER}" "${VENV}/bin/pip" install -q -e "${APP_DIR}[dev,ml]"

# 5. data: the ML outputs (vol_dataset Parquet, ml/shared/paths.DATA_DIR = app/data) live on the big
#    /data disk. Create /data/marketdata-python (mdapps) and point app/data at it. Refuse to clobber a
#    real app/data with content (an empty real dir is replaced).
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${DATA_DIR}"
APP_DATA="${APP_DIR}/data"
if [[ -e "${APP_DATA}" && ! -L "${APP_DATA}" ]]; then
  if [[ -d "${APP_DATA}" && -z "$(ls -A "${APP_DATA}" 2>/dev/null)" ]]; then
    rmdir "${APP_DATA}"
  else
    echo "${APP_DATA} already exists as a real directory with content — refusing to replace it with a"
    echo "symlink. Move it aside (its contents belong under ${DATA_DIR}) and re-run."
    exit 1
  fi
fi
ln -sfn "${DATA_DIR}" "${APP_DATA}"
chown -h "${SERVICE_USER}:${SERVICE_USER}" "${APP_DATA}"

# 6. FINAL ACT: certify the GPU is usable by XGBoost, then run the vol_dataset tests.
echo ">>> nvidia-smi:"
nvidia-smi

echo ">>> XGBoost CUDA check (as ${SERVICE_USER}) — trains a tiny reg:quantileerror model on the GPU ..."
sudo -u "${SERVICE_USER}" "${VENV}/bin/python" - <<'PY'
import json
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(256, 4))
y = X[:, 0] + rng.normal(size=256) * 0.1
dtrain = xgb.QuantileDMatrix(X, label=y)
# device="cuda" raises here if CUDA is not available to XGBoost — which is the point.
booster = xgb.train(
    {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "device": "cuda", "tree_method": "hist"},
    dtrain, num_boost_round=5)
device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
print("XGBoost trained on device:", device)
assert "cuda" in device, f"expected a CUDA device, got {device!r} — CUDA is not available to XGBoost"
print("CUDA OK")
PY

echo ">>> vol_dataset tests with ${VENV}/bin/python ..."
( cd "${APP_DIR}" && sudo -u "${SERVICE_USER}" "${VENV}/bin/python" -m pytest tests/vol_dataset -q )

cat <<NEXT
>>> Bootstrap PASSED on $(hostname) (python $(${VENV}/bin/python -c 'import sys; print(sys.version.split()[0])')).
>>> balerion runs no services; nothing was installed as a unit or started.

Next steps — each command is on its own line:

  1) Put any batch credentials in place (0600, mdapps) if a job needs them, e.g. ClickHouse:
     sudoedit /etc/marketdata-python/greeks-server.env

  2) Build the vol dataset onto the /data disk (on balerion, as mdapps):
     sudo -u ${SERVICE_USER} ${VENV}/bin/python -m vol_dataset build 2016-01-01 2026-12-31

  Ongoing updates run on the Mac from the repo (ships committed HEAD, no balerion checkout):
     ./deploy-python balerion
NEXT
