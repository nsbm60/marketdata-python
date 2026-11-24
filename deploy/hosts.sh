#!/usr/bin/env bash
# Per-host deploy settings — the ONE place that knows each deploy target's managed services and pip
# extras. Sourced by BOTH ./deploy-python (on the Mac) and deploy/deploy-python-remote.sh (on the
# target), so the two never drift. bash 3.2-safe (the Mac's /bin/bash): no associative arrays.
#
#   drogon   — the ClickHouse-side box: runs the long-running services (pubsub-broker, greeks-server).
#   balerion — the GPU box for batch/ML work: NO services; installs the `ml` pip extra (GPU XGBoost).

ACCEPTED_HOSTS="drogon balerion"

# host_config <host>: sets HOST_SERVICES (indexed array — the host's managed systemd services) and
# HOST_EXTRAS (the pip extras name for `-e .[<extras>]`, '' for none). Returns non-zero for an
# unknown host so callers can print the accepted list and exit.
host_config() {
  HOST_SERVICES=()
  HOST_EXTRAS=""
  case "${1:-}" in
    drogon)   HOST_SERVICES=(pubsub-broker greeks-server) ;;
    balerion) HOST_EXTRAS="ml" ;;   # no services
    *) return 1 ;;
  esac
  return 0
}

# svc_in_host <svc>: true when <svc> is one of this host's managed services (call after host_config).
# Uses "[*]" with a :- default so it is safe for a host with no services under `set -u`.
svc_in_host() {
  case " ${HOST_SERVICES[*]:-} " in
    *" ${1} "*) return 0 ;;
    *) return 1 ;;
  esac
}
