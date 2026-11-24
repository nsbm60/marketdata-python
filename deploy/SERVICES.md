# Python deployment convention — hosts, services

This is the pattern for shipping the `marketdata-python` tree to its deploy targets and running the
long-running Python **services**. Follow it so each new service (and host) is a drop-in, not a redesign.

## Deploy targets (hosts)

`./deploy-python <host>` ships the committed tree to one of two machines. Each host's managed services
and pip extras live in **`deploy/hosts.sh`** — the one place, sourced by both `./deploy-python` (Mac)
and `deploy/deploy-python-remote.sh` (the target), so they never drift. A missing or unknown host exits
non-zero with the accepted list.

| Host | Role | Services | pip extras | Bootstrap |
|------|------|----------|------------|-----------|
| `drogon`   | the ClickHouse-side box — long-running services **and** the ClickHouse-side batch/ML jobs (rv, jumps, iv_curve, iv_event, vol_dataset builds) | `pubsub-broker`, `greeks-server` | none | `sudo ./deploy/setup-drogon-services.sh` |
| `balerion` | the GPU box — ML training and GPU batch work; **no services** | none | `ml` (GPU XGBoost) | `sudo PYTHON=/path/to/python3.14 ./deploy/setup-balerion.sh` |

Both hosts share the same `/opt/marketdata-python/{app,venv}` layout and the `mdapps` user. balerion
additionally points `app/data` at the big `/data` disk (`/data/marketdata-python`) for dataset/model
output, and installs no systemd unit.

## The one fact everything follows from

All these services are the **same `marketdata-python` repo** run at different entry
points (`python -m service.pubsub_broker`, `python -m greeks.…`, …). So they share
**one app tree and one venv**; a service is just a **systemd unit** (its own
`ExecStart`, env file, and `RuntimeDirectory`). Per-service trees would be N copies of
identical code and N venvs that drift — we don't do that.

Batch jobs are different: `option_archive` is a one-shot timer job and keeps its own
home at `/opt/option_archive` with its own `setup-drogon.sh` / `deploy.sh`. This
convention is for services only and does not touch it.

## Layout on drogon

| Path | What |
|------|------|
| `/opt/marketdata-python/app`  | a **frozen plain copy** of the committed tree (no `.git`); the service runs from here, never from the repo — updated only by a deliberate `./deploy-python` |
| `/opt/marketdata-python/deploy-record` | the last deploy's commit + deploy time, written next to `app/` by each `./deploy-python` |
| `/opt/marketdata-python/venv` | the shared venv, built on the pinned `/usr/bin/python3.14`, OUTSIDE the app tree |
| `/etc/marketdata-python/<service>.env` | per-service config/credentials, `0600`, `mdapps` (optional — the broker needs none) |
| `/etc/systemd/system/<service>.service` | one unit per service, installed from `app/deploy/<service>.service` |
| `/run/<service runtime dir>` | per-service `RuntimeDirectory` (e.g. `md-pubsub` for the broker's ipc sockets) |

The venv lives OUTSIDE `app/`, so a `git pull`/checkout reset never touches it. Runs
as `User=mdapps` (shared with the Scala batch and the archive).

## The scripts

- **`deploy/hosts.sh`** — the single source of per-host settings (`ACCEPTED_HOSTS`, and
  `host_config <host>` setting `HOST_SERVICES` + `HOST_EXTRAS`). Sourced by `./deploy-python` and by
  `deploy/deploy-python-remote.sh`. Adding a host or changing a host's services/extras happens here.
- **`setup-drogon-services.sh`** — one-time bootstrap for drogon, run as root from a
  checkout. Creates the user/dirs/venv, lays down the first plain copy of the checkout into
  `app/` (no `.git`), `pip install -e .[dev]`, installs every managed unit, `daemon-reload`,
  then **certifies each service**. It does **not** enable or start anything (don't start
  before you look). Idempotent.
- **`setup-balerion.sh`** — one-time bootstrap for balerion (the GPU box), run as root from a
  checkout with `PYTHON=/path/to/python3.14` set (**no default** — it exits if unset). Same
  user/dirs/venv, `pip install -e .[dev,ml]`, points `app/data` at `/data/marketdata-python`, checks
  the C++ toolchain, and as its FINAL ACT certifies CUDA is available to XGBoost (trains a tiny GPU
  `reg:quantileerror` model) and runs the `vol_dataset` tests. Installs **no** systemd unit. Idempotent.
- **`./deploy-python <host>`** — ongoing update, run **on the Mac from the repo** (mirrors the
  Scala `./deploy-jar`). Packs the committed tree with `git archive HEAD` (refuses if HEAD
  isn't a commit; a dirty working copy is never shipped), records the commit, scps the pack
  + the remote script + `hosts.sh` to `~/deploy-staging` on the host, then over `ssh -t … sudo` runs
  **`deploy/deploy-python-remote.sh`** there: unpack into `app/` (overlay copy, no `.git` in
  `/opt`), `pip install -e .[<extras>]`, refresh that host's unit files, `daemon-reload` (skipped on a
  host with no services), write the deploy record next to `app/`, and restart only the named services.
  The host's home checkout is not used.
- **`deploy/deploy-python-remote.sh`** — the one implementation of the deploy mechanics,
  staged and run on the host by `./deploy-python`. It takes the host as an argument and reads the
  host's services/extras from `hosts.sh`; `--restart` / `--restart-all` accept only that host's services.

The deploy path reads services from `hosts.sh`; `setup-drogon-services.sh` carries its own
`SERVICES=(…)` list for the bootstrap/certification concern (a separate step). Keep the drogon entries
consistent between the two.

## Adding a service (greeks, optimizer, …)

1. Add the entry point to the repo (`python -m <module>`).
2. Drop `deploy/<name>.service` (copy `pubsub-broker.service`; change `Description`,
   `ExecStart`, `RuntimeDirectory`, and the env path). Keep `User=mdapps`, the
   `/opt/marketdata-python/{app,venv}` paths, `Restart=always`, `WantedBy=multi-user.target`.
3. If it needs config/secrets, add `deploy/<name>.env.example` (template only, never
   real keys). `setup` installs it to `/etc/marketdata-python/<name>.env` at `0600`.
4. Add `<name>` to the host's `HOST_SERVICES` in **`deploy/hosts.sh`** (the deploy path) and to the
   `SERVICES=(…)` array in **`setup-drogon-services.sh`** (the bootstrap/certification path).
5. Add a certification command for it in `certify_service()` in `setup` (e.g.
   greeks → `scripts/check_greeks.sh`). The commented stubs are there.

No new scripts. That is the whole point of the shared pair.

## Boot & ordering

- **Enable on boot** per service: `systemctl enable --now <name>`. `setup` installs
  but never enables — enabling is the operator's explicit, reviewed step.
- **No startup ordering toward discovered services.** A service either advertises
  (UDP broadcast, retried) or resolves-with-retry (the ServiceLocator connector backs
  off and waits). Both converge under any boot order, so units carry only
  `After=/Wants=network-online.target` — never `After=` the discovery-router or another
  service. (The broker is advertise-only: the discovery-router can be down at boot and
  the broker still runs and forwards frames; its advertisement is seen once the router
  is up. Verified in `service/advertiser.py` — UDP `sendto` to the broadcast address
  succeeds with no receiver.) This is the service-resolution rule in
  `docs/architecture/principles.md`.

## Commands

The bootstrap and the `systemctl`/`journalctl` lines run ON drogon. The ongoing update
(`./deploy-python`) runs on the **Mac** from the repo. Each line is a standalone,
copy-paste-safe command.

```bash
# one-time bootstrap, as root, ON drogon from a checkout of this repo:
sudo ./deploy/setup-drogon-services.sh

# one-time bootstrap, as root, ON balerion from a checkout (pinned interpreter required, no default):
sudo PYTHON=/usr/bin/python3.14 ./deploy/setup-balerion.sh

# enable the broker on boot and start it now (on drogon):
sudo systemctl enable --now pubsub-broker

# watch it (on drogon):
journalctl -u pubsub-broker -f

# ongoing update — on the MAC, from the repo (commit first; ships committed HEAD). <host> is required:
# copy -> /opt + reinstall, restart nothing:
./deploy-python drogon

# ...and restart the broker (stateless, lossless):
./deploy-python drogon --restart pubsub-broker

# ...and restart every managed service on the host:
./deploy-python drogon --restart-all

# ship to the GPU box (no services, installs the ml extra):
./deploy-python balerion
```

## greeks-server

`python -m greeks_server` (spec: Scala `docs/specs/greeks-server-spec.md`). Needs
`CLICKHOUSE_PASSWORD` in `/etc/marketdata-python/greeks-server.env`. Certified by
`scripts/check_greeks.sh`. Phase 1 publishes on `shadow.md.option.greeks.*`
(`server.topic_prefix` in `config/greeks.yaml`).

```bash
# first time on drogon, from the pulled home checkout (creates the env template, installs the unit, certifies):
sudo ./deploy/setup-drogon-services.sh

# put the ClickHouse password in the env file:
sudoedit /etc/marketdata-python/greeks-server.env

# enable on boot and start now:
sudo systemctl enable --now greeks-server

# watch it:
journalctl -u greeks-server -f

# later updates (on the Mac, from the repo):
./deploy-python drogon --restart greeks-server
```
