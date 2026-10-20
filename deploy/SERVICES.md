# drogon Python services — deployment convention

This is the pattern for long-running Python **services** on drogon: the pub/sub
broker now, the greeks server and optimizer next. Follow it so each new service is a
drop-in, not a redesign.

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
| `/opt/marketdata-python/app`  | a **frozen plain copy** of the checkout (no `.git`); the service runs from here, never from the repo — updated only by a deliberate `deploy-services.sh` |
| `/opt/marketdata-python/venv` | the shared venv, built on the pinned `/usr/bin/python3.14`, OUTSIDE the app tree |
| `/etc/marketdata-python/<service>.env` | per-service config/credentials, `0600`, `mdapps` (optional — the broker needs none) |
| `/etc/systemd/system/<service>.service` | one unit per service, installed from `app/deploy/<service>.service` |
| `/run/<service runtime dir>` | per-service `RuntimeDirectory` (e.g. `md-pubsub` for the broker's ipc sockets) |

The venv lives OUTSIDE `app/`, so a `git pull`/checkout reset never touches it. Runs
as `User=mdapps` (shared with the Scala batch and the archive).

## The two scripts (one pair for ALL services)

- **`setup-drogon-services.sh`** — one-time bootstrap, run as root on drogon from a
  checkout. Creates the user/dirs/venv, lays down the first plain copy of the checkout into
  `app/` (no `.git`), `pip install -e .[dev]`, installs every managed unit, `daemon-reload`,
  then **certifies each service**. It does **not** enable or start anything (don't start
  before you look). Idempotent.
- **`deploy-services.sh`** — ongoing update, run **ON drogon as root, from the home
  checkout**: you `git pull` the home checkout yourself, then this copies that checkout
  into `app/` (a plain local copy — no rsync, no `.git` in `/opt`), `pip install -e`,
  refreshes the unit files, `daemon-reload`, and optionally restarts named services. The
  service never runs from the repo; `/opt` changes only on a deliberate deploy.

Both carry a `SERVICES=(…)` array. It is the single list of what we manage.

## Adding a service (greeks, optimizer, …)

1. Add the entry point to the repo (`python -m <module>`).
2. Drop `deploy/<name>.service` (copy `pubsub-broker.service`; change `Description`,
   `ExecStart`, `RuntimeDirectory`, and the env path). Keep `User=mdapps`, the
   `/opt/marketdata-python/{app,venv}` paths, `Restart=always`, `WantedBy=multi-user.target`.
3. If it needs config/secrets, add `deploy/<name>.env.example` (template only, never
   real keys). `setup` installs it to `/etc/marketdata-python/<name>.env` at `0600`.
4. Add `<name>` to the `SERVICES=(…)` array in **both** scripts.
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

All run ON drogon. Each line is a standalone, copy-paste-safe command.

```bash
# one-time bootstrap, as root, from a checkout of this repo:
sudo ./deploy/setup-drogon-services.sh

# enable the broker on boot and start it now:
sudo systemctl enable --now pubsub-broker

# watch it:
journalctl -u pubsub-broker -f

# ongoing update — run from the HOME checkout after pulling it:
cd ~/marketdata-python && git pull

# copy -> /opt + reinstall, restart nothing:
sudo ./deploy/deploy-services.sh

# ...and restart the broker (stateless, lossless):
sudo ./deploy/deploy-services.sh --restart pubsub-broker

# ...and restart every managed service:
sudo ./deploy/deploy-services.sh --restart-all
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

# later updates:
sudo ./deploy/deploy-services.sh --restart greeks-server
```
