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
| `/opt/marketdata-python/app`  | a **git checkout** of the repo; updated by `git pull` (run-from-development-checkout model, like the Scala services) |
| `/opt/marketdata-python/venv` | the shared venv, built on the pinned `/usr/bin/python3.14`, OUTSIDE the app tree |
| `/etc/marketdata-python/<service>.env` | per-service config/credentials, `0600`, `mdapps` (optional — the broker needs none) |
| `/etc/systemd/system/<service>.service` | one unit per service, installed from `app/deploy/<service>.service` |
| `/run/<service runtime dir>` | per-service `RuntimeDirectory` (e.g. `md-pubsub` for the broker's ipc sockets) |

The venv lives OUTSIDE `app/`, so a `git pull`/checkout reset never touches it. Runs
as `User=mdapps` (shared with the Scala batch and the archive).

## The two scripts (one pair for ALL services)

- **`setup-drogon-services.sh`** — one-time bootstrap, run as root on drogon from a
  checkout. Creates the user/dirs/venv, clones the repo into `app/` if absent (so it's
  a git checkout), `pip install -e .[dev]`, installs every managed unit, `daemon-reload`,
  then **certifies each service**. It does **not** enable or start anything (don't start
  before you look). Idempotent.
- **`deploy-services.sh`** — ongoing update, run **ON drogon as root**: `git pull` the
  `app/` checkout (fast-forward only), `pip install -e`, refresh the unit files from the
  checkout, `daemon-reload`, and optionally restart named services. No rsync, no "from a
  dev checkout".

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

# ongoing update: pull + reinstall, restart nothing:
sudo /opt/marketdata-python/app/deploy/deploy-services.sh

# ...and restart the broker (stateless, lossless):
sudo /opt/marketdata-python/app/deploy/deploy-services.sh --restart pubsub-broker

# ...and restart every managed service:
sudo /opt/marketdata-python/app/deploy/deploy-services.sh --restart-all
```
