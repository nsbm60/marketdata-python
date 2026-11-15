# Session handoff — big picture for a fresh session

Purpose: let a new session (or a new day) pick up without re-deriving the
whole platform. This is the mental model and the in-flight state, not a
commit log. Read this, then read the actual code/scripts named here — do
not act from this summary alone.

## What this is
Single-operator, theta-harvesting options trading platform. One person
(Neil) builds and runs it. Claude is the design/architecture authority;
Claude Code (CC) is the scoped implementer, driven by detailed briefs.
Rural Starlink in Mimbres NM — power and network are genuinely flaky, so
reproducibility-after-reboot and resilient reconnect are first-class
concerns, not polish.

## Stack
- Scala services: MDS (market data), CalcServer (reports/calc), IBServer
  (execution), UIServer. Repo: ~/Projects/Scala/MarketData.
- Python analytics/services: option_archive (backfill), pub/sub broker,
  greeks server (to build), portfolio optimizer. Repo:
  ~/Projects/Python/marketdata-python.
- React/TypeScript UI — display only, no domain knowledge (server owns
  domain decisions).
- ClickHouse on "drogon" (Linux box, 192.168.37.163). Execution+data via
  Interactive Brokers and Alpaca.
- Messaging: ZMQ — pub/sub for market data, DEALER/ROUTER for control.
  Preferred over REST for inter-service comms.

## Hosts
- drogon — the Linux server: ClickHouse, IB Gateway, the Python services,
  the discovery-router (discovery-router.jar as systemd discovery-router.service).
- mac-studio — Neil's Mac: development, a second discovery-router started
  by hand (not launchd — it errors if it starts before the network is up).

## Standing conventions (these constrain all future work)
- Discovery: ZMQ ServiceLocator for ALL processes (batch included).
  Advertisements carry LOCATION ONLY (host/ports) — never credentials.
  Credentials come from each side's deploy config (Scala config files;
  Python reads them from its systemd EnvironmentFile). The discovery
  ENGINE is frozen; non-standard shapes get their own per-service locator
  (like IBExecLocator, ClickHouseUtils) rather than changing the engine.
- Schema DDL: one CREATE per table, only in database_schema.sql (canonical).
  Migration files carry only idempotent ALTERs, applied by one stated
  command, never "remembered at restart". (migrations/option_archive_tables.sql
  is applied with: clickhouse-client --multiquery --database trading < that file.)
- Git: commit straight to master, push to origin routinely (master is the
  running system AND the off-machine backup). Branches only for risky work.
- Reconnect rule: a reconnect is never conditional — reconnect exactly like
  a fresh process start, same code path.
- Tests: Neil is a self-described test skeptic; don't over-impose test
  ceremony, but the price-selection and topic-builder work IS pinned by tests.

## Deploy models (THREE different ones — do not conflate)
1. Scala services: still run from the development checkout; restart picks up
   the working copy. (A cleanup to give them committed units + per-jar deploy
   is scoped-but-not-built — deploy is per-JAR, `deploy-jar <path>`, restart
   resolved from the units on drogon; service is an operational identity
   above the jar.)
2. option_archive (batch): ~/marketdata-python checkout, deploy.sh rsyncs
   Mac→drogon into /opt/option_archive/app, setup-drogon.sh bootstraps +
   certifies. This one DOES use rsync and runs from the Mac.
3. Python SERVICES (broker/greeks/optimizer) — /opt/marketdata-python/app is a
   FROZEN PLAIN COPY (no .git) the service runs from. Ongoing deploy is
   ./deploy-python on the MAC (mirrors ./deploy-jar): `git archive HEAD` →
   scp pack + deploy/deploy-python-remote.sh to ~/deploy-staging on drogon →
   `ssh -t … sudo` runs the remote script: unpack→/opt (local tar, no rsync,
   no git in /opt), reinstall into the shared venv, refresh units, write the
   deploy record next to app/, opt-in per-service restart. The drogon home
   checkout is no longer used for deploying. Rationale: a live service must
   not run from modifiable source. setup-drogon-services.sh bootstraps
   (one-time, on drogon). SERVICES.md documents it.

## The pub/sub bus (settled, broker deployed)
- XSUB/XPUB proxy (zmq.proxy), ~40 lines, stateless. Python. Advertises as
  "pubsubbus" with EXPLICIT xsub/xpub fields + pubSub, NO router (it has no
  control plane). This is LIVE and verified correct in the discovery-router log.
- Broker binds ipc:// (drogon-local) + tcp:// (off-box). Producers connect to
  XSUB; subscribers connect to XPUB.
- Migration is PRODUCER-side (handful of producers) not consumer-side (~46
  sites). First real migration: MDS publish → broker XSUB, CalcServer
  subscribe → broker XPUB. Not done yet.

## Greeks server (designed, not built)
- Pure pub/sub producer on the bus: subscribe price stream, compute greeks,
  publish md.option.greeks.*. Market-implied vol per option (invert price →
  IV → greeks via certified py_vollib pricer). IV is a PLUGGABLE input (future
  vol model slots in). Triggers: option tick (re-imply), underlying tick
  (coalesced), 5s heartbeat (theta is continuous). No greeks display off-hours.
- Snapshot CLI already built+committed (greeks_snapshot) for Fidelity
  comparison validation.

## Price-selection work (DONE, Scala, committed)
- Baseline-jump bug fixed: last-trade used as price only if it printed in the
  current session (sessionPrice gate on LiveOptionPrice), else quote mid.
- LiveEquityPrice.last is now Option[Double] — 0.0 sentinel killed; no
  synthetic last from quote mid; withQuote/withTrade enforce "quote never
  writes last". ~549 Scala tests pass; 2 pre-existing unrelated failures in
  MdsOpsSpec (wire-decode payload-type mismatch) — known, parked.

## Topic builder (DONE, both languages)
- Python topic_builder.py has the producer half; 14-row golden fixture,
  byte-identical cross-language, tested both sides.

## Archive (running)
- option_archive: single-program archive.py, ledger-as-state, crash-safe,
  ReplacingMergeTree. Trades + inline NBBO quotes (pool 16, session-gated
  price selection). Drain running on drogon, ~2024-Q2+ at last check.
  option_contract_asof weekly as-of reference. ~1B+ rows in option_trade.

## The research program (WHY the plumbing exists — the point of all of it)
The archive, greeks pipeline, and GPU are not ends in themselves; they are
the substrate for ML over historical options data, which in turn feeds the
optimizer that constructs the actual trading book. The dependency chain:

  backfill archive (trades + NBBO quotes, ~5yr)
    -> greeks pipeline: compute greeks ANALYTICALLY from market prices
       (in-house, CPU/py_vollib, Black-76). This is deterministic MATH, NOT ML.
       Output = versioned greeks tables in ClickHouse.
    -> ML TRAINS ON the greeks tables (RTX 5090 GPU). ML CONSUMES greeks; it
       does NOT produce them. What the ML's OUTPUT is (vol forecast? move
       probability? per-candidate edge? regime?) is NOT yet pinned here —
       ASK Neil; it is the left side of the ML->optimizer seam below.
    -> portfolio optimizer (CVXPY) constructs/rebalances the book
    -> IBServer combo/BAG orders stage the trade set for one-button execution

- WHY build the archive at all: to own a training corpus that OUTLASTS the
  vendor. Massive retains ~5yr rolling; Neil accretes older history locally
  before it rolls off, so the training archive grows beyond the vendor window.
  Massive is the permanent PRIMITIVES vendor (trades, quotes, reference) —
  greeks and all computed layers are in-house (vendor-exit plan was dropped;
  Massive is structurally required for the quote layer, Alpaca licenses none).
- Greeks from PRICES, not stored vendor greeks: forward via escrowed-dividend
  Black-76, IV via py_vollib, versioned in ClickHouse. Own solver certified
  (greeks validation PASSED). This is deliberate — the training target must be
  a defined, reproducible computation, not an undocumented vendor mark.
- GPU split: py_vollib greeks on CPU; the RTX 5090 is reserved for ML TRAINING
  on the greeks tables, not for generating them.
- A concrete first study is scoped+partially-implemented: an intraday theta
  study on NVDA 3-DTE Friday weeklies. (Theta harvesting is the strategy; this
  studies its intraday behavior.)
- A future vol-model hook already exists by design: greeks-server IV is a
  PLUGGABLE input, so a learned vol model (e.g. NVDA IV response to QQQ/SPY/SMH)
  slots in without re-architecting.

## The optimizer (the trade-construction end)
- Own Python server; shell exists, no functionality yet. CVXPY-based solver,
  detailed design done, through multiple red-team cycles.
- Constraint-categories framework established. Formulation bugs already caught:
  gamma penalty per-UNDERLYING not per-position; spread cost on the TRADE not
  the target; yield-floor and concentration constraints removed as redundant.
- Phased: Phase 1 dashboard built; Phase 2 candidate-pool architecture built.
- End state: optimizer prepares trade SETS (closes, opens, rolls) staged for
  review, initiated from a single button — executed via the IBServer combo/BAG
  order path (which is being built as a manual rolling tool first).

## OPEN & MOST IMPORTANT: the ML -> optimizer seam (UNDESIGNED)
Neil's stated top concern. As of 2026-10-18 this interface does NOT exist —
not in code, not in a plan doc. The optimizer package is empty scaffolding
(candidates/, constraints/, solver/ are 0-line __init__.py) plus toy CVXPY
exercises and the intraday theta study tool. The two endpoints (the ML model's OUTPUT — not greeks; greeks are analytic,
not ML — and the CVXPY optimizer design) are each worked on but UNCONNECTED,
and HOW an
ML output becomes an optimizer input is the single biggest undecided design
question in the program. The real fork:
  1. ML as the OBJECTIVE'S EDGE TERM — model emits a per-candidate expected
     return/edge; objective maximizes edge-weighted exposure minus the existing
     penalties (gamma per-underlying, trade-spread). Optimizer SIZES against the
     prediction -> model error is leveraged by sizing.
  2. ML as CANDIDATE SELECTION — model ranks/filters the Phase-2 candidate pool;
     optimizer does pure risk-constrained construction over a pre-screened set,
     NO learned edge inside the objective -> optimizer insulated from model
     error; ML can only admit/exclude, never oversize.
These have opposite risk profiles. (2) is the more defensible default for a
one-operator theta book on flaky infra (bounds ML influence to a gate, keeps
the solver's job "construct a risk-sane book from these candidates") — but it
is a deliberate call to make, NOT settled. A fresh session must treat this as
OPEN and design it with Neil before writing optimizer objective code. Do not
infer the coupling from the gamma/spread penalty terms (consistent with either).

## Open / owed
- Credential rotation owed (Massive API key printed in logs; CH password).
- SplitJob (Scala): mirror DividendJob, /v3/reference/splits → trading.split.
  Table exists, job not built.
- PubSubBusLocator (Scala): resolve xsub/xpub, following IBExecLocator pattern.
- Scala PUBSUB_BUS constant = "pubsubbus" before CalcServer resolves the bus.
- option_contract_asof catch-up seeder re-run after drain.
- Fidelity greeks comparison (deferred — their chain page was broken).

## How Neil works (apply this)
- Answer-first, expert level, no hedging/meta-commentary, single recommended
  path not a menu, no unsolicited advice. Wants to float ideas and develop
  them before a verdict. Dislikes concede-then-reframe. CRITICAL: read the
  actual repo before specifying or reviewing code — do not reason from memory
  about his code (this session drifted badly by doing exactly that).
