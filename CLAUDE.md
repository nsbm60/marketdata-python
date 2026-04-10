# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Mandatory reading by subsystem:**

| Subsystem | Document |
|-----------|----------|
| Breakout detector | `docs/plans/breakout_detector_refactor_spec.md` |

## Related Projects

- **Scala Backend**: `/Users/nsm/Projects/Scala/MarketData` — MDS, CalcServer, IBServer
- **UI**: `/Users/nsm/Projects/js/marketdata-ui` — React trading UI

## Development Process

All significant changes follow this process:
1. Write a spec or plan document before any code is written
2. CC shows proposed signatures/design and waits for explicit approval
3. CC implements one step at a time, showing diffs before committing
4. Architectural decisions are made explicitly by the developer, not opaquely by CC

When given a spec or task, CC must always respond with a written implementation
plan first — listing each step, files to be changed, and key design decisions.
CC must wait for explicit approval of the plan before writing any code. This
applies even when the spec appears complete and unambiguous.

CC must never implement without an approved plan. CC must never proceed to the
next step without explicit approval of the current step.

## Cross-Repo Changes

When modifying table names, column names, or ZMQ topics, grep ALL THREE repos
(Scala, Python, UI) before making any changes. Do not assume a change is
isolated to one repo.

## Data Storage Rules

All ClickHouse timestamps MUST be stored as `DateTime64(3, 'UTC')`. Never use
a named timezone in a DateTime column. ET is a display/calculation convention,
not a storage convention. Convert at read time using
`toTimezone(ts, 'America/New_York')`.
