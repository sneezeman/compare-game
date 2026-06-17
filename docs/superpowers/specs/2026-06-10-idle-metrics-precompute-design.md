# Idle metrics precompute — design

**Date:** 2026-06-10
**Status:** approved (design)

## Problem

Whole-frame image-quality metrics are computed on demand (pre-filter screen,
cross-view consensus, tournament ranking), so users wait behind a "Computing
metrics…" bar. The cache holding them is **in-memory, bounded to 5000 entries,
and wiped on every restart** — and the dataset (513 GIFs, ~8k non-RAW
(GIF, epoch) pairs) both exceeds the cap and is re-uploaded/restarted often.

## Goal

When the VPS is idle, precompute whole-frame metrics for every GIF/epoch in the
background and persist them to disk, so the wait disappears and the work survives
restarts. Never compete with active users.

## Decisions (confirmed)

- **Persist to disk** (SQLite), keyed by GIF **filename** (stable across scans;
  `exp_id` is reassigned each scan), so precompute runs once per GIF ever.
- **Scope:** whole-frame (no ROI) at the **default `r_o` (0.5)**, every **non-RAW**
  epoch of every GIF. Custom-ROI / non-default-`r_o` metrics stay on-demand
  (can't be known ahead).
- **Idle = quiet period, user-first:** run only after ~60 s with no user request;
  pause the instant any request arrives; resume when quiet again.

## Non-goals

- Precomputing ROI-specific or non-default-`r_o` metrics.
- A UI for the precompute (a status log line / optional status endpoint only).
- Distributed/multi-worker computation (single-worker gunicorn stays).

## Architecture

### 1. Persistent store — `metrics_store.py` (pure, no Flask)
SQLite at `COMPARE_METRICS_DB` (default `/opt/compare-game/metrics_cache.sqlite`,
gitignored). WAL mode for concurrent reads; writes serialized by a module lock.
```
CREATE TABLE metrics (
  gif_filename TEXT, epoch INTEGER, r_o REAL,
  metrics_json TEXT NOT NULL, computed_at TEXT,
  PRIMARY KEY (gif_filename, epoch, r_o)
);
```
Interface (all pure functions over a connection/path):
- `get(filename, epoch, r_o) -> list | None`
- `put(filename, epoch, r_o, metrics) -> None`
- `has(filename, epoch, r_o) -> bool`
- `present_pairs(r_o) -> set[(filename, epoch)]`  (for the work-list)

`metrics` is the same list-of-`(name, value, higher_is_better)` the app already
caches; stored as JSON.

### 2. Cache integration (`_get_cached_metrics`)
Add an L2 tier for **whole-frame, default-`r_o`** requests only:
in-memory → **persistent store** (translate `exp_id`→filename) → compute.
Any whole-frame/default-`r_o` compute (precompute **or** a live pre-filter/
cross-view request) write-throughs to the store. ROI / non-default `r_o` stay
in-memory-only, unchanged. This keeps one warming path for both the worker and
live traffic.

### 3. Idle precompute worker — `_precompute_worker` daemon thread
- `before_request` stamps `_last_request_time` (skip `/health`).
- Build a work-list of `(exp_id, filename, epoch)` **missing from the store**
  (`present_pairs` minus all non-RAW pairs), **ordered by GIF** so every epoch of
  a loaded GIF is computed before LRU eviction (avoids reload churn). RAW frame
  (epoch 0 when `raw_first`) is skipped.
- Loop: if `now − _last_request_time ≥ IDLE_SECONDS (60)`, compute the next item
  via the shared metrics path (which write-throughs to the store); **re-check the
  idle gate before every item**; if active, sleep briefly and re-check. Rebuild
  the work-list when exhausted (picks up newly uploaded GIFs) and otherwise sleep
  long. Log progress every N items (e.g. 50) and on completion.
- Starts as a `daemon=True` thread next to `_cross_view_thread`.

## Data flow

1. Boot → worker builds work-list from `present_pairs` vs current experiments.
2. Quiet ≥60 s → compute missing item → `compute_all` → write-through to store +
   in-memory cache.
3. User request → `before_request` stamps time → worker pauses before next item.
4. Live pre-filter/cross-view whole-frame request → store hit → instant; miss →
   compute + store (so the worker later finds fewer gaps).
5. Restart → store already populated → work-list near-empty (the payoff).

## Error handling

- Per-item `try/except`: log and skip a bad GIF/epoch; never kill the worker.
- DB locked / IO error on `put`: log, keep the in-memory result, continue.
- Corrupt/missing DB file: recreate schema on open.
- A stored entry whose GIF no longer exists is simply never read (harmless);
  optional periodic prune is out of scope.

## Testing

- `metrics_store`: put/get/has roundtrip; missing → None; survives reopen
  (persistence); `present_pairs` returns exactly stored pairs for an `r_o`.
- Work-list builder: excludes present pairs and the RAW frame; orders by GIF.
- Idle gate (fake clock): computes when `now − last_request ≥ 60`; pauses when a
  request was recent.
- Integration smoke (on the 513-GIF dataset, manual): worker fills the store while
  idle, `/api/prefilter-metric` whole-frame becomes a store hit, store survives a
  restart.

## Files touched (work in `/opt/compare-game` directly)

- `compare_tool/metrics_store.py` — new SQLite store (pure).
- `compare_tool/app.py` — L2 in `_get_cached_metrics`; `_last_request_time` in
  `before_request`; `_precompute_worker` + thread; work-list builder.
- `compare_tool/test_metrics_store.py`, `test_precompute.py` — new tests.
- `.gitignore` — ignore `metrics_cache.sqlite*`.
