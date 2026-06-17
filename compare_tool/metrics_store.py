"""Disk-persisted store for whole-frame image-quality metrics.

Pure (no Flask) so it is unit-testable in isolation. Keyed by GIF *filename*
(stable across rescans, unlike the in-memory exp_id) + epoch + r_o. Holds the
same list-of-(name, value, higher_is_better) the app caches, as JSON.

Concurrency: WAL mode allows concurrent reads from request threads while the
precompute daemon writes; writes are serialized by a module-level lock.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

_write_lock = threading.Lock()


def connect(path):
    """Open (creating dirs + schema if needed) and return a connection.
    check_same_thread=False so the request threads and the precompute daemon
    can share one connection (writes are lock-guarded)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS metrics (
               gif_filename TEXT NOT NULL,
               epoch        INTEGER NOT NULL,
               r_o          REAL NOT NULL,
               metrics_json TEXT NOT NULL,
               computed_at  TEXT,
               PRIMARY KEY (gif_filename, epoch, r_o)
           )"""
    )
    conn.commit()
    return conn


def get(conn, filename, epoch, r_o):
    """Return the metrics list, or None if absent."""
    cur = conn.execute(
        "SELECT metrics_json FROM metrics WHERE gif_filename=? AND epoch=? AND r_o=?",
        (filename, int(epoch), float(r_o)),
    )
    row = cur.fetchone()
    if not row:
        return None
    return [tuple(item) for item in json.loads(row[0])]


def has(conn, filename, epoch, r_o):
    cur = conn.execute(
        "SELECT 1 FROM metrics WHERE gif_filename=? AND epoch=? AND r_o=?",
        (filename, int(epoch), float(r_o)),
    )
    return cur.fetchone() is not None


def put(conn, filename, epoch, r_o, metrics, computed_at=None):
    """Insert or replace the metrics for (filename, epoch, r_o)."""
    # Cast value to a plain float — compute_all may hand back numpy scalars,
    # which json.dumps cannot serialize.
    payload = json.dumps([[str(name), float(value), bool(hib)] for name, value, hib in metrics])
    stamp = computed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _write_lock:
        conn.execute(
            "INSERT OR REPLACE INTO metrics"
            " (gif_filename, epoch, r_o, metrics_json, computed_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (filename, int(epoch), float(r_o), payload, stamp),
        )
        conn.commit()


def present_pairs(conn, r_o):
    """Set of (filename, epoch) already stored at this r_o — for the work-list."""
    cur = conn.execute(
        "SELECT gif_filename, epoch FROM metrics WHERE r_o=?", (float(r_o),)
    )
    return {(name, epoch) for name, epoch in cur.fetchall()}
