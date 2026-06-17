"""Pure helpers for the idle metrics-precompute worker.

The worker loop and shared state live in app.py (they touch the experiments
table, the metrics store, and the request clock); these functions are the
testable decision logic.
"""

import re

_VIEW_RE = re.compile(r"view[012]")


def sibling_filenames(filename):
    """The same GIF across all three views (view0/view1/view2).

    A tournament runs on one view, but that processes the whole experiment, so
    its sibling views should be skipped too. Returns just {filename} when there
    is no view token.
    """
    if not _VIEW_RE.search(filename):
        return {filename}
    return {_VIEW_RE.sub("view%d" % v, filename, count=1) for v in (0, 1, 2)}


def build_worklist(exps, present_pairs, skip_filenames=None):
    """Ordered list of (exp_id, filename, epoch) still needing metrics.

    exps: iterable of dicts with exp_id, filename, num_epochs, raw_first.
    present_pairs: set of (filename, epoch) already in the store.
    skip_filenames: filenames to skip entirely (e.g. experiments that already
        have tournament results — no point warming what's been processed).
    Skips the RAW frame (epoch 0 when raw_first) and anything already present.
    Ordered by filename then epoch so every epoch of one GIF is computed while
    it is loaded (before LRU eviction), avoiding reload churn.
    """
    skip = skip_filenames or set()
    work = []
    for e in sorted(exps, key=lambda e: e["filename"]):
        if e["filename"] in skip:
            continue
        start = 1 if e.get("raw_first") else 0
        for epoch in range(start, e["num_epochs"]):
            if (e["filename"], epoch) not in present_pairs:
                work.append((e["exp_id"], e["filename"], epoch))
    return work


def is_idle(now, last_request_time, idle_seconds=60):
    """True when no user request has arrived for at least idle_seconds."""
    return (now - last_request_time) >= idle_seconds


def scan_is_settled(prev_count, curr_count):
    """True once the experiment count is non-zero and has stopped growing.

    The cold-start GIF scan populates `experiments` incrementally over minutes;
    polling this lets the worker wait for the full set before building its first
    work-list (so the first pass isn't a partial slice of a still-loading scan).
    """
    return curr_count > 0 and curr_count == prev_count
