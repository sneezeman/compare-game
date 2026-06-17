"""Pure helpers for the idle metrics-precompute worker.

The worker loop and shared state live in app.py (they touch the experiments
table, the metrics store, and the request clock); these two functions are the
testable decision logic.
"""


def build_worklist(exps, present_pairs):
    """Ordered list of (exp_id, filename, epoch) still needing metrics.

    exps: iterable of dicts with exp_id, filename, num_epochs, raw_first.
    present_pairs: set of (filename, epoch) already in the store.
    Skips the RAW frame (epoch 0 when raw_first) and anything already present.
    Ordered by filename then epoch so every epoch of one GIF is computed while
    it is loaded (before LRU eviction), avoiding reload churn.
    """
    work = []
    for e in sorted(exps, key=lambda e: e["filename"]):
        start = 1 if e.get("raw_first") else 0
        for epoch in range(start, e["num_epochs"]):
            if (e["filename"], epoch) not in present_pairs:
                work.append((e["exp_id"], e["filename"], epoch))
    return work


def is_idle(now, last_request_time, idle_seconds=60):
    """True when no user request has arrived for at least idle_seconds."""
    return (now - last_request_time) >= idle_seconds
