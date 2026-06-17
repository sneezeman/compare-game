"""Unit tests for the pure precompute helpers (work-list + idle gate)."""

from precompute import build_worklist, is_idle


def _exp(exp_id, filename, num_epochs, raw_first=False):
    return {"exp_id": exp_id, "filename": filename,
            "num_epochs": num_epochs, "raw_first": raw_first}


def test_worklist_all_epochs_when_empty_store():
    exps = [_exp("e1", "g.gif", 3)]
    assert build_worklist(exps, set()) == [
        ("e1", "g.gif", 0), ("e1", "g.gif", 1), ("e1", "g.gif", 2)]


def test_worklist_skips_raw_frame():
    exps = [_exp("e1", "g.gif", 3, raw_first=True)]
    # epoch 0 is the RAW frame → skipped
    assert build_worklist(exps, set()) == [("e1", "g.gif", 1), ("e1", "g.gif", 2)]


def test_worklist_excludes_present_pairs():
    exps = [_exp("e1", "g.gif", 3)]
    present = {("g.gif", 0), ("g.gif", 2)}
    assert build_worklist(exps, present) == [("e1", "g.gif", 1)]


def test_worklist_ordered_by_filename_then_epoch():
    exps = [_exp("e2", "b.gif", 2), _exp("e1", "a.gif", 2)]
    work = build_worklist(exps, set())
    assert work == [("e1", "a.gif", 0), ("e1", "a.gif", 1),
                    ("e2", "b.gif", 0), ("e2", "b.gif", 1)]


def test_worklist_empty_when_all_present():
    exps = [_exp("e1", "g.gif", 2)]
    assert build_worklist(exps, {("g.gif", 0), ("g.gif", 1)}) == []


def test_is_idle_gate():
    assert is_idle(now=1000.0, last_request_time=900.0, idle_seconds=60) is True
    assert is_idle(now=1000.0, last_request_time=970.0, idle_seconds=60) is False
    # exactly at the threshold counts as idle
    assert is_idle(now=1000.0, last_request_time=940.0, idle_seconds=60) is True
