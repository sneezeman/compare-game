"""Unit tests for the persistent metrics store (pure SQLite, no Flask)."""

import metrics_store as ms

M = [("ssim", 0.9, True), ("mse", 0.01, False), ("entropy", 7.5, True)]


def test_put_get_roundtrip(tmp_path):
    conn = ms.connect(str(tmp_path / "m.sqlite"))
    assert ms.get(conn, "a/b.gif", 5, 0.5) is None
    ms.put(conn, "a/b.gif", 5, 0.5, M)
    got = ms.get(conn, "a/b.gif", 5, 0.5)
    assert got == M
    assert all(isinstance(t, tuple) for t in got)
    assert isinstance(got[0][2], bool)


def test_has(tmp_path):
    conn = ms.connect(str(tmp_path / "m.sqlite"))
    assert not ms.has(conn, "x.gif", 0, 0.5)
    ms.put(conn, "x.gif", 0, 0.5, M)
    assert ms.has(conn, "x.gif", 0, 0.5)
    assert not ms.has(conn, "x.gif", 1, 0.5)


def test_persists_across_reopen(tmp_path):
    db = str(tmp_path / "m.sqlite")
    c1 = ms.connect(db)
    ms.put(c1, "p.gif", 2, 0.5, M)
    c1.close()
    c2 = ms.connect(db)
    assert ms.get(c2, "p.gif", 2, 0.5) == M


def test_present_frames_filtered_by_ro(tmp_path):
    conn = ms.connect(str(tmp_path / "m.sqlite"))
    ms.put(conn, "g1.gif", 1, 0.5, M)
    ms.put(conn, "g1.gif", 2, 0.5, M)
    ms.put(conn, "g2.gif", 1, 0.5, M)
    ms.put(conn, "g1.gif", 9, 0.7, M)  # different r_o — excluded at 0.5
    assert ms.present_frames(conn, 0.5) == {("g1.gif", 1), ("g1.gif", 2), ("g2.gif", 1)}
    assert ms.present_frames(conn, 0.7) == {("g1.gif", 9)}


def test_replace_overwrites(tmp_path):
    conn = ms.connect(str(tmp_path / "m.sqlite"))
    ms.put(conn, "r.gif", 0, 0.5, M)
    m2 = [("ssim", 0.1, True)]
    ms.put(conn, "r.gif", 0, 0.5, m2)
    assert ms.get(conn, "r.gif", 0, 0.5) == m2


def test_connect_creates_missing_db_and_schema(tmp_path):
    # Opening a brand-new path must just work (schema created on open).
    conn = ms.connect(str(tmp_path / "nested" / "new.sqlite"))
    assert ms.present_frames(conn, 0.5) == set()


def test_forget_drops_only_that_gif(tmp_path):
    conn = ms.connect(str(tmp_path / "m.sqlite"))
    ms.put(conn, "a.gif", 0, 0.5, M)
    ms.put(conn, "a.gif", 1, 0.5, M)
    ms.put(conn, "b.gif", 0, 0.5, M)
    ms.forget(conn, "a.gif")
    assert ms.present_frames(conn, 0.5) == {("b.gif", 0)}
