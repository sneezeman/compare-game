"""Unit tests for the data-dir diff and the saved-result parser."""

import os

from rescan import diff_scan, snapshot
from results_view import parse_result_tsv, split_full_label

NOW = 1000.0


def test_diff_detects_added_changed_removed():
    known = {'a.gif': (10, 1.0), 'b.gif': (20, 2.0), 'c.gif': (30, 3.0)}
    disk = {'a.gif': (10, 1.0), 'b.gif': (25, 5.0), 'd.gif': (40, 4.0)}
    assert diff_scan(known, disk, NOW) == (['d.gif'], ['b.gif'], ['c.gif'], [])


def test_diff_defers_files_still_being_written():
    disk = {'new.gif': (10, NOW - 5), 'old.gif': (10, NOW - 60)}
    assert diff_scan({}, disk, NOW) == (['old.gif'], [], [], ['new.gif'])


def test_diff_unchanged_is_empty():
    sigs = {'a.gif': (10, 1.0)}
    assert diff_scan(sigs, dict(sigs), NOW) == ([], [], [], [])


def test_snapshot_skips_hidden_and_non_gif(tmp_path):
    (tmp_path / 'ds' / 'exp').mkdir(parents=True)
    (tmp_path / 'ds' / 'exp' / 'v0.gif').write_bytes(b'x')
    (tmp_path / 'ds' / 'exp' / '.v1.gif.Xa12').write_bytes(b'x')
    (tmp_path / 'ds' / 'exp' / 'notes.txt').write_bytes(b'x')
    (tmp_path / '.trash').mkdir()
    (tmp_path / '.trash' / 'old.gif').write_bytes(b'x')
    assert list(snapshot(str(tmp_path))) == [os.path.join('ds', 'exp', 'v0.gif')]


TSV = """Compare Game — Tournament Results
Date: 2026-10-02T15:33:50.330020
User: artem
Model: Merge-sort tournament
GIFs: ls3405/T02_rec_/v1.gif, ls3405/T02_rec_/v2.gif
OTF radius: 0.5
ROI: 128,128,256,256
Total comparisons: 78

Top 3: ls3405/T02_rec_/v1.gif Ep.85, ls3405/T02_rec_/v1.gif RAW

Ranking Confidence
1\tls3405/T02_rec_/v1.gif Ep.85

Rank\tCandidate\tBrenner\tTenengrad
1\tls3405/T02_rec_/v1.gif Ep.85\t0.500000\t1.250000
2\tls3405/T02_rec_/v1.gif RAW\t0.400000\t1.000000

Metric vs Manual Rank (Spearman correlation)
Brenner\trho=1.0000\tp=0.0000
"""


def test_parse_result_tsv():
    r = parse_result_tsv(TSV)
    assert r['user'] == 'artem'
    assert r['roi'] == '128,128,256,256'
    assert r['gifs'] == ['ls3405/T02_rec_/v1.gif', 'ls3405/T02_rec_/v2.gif']
    assert r['metric_names'] == ['Brenner', 'Tenengrad']
    assert [c['full_label'] for c in r['ranking']] == [
        'ls3405/T02_rec_/v1.gif Ep.85', 'ls3405/T02_rec_/v1.gif RAW']
    assert r['ranking'][0]['values'] == [0.5, 1.25]


def test_parse_result_tsv_roi_none_and_no_table():
    r = parse_result_tsv('Date: 2026-01-01\nROI: none\n')
    assert r['roi'] == ''
    assert r['ranking'] == []


def test_split_full_label():
    assert split_full_label('a b/c.gif Ep.85') == ('a b/c.gif', 'Ep.85')
    assert split_full_label('x.gif RAW') == ('x.gif', 'RAW')
