"""Integration tests: live rescan, users.json reload, reopening a saved result."""

import json
import os
import time

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def app_mod(tmp_path, monkeypatch):
    monkeypatch.setenv('COMPARE_METRICS_DB', str(tmp_path / 'm.sqlite'))
    import app as app_mod
    app_mod.experiments.clear()
    app_mod.tournaments.clear()
    app_mod._unreadable_gifs.clear()
    app_mod._metrics_db = app_mod.metrics_store.connect(str(tmp_path / 'm.sqlite'))
    app_mod.results_dir = str(tmp_path / 'results')
    app_mod._user_config = {}
    yield app_mod
    app_mod.experiments.clear()
    app_mod._user_config = {}


def _gif(path, n_frames=3, age=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(np.full((8, 8), 40 * i, dtype=np.uint8)) for i in range(n_frames)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    t = time.time() - age
    os.utime(path, (t, t))


def _ids(app_mod):
    return {e['filename']: eid for eid, e in app_mod.experiments.items()}


def test_rescan_adds_keeps_ids_and_removes(app_mod, tmp_path):
    data = tmp_path / 'data'
    _gif(data / 'ds' / 'T000_rec_' / 'all_epochs_view0_0-2.gif')
    app_mod.load_data_dir(str(data))
    before = _ids(app_mod)

    _gif(data / 'ds' / 'T001_rec_' / 'all_epochs_view0_0-2.gif')
    _gif(data / 'ds' / 'T002_rec_' / 'all_epochs_view0_0-2.gif', age=0)  # still uploading
    applied = app_mod.rescan_data_dir(str(data))
    after = _ids(app_mod)
    assert applied == '+1 new, 1 still uploading'
    assert after[next(iter(before))] == next(iter(before.values()))  # id kept
    assert len(after) == 2

    os.remove(data / 'ds' / 'T000_rec_' / 'all_epochs_view0_0-2.gif')
    assert app_mod.rescan_data_dir(str(data)).startswith('-1 removed')
    assert 'ds/T000_rec_/all_epochs_view0_0-2.gif' not in _ids(app_mod)


def test_rescan_keeps_gif_used_by_running_tournament(app_mod, tmp_path):
    data = tmp_path / 'data'
    gif = data / 'ds' / 'T000_rec_' / 'all_epochs_view0_0-2.gif'
    _gif(gif)
    app_mod.load_data_dir(str(data))
    eid = next(iter(app_mod.experiments))
    t = app_mod.Tournament([0, 1])
    app_mod.tournaments['s1'] = {'tournament': t, 'selected_exp_ids': [eid], 'candidates': []}
    os.remove(gif)
    assert app_mod.rescan_data_dir(str(data)) == '1 waiting for a running tournament'
    assert eid in app_mod.experiments
    t.force_finish()
    assert app_mod.rescan_data_dir(str(data)) == '-1 removed'
    assert eid not in app_mod.experiments


def test_rescan_retries_unreadable_only_after_change(app_mod, tmp_path):
    data = tmp_path / 'data'
    bad = data / 'ds' / 'x.gif'
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b'not a gif')
    t = time.time() - 60
    os.utime(bad, (t, t))
    assert app_mod.rescan_data_dir(str(data)) == ''
    assert 'ds/x.gif' in app_mod._unreadable_gifs
    assert app_mod.rescan_data_dir(str(data)) == ''
    _gif(bad)
    assert app_mod.rescan_data_dir(str(data)) == '+1 new'


def test_reload_users_keeps_config_on_bad_file(app_mod, tmp_path):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'a': {'password': 'p', 'dirs': ['ds']}}))
    assert app_mod.reload_users_config(str(users))
    users.write_text('{broken')
    assert not app_mod.reload_users_config(str(users))
    users.write_text('{}')
    assert not app_mod.reload_users_config(str(users))
    assert list(app_mod._user_config) == ['a']


def test_view_saved_result(app_mod, tmp_path):
    data = tmp_path / 'data'
    _gif(data / 'ds' / 'T000_rec_' / 'all_epochs_view0_1-2_raw.gif')
    app_mod.load_data_dir(str(data))
    eid = next(iter(app_mod.experiments))
    labels = app_mod.experiments[eid]['epoch_labels']
    assert labels == ['RAW', 'Ep.1', 'Ep.2']

    g = 'ds/T000_rec_/all_epochs_view0_1-2_raw.gif'
    tsv = (f'Date: 2026-10-02T15:33:50\nUser: artem\nGIFs: {g}\nROI: 1,2,3,4\n\n'
           f'Rank\tCandidate\tBrenner\tTenengrad\n'
           f'1\t{g} Ep.2\t0.5\t1.0\n2\t{g} Ep.1\t0.4\t0.9\n3\tgone/x.gif Ep.7\t0.1\t0.2\n')
    (tmp_path / 'results' / 'ds').mkdir(parents=True)
    (tmp_path / 'results' / 'ds' / 'tournament_x.tsv').write_text(tsv)
    app_mod.scan_past_results()

    client = app_mod.app.test_client()
    r = client.get('/api/past-results/view/ds/tournament_x.tsv')
    assert r.status_code == 200
    d = r.get_json()
    assert d['user'] == 'artem' and d['roi'] == '1,2,3,4'
    assert [(c['exp_id'], c['epoch']) for c in d['ranking']] == [
        (eid, 2), (eid, 1), (None, None)]
    assert d['all_metrics'][0][1] == {'name': 'Tenengrad', 'value': 1.0,
                                      'higher_is_better': True}
    assert len(d['top3']) == 3

    assert client.get('/api/past-results/view/../etc/passwd').status_code in (400, 404)
    assert client.get('/api/past-results/ds/tournament_x.tsv').status_code == 200


def test_view_saved_result_access_denied(app_mod, tmp_path):
    (tmp_path / 'results' / 'ds').mkdir(parents=True)
    (tmp_path / 'results' / 'ds' / 't.tsv').write_text(
        'GIFs: ds/a.gif\n\nRank\tCandidate\tBrenner\n1\tds/a.gif Ep.1\t0.5\n')
    app_mod.scan_past_results()
    app_mod._user_config = {'u': {'password': 'p', 'dirs': ['other']}}
    client = app_mod.app.test_client()
    r = client.get('/api/past-results/view/ds/t.tsv',
                   headers={'Authorization': 'Basic dTpw'})  # u:p
    assert r.status_code == 403


def test_rescan_forgets_stored_metrics_of_readded_gif(app_mod, tmp_path):
    data = tmp_path / 'data'
    gif = data / 'ds' / 'a.gif'
    _gif(gif)
    app_mod.load_data_dir(str(data))
    app_mod.metrics_store.put(app_mod._metrics_db, 'ds/a.gif', 0, 0.5, [('m', 1.0, True)])
    os.remove(gif)
    app_mod.rescan_data_dir(str(data))
    _gif(gif)
    app_mod.rescan_data_dir(str(data))
    assert app_mod.metrics_store.present_frames(app_mod._metrics_db, 0.5) == set()


def test_saved_result_dot_path_cannot_dodge_acl(app_mod, tmp_path):
    (tmp_path / 'results' / 'secret').mkdir(parents=True)
    (tmp_path / 'results' / 'secret' / 't.tsv').write_text(
        'GIFs: secret/a.gif\n\nRank\tCandidate\tm\n1\tsecret/a.gif Ep.1\t0.5\n')
    (tmp_path / 'results' / 'stray.tsv').write_text('GIFs: \n')
    app_mod.scan_past_results()
    app_mod._user_config = {'u': {'password': 'p', 'dirs': ['ls3231']}}
    client = app_mod.app.test_client()
    auth = {'Authorization': 'Basic dTpw'}  # u:p
    for url in ['/api/past-results/view/secret/./t.tsv',
                '/api/past-results/secret/./t.tsv',
                '/api/past-results/view/x/../secret/t.tsv',
                '/api/past-results/stray.tsv']:
        assert client.get(url, headers=auth).status_code == 403, url


def test_users_file_rejects_string_dirs(app_mod, tmp_path):
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'a': {'password': 'p', 'dirs': ['ds']}}))
    assert app_mod.reload_users_config(str(users))
    users.write_text(json.dumps({'a': {'password': 'p', 'dirs': 'ls3231'}}))
    assert not app_mod.reload_users_config(str(users))
    users.write_text(json.dumps({'a': {'password': 5}}))
    assert not app_mod.reload_users_config(str(users))
    assert app_mod._user_config['a']['dirs'] == ['ds']
