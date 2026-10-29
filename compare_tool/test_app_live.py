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


def _two_tenants(app_mod, tmp_path):
    """GIFs in ds and ds2; user u (password p) may see only ds."""
    data = tmp_path / 'data'
    _gif(data / 'ds' / 'a_rec_' / 'all_epochs_view0_0-3.gif', n_frames=4)
    _gif(data / 'ds2' / 'b_rec_' / 'all_epochs_view0_0-3.gif', n_frames=4)
    app_mod.load_data_dir(str(data))
    app_mod._user_config = {'u': {'password': 'p', 'dirs': ['ds']}}
    ids = _ids(app_mod)
    return (ids['ds/a_rec_/all_epochs_view0_0-3.gif'],
            ids['ds2/b_rec_/all_epochs_view0_0-3.gif'],
            {'Authorization': 'Basic dTpw'})  # u:p


def test_prefix_grant_matches_whole_path_components(app_mod):
    app_mod._user_config = {'u': {'password': 'p', 'dirs': ['ls3231']}}
    assert app_mod._user_can_access('u', 'ls3231/a.gif')
    assert not app_mod._user_can_access('u', 'ls32310/a.gif')
    assert not app_mod._user_can_access('u', 'ls3231_extra/a.gif')


def test_epoch_config_rejects_markup(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    bad = {'experiments': {own: {'start': 0, 'end': '"><img src=x onerror=alert(1)>'}}}
    r = client.post('/api/epoch-config', json=bad, headers=auth)
    assert r.status_code == 400
    assert app_mod.experiments[own]['epoch_config'].get('end') != bad['experiments'][own]['end']

    ok = {'experiments': {own: {'start': '10', 'end': 13, 'step': None, 'raw_first': 'yes'}}}
    assert client.post('/api/epoch-config', json=ok, headers=auth).status_code == 200
    assert app_mod.experiments[own]['epoch_config'] == {
        'source': 'user', 'raw_first': False, 'start': 10, 'end': 13}


def test_cross_view_checks_access_and_owner(app_mod, tmp_path):
    own, foreign, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    r = client.post('/api/cross-view/start', headers=auth,
                    json={'candidates': [{'exp_id': foreign, 'epoch': 0}]})
    assert r.status_code == 403

    r = client.post('/api/cross-view/start', headers=auth,
                    json={'candidates': [{'exp_id': own, 'epoch': 0}]})
    assert r.status_code == 200
    job = r.get_json()['job_id']
    app_mod._user_config['v'] = {'password': 'p', 'dirs': ['*']}
    other = {'Authorization': 'Basic djpw'}  # v:p
    assert client.get(f'/api/cross-view/status/{job}', headers=other).status_code == 404
    assert client.get(f'/api/cross-view/status/{job}', headers=auth).status_code == 200


def test_bad_params_get_400_not_500(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    assert client.get(f'/api/metrics/{own}/1?r_o=abc', headers=auth).status_code == 400
    assert client.get(f'/api/metrics/{own}/1?roi=1,2', headers=auth).status_code == 400
    assert client.get(f'/api/metrics/{own}/1?r_o=0.52', headers=auth).status_code == 200
    r = client.get(f'/api/frame/{own}/1?w=100000', headers=auth)
    assert r.status_code == 200
    full_w = app_mod._load(own)[1][0].shape[1]
    assert Image.open(__import__('io').BytesIO(r.data)).size[0] == full_w
    assert client.get(f'/api/frame/{own}/1?w=x', headers=auth).status_code == 400


def test_cross_site_post_refused(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    body = {'experiments': {own: {'start': 0}}}
    r = client.post('/api/epoch-config', json=body,
                    headers={**auth, 'Sec-Fetch-Site': 'cross-site'})
    assert r.status_code == 403
    r = client.post('/api/epoch-config', data=json.dumps(body),
                    headers={**auth, 'Content-Type': 'text/plain'})
    assert r.status_code == 400


def test_resume_returns_current_pair_to_owner_only(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    r = client.post('/api/tournament/start', headers=auth,
                    json={'experiments': [{'exp_id': own}]})
    sid = r.get_json()['session_id']
    state = client.get(f'/api/tournament/{sid}', headers=auth).get_json()
    assert state['pair'] == r.get_json()['pair'] and not state['done']
    assert state['selected'][0]['expId'] == own
    app_mod._user_config['v'] = {'password': 'p', 'dirs': ['*']}
    assert client.get(f'/api/tournament/{sid}', headers={'Authorization': 'Basic djpw'}).status_code == 404


def test_same_second_results_do_not_overwrite(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    for _ in range(2):
        sid = client.post('/api/tournament/start', headers=auth,
                          json={'experiments': [{'exp_id': own}]}).get_json()['session_id']
        assert client.post(f'/api/tournament/{sid}/finish', headers=auth).status_code == 200
    saved = list((tmp_path / 'results').rglob('tournament_*.tsv'))
    assert len(saved) == 2


def test_thumbnail_of_narrow_roi_stays_small(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    r = client.get(f'/api/frame/{own}/0?roi=0,0,1,8&w=2000', headers=auth)
    assert r.status_code == 200
    assert Image.open(__import__('io').BytesIO(r.data)).size == (1, 8)


def test_epoch_config_is_all_or_nothing(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    before = dict(app_mod.experiments[own]['epoch_config'])
    body = {'experiments': {own: {'start': 5}, 'x': {'start': 'bad'}}}
    assert client.post('/api/epoch-config', json=body, headers=auth).status_code == 400
    assert app_mod.experiments[own]['epoch_config'] == before


def test_finish_saves_even_if_a_gif_left(app_mod, tmp_path):
    own, _, auth = _two_tenants(app_mod, tmp_path)
    client = app_mod.app.test_client()
    sid = client.post('/api/tournament/start', headers=auth,
                      json={'experiments': [{'exp_id': own}]}).get_json()['session_id']
    app_mod._metrics_cache.clear()
    app_mod.experiments.pop(own)  # what a rescan does when the GIF is replaced
    r = client.post(f'/api/tournament/{sid}/finish', headers=auth)
    assert r.status_code == 200
    assert list((tmp_path / 'results').rglob('tournament_*.tsv'))
