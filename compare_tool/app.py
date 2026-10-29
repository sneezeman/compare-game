"""
Flask server for the Compare Game — epoch comparison tool.
"""

import argparse
import hmac
import io
import itertools
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime

import glob as glob_mod

import numpy as np
from flask import (Flask, Response, g, jsonify, render_template, request,
                   send_file)
from PIL import Image

import metrics_store
from eval_export import (build_csv, build_sh, collect_eval_rows,
                         command_for_ranking, is_raw_label)
from precompute import (build_worklist, is_idle, scan_is_settled,
                        sibling_filenames)
from gif_loader import (compute_label_exclusion, compute_variability_map,
                        detect_label_present, extract_frames_u8, frame_to_float,
                        suggest_rois,
                        variability_to_heatmap_rgba)
from metrics import (compute_all, compute_metric_zscores, METRIC_GROUPS,
                     METRICS, ELIMINATION_METRICS, ELIMINATION_SETS,
                     ELIMINATION_METRIC_UNION, DEFAULT_ELIMINATION_SET)
from rescan import diff_scan, file_mtime, snapshot
from results_view import parse_result_tsv, split_full_label
from tournament import Tournament

app = Flask(__name__)

# Thread safety lock for shared mutable state
_state_lock = threading.RLock()  # reentrant — cleanup is called from locked contexts

# User config: {username: {password, dirs}} loaded from JSON or env var
_user_config = {}


def _read_users_file(config_path):
    """Parse users.json into {username: {password, dirs}}.

    Raises ValueError on a wrong shape. A string `dirs` would otherwise be
    matched character by character and widen access.
    """
    with open(config_path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('top level must be an object')
    users = {}
    for username, cfg in data.items():
        if not isinstance(cfg, dict) or not isinstance(cfg.get('password'), str):
            raise ValueError(f'user {username!r}: password must be a string')
        dirs = cfg.get('dirs', ['*'])
        if not isinstance(dirs, list) or not all(isinstance(d, str) for d in dirs):
            raise ValueError(f'user {username!r}: dirs must be a list of strings')
        users[username] = {'password': cfg['password'], 'dirs': dirs}
    return users


def load_users_config(config_path=None):
    """Load user config from JSON file or COMPARE_USERS env var."""
    global _user_config

    # Check env var for config path if not provided
    if not config_path:
        config_path = os.environ.get('COMPARE_USERS_CONFIG')

    if config_path and os.path.isfile(config_path):
        _user_config = _read_users_file(config_path)
        print(f'  Loaded {len(_user_config)} user(s) from {config_path}')
        return

    # Fallback: COMPARE_USERS env var (all users get full access)
    users = {}
    for pair in os.environ.get('COMPARE_USERS', '').split(','):
        pair = pair.strip()
        if ':' in pair:
            u, p = pair.split(':', 1)
            users[u] = {'password': p, 'dirs': ['*']}
    _user_config = users


def reload_users_config(config_path):
    """Swap in an edited users.json while the app runs.

    Keeps the current config when the file is missing, invalid or empty,
    because an empty config switches authentication off.
    """
    global _user_config
    try:
        users = _read_users_file(config_path)
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        print(f'  [rescan] users.json NOT reloaded ({e}); keeping {len(_user_config)} user(s)',
              flush=True)
        return False
    if not users:
        print('  [rescan] users.json is empty; keeping the current users', flush=True)
        return False
    _user_config = users  # one assignment, so no request sees a half-built config
    print(f'  [rescan] users.json reloaded: {len(users)} user(s)', flush=True)
    return True


def _user_can_access(username, filename):
    """Check if a user can access an experiment by its filename."""
    cfg = _user_config.get(username, {})
    dirs = cfg.get('dirs', [])
    if '*' in dirs:
        return True

    # Match whole path components, so 'ls3231' never grants 'ls32310/'.
    return any(filename == d.rstrip('/') or filename.startswith(d.rstrip('/') + '/')
               for d in dirs if d.strip('/'))


# Rate limiting / auto-ban for failed auth attempts
_fail_counts = {}  # ip -> (count, first_fail_time)
_banned_ips = {}   # ip -> ban_timestamp
_BAN_THRESHOLD = 10  # ban after this many failed attempts
_BAN_WINDOW = 300    # within this many seconds


@app.before_request
def check_auth():
    ip = request.remote_addr

    # Mark user activity so the idle precompute worker backs off. /health is a
    # liveness probe, not a user — don't let it keep the worker paused forever.
    if request.path != '/health':
        global _last_request_time
        _last_request_time = time.time()

    # Health check and logout — exempt from auth
    if request.path in ('/health', '/logout'):
        return

    # Browsers resend cached Basic credentials on cross-site POSTs (CSRF).
    if (request.method == 'POST'
            and request.headers.get('Sec-Fetch-Site', 'same-origin') not in ('same-origin', 'none')):
        return ('Cross-site request refused', 403)

    # Drop banned IPs (expire after 1 hour)
    with _state_lock:
        ban_time = _banned_ips.get(ip)
        if ban_time:
            if time.time() - ban_time < 3600:
                return ('', 403)
            else:
                del _banned_ips[ip]  # ban expired

    if not _user_config:
        return  # no auth configured

    auth = request.authorization
    if not auth or not auth.username or auth.username not in _user_config:
        user_cfg = None
    else:
        user_cfg = _user_config[auth.username]

    if not user_cfg or not auth.password or not hmac.compare_digest(user_cfg['password'], auth.password):
        # Track failures
        now = time.time()
        with _state_lock:
            count, first = _fail_counts.get(ip, (0, now))
            if now - first > _BAN_WINDOW:
                count, first = 0, now
            count += 1
            _fail_counts[ip] = (count, first)
            if count >= _BAN_THRESHOLD:
                _banned_ips[ip] = now
                print(f'  Banned {ip} after {count} failed auth attempts')
                return ('', 403)
        return ('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Compare Game"'})

    g.username = auth.username

@app.after_request
def security_headers(resp):
    # Inline handlers still need 'unsafe-inline'. The policy's value is that
    # an injected script can't load code from, or send data to, another host.
    resp.headers.setdefault('Content-Security-Policy',
                            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                            "style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
                            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                            "form-action 'self'")
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    return resp


@app.route('/logout')
def logout():
    """Force the browser to drop its cached Basic-auth credentials."""
    return ('Logged out. <a href="/">Log in again</a>', 401,
            {'WWW-Authenticate': 'Basic realm="Compare Game"'})


def _check_exp_access(exp_id):
    """Check if current user can access this experiment. Returns (exp, None) or (None, (error_response, code))."""
    exp = experiments.get(exp_id)
    if not exp:
        return None, (jsonify(error='Experiment not found'), 404)
    username = getattr(g, 'username', None)
    if username and not _user_can_access(username, exp['filename']):
        return None, (jsonify(error='Access denied'), 403)
    return exp, None


def _json_body():
    """The request's JSON object, or {}.

    Only application/json is parsed, so a cross-site form can't post here
    without a CORS preflight.
    """
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _parse_r_o(value):
    """r_o snapped to the UI's 0.05 grid in [0.05, 1.0], or None if invalid.

    Snapping bounds the metrics cache to 20 keys per frame and ROI.
    """
    try:
        r = round(float(value) * 20) / 20
    except (TypeError, ValueError, OverflowError):
        return None
    return r if 0.05 <= r <= 1.0 else None


def _parse_roi(value):
    """ROI as a canonical 'x,y,w,h' string, '' for none, or None if invalid."""
    if not value:
        return ''
    try:
        x, y, w, h = (int(p) for p in str(value).split(','))
    except ValueError:
        return None
    if min(x, y) < 0 or min(w, h) < 1 or max(x, y, w, h) > 100_000:
        return None
    return f'{x},{y},{w},{h}'


def _bad_params(r_o, roi):
    """400 response when r_o or roi failed to parse, else None."""
    if r_o is None:
        return jsonify(error='Invalid r_o'), 400
    if roi is None:
        return jsonify(error='Invalid roi'), 400
    return None


def _format_exp_label(filename):
    """Extract a readable short label from an experiment filename.
    'ls3639/NM0029/finetune/NM0029_HT_100nm_T008_0001_rec__from_NM0029_HT_100nm_T000_0001_rec__140/all_epochs_view1_141-150.gif'
    → 'T008←T000@140 view1'
    """
    parts = filename.replace('\\', '/').split('/')
    gif_name = parts[-1] if parts else filename
    dir_name = parts[-2] if len(parts) >= 2 else ''

    # Extract view from gif name
    view = ''
    vm = re.search(r'view(\d+)', gif_name)
    if vm:
        view = f' view{vm.group(1)}'

    # Finetuning: ..._T001_0001_rec__from_..._T000_0001_rec__155
    ft = re.search(r'_?(T\d+)(?:_\d+)?_rec__from_.+?_(T\d+)(?:_\d+)?_rec__(\d+)', dir_name)
    if ft:
        return f'{ft.group(1)}\u2190{ft.group(2)}@{ft.group(3)}{view}'

    # Training tile: ..._T000_0001_rec_
    tr = re.search(r'_?(T\d+)(?:_\d+)?_rec_?$', dir_name)
    if tr:
        return f'{tr.group(1)}{view}'

    return gif_name


def _cleanup_old_tournaments():
    """Remove tournaments older than 24 hours and stale rate-limit entries."""
    now = time.time()
    cutoff = datetime.now().timestamp() - 86400
    with _state_lock:
        to_remove = [sid for sid, td in tournaments.items()
                     if td.get('created_at', 0) < cutoff]
        for sid in to_remove:
            del tournaments[sid]
        # Clean up stale fail counts and expired bans
        stale = [ip for ip, (count, first) in _fail_counts.items()
                 if now - first > _BAN_WINDOW * 2]
        for ip in stale:
            del _fail_counts[ip]
        expired_bans = [ip for ip, t in _banned_ips.items() if now - t > 3600]
        for ip in expired_bans:
            del _banned_ips[ip]


# In-memory store: exp_id -> experiment data
experiments = {}
# Tournament store: session_id -> tournament data
tournaments = {}
# Metrics cache: (exp_id, epoch, roi, r_o) -> metrics list
_metrics_cache = {}
_load_locks = {}  # exp_id -> lock held while that GIF decodes
# LRU tracking: exp_id -> last_access_time (for frame eviction)
_access_times = {}
_MAX_LOADED = 3  # max experiments with frames in memory at once
_MAX_METRICS_CACHE = 5000  # max entries in metrics cache

# --- Idle metrics precompute (disk-persisted whole-frame metrics) ---
DEFAULT_R_O = 0.5  # whole-frame metrics at this r_o are persisted to the store
_metrics_db = metrics_store.connect(
    os.environ.get('COMPARE_METRICS_DB', '/opt/compare-game/metrics_cache.sqlite'))
_last_request_time = time.time()  # stamped by before_request; gates the worker
_PRECOMPUTE_IDLE_SECONDS = 60
# Past results: [{filename, date, gifs, roi, top3, path}, ...]
past_results = []
# Filenames that have completed comparisons: {gif_filename: user_name}
completed_gifs = {}
# Results save directory (set in __main__)
results_dir = 'results'


# ---------------------------------------------------------------------------
# Epoch config helpers
# ---------------------------------------------------------------------------

def parse_epoch_config_from_name(name):
    """Parse epoch config from filename/dirname.

    Supported patterns:
    - all_epochs_view0_0-45_raw.gif -> start=0, end=45, raw_first=True
    - all_epochs_view0_5-180.gif   -> start=5, end=180
    - E{start}-{end}-{step}        -> explicit with optional step
    - '_raw' or '_raw.' in name     -> first frame is RAW
    """
    config = {}

    # Try E-prefix pattern first (e.g. E100-110 or E0-180-5)
    m = re.search(r'E(\d+)-(\d+)(?:-(\d+))?', name)
    if m:
        config['start'] = int(m.group(1))
        config['end'] = int(m.group(2))
        if m.group(3):
            config['step'] = int(m.group(3))
    else:
        # Try underscore-delimited range in filename (e.g. _0-45_ or _5-180.)
        # Only match in the basename to avoid false positives from directory names
        basename = os.path.basename(name)
        m = re.search(r'_(\d+)-(\d+)(?:_|\.)', basename)
        if m:
            config['start'] = int(m.group(1))
            config['end'] = int(m.group(2))

    # Detect RAW: _raw_ or _raw. in filename
    if re.search(r'[_.]raw[_.]', name, re.IGNORECASE) or name.lower().endswith('_raw.gif'):
        config['raw_first'] = True

    return config if config else None


def _clean_epoch_config(posted):
    """A posted epoch config reduced to bounded ints and a bool, or None if invalid."""
    if not isinstance(posted, dict):
        return None
    config = {'source': 'user', 'raw_first': posted.get('raw_first') is True}
    for key, lo in (('start', 0), ('end', 0), ('step', 1)):
        val = posted.get(key)
        if val is None or val == '':
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float, str)):
            return None
        try:
            val = int(val)
        except (ValueError, OverflowError):
            return None
        if not lo <= val <= 100_000:
            return None
        config[key] = val
    return config


def generate_epoch_labels(num_epochs, start=None, end=None, step=None, raw_first=False):
    """Generate epoch labels for display."""
    if start is None:
        return [f'Ep.{i + 1}' for i in range(num_epochs)]

    effective_epochs = num_epochs - (1 if raw_first else 0)
    if step is None and end is not None and effective_epochs > 1:
        step = (end - start) // (effective_epochs - 1)
    if step is None or step < 1:
        step = 1

    labels = []
    for i in range(num_epochs):
        if raw_first and i == 0:
            labels.append('RAW')
        else:
            epoch_idx = i - (1 if raw_first else 0)
            labels.append(f'Ep.{start + epoch_idx * step}')
    return labels


# ---------------------------------------------------------------------------
# Cached metrics
# ---------------------------------------------------------------------------

def _eval_command_payload(ranking):
    """Eval command for a finished tournament, plus why it may not be the winner.

    Walks the ranking rather than taking only the top entry: a RAW frame (the
    un-denoised input) can win, and it has no epoch or checkpoint to evaluate.
    In that case we surface the best actual epoch and say what happened, instead
    of rendering nothing and leaving the user guessing.
    """
    labels = [c.get('full_label') for c in (ranking or [])]
    cmd, skipped = command_for_ranking(labels)
    return {
        'eval_command': cmd,
        'eval_skipped': skipped,
        'eval_winner_is_raw': bool(labels) and is_raw_label(labels[0]),
    }


def _get_cached_metrics(exp_id, epoch, roi_str, r_o):
    """Get metrics with caching.

    L1 = in-memory cache (all ROIs / r_o). L2 = persistent disk store, but only
    for whole-frame (no ROI) at the default r_o — those are what the idle
    precompute warms. An L2 hit also skips the expensive GIF decode entirely.
    """
    key = (exp_id, epoch, roi_str or '', r_o)
    with _state_lock:
        if key in _metrics_cache:
            return _metrics_cache[key]

    whole_frame = (not roi_str) and r_o == DEFAULT_R_O
    meta = experiments.get(exp_id)
    if meta is None:
        return None

    # L2: persistent store (whole-frame, default r_o) — no frame load on a hit.
    if whole_frame and meta.get('filename') and 0 <= epoch < meta.get('num_epochs', 0):
        stored = metrics_store.get(_metrics_db, meta['filename'], epoch, r_o)
        if stored is not None:
            with _state_lock:
                _metrics_cache[key] = stored
            return stored

    loaded = _load(exp_id)
    if not loaded or not 0 <= epoch < len(loaded[1]):
        return None
    exp, frames, _ = loaded

    img = frame_to_float(_apply_roi(frames[epoch], roi_str))
    results = compute_all(img, r_o=r_o)
    with _state_lock:
        if len(_metrics_cache) >= _MAX_METRICS_CACHE:
            # Evict oldest half
            keys = list(_metrics_cache.keys())
            for k in keys[:len(keys) // 2]:
                del _metrics_cache[k]
        _metrics_cache[key] = results
    # Write-through to the persistent store for whole-frame default-r_o results,
    # so the worker (and future restarts) find them already warm.
    # Skip it when a rescan dropped this entry meanwhile (the GIF was replaced),
    # so old pixels' metrics never land under the filename again.
    if whole_frame and exp.get('filename'):
        try:
            with _state_lock:
                if experiments.get(exp_id) is exp:
                    metrics_store.put(_metrics_db, exp['filename'], epoch, r_o, results)
        except Exception as e:
            print(f'  metrics_store put failed for {exp["filename"]} ep{epoch}: {e}')
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health')
def health():
    with _state_lock:
        loaded = sum(1 for e in experiments.values() if 'frames' in e)
        total = len(experiments)
    return jsonify(status='ok', experiments=total, loaded=loaded)


@app.template_global('asset_version')
def _asset_version(filename):
    """Return an mtime-based cache-bust token for a static file."""
    try:
        path = os.path.join(app.static_folder, filename)
        return str(int(os.path.getmtime(path)))
    except OSError:
        return '0'


@app.route('/')
def index():
    # Pass the authenticated name so the UI can skip asking for it.
    resp = app.make_response(
        render_template('index.html', username=getattr(g, 'username', '') or ''))
    # Force the HTML to always re-validate so new ?v=... tokens are picked up
    resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return resp


@app.route('/api/experiments')
def list_experiments():
    """List all loaded experiments with completion status, filtered by user access."""
    username = getattr(g, 'username', None)
    result = []
    with _state_lock:
        snapshot_items = list(experiments.items())
    for exp_id, exp in snapshot_items:
        if username and not _user_can_access(username, exp['filename']):
            continue
        result.append({
            'exp_id': exp_id,
            'filename': exp['filename'],
            'num_epochs': exp['num_epochs'],
            'height': exp['height'],
            'width': exp['width'],
            'has_results': exp['filename'] in completed_gifs,
            'done_by': completed_gifs.get(exp['filename'], ''),
        })
    return jsonify(experiments=result)


@app.route('/api/past-results')
def list_past_results():
    """List all past tournament results, filtered by user access."""
    username = getattr(g, 'username', None)
    if not username:
        return jsonify(results=past_results)

    filtered = []
    for r in past_results:
        # Include result if any of its GIFs are accessible to the user
        gifs = r.get('gifs', [])
        if not gifs or any(_user_can_access(username, gif) for gif in gifs):
            filtered.append(r)
    return jsonify(results=filtered)


def _resolve_past_result(filename):
    """Path of a saved result TSV, or (None, error response) if not allowed."""
    root = os.path.normpath(results_dir)
    path = os.path.normpath(os.path.join(root, filename))
    if not path.startswith(root + os.sep):
        return None, (jsonify(error='Invalid path'), 400)
    if not os.path.isfile(path):
        return None, (jsonify(error='Result not found'), 404)
    # Match on the canonical path, so 'a/./b.tsv' cannot dodge the check.
    # A file missing from past_results is visible to full-access users only.
    username = getattr(g, 'username', None)
    if username:
        rel = os.path.relpath(path, root)
        matching = [r for r in past_results if r['filename'] == rel]
        gifs = matching[0].get('gifs', []) if matching else []
        full = '*' in _user_config.get(username, {}).get('dirs', [])
        if not full and not any(_user_can_access(username, gif) for gif in gifs):
            return None, (jsonify(error='Access denied'), 403)
    return path, None


@app.route('/api/past-results/<path:filename>')
def get_past_result(filename):
    """Download a past result TSV file (access-checked)."""
    path, err = _resolve_past_result(filename)
    if err:
        return err
    return send_file(path, mimetype='text/tab-separated-values',
                     as_attachment=True, download_name=os.path.basename(filename))


@app.route('/api/past-results/view/<path:filename>')
def view_past_result(filename):
    """A saved result in the shape the results screen renders.

    Candidates whose GIF is still loaded get an exp_id and frame index, so
    the top-3 viewer can show them; the rest carry labels and metrics only.
    """
    path, err = _resolve_past_result(filename)
    if err:
        return err
    with open(path, encoding='utf-8', errors='replace') as f:
        saved = parse_result_tsv(f.read())

    username = getattr(g, 'username', None)
    with _state_lock:
        loaded = {e['filename']: (eid, list(e.get('epoch_labels', [])))
                  for eid, e in experiments.items()}

    ranking = []
    for row in saved['ranking']:
        gif, epoch_label = split_full_label(row['full_label'])
        cand = {'label': f'{_format_exp_label(gif)} {epoch_label}',
                'full_label': row['full_label'], 'exp_id': None, 'epoch': None}
        hit = loaded.get(gif)
        if (hit and epoch_label in hit[1]
                and (not username or _user_can_access(username, gif))):
            cand['exp_id'], cand['epoch'] = hit[0], hit[1].index(epoch_label)
        ranking.append(cand)

    names = saved['metric_names']
    higher = {name: hib for name, _, hib in METRICS}
    if all(len(row['values']) == len(names) for row in saved['ranking']):
        all_metrics = [[{'name': n, 'value': _json_number(v), 'higher_is_better': higher.get(n, True)}
                        for n, v in zip(names, row['values'])]
                       for row in saved['ranking']]
    else:
        all_metrics = [[] for _ in saved['ranking']]

    return jsonify(
        ranking=ranking,
        top3=ranking[:3],
        all_metrics=all_metrics,
        **_eval_command_payload(ranking),
        roi=saved['roi'],
        date=saved['date'],
        user=saved['user'],
        filename=os.path.relpath(path, os.path.normpath(results_dir)),
    )


def _accessible_results():
    """Past results visible to the current user (mirrors list_past_results)."""
    username = getattr(g, 'username', None)
    if not username:
        return list(past_results)
    return [r for r in past_results
            if not r.get('gifs')
            or any(_user_can_access(username, gif) for gif in r.get('gifs', []))]


def _selected_filenames(exp_ids):
    """Map requested experiment ids to their GIF filenames, or None for 'all'."""
    if not exp_ids:
        return None
    names = set()
    for eid in exp_ids:
        exp = experiments.get(eid)
        if exp and 'filename' in exp:
            names.add(exp['filename'])
    return names


def _eval_rows_for_request():
    data = _json_body()
    selected = _selected_filenames(data.get('exp_ids'))
    return collect_eval_rows(_accessible_results(), selected_filenames=selected)


@app.route('/api/export/eval-sheet', methods=['POST'])
def export_eval_sheet():
    """CSV of selected finetuning winners (tile, metadata, finetune tile, epoch)."""
    csv_text = build_csv(_eval_rows_for_request())
    return Response(csv_text, mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename="eval_selection.csv"'})


@app.route('/api/export/eval-script', methods=['POST'])
def export_eval_script():
    """A .sh of submit_eval.sh calls — one per selected finetuning winner."""
    sh_text = build_sh(_eval_rows_for_request())
    return Response(sh_text, mimetype='text/x-shellscript', headers={
        'Content-Disposition': 'attachment; filename="eval_selection.sh"'})


@app.route('/api/frame/<exp_id>/<int:epoch>')
def get_frame(exp_id, epoch):
    """Return a frame as PNG. Optional ?roi=x,y,w,h for cropping."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    loaded = _load(exp_id)
    if not loaded:
        return jsonify(error='Experiment not found'), 404
    frames = loaded[1]
    if not 0 <= epoch < len(frames):
        return jsonify(error='Invalid epoch'), 400

    img = _apply_roi(frames[epoch], request.args.get('roi'))

    # Optional resize for thumbnails: ?w=140
    w = request.args.get('w')
    if w:
        try:
            w = int(w)
        except ValueError:
            return jsonify(error='Invalid w'), 400
        h, orig_w = img.shape[:2]

        # Thumbnails only shrink, which also bounds new_h by the crop height.
        w = min(max(w, 16), orig_w)
        new_h = max(1, int(h * w / orig_w))
        img = np.asarray(Image.fromarray(img).resize((w, new_h), Image.LANCZOS))

    return _numpy_to_png_response(img)


@app.route('/api/variability/<exp_id>')
def get_variability(exp_id):
    """Return variability heatmap as RGBA PNG overlay."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    loaded = _load(exp_id)
    if not loaded:
        return jsonify(error='Experiment not found'), 404

    rgba = variability_to_heatmap_rgba(loaded[2])
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/first_frame/<exp_id>')
def get_first_frame(exp_id):
    """Return the first frame as PNG (for ROI selection background)."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    loaded = _load(exp_id)
    if not loaded:
        return jsonify(error='Experiment not found'), 404
    return _numpy_to_png_response(loaded[1][0])


@app.route('/api/roi-suggestions/<exp_id>')
def get_roi_suggestions(exp_id):
    """Return auto-suggested ROI regions based on variability."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    loaded = _load(exp_id)
    if not loaded:
        return jsonify(error='Experiment not found'), 404
    exp, _, variability = loaded

    H, W = variability.shape
    ex, ey, ew, eh = compute_label_exclusion(W, H)
    label_present = exp.get('label_present', True)
    # Only restrict the corner when a label was actually detected. When absent,
    # the corner is free for ROI selection (no exclusion, and suggest_rois may
    # use it). The frontend defaults its manual toggle to label_detected.
    exclusion_box = (ex, ey, ew, eh) if label_present else None
    rois = suggest_rois(variability, exclusion_box=exclusion_box)
    return jsonify(
        suggestions=rois,
        label_detected=bool(label_present),
        exclusion=({'x': ex, 'y': ey, 'w': ew, 'h': eh}
                   if label_present else None),
        # The box geometry is always sent so the manual toggle can re-impose
        # the restriction even when detection said "absent".
        label_box={'x': ex, 'y': ey, 'w': ew, 'h': eh})


@app.route('/api/metrics/<exp_id>/<int:epoch>')
def get_metrics(exp_id, epoch):
    """Return metric values for a frame (optionally cropped by ROI)."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    if epoch < 0 or epoch >= exp['num_epochs']:
        return jsonify(error='Invalid epoch'), 400

    r_o = _parse_r_o(request.args.get('r_o', DEFAULT_R_O))
    roi_str = _parse_roi(request.args.get('roi'))
    bad = _bad_params(r_o, roi_str)
    if bad:
        return bad

    results = _get_cached_metrics(exp_id, epoch, roi_str, r_o)
    if results is None:
        return jsonify(error='Experiment changed on disk; reload the page'), 409

    # Build lookup: metric name -> group info
    group_lookup = {}
    for grp in METRIC_GROUPS:
        for m in grp['members']:
            group_lookup[m] = {'group': grp['name'], 'representative': m == grp['representative']}

    return jsonify(metrics=[
        {
            'name': name, 'value': round(val, 6), 'higher_is_better': hib,
            **group_lookup.get(name, {}),
        }
        for name, val, hib in results
    ])


@app.route('/api/epoch-config')
def get_epoch_config():
    """Return epoch config for experiments accessible to current user."""
    username = getattr(g, 'username', None)
    result = {}
    with _state_lock:
        snapshot_items = list(experiments.items())
    for exp_id, exp in snapshot_items:
        if username and not _user_can_access(username, exp['filename']):
            continue
        result[exp_id] = {
            'filename': exp['filename'],
            'num_epochs': exp['num_epochs'],
            'epoch_config': exp.get('epoch_config', {}),
            'epoch_labels': exp.get('epoch_labels', []),
        }
    return jsonify(experiments=result)


@app.route('/api/epoch-config', methods=['POST'])
def set_epoch_config():
    """Update epoch config for experiments (access-checked)."""
    username = getattr(g, 'username', None)
    data = _json_body()
    items = data.get('experiments')
    if not isinstance(items, dict):
        return jsonify(error='No experiments specified'), 400
    cleaned = {}
    for exp_id, posted in items.items():
        config = _clean_epoch_config(posted)
        if config is None:
            return jsonify(error=f'Invalid epoch config for {exp_id}'), 400
        cleaned[exp_id] = config

    for exp_id, config in cleaned.items():
        exp = experiments.get(exp_id)
        if not exp:
            continue
        if username and not _user_can_access(username, exp['filename']):
            continue
        exp['epoch_config'] = config
        exp['epoch_labels'] = generate_epoch_labels(
            exp['num_epochs'],
            config.get('start'),
            config.get('end'),
            config.get('step'),
            config.get('raw_first', False),
        )
    return jsonify(ok=True)


@app.route('/api/prefilter-list', methods=['POST'])
def prefilter_list():
    """Fast: return just the candidate list (no metric computation) plus
    ROI flag. The frontend then fetches metrics per-candidate with progress."""
    data = _json_body()
    exps_config = data.get('experiments', [])
    roi = data.get('roi')
    if not exps_config:
        return jsonify(error='No experiments specified'), 400

    candidates = []
    for ec in exps_config:
        exp, err = _check_exp_access(ec['exp_id'])
        if err:
            return err
        exp = _ensure_loaded(ec['exp_id'])
        if not exp:
            return jsonify(error=f'Experiment {ec["exp_id"]} not found'), 404
        exp_label = _format_exp_label(exp['filename'])
        epoch_labels = exp.get('epoch_labels',
                               [f'Ep.{i + 1}' for i in range(exp['num_epochs'])])
        for epoch in range(exp['num_epochs']):
            epoch_label = epoch_labels[epoch] if epoch < len(epoch_labels) else f'Ep.{epoch + 1}'
            candidates.append({
                'exp_id': ec['exp_id'],
                'epoch': epoch,
                'label': f'{exp_label} {epoch_label}',
            })

    if len(candidates) < 4:
        return jsonify(error='Too few candidates for pre-filtering'), 400

    return jsonify(candidates=candidates, has_roi=bool(roi),
                   # Kept for older clients; equals the default set.
                   elimination_metrics=ELIMINATION_METRICS,
                   elimination_sets=ELIMINATION_SETS,
                   default_elimination_set=DEFAULT_ELIMINATION_SET)


@app.route('/api/prefilter-metric', methods=['POST'])
def prefilter_metric():
    """Return raw elimination-metric values for one candidate.
    Body: {exp_id, epoch, r_o, roi, use_roi: bool}"""
    data = _json_body()
    exp_id = data.get('exp_id')
    try:
        epoch = int(data.get('epoch', 0))
    except (TypeError, ValueError):
        return jsonify(error='Invalid epoch'), 400
    r_o = _parse_r_o(data.get('r_o', DEFAULT_R_O))
    roi = _parse_roi(data.get('roi')) if data.get('use_roi', True) else ''
    bad = _bad_params(r_o, roi)
    if bad:
        return bad

    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    if epoch < 0 or epoch >= exp['num_epochs']:
        return jsonify(error='Invalid epoch'), 400

    metrics = _get_cached_metrics(exp_id, epoch, roi, r_o)
    if metrics is None:
        return jsonify(error='Experiment changed on disk; reload the page'), 409
    # Union of all sets, not just the active one: compute_all produces every
    # metric anyway, so returning them lets the client re-rank on a set switch
    # without another round of (expensive) computation.
    values = {name: round(val, 6) for name, val, _ in metrics
              if name in ELIMINATION_METRIC_UNION}
    return jsonify(values=values)


# ---------------------------------------------------------------------------
# Cross-view background computation
# ---------------------------------------------------------------------------
# While the user is busy with the tournament, compute elimination metrics on
# the sibling views (e.g. view0 and view2 when they're comparing view1) using
# whole-frame (no ROI). A single daemon thread processes jobs serially to
# avoid CPU contention during the tournament.

_cross_view_queue = queue.Queue()
_cross_view_jobs = {}  # job_id -> dict
_CROSS_VIEW_MAX_CANDIDATES = 500
_CROSS_VIEW_JOB_TTL = 3600  # seconds


def _find_sibling_exp_id(exp_id, target_view):
    """Given an experiment, find the exp_id of the same GIF but with a
    different view (e.g. 'view0' instead of 'view1'). Returns None if not found."""
    exp = experiments.get(exp_id)
    if not exp:
        return None
    name = exp['filename']
    m = re.search(r'view([012])', name)
    if not m or m.group(1) == target_view[-1]:
        return exp_id if m and m.group(1) == target_view[-1] else None
    sibling_name = re.sub(r'view[012]', target_view, name, count=1)
    with _state_lock:
        snapshot_items = list(experiments.items())
    for other_id, other_exp in snapshot_items:
        if other_exp['filename'] == sibling_name:
            return other_id
    return None


def _cross_view_worker():
    while True:
        job = _cross_view_queue.get()
        job_id = job['job_id']
        state = _cross_view_jobs.get(job_id)
        if not state or state['status'] != 'pending':
            continue
        try:
            state['status'] = 'running'
            candidates = job['candidates']
            r_o = job['r_o']

            # Find which views are available for these candidates
            views_present = set()
            per_view_exp = {}  # (view, candidate_idx) -> exp_id
            for i, c in enumerate(candidates):
                for v in ('view0', 'view1', 'view2'):
                    sib = _find_sibling_exp_id(c['exp_id'], v)
                    if sib:
                        views_present.add(v)
                        per_view_exp[(v, i)] = sib

            if not views_present:
                state['status'] = 'missing'
                continue

            views = sorted(views_present)
            total = len(views) * len(candidates)
            state['progress'] = {'done': 0, 'total': total}

            # Collect raw elimination-metric values per view
            raw_by_view = {v: [] for v in views}
            for v in views:
                for i, c in enumerate(candidates):
                    sib = per_view_exp.get((v, i))
                    if sib is None:
                        raw_by_view[v].append({})
                    else:
                        try:
                            metrics = _get_cached_metrics(sib, c['epoch'], None, r_o)
                            raw_by_view[v].append({name: val for name, val, _ in metrics
                                                   if name in ELIMINATION_METRICS})
                        except Exception:
                            raw_by_view[v].append({})
                    state['progress']['done'] += 1

            # Compute z-score consensus per view
            results = {}
            for v in views:
                raw = raw_by_view[v]
                zsum = [0.0] * len(candidates)
                zcount = [0] * len(candidates)
                for mname, weight in ELIMINATION_METRICS.items():
                    vals = [r.get(mname, 0) for r in raw]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    for i, val in enumerate(vals):
                        z = (val - mean) / std if std > 1e-10 else 0.0
                        zsum[i] += z * weight
                        zcount[i] += 1
                scores = [zsum[i] / zcount[i] if zcount[i] else 0.0
                          for i in range(len(candidates))]
                ranked = sorted(
                    [{'exp_id': candidates[i]['exp_id'],
                      'epoch': candidates[i]['epoch'],
                      'label': candidates[i].get('label', ''),
                      'score': round(scores[i], 3)}
                     for i in range(len(candidates))],
                    key=lambda x: x['score'], reverse=True,
                )
                results[v] = ranked

            state['views'] = results
            state['status'] = 'done'
        except Exception as e:
            state['status'] = 'error'
            state['error'] = str(e)


_cross_view_thread = threading.Thread(target=_cross_view_worker, daemon=True)
_cross_view_thread.start()


def _precompute_worker():
    """When the VPS is idle, warm the persistent store with whole-frame metrics
    for every non-RAW (GIF, epoch) at the default r_o, so the pre-filter and
    cross-view never wait on a cold cache. Pauses the instant a user request
    arrives and resumes after a quiet period. Survives restarts (the store is
    on disk), so each GIF is computed once ever, not after every upload."""
    # Wait for the cold-start scan to finish populating `experiments` (the count
    # stops growing) so the first work-list is the full set, not a partial slice
    # of a still-loading scan. Capped so it can never hang forever.
    prev_count, waited = -1, 0
    while waited < 900:
        time.sleep(15)
        waited += 15
        with _state_lock:
            curr_count = sum(1 for e in experiments.values() if e.get('num_epochs'))
        if scan_is_settled(prev_count, curr_count):
            break
        prev_count = curr_count

    while True:
        try:
            present = metrics_store.present_frames(_metrics_db, DEFAULT_R_O)
            with _state_lock:
                exps = [{'exp_id': eid,
                         'filename': e['filename'],
                         'num_epochs': e.get('num_epochs', 0),
                         'raw_first': bool(e.get('epoch_config', {}).get('raw_first'))}
                        for eid, e in experiments.items()
                        if e.get('num_epochs') and e.get('filename')]
                # Skip experiments that already have tournament results — and
                # their sibling views (one processed view = the whole experiment
                # is done, so there's no need to warm view0/1/2).
                skip = set()
                for f in list(completed_gifs):
                    skip |= sibling_filenames(f)
            work = build_worklist(exps, present, skip_filenames=skip)
            if not work:
                time.sleep(120)  # nothing missing; re-check for new uploads later
                continue
            total = len(work)
            # flush=True: the worker's output is sparse, so without it the lines
            # sit in gunicorn's block-buffered stdout and never reach the journal.
            print(f'  [precompute] warming {total} frames', flush=True)
            done = 0
            for exp_id, filename, epoch in work:
                # Idle gate: wait while a user is active; pause before each item.
                while not is_idle(time.time(), _last_request_time,
                                  _PRECOMPUTE_IDLE_SECONDS):
                    time.sleep(5)
                try:
                    _get_cached_metrics(exp_id, epoch, '', DEFAULT_R_O)
                    done += 1
                    if done % 50 == 0:
                        print(f'  [precompute] {done}/{total} warmed', flush=True)
                except Exception as e:
                    print(f'  [precompute] error {filename} ep{epoch}: {e}', flush=True)
            print(f'  [precompute] pass complete: {done}/{total} warmed', flush=True)
        except Exception as e:
            print(f'  [precompute] worker error: {e}', flush=True)
            time.sleep(60)


_precompute_thread = threading.Thread(target=_precompute_worker, daemon=True)
_precompute_thread.start()


@app.route('/api/cross-view/start', methods=['POST'])
def cross_view_start():
    """Kick off background computation of sibling-view metrics.
    Body: {candidates: [{exp_id, epoch, label}, ...], r_o: 0.5}"""
    data = _json_body()
    posted = data.get('candidates')
    r_o = _parse_r_o(data.get('r_o', DEFAULT_R_O))
    if r_o is None:
        return jsonify(error='Invalid r_o'), 400
    if not isinstance(posted, list) or not posted:
        return jsonify(error='No candidates'), 400
    if len(posted) > _CROSS_VIEW_MAX_CANDIDATES:
        return jsonify(error='Too many candidates'), 400

    candidates = []
    for c in posted:
        if not isinstance(c, dict) or not isinstance(c.get('epoch'), int):
            return jsonify(error='Invalid candidate'), 400
        _, err = _check_exp_access(c.get('exp_id'))
        if err:
            return err
        candidates.append({'exp_id': c['exp_id'], 'epoch': c['epoch'],
                           'label': str(c.get('label', ''))})

    now = time.time()
    job_id = str(uuid.uuid4())[:8]
    with _state_lock:
        for old_id in [j for j, st in _cross_view_jobs.items()
                       if now - st['created'] > _CROSS_VIEW_JOB_TTL]:
            del _cross_view_jobs[old_id]

        # One queued job per user: a newer request supersedes the older one.
        for st in _cross_view_jobs.values():
            if st['username'] == getattr(g, 'username', None) and st['status'] == 'pending':
                st['status'] = 'superseded'
        _cross_view_jobs[job_id] = {
            'status': 'pending',
            'progress': {'done': 0, 'total': 0},
            'views': {},
            'username': getattr(g, 'username', None),
            'created': now,
        }
    _cross_view_queue.put({'job_id': job_id, 'candidates': candidates, 'r_o': r_o})
    return jsonify(job_id=job_id)


@app.route('/api/cross-view/status/<job_id>')
def cross_view_status(job_id):
    state = _cross_view_jobs.get(job_id)
    if not state or state['username'] != getattr(g, 'username', None):
        return jsonify(error='Unknown job'), 404
    return jsonify(status=state['status'], progress=state['progress'],
                   views=state['views'], error=state.get('error'))


@app.route('/api/tournament/start', methods=['POST'])
def start_tournament():
    """Start a tournament with one or more experiments.
    Body: {experiments: [{exp_id}, ...], r_o: 0.5, roi: "x,y,w,h"}
    Each (exp_id, epoch) pair is a separate candidate."""
    data = _json_body()
    exps_config = data.get('experiments', [])
    r_o = _parse_r_o(data.get('r_o', DEFAULT_R_O))
    roi = _parse_roi(data.get('roi'))
    bad = _bad_params(r_o, roi)
    if bad:
        return bad
    # The logged-in identity wins: it is authenticated, so results can't be
    # attributed to someone else. The posted name is only a fallback for
    # deployments running without auth (no users.json configured).
    user_name = getattr(g, 'username', None) or data.get('user_name', '')
    # Optional: exclude specific (exp_id, epoch) pairs (from pre-filter)
    exclude = set()
    for ex in data.get('exclude', []):
        exclude.add((ex['exp_id'], ex['epoch']))

    if not exps_config:
        return jsonify(error='No experiments specified'), 400

    # Cleanup old tournaments
    with _state_lock:
        _cleanup_old_tournaments()

    # Build flat list of candidates: [{exp_id, epoch, label}, ...]
    candidates = []
    for ec in exps_config:
        # Check access for each experiment
        exp, err = _check_exp_access(ec['exp_id'])
        if err:
            return err
        exp = _ensure_loaded(ec['exp_id'])
        if not exp:
            return jsonify(error=f'Experiment {ec["exp_id"]} not found'), 404
        exp_label = _format_exp_label(exp['filename'])
        epoch_labels = exp.get('epoch_labels',
                               [f'Ep.{i + 1}' for i in range(exp['num_epochs'])])
        for epoch in range(exp['num_epochs']):
            epoch_label = epoch_labels[epoch] if epoch < len(epoch_labels) else f'Ep.{epoch + 1}'
            # Skip pre-filtered candidates
            if (ec['exp_id'], epoch) in exclude:
                continue
            candidates.append({
                'exp_id': ec['exp_id'],
                'epoch': epoch,
                'label': f'{exp_label} {epoch_label}',
                'full_label': f'{exp["filename"]} {epoch_label}',
            })

    # Sort candidates by epoch descending — later epochs are likely better,
    # so placing them first helps merge-sort find the winner faster
    if not candidates:
        return jsonify(error='No candidates'), 400

    candidates.sort(key=lambda c: c['epoch'], reverse=True)

    session_id = str(uuid.uuid4())[:8]
    t = Tournament(list(range(len(candidates))))

    # Record every experiment the user originally selected, even ones that got
    # entirely pre-filtered out — so the result file still lists them as done.
    selected_exp_ids = [ec['exp_id'] for ec in exps_config]

    with _state_lock:
        # A rescan may have dropped a selected GIF while frames were loading.
        if any(c['exp_id'] not in experiments for c in candidates):
            return jsonify(error='A selected GIF changed on disk; reload the page'), 409
        tournaments[session_id] = {
            'tournament': t,
            'candidates': candidates,
            'selected_exp_ids': selected_exp_ids,
            'roi': roi,
            'r_o': r_o,
            'user_name': user_name,
            'username': getattr(g, 'username', None),
            'created_at': datetime.now().timestamp(),
        }

    pair = t.current_pair()
    progress = t.progress()

    return jsonify(
        session_id=session_id,
        pair=_pair_info(candidates, pair),
        progress={'current': progress[0], 'total': progress[1]},
        done=t.is_done(),
    )


def _pair_info(candidates, pair):
    """Convert a pair of candidate indices to rich info for the frontend."""
    if pair is None:
        return None
    a, b = pair
    return {
        'left': {'index': a, **candidates[a]},
        'right': {'index': b, **candidates[b]},
    }


@app.route('/api/tournament/<session_id>')
def tournament_state(session_id):
    """Current pair and settings of a tournament, so a reloaded page can resume."""
    tdata = tournaments.get(session_id)
    if not tdata or (tdata.get('username') is not None
                     and tdata.get('username') != getattr(g, 'username', None)):
        return jsonify(error='No active tournament'), 404

    t = tdata['tournament']
    with _state_lock:
        pair = t.current_pair()
        current, total = t.progress()
        done = t.is_done()
        selected = [{'expId': eid, 'filename': experiments[eid]['filename']}
                    for eid in tdata['selected_exp_ids'] if eid in experiments]
    return jsonify(
        pair=_pair_info(tdata['candidates'], pair),
        progress={'current': current, 'total': total},
        done=done,
        roi=tdata['roi'] or None,
        r_o=tdata['r_o'],
        selected=selected,
    )


@app.route('/api/tournament/<session_id>/choice', methods=['POST'])
def tournament_choice(session_id):
    """Submit a choice: {winner: "left"|"right"|"tie"}.
    On tie, z-score-normalized metrics break it."""
    tdata = tournaments.get(session_id)
    if not tdata:
        return jsonify(error='No active tournament'), 404

    # Verify tournament ownership
    if tdata.get('username') is not None and tdata.get('username') != getattr(g, 'username', None):
        return jsonify(error='Access denied'), 403

    t = tdata['tournament']
    candidates = tdata['candidates']
    data = _json_body()
    winner = data.get('winner')
    tie_explanation = None

    if winner == 'tie':
        pair = t.current_pair()
        if pair:
            r_o = tdata['r_o']
            roi = tdata['roi']
            ca, cb = candidates[pair[0]], candidates[pair[1]]

            metrics_a = _get_cached_metrics(ca['exp_id'], ca['epoch'], roi, r_o)
            metrics_b = _get_cached_metrics(cb['exp_id'], cb['epoch'], roi, r_o)
            if metrics_a is None or metrics_b is None:
                return jsonify(error='A GIF in this pair changed on disk; pick a winner instead'), 409

            # Z-score normalization: only count votes where |z| > 0.5.
            # Each redundancy cluster votes as one unit (via its representative)
            # to avoid large clusters dominating the tie-break.
            zscored = compute_metric_zscores([metrics_a, metrics_b])
            za, zb = zscored[0], zscored[1]

            # Build set of representatives + standalone metric names
            rep_names = set()
            grouped_names = set()
            for grp in METRIC_GROUPS:
                rep_names.add(grp['representative'])
                grouped_names.update(grp['members'])

            a_wins = 0
            b_wins = 0
            a_voters = []
            b_voters = []
            for i, ((name, z_a, hib), (_, z_b, _)) in enumerate(zip(za, zb)):
                # Skip non-representative cluster members
                if name in grouped_names and name not in rep_names:
                    continue
                # Only vote if difference is meaningful (|z| > 0.5)
                diff = abs(z_a - z_b)
                if diff < 0.5:
                    continue
                if hib:
                    if z_a > z_b:
                        a_wins += 1
                        a_voters.append(name)
                    elif z_b > z_a:
                        b_wins += 1
                        b_voters.append(name)
                else:
                    if z_a < z_b:
                        a_wins += 1
                        a_voters.append(name)
                    elif z_b < z_a:
                        b_wins += 1
                        b_voters.append(name)

            if a_wins == 0 and b_wins == 0:
                # True tie — no metric has meaningful difference
                winner = 'left'  # arbitrary
                tie_explanation = f'True tie: no metric showed meaningful difference'
            elif a_wins >= b_wins:
                winner = 'left'
                tie_explanation = f'Metrics {a_wins}-{b_wins} for left ({", ".join(a_voters[:3])})'
            else:
                winner = 'right'
                tie_explanation = f'Metrics {b_wins}-{a_wins} for right ({", ".join(b_voters[:3])})'

    if winner not in ('left', 'right'):
        return jsonify(error='Invalid winner'), 400

    with _state_lock:
        t.choose(winner)
        pair = t.current_pair()
        progress = t.progress()
        done = t.is_done()
        results = list(t.results) if t.results else None
        top_k = list(t.get_top_k()) if t.get_top_k() else None

    result = {
        'pair': _pair_info(candidates, pair),
        'progress': {'current': progress[0], 'total': progress[1]},
        'done': done,
    }

    if tie_explanation:
        result['tie_explanation'] = tie_explanation

    if done and results:
        ranking = [candidates[i] for i in results]
        top3 = [candidates[i] for i in (top_k or [])]

        # Compute metrics for all ranked candidates
        roi = tdata['roi']
        r_o = tdata['r_o']
        all_metrics = _ranking_metrics(ranking, roi, r_o)

        # Auto-save (includes choice history)
        save_path = _auto_save_results(ranking, all_metrics, tdata)

        result['ranking'] = ranking
        result['top3'] = top3
        # Copy-pastable eval command for the winner (None for training tiles,
        # which have no checkpoint to evaluate against).
        result.update(_eval_command_payload(ranking))
        result['all_metrics'] = [
            [{'name': n, 'value': _json_number(v), 'higher_is_better': h} for n, v, h in m]
            for m in all_metrics
        ]
        if save_path:
            result['save_path'] = os.path.basename(save_path)

    return jsonify(**result)


@app.route('/api/tournament/<session_id>/undo', methods=['POST'])
def tournament_undo(session_id):
    """Undo last tournament choice."""
    tdata = tournaments.get(session_id)
    if not tdata:
        return jsonify(error='No active tournament'), 404

    # Verify tournament ownership
    if tdata.get('username') is not None and tdata.get('username') != getattr(g, 'username', None):
        return jsonify(error='Access denied'), 403

    t = tdata['tournament']
    candidates = tdata['candidates']
    with _state_lock:
        pair = t.undo()
        progress = t.progress()
        done = t.is_done()

    return jsonify(
        pair=_pair_info(candidates, pair),
        progress={'current': progress[0], 'total': progress[1]},
        done=done,
        undone=pair is not None,
    )


@app.route('/api/tournament/<session_id>/finish', methods=['POST'])
def tournament_finish(session_id):
    """Force-finish the tournament with current rankings."""
    tdata = tournaments.get(session_id)
    if not tdata:
        return jsonify(error='No active tournament'), 404

    # Verify tournament ownership
    if tdata.get('username') is not None and tdata.get('username') != getattr(g, 'username', None):
        return jsonify(error='Access denied'), 403

    t = tdata['tournament']
    candidates = tdata['candidates']
    with _state_lock:
        t.force_finish()
        results = list(t.results) if t.results else []
        top_k = list(t.get_top_k()) if t.get_top_k() else []
        current, total = t.progress()

    ranking = [candidates[i] for i in results]
    top3 = [candidates[i] for i in top_k]

    roi = tdata['roi']
    r_o = tdata['r_o']
    all_metrics = _ranking_metrics(ranking, roi, r_o)

    save_path = _auto_save_results(ranking, all_metrics, tdata)

    result = {
        'pair': None,
        'progress': {'current': current, 'total': total},
        'done': True,
        'ranking': ranking,
        'top3': top3,
        # Copy-pastable eval command for the winner (see tournament_choice).
        **_eval_command_payload(ranking),
        'all_metrics': [
            [{'name': n, 'value': _json_number(v), 'higher_is_better': h} for n, v, h in m]
            for m in all_metrics
        ],
    }
    if save_path:
        result['save_path'] = os.path.basename(save_path)

    return jsonify(**result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_roi(img, roi_str):
    """Crop image using ROI string 'x,y,w,h'."""
    if not roi_str:
        return img
    try:
        x, y, w, h = [int(v) for v in roi_str.split(',')]
        H, W = img.shape
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return img[y:y + h, x:x + w]
    except (ValueError, TypeError):
        return img


def _numpy_to_png_response(img):
    """Convert a uint8 grayscale numpy array to a PNG response."""
    pil_img = Image.fromarray(img, 'L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def _ranking_metrics(ranking, roi, r_o):
    """Metrics for each ranked candidate.

    A GIF that a rescan dropped mid-tournament gets NaN values, so the result
    and its choice history are still saved.
    """
    all_metrics = [_get_cached_metrics(c['exp_id'], c['epoch'], roi, r_o) for c in ranking]
    template = next((m for m in all_metrics if m), [])
    return [m if m is not None else [(n, float('nan'), h) for n, _, h in template]
            for m in all_metrics]


def _json_number(v):
    """A metric value rounded for JSON; NaN becomes null, which JSON can carry."""
    return None if v != v else round(v, 6)


def _auto_save_results(ranking, all_metrics, tdata):
    """Save tournament results and the choice history to a TSV.

    Only the top 3 are ordered. The tournament stops merging once it knows
    them, so later rows keep a partial order.
    """
    try:
        t = tdata['tournament']
        roi = tdata.get('roi')
        r_o = tdata.get('r_o', 0.5)
        user_name = tdata.get('user_name', '')

        # Prefer the originally selected experiments (covers the case where
        # pre-filter removed every candidate from a GIF, so it wouldn't appear
        # in the ranking). Fall back to ranking-derived IDs for legacy sessions.
        exp_ids = list(tdata.get('selected_exp_ids') or [])
        if not exp_ids:
            for c in ranking:
                if c['exp_id'] not in exp_ids:
                    exp_ids.append(c['exp_id'])
        gif_names = [experiments[eid]['filename']
                     for eid in exp_ids if eid in experiments]

        lines = []
        lines.append('Compare Game \u2014 Tournament Results')
        lines.append(f'Date: {datetime.now().isoformat()}')
        lines.append(f'User: {user_name}')
        lines.append(f'Model: Merge-sort tournament')
        lines.append(f'GIFs: {", ".join(gif_names)}')
        lines.append(f'OTF radius: {r_o}')
        lines.append(f'ROI: {roi or "none"}')
        lines.append(f'Total comparisons: {t.comparison_count}')
        lines.append('')

        top3_labels = [c.get('full_label', c['label']) for c in ranking[:3]]
        lines.append(f'Top 3: {", ".join(top3_labels)}')
        lines.append('')

        # TSV table
        if not all_metrics:
            return None
        metric_names = [n for n, _, _ in all_metrics[0]]
        lines.append('\t'.join(['Rank', 'Candidate'] + metric_names))
        for i, (c, metrics) in enumerate(zip(ranking, all_metrics)):
            label = c.get('full_label', c['label'])
            vals = [f'{v:.6f}' for _, v, _ in metrics]
            lines.append('\t'.join([str(i + 1), label] + vals))

        # Pairwise choice history
        history = t.get_history()
        if history:
            candidates = tdata['candidates']
            lines.append('')
            lines.append('Pairwise Choice History')
            lines.append('\t'.join(['#', 'Left', 'Right', 'Winner', 'Timestamp']))
            for i, h in enumerate(history):
                left_label = candidates[h['left']].get('label', str(h['left']))
                right_label = candidates[h['right']].get('label', str(h['right']))
                lines.append('\t'.join([
                    str(i + 1), left_label, right_label,
                    h['winner'], h.get('timestamp', ''),
                ]))

        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

        # Mirror the experiment folder structure inside results_dir
        gif_dirs = [os.path.dirname(gn) for gn in gif_names]
        sub_dir = os.path.commonpath(gif_dirs) if gif_dirs else ''
        dest_dir = os.path.join(results_dir, sub_dir)
        os.makedirs(dest_dir, exist_ok=True)

        # Two tournaments finishing in the same second get _2, _3, ...
        for n in itertools.count(1):
            suffix = '' if n == 1 else f'_{n}'
            save_path = os.path.join(dest_dir, f'tournament_{timestamp}{suffix}.tsv')
            try:
                with open(save_path, 'x') as f:
                    f.write('\n'.join(lines) + '\n')
                break
            except FileExistsError:
                continue

        print(f'Results saved to {save_path}')

        # Refresh tracking
        scan_past_results()

        return save_path
    except Exception as e:
        print(f'Warning: could not save results: {e}')
        return None


# ---------------------------------------------------------------------------
# Data directory loading
# ---------------------------------------------------------------------------

def _evict_lru():
    """Evict least recently used experiment frames to free memory.
    Must be called with _state_lock held."""
    loaded = [eid for eid, exp in experiments.items() if 'frames' in exp]
    if len(loaded) <= _MAX_LOADED:
        return

    # Sort by access time, evict oldest
    loaded.sort(key=lambda eid: _access_times.get(eid, 0))
    while len(loaded) > _MAX_LOADED:
        evict_id = loaded.pop(0)
        exp = experiments[evict_id]
        del exp['frames']
        del exp['variability']
        # Clear metrics cache for this experiment
        to_del = [k for k in _metrics_cache if k[0] == evict_id]
        for k in to_del:
            del _metrics_cache[k]
        print(f'  Evicted {exp["filename"]} to free memory')


def _load(exp_id):
    """(exp, frames, variability) with the GIF decoded, or None.

    Callers keep the returned arrays, so a concurrent LRU eviction can't pull
    them away mid-request. Decoding runs outside _state_lock so other requests
    keep flowing, and a per-GIF lock stops two requests decoding one GIF twice.
    """
    with _state_lock:
        exp = experiments.get(exp_id)
        if not exp:
            return None
        if 'frames' in exp:
            _access_times[exp_id] = time.time()
            return exp, exp['frames'], exp['variability']
        load_lock = _load_locks.setdefault(exp_id, threading.Lock())

    with load_lock:
        with _state_lock:
            if experiments.get(exp_id) is not exp:
                return None
            if 'frames' in exp:
                _access_times[exp_id] = time.time()
                return exp, exp['frames'], exp['variability']

        print(f'  Loading {exp["filename"]}...', flush=True)
        frames = extract_frames_u8(exp['gif_path'])
        if len(frames) < 2:
            return None
        variability = compute_variability_map(frames)
        H0, W0 = frames[0].shape

        # The highest-epoch frame carries a burned-in label if any frame does.
        label_present = detect_label_present(
            frame_to_float(frames[-1]), compute_label_exclusion(W0, H0))

        with _state_lock:
            # A rescan dropped or replaced this GIF while it was decoding.
            if experiments.get(exp_id) is not exp:
                return None
            exp['frames'] = frames
            exp['variability'] = variability
            exp['num_epochs'] = len(frames)
            exp['height'], exp['width'] = H0, W0
            exp['label_present'] = label_present
            _access_times[exp_id] = time.time()
            _evict_lru()

            # Regenerate labels if frame count changed
            if len(exp.get('epoch_labels', [])) != len(frames):
                ec = exp.get('epoch_config', {})
                # Re-check raw_first auto-detection with actual frame count
                if (not ec.get('raw_first')
                        and ec.get('start') is not None
                        and ec.get('end') is not None):
                    expected = ec['end'] - ec['start'] + 1
                    step = ec.get('step')
                    if step and step > 1:
                        expected = (ec['end'] - ec['start']) // step + 1
                    if len(frames) == expected + 1:
                        ec['raw_first'] = True
                exp['epoch_labels'] = generate_epoch_labels(
                    len(frames), ec.get('start'), ec.get('end'),
                    ec.get('step'), ec.get('raw_first', False),
                )
        print(f'  Loaded {exp["filename"]} ({len(frames)} epochs, {W0}x{H0})', flush=True)
    return exp, frames, variability


def _ensure_loaded(exp_id):
    """Decode an experiment's GIF if needed; the experiment dict, or None."""
    loaded = _load(exp_id)
    return loaded[0] if loaded else None


def scan_past_results():
    """Scan results directory for past tournament TSVs and track which GIFs were compared."""
    global past_results, completed_gifs
    past_results = []
    completed_gifs = {}

    if not os.path.isdir(results_dir):
        return


    for tsv_path in sorted(glob_mod.glob(os.path.join(results_dir, '**', '*.tsv'), recursive=True), reverse=True):
        try:
            with open(tsv_path) as f:
                lines = f.readlines()

            info = {
                'filename': os.path.relpath(tsv_path, results_dir),
                'date': '',
                'user': '',
                'gifs': [],
                'roi': '',
                'top3': '',
            }
            for line in lines[:15]:
                line = line.strip()
                if line.startswith('Date:'):
                    info['date'] = line[5:].strip()
                elif line.startswith('User:'):
                    info['user'] = line[5:].strip()
                elif line.startswith('GIFs:'):
                    info['gifs'] = [g.strip() for g in line[5:].split(',')]
                    for g in info['gifs']:
                        completed_gifs[g] = info.get('user', '')
                elif line.startswith('ROI:'):
                    info['roi'] = line[4:].strip()
                elif line.startswith('Top 3:'):
                    info['top3'] = line[6:].strip()

            past_results.append(info)
        except Exception:
            continue

    # Sort by date descending (newest first) — falls back to filename for
    # entries without a parseable date
    past_results.sort(key=lambda r: r.get('date') or r.get('filename', ''), reverse=True)

    print(f'  Found {len(past_results)} past result(s)')


def _probe_gif(gif_path, rel_path, cli_config=None):
    """Read a GIF's metadata (no pixels) into an experiment entry.

    Returns None for unreadable files and GIFs with fewer than two frames.
    """
    cli_config = cli_config or {}
    try:
        st = os.stat(gif_path)
        img = Image.open(gif_path)
        try:
            w, h = img.size
            n_frames = 0
            try:
                while True:
                    n_frames += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
        finally:
            img.close()
        if n_frames < 2:
            return None
    except Exception:
        return None

    # Epoch config: CLI > filename > default
    epoch_config = {'source': 'default'}
    if cli_config.get('start') is not None:
        epoch_config = {**cli_config, 'source': 'cli'}
    else:
        parsed = parse_epoch_config_from_name(rel_path)
        if parsed:
            epoch_config = {**parsed, 'source': 'filename'}

    # Auto-detect raw_first: if the filename says start-end (N values)
    # but the GIF has N+1 frames, the first frame is a RAW reference
    if (not epoch_config.get('raw_first')
            and epoch_config.get('start') is not None
            and epoch_config.get('end') is not None):
        expected = epoch_config['end'] - epoch_config['start'] + 1
        step = epoch_config.get('step')
        if step and step > 1:
            expected = (epoch_config['end'] - epoch_config['start']) // step + 1
        if n_frames == expected + 1:
            epoch_config['raw_first'] = True
            print(f'  Auto-detected raw_first for {rel_path} '
                  f'({n_frames} frames, range {epoch_config["start"]}-{epoch_config["end"]})')

    labels = generate_epoch_labels(
        n_frames,
        epoch_config.get('start'),
        epoch_config.get('end'),
        epoch_config.get('step'),
        epoch_config.get('raw_first', False),
    )
    return {
        'filename': rel_path,
        'gif_path': gif_path,
        'sig': (st.st_size, st.st_mtime),
        'num_epochs': n_frames,
        'height': h,
        'width': w,
        'epoch_config': epoch_config,
        'epoch_labels': labels,
    }


def load_data_dir(data_dir, cli_config=None):
    """Scan a data directory for GIFs (lazy — only metadata, no pixel loading)."""
    for rel_path in sorted(snapshot(data_dir)):
        entry = _probe_gif(os.path.join(data_dir, rel_path), rel_path, cli_config)
        if entry is None:
            continue
        experiments[str(uuid.uuid4())[:8]] = entry
        print(f'  Found {rel_path} ({entry["num_epochs"]} epochs, '
              f'{entry["width"]}x{entry["height"]})')


def snapshot_sig(data_dir, rel_path):
    """(size, mtime) of one GIF, or None when it has vanished."""
    try:
        st = os.stat(os.path.join(data_dir, rel_path))
    except OSError:
        return None
    return (st.st_size, st.st_mtime)


# Files that failed to probe: {rel_path: sig}. Retried only once they change.
_unreadable_gifs = {}


def _drop_experiment(exp_id):
    """Forget an experiment and its in-memory caches. Caller holds _state_lock."""
    experiments.pop(exp_id, None)
    _access_times.pop(exp_id, None)
    for k in [k for k in _metrics_cache if k[0] == exp_id]:
        del _metrics_cache[k]


def _busy_exp_ids():
    """exp_ids used by unfinished tournaments. Caller holds _state_lock."""
    busy = set()
    for td in tournaments.values():
        t = td.get('tournament')
        if t is not None and t.is_done():
            continue
        busy.update(td.get('selected_exp_ids') or [])
        busy.update(c['exp_id'] for c in td.get('candidates', []))
    return busy


def rescan_data_dir(data_dir):
    """Apply GIFs added, replaced or deleted since the last scan.

    Unchanged files keep their exp_id, so open pages and running tournaments
    stay valid. A GIF that an unfinished tournament uses is left alone until
    that tournament finishes. Returns a summary string, empty when nothing
    changed, which also notes files still uploading or waiting on a tournament.
    """
    with _state_lock:
        by_name = {e['filename']: eid for eid, e in experiments.items()}
        known = {e['filename']: e.get('sig') for e in experiments.values()}
    known.update(_unreadable_gifs)
    added, changed, removed, deferred = diff_scan(known, snapshot(data_dir), time.time())
    # A previously unreadable file that changed is new to the app, not replaced.
    added += [rel for rel in changed if rel not in by_name]
    changed = [rel for rel in changed if rel in by_name]
    if not (added or changed or removed):
        return f'{len(deferred)} still uploading' if deferred else ''

    # Probing reads every frame header, so do it outside the lock.
    probed = {rel: _probe_gif(os.path.join(data_dir, rel), rel)
              for rel in added + changed}

    n_add = n_change = n_remove = n_wait = 0
    with _state_lock:
        busy = _busy_exp_ids()
        for rel in removed + changed + added:
            eid = by_name.get(rel)
            if eid in busy:
                n_wait += 1
                continue
            if eid:
                _drop_experiment(eid)
            _unreadable_gifs.pop(rel, None)
            # The store is keyed by filename, so stored metrics may describe
            # an earlier file under the same name.
            metrics_store.forget(_metrics_db, rel)
            if rel in removed:
                n_remove += 1
                continue
            entry = probed[rel]
            if entry is None:
                _unreadable_gifs[rel] = snapshot_sig(data_dir, rel)
                continue
            experiments[str(uuid.uuid4())[:8]] = entry
            if rel in added:
                n_add += 1
            else:
                n_change += 1

    parts = []
    if n_add:
        parts.append(f'+{n_add} new')
    if n_change:
        parts.append(f'{n_change} replaced')
    if n_remove:
        parts.append(f'-{n_remove} removed')
    if n_wait:
        parts.append(f'{n_wait} waiting for a running tournament')
    if deferred:
        parts.append(f'{len(deferred)} still uploading')
    return ', '.join(parts)


_RESCAN_INTERVAL = 15  # seconds between data-dir / users.json polls


def _rescan_worker(data_dir, users_path, trigger_path):
    """Poll for new GIFs and an edited users.json, and apply them live.

    Touching `trigger_path` forces a status line even when nothing changed
    or the scan failed, which is how update-compare-game-data knows the
    scan has run.
    """
    users_mtime = file_mtime(users_path) if users_path else None
    trigger_mtime = file_mtime(trigger_path)
    last_summary = ''
    while True:
        time.sleep(_RESCAN_INTERVAL)
        t = file_mtime(trigger_path)
        forced = t is not None and t != trigger_mtime
        trigger_mtime = t
        try:
            if users_path:
                m = file_mtime(users_path)
                if m is not None and m != users_mtime:
                    users_mtime = m
                    reload_users_config(users_path)
            summary = rescan_data_dir(data_dir)
            # Waiting/uploading notes repeat every poll; log them once.
            if forced or (summary and summary != last_summary):
                with _state_lock:
                    n = len(experiments)
                note = f' ({summary})' if summary else ''
                print(f'  [rescan] experiments has {n} GIF(s){note}', flush=True)
            last_summary = summary
        except Exception as e:
            print(f'  [rescan] error: {e}', flush=True)
            if forced:
                print('  [rescan] FAILED, see error above', flush=True)


def start_rescan_worker(data_dir, users_path, trigger_path):
    threading.Thread(target=_rescan_worker, args=(data_dir, users_path, trigger_path),
                     daemon=True).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Game \u2014 Epoch Comparison Tool')
    parser.add_argument('data_dir', type=str,
                        help='Directory with GIF files to load')
    parser.add_argument('--epoch-start', type=int, default=None,
                        help='First epoch number (applies to all GIFs)')
    parser.add_argument('--epoch-end', type=int, default=None,
                        help='Last epoch number (used to derive step)')
    parser.add_argument('--epoch-step', type=int, default=None,
                        help='Step between epochs (default: 1)')
    parser.add_argument('--raw-first', action='store_true',
                        help='First GIF frame is RAW (no denoising)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results (default: DATA_DIR/results)')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--ssl-cert', type=str, default=None,
                        help='Path to SSL certificate file for HTTPS')
    parser.add_argument('--ssl-key', type=str, default=None,
                        help='Path to SSL private key file for HTTPS')
    parser.add_argument('--users-config', type=str, default=None,
                        help='Path to users.json config file')
    args = parser.parse_args()

    load_users_config(args.users_config)

    cli_epoch_config = {}
    if args.epoch_start is not None:
        cli_epoch_config['start'] = args.epoch_start
    if args.epoch_end is not None:
        cli_epoch_config['end'] = args.epoch_end
    if args.epoch_step is not None:
        cli_epoch_config['step'] = args.epoch_step
    if args.raw_first:
        cli_epoch_config['raw_first'] = True

    import sys
    sys.modules[__name__].results_dir = (args.results_dir
                                         or os.environ.get('COMPARE_RESULTS_DIR')
                                         or os.path.join(args.data_dir, 'results'))

    print(f'Scanning GIFs from {args.data_dir}...')
    load_data_dir(args.data_dir, cli_epoch_config)
    print(f'Found {len(experiments)} experiment(s)')

    print(f'Scanning past results from {results_dir}...')
    scan_past_results()

    ssl_ctx = None
    if args.ssl_cert and args.ssl_key:
        ssl_ctx = (args.ssl_cert, args.ssl_key)

    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            ssl_context=ssl_ctx)
