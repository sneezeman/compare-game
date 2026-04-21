"""
Flask server for the Compare Game — epoch comparison tool.
"""

import argparse
import hmac
import io
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime

import glob as glob_mod

import numpy as np
from flask import Flask, g, jsonify, render_template, request, send_file
from PIL import Image

from gif_loader import (compute_variability_map, extract_frames,
                        suggest_rois, variability_to_heatmap_rgba)
from metrics import compute_all, compute_metric_zscores, METRIC_GROUPS, ELIMINATION_METRICS
from tournament import Tournament

app = Flask(__name__)

# Thread safety lock for shared mutable state
_state_lock = threading.RLock()  # reentrant — cleanup is called from locked contexts

# User config: {username: {password, dirs}} loaded from JSON or env var
_user_config = {}


def load_users_config(config_path=None):
    """Load user config from JSON file or COMPARE_USERS env var."""
    global _user_config
    _user_config = {}

    # Check env var for config path if not provided
    if not config_path:
        config_path = os.environ.get('COMPARE_USERS_CONFIG')

    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            data = json.load(f)
        for username, cfg in data.items():
            _user_config[username] = {
                'password': cfg['password'],
                'dirs': cfg.get('dirs', ['*']),
            }
        print(f'  Loaded {len(_user_config)} user(s) from {config_path}')
        return

    # Fallback: COMPARE_USERS env var (all users get full access)
    for pair in os.environ.get('COMPARE_USERS', '').split(','):
        pair = pair.strip()
        if ':' in pair:
            u, p = pair.split(':', 1)
            _user_config[u] = {'password': p, 'dirs': ['*']}


def _user_can_access(username, filename):
    """Check if a user can access an experiment by its filename."""
    cfg = _user_config.get(username, {})
    dirs = cfg.get('dirs', [])
    if '*' in dirs:
        return True
    return any(filename.startswith(d) for d in dirs)


# Rate limiting / auto-ban for failed auth attempts
_fail_counts = {}  # ip -> (count, first_fail_time)
_banned_ips = {}   # ip -> ban_timestamp
_BAN_THRESHOLD = 10  # ban after this many failed attempts
_BAN_WINDOW = 300    # within this many seconds


@app.before_request
def check_auth():
    ip = request.remote_addr

    # Health check and logout — exempt from auth
    if request.path in ('/health', '/logout'):
        return

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
# LRU tracking: exp_id -> last_access_time (for frame eviction)
_access_times = {}
_MAX_LOADED = 3  # max experiments with frames in memory at once
_MAX_METRICS_CACHE = 5000  # max entries in metrics cache
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

def _get_cached_metrics(exp_id, epoch, roi_str, r_o):
    """Get metrics with caching."""
    key = (exp_id, epoch, roi_str or '', r_o)
    with _state_lock:
        if key in _metrics_cache:
            return _metrics_cache[key]

    exp = _ensure_loaded(exp_id)
    if not exp or epoch < 0 or epoch >= exp['num_epochs']:
        return None

    img = exp['frames'][epoch]
    img = _apply_roi(img, roi_str)
    results = compute_all(img, r_o=r_o)
    with _state_lock:
        if len(_metrics_cache) >= _MAX_METRICS_CACHE:
            # Evict oldest half
            keys = list(_metrics_cache.keys())
            for k in keys[:len(keys) // 2]:
                del _metrics_cache[k]
        _metrics_cache[key] = results
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/health')
def health():
    return jsonify(status='ok', experiments=len(experiments),
                   loaded=sum(1 for e in experiments.values() if 'frames' in e))


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
    resp = app.make_response(render_template('index.html'))
    # Force the HTML to always re-validate so new ?v=... tokens are picked up
    resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return resp


@app.route('/api/experiments')
def list_experiments():
    """List all loaded experiments with completion status, filtered by user access."""
    username = getattr(g, 'username', None)
    result = []
    for exp_id, exp in experiments.items():
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


@app.route('/api/past-results/<path:filename>')
def get_past_result(filename):
    """Download a past result TSV file (access-checked)."""
    path = os.path.normpath(os.path.join(results_dir, filename))
    if not path.startswith(os.path.normpath(results_dir) + os.sep):
        return jsonify(error='Invalid path'), 400
    if not os.path.isfile(path):
        return jsonify(error='Result not found'), 404
    # Check if user can access any GIF in this result
    username = getattr(g, 'username', None)
    if username:
        matching = [r for r in past_results if r['filename'] == filename]
        if matching:
            gifs = matching[0].get('gifs', [])
            if not any(_user_can_access(username, gif) for gif in gifs):
                return jsonify(error='Access denied'), 403
    return send_file(path, mimetype='text/tab-separated-values',
                     as_attachment=True, download_name=os.path.basename(filename))


@app.route('/api/frame/<exp_id>/<int:epoch>')
def get_frame(exp_id, epoch):
    """Return a frame as PNG. Optional ?roi=x,y,w,h for cropping."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    if epoch < 0 or epoch >= exp['num_epochs']:
        return jsonify(error='Invalid epoch'), 400

    img = exp['frames'][epoch]
    img = _apply_roi(img, request.args.get('roi'))

    # Optional resize for thumbnails: ?w=140
    w = request.args.get('w')
    if w:
        w = int(w)
        h, orig_w = img.shape[:2]
        new_h = max(1, int(h * w / orig_w))
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil_img = pil_img.resize((w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    return _numpy_to_png_response(img)


@app.route('/api/variability/<exp_id>')
def get_variability(exp_id):
    """Return variability heatmap as RGBA PNG overlay."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404

    rgba = variability_to_heatmap_rgba(exp['variability'])
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
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    return _numpy_to_png_response(exp['frames'][0])


@app.route('/api/roi-suggestions/<exp_id>')
def get_roi_suggestions(exp_id):
    """Return auto-suggested ROI regions based on variability."""
    exp, err = _check_exp_access(exp_id)
    if err:
        return err
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404

    rois = suggest_rois(exp['variability'])
    return jsonify(suggestions=rois)


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

    r_o = float(request.args.get('r_o', 0.5))
    roi_str = request.args.get('roi')

    results = _get_cached_metrics(exp_id, epoch, roi_str, r_o)

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
    for exp_id, exp in experiments.items():
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
    data = request.get_json(force=True)
    for exp_id, config in data.get('experiments', {}).items():
        exp = experiments.get(exp_id)
        if not exp:
            continue
        if username and not _user_can_access(username, exp['filename']):
            continue
        config['source'] = 'user'
        exp['epoch_config'] = config
        exp['epoch_labels'] = generate_epoch_labels(
            exp['num_epochs'],
            config.get('start'),
            config.get('end'),
            config.get('step'),
            config.get('raw_first', False),
        )
    return jsonify(ok=True)


@app.route('/api/prefilter', methods=['POST'])
def prefilter():
    """Compute metrics for all candidates and return them sorted by consensus score.
    Body: {experiments: [{exp_id}, ...], r_o: 0.5, roi: "x,y,w,h", cut_frac: 0.25}
    Returns sorted candidates with scores and a cutoff index."""
    data = request.get_json(force=True) if request.data else {}
    exps_config = data.get('experiments', [])
    r_o = float(data.get('r_o', 0.5))
    roi = data.get('roi')
    cut_frac = float(data.get('cut_frac', 0.25))

    if not exps_config:
        return jsonify(error='No experiments specified'), 400

    # Build candidate list (same logic as start_tournament)
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
        raw_first = exp.get('epoch_config', {}).get('raw_first', False)
        for epoch in range(exp['num_epochs']):
            if raw_first and epoch == 0:
                continue
            epoch_label = epoch_labels[epoch] if epoch < len(epoch_labels) else f'Ep.{epoch + 1}'
            candidates.append({
                'exp_id': ec['exp_id'],
                'epoch': epoch,
                'label': f'{exp_label} {epoch_label}',
            })

    if len(candidates) < 4:
        return jsonify(error='Too few candidates for pre-filtering'), 400

    # Compute metrics for each candidate using elimination-proven metrics only.
    # ELIMINATION_METRICS maps metric name -> weight (negative = inverted).
    scored = []
    for c in candidates:
        metrics = _get_cached_metrics(c['exp_id'], c['epoch'], roi, r_o)
        values = {}
        for name, val, hib in metrics:
            if name in ELIMINATION_METRICS:
                values[name] = val
        scored.append({**c, '_values': values})

    # Z-score each metric across all candidates, then weighted average
    for mname, weight in ELIMINATION_METRICS.items():
        vals = [s['_values'].get(mname, 0) for s in scored]
        mean = np.mean(vals)
        std = np.std(vals)
        for s in scored:
            raw = s['_values'].get(mname, 0)
            z = (raw - mean) / std if std > 1e-10 else 0.0
            # weight sign handles inversion (Spec. Struct. has weight -1)
            s['_values'][mname] = z * weight

    # Consensus = weighted mean of z-scores (higher = better)
    for s in scored:
        s['score'] = float(np.mean(list(s['_values'].values())))
        del s['_values']

    # Sort best → worst
    scored.sort(key=lambda s: s['score'], reverse=True)

    # Cutoff index: keep top (1 - cut_frac), at least 3
    import math
    n_keep = max(3, math.ceil(len(scored) * (1 - cut_frac)))
    cutoff = n_keep

    return jsonify(
        candidates=[{
            'exp_id': s['exp_id'],
            'epoch': s['epoch'],
            'label': s['label'],
            'score': round(s['score'], 3),
        } for s in scored],
        cutoff=cutoff,
        total=len(scored),
    )


@app.route('/api/tournament/start', methods=['POST'])
def start_tournament():
    """Start a tournament with one or more experiments.
    Body: {experiments: [{exp_id}, ...], r_o: 0.5, roi: "x,y,w,h"}
    Each (exp_id, epoch) pair is a separate candidate."""
    data = request.get_json(force=True) if request.data else {}
    exps_config = data.get('experiments', [])
    r_o = float(data.get('r_o', 0.5))
    roi = data.get('roi')
    user_name = data.get('user_name', '')
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
        raw_first = exp.get('epoch_config', {}).get('raw_first', False)
        for epoch in range(exp['num_epochs']):
            epoch_label = epoch_labels[epoch] if epoch < len(epoch_labels) else f'Ep.{epoch + 1}'
            # Skip RAW frame from tournament — it's unprocessed reference, not a candidate
            if raw_first and epoch == 0:
                continue
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
        return jsonify(error='No candidates after filtering (all frames may be RAW)'), 400

    candidates.sort(key=lambda c: c['epoch'], reverse=True)

    session_id = str(uuid.uuid4())[:8]
    t = Tournament(list(range(len(candidates))))

    tournaments[session_id] = {
        'tournament': t,
        'candidates': candidates,
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
    data = request.get_json(force=True)
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
        ci = t.get_confidence_intervals()
        history = t.get_history()

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

        confidence = []
        for item_idx in results:
            if item_idx in ci:
                confidence.append(ci[item_idx])
            else:
                confidence.append(None)

        # Compute metrics for all ranked candidates
        roi = tdata['roi']
        r_o = tdata['r_o']
        all_metrics = []
        for c in ranking:
            metrics = _get_cached_metrics(c['exp_id'], c['epoch'], roi, r_o)
            all_metrics.append(metrics)

        # Auto-save (includes choice history)
        save_path = _auto_save_results(ranking, all_metrics, tdata, confidence)

        result['ranking'] = ranking
        result['top3'] = top3
        result['confidence'] = confidence
        result['all_metrics'] = [
            [{'name': n, 'value': round(v, 6), 'higher_is_better': h} for n, v, h in m]
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
        ci = t.get_confidence_intervals()

    ranking = [candidates[i] for i in results]
    top3 = [candidates[i] for i in top_k]
    confidence = []
    for item_idx in t.results:
        confidence.append(ci.get(item_idx))

    roi = tdata['roi']
    r_o = tdata['r_o']
    all_metrics = []
    for c in ranking:
        metrics = _get_cached_metrics(c['exp_id'], c['epoch'], roi, r_o)
        all_metrics.append(metrics)

    save_path = _auto_save_results(ranking, all_metrics, tdata, confidence)

    result = {
        'pair': None,
        'progress': t.progress(),
        'done': True,
        'ranking': ranking,
        'top3': top3,
        'confidence': confidence,
        'all_metrics': [
            [{'name': n, 'value': round(v, 6), 'higher_is_better': h} for n, v, h in m]
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
    """Convert a [0,1] grayscale numpy array to a PNG response."""
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, 'L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def _rank_values(values, higher_is_better=True):
    """Rank values (1 = best)."""
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1], reverse=higher_is_better)
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


def _auto_save_results(ranking, all_metrics, tdata, confidence=None):
    """Save tournament results to TSV with Spearman correlations and choice history."""
    try:
        from scipy.stats import spearmanr

        t = tdata['tournament']
        roi = tdata.get('roi')
        r_o = tdata.get('r_o', 0.5)
        user_name = tdata.get('user_name', '')

        exp_ids = []
        for c in ranking:
            if c['exp_id'] not in exp_ids:
                exp_ids.append(c['exp_id'])
        gif_names = [experiments[eid]['filename'] for eid in exp_ids]

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

        # Confidence intervals
        if confidence:
            lines.append('Ranking Confidence')
            for i, (c, ci) in enumerate(zip(ranking, confidence)):
                label = c.get('full_label', c['label'])
                if ci:
                    lines.append(f'{i+1}\t{label}\trank {ci["rank_lower"]}-{ci["rank_upper"]}\t'
                                 f'strength={ci["strength"]:.4f}\tconf={ci["confidence"]:.2f}')
                else:
                    lines.append(f'{i+1}\t{label}')
            lines.append('')

        # TSV table
        if not all_metrics:
            return save_path  # nothing to save
        metric_names = [n for n, _, _ in all_metrics[0]]
        lines.append('\t'.join(['Rank', 'Candidate'] + metric_names))
        for i, (c, metrics) in enumerate(zip(ranking, all_metrics)):
            label = c.get('full_label', c['label'])
            vals = [f'{v:.6f}' for _, v, _ in metrics]
            lines.append('\t'.join([str(i + 1), label] + vals))

        # Metric vs manual rank correlations
        n = len(ranking)
        if n >= 4:
            lines.append('')
            lines.append('Metric vs Manual Rank (Spearman correlation)')
            manual_ranks = list(range(1, n + 1))
            for mi, (name, _, hib) in enumerate(all_metrics[0]):
                metric_vals = [all_metrics[i][mi][1] for i in range(n)]
                metric_ranks = _rank_values(metric_vals, higher_is_better=hib)
                rho, pval = spearmanr(manual_ranks, metric_ranks)
                lines.append(f'{name}\trho={rho:.4f}\tp={pval:.4f}')

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
        filename = f'tournament_{timestamp}.tsv'

        # Mirror the experiment folder structure inside results_dir
        gif_dirs = [os.path.dirname(gn) for gn in gif_names]
        sub_dir = os.path.commonpath(gif_dirs) if gif_dirs else ''
        dest_dir = os.path.join(results_dir, sub_dir)
        os.makedirs(dest_dir, exist_ok=True)
        save_path = os.path.join(dest_dir, filename)

        with open(save_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

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


def _ensure_loaded(exp_id):
    """Lazy-load frames for an experiment if not yet loaded."""
    exp = experiments.get(exp_id)
    if not exp:
        return None
    if 'frames' in exp:
        with _state_lock:
            _access_times[exp_id] = time.time()
        return exp  # already loaded

    with _state_lock:
        # Double-check after acquiring lock
        if 'frames' in exp:
            _access_times[exp_id] = time.time()
            return exp

        # Evict old experiments before loading new one
        _evict_lru()

        # Load from path
        print(f'  Loading {exp["filename"]}...')
        frames = extract_frames(exp['gif_path'])
        if len(frames) < 2:
            return None
        variability = compute_variability_map(frames)
        exp['frames'] = frames
        exp['variability'] = variability
        exp['num_epochs'] = len(frames)
        exp['height'] = frames[0].shape[0]
        exp['width'] = frames[0].shape[1]
        _access_times[exp_id] = time.time()
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
        print(f'  Loaded {exp["filename"]} ({len(frames)} epochs, {exp["width"]}x{exp["height"]})')
    return exp


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


def load_data_dir(data_dir, cli_config=None):
    """Scan a data directory for GIFs (lazy — only metadata, no pixel loading)."""


    if cli_config is None:
        cli_config = {}

    gif_files = glob_mod.glob(os.path.join(data_dir, '**', '*.gif'), recursive=True)
    for gif_path in gif_files:
        exp_id = str(uuid.uuid4())[:8]
        rel_path = os.path.relpath(gif_path, data_dir)
        # Peek at the GIF to get frame count and dimensions without loading all frames
        try:
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
                continue
        except Exception:
            continue

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

        experiments[exp_id] = {
            'filename': rel_path,
            'gif_path': gif_path,
            'num_epochs': n_frames,
            'height': h,
            'width': w,
            'epoch_config': epoch_config,
            'epoch_labels': labels,
        }
        print(f'  Found {rel_path} ({n_frames} epochs, {w}x{h})')


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
