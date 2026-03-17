"""
Flask server for the Compare Game — epoch comparison tool.
"""

import argparse
import io
import os
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from gif_loader import compute_variability_map, extract_frames, variability_to_heatmap_rgba
from metrics import compute_all
from tournament import Tournament

app = Flask(__name__)

# In-memory store: exp_id -> experiment data
experiments = {}
# Tournament store: session_id -> tournament data
tournaments = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/experiments')
def list_experiments():
    """List all loaded experiments."""
    result = []
    for exp_id, exp in experiments.items():
        result.append({
            'exp_id': exp_id,
            'filename': exp['filename'],
            'num_epochs': exp['num_epochs'],
            'height': exp['height'],
            'width': exp['width'],
        })
    return jsonify(experiments=result)


@app.route('/api/upload', methods=['POST'])
def upload():
    """Accept GIF file(s), extract frames, return experiment ID."""
    if 'file' not in request.files:
        return jsonify(error='No file uploaded'), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.gif'):
        return jsonify(error='Only GIF files are supported'), 400

    exp_id = str(uuid.uuid4())[:8]

    # Save temporarily and extract frames
    tmp_path = os.path.join('/tmp', f'{exp_id}_{f.filename}')
    f.save(tmp_path)

    frames = extract_frames(tmp_path)
    os.remove(tmp_path)

    if len(frames) < 2:
        return jsonify(error='GIF must have at least 2 frames'), 400

    variability = compute_variability_map(frames)

    experiments[exp_id] = {
        'filename': f.filename,
        'frames': frames,
        'variability': variability,
        'num_epochs': len(frames),
        'height': frames[0].shape[0],
        'width': frames[0].shape[1],
    }

    return jsonify(
        exp_id=exp_id,
        filename=f.filename,
        num_epochs=len(frames),
        height=frames[0].shape[0],
        width=frames[0].shape[1],
    )


@app.route('/api/frame/<exp_id>/<int:epoch>')
def get_frame(exp_id, epoch):
    """Return a frame as PNG. Optional ?roi=x,y,w,h for cropping."""
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    if epoch < 0 or epoch >= exp['num_epochs']:
        return jsonify(error='Invalid epoch'), 400

    img = exp['frames'][epoch]
    img = _apply_roi(img, request.args.get('roi'))

    return _numpy_to_png_response(img)


@app.route('/api/variability/<exp_id>')
def get_variability(exp_id):
    """Return variability heatmap as RGBA PNG overlay."""
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
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    return _numpy_to_png_response(exp['frames'][0])


@app.route('/api/metrics/<exp_id>/<int:epoch>')
def get_metrics(exp_id, epoch):
    """Return metric values for a frame (optionally cropped by ROI)."""
    exp = _ensure_loaded(exp_id)
    if not exp:
        return jsonify(error='Experiment not found'), 404
    if epoch < 0 or epoch >= exp['num_epochs']:
        return jsonify(error='Invalid epoch'), 400

    r_o = float(request.args.get('r_o', 0.5))
    img = exp['frames'][epoch]
    img = _apply_roi(img, request.args.get('roi'))

    results = compute_all(img, r_o=r_o)
    return jsonify(metrics=[
        {'name': name, 'value': round(val, 6), 'higher_is_better': hib}
        for name, val, hib in results
    ])


@app.route('/api/tournament/start', methods=['POST'])
def start_tournament():
    """Start a tournament with one or more experiments.
    Body: {experiments: [{exp_id, roi}, ...], r_o: 0.5}
    Each (exp_id, epoch) pair is a separate candidate."""
    data = request.get_json(force=True) if request.data else {}
    exps_config = data.get('experiments', [])
    r_o = float(data.get('r_o', 0.5))
    roi = data.get('roi')

    if not exps_config:
        return jsonify(error='No experiments specified'), 400

    # Build flat list of candidates: [{exp_id, epoch, label}, ...]
    candidates = []
    for ec in exps_config:
        exp = _ensure_loaded(ec['exp_id'])
        if not exp:
            return jsonify(error=f'Experiment {ec["exp_id"]} not found'), 404
        short_name = os.path.basename(exp['filename'])
        for epoch in range(exp['num_epochs']):
            candidates.append({
                'exp_id': ec['exp_id'],
                'epoch': epoch,
                'label': f'{short_name} Ep.{epoch + 1}',
                'full_label': f'{exp["filename"]} Ep.{epoch + 1}',
            })

    # Shuffle so the user compares random pairs, not sequential ones
    import random
    indices = list(range(len(candidates)))
    random.shuffle(indices)
    candidates = [candidates[i] for i in indices]

    session_id = str(uuid.uuid4())[:8]
    t = Tournament(list(range(len(candidates))))

    tournaments[session_id] = {
        'tournament': t,
        'candidates': candidates,
        'roi': roi,
        'r_o': r_o,
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
    On tie, metrics break it."""
    tdata = tournaments.get(session_id)
    if not tdata:
        return jsonify(error='No active tournament'), 404

    t = tdata['tournament']
    candidates = tdata['candidates']
    data = request.get_json(force=True)
    winner = data.get('winner')

    if winner == 'tie':
        pair = t.current_pair()
        if pair:
            r_o = tdata['r_o']
            roi = tdata['roi']
            ca, cb = candidates[pair[0]], candidates[pair[1]]

            img_a = experiments[ca['exp_id']]['frames'][ca['epoch']]
            img_b = experiments[cb['exp_id']]['frames'][cb['epoch']]
            if roi:
                img_a = _apply_roi(img_a, roi)
                img_b = _apply_roi(img_b, roi)

            metrics_a = compute_all(img_a, r_o=r_o)
            metrics_b = compute_all(img_b, r_o=r_o)

            a_wins = 0
            b_wins = 0
            for (_, va, hib), (_, vb, _) in zip(metrics_a, metrics_b):
                if hib:
                    if va > vb: a_wins += 1
                    elif vb > va: b_wins += 1
                else:
                    if va < vb: a_wins += 1
                    elif vb < va: b_wins += 1

            winner = 'left' if a_wins >= b_wins else 'right'

    if winner not in ('left', 'right'):
        return jsonify(error='Invalid winner'), 400

    t.choose(winner)
    pair = t.current_pair()
    progress = t.progress()

    result = {
        'pair': _pair_info(candidates, pair),
        'progress': {'current': progress[0], 'total': progress[1]},
        'done': t.is_done(),
    }

    if t.is_done():
        result['ranking'] = [candidates[i] for i in t.results]
        result['top3'] = [candidates[i] for i in t.get_top_k()]

    return jsonify(**result)


@app.route('/api/tournament/<session_id>/undo', methods=['POST'])
def tournament_undo(session_id):
    """Undo last tournament choice."""
    tdata = tournaments.get(session_id)
    if not tdata:
        return jsonify(error='No active tournament'), 404

    t = tdata['tournament']
    candidates = tdata['candidates']
    pair = t.undo()
    progress = t.progress()

    return jsonify(
        pair=_pair_info(candidates, pair),
        progress={'current': progress[0], 'total': progress[1]},
        done=t.is_done(),
        undone=pair is not None,
    )


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


# ---------------------------------------------------------------------------
# Data directory loading
# ---------------------------------------------------------------------------

def _ensure_loaded(exp_id):
    """Lazy-load frames for an experiment if not yet loaded."""
    exp = experiments.get(exp_id)
    if not exp:
        return None
    if 'frames' in exp:
        return exp  # already loaded
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
    print(f'  Loaded {exp["filename"]} ({len(frames)} epochs, {exp["width"]}x{exp["height"]})')
    return exp


def load_data_dir(data_dir):
    """Scan a data directory for GIFs (lazy — only metadata, no pixel loading)."""
    import glob as glob_mod

    gif_files = glob_mod.glob(os.path.join(data_dir, '**', '*.gif'), recursive=True)
    for gif_path in gif_files:
        exp_id = str(uuid.uuid4())[:8]
        rel_path = os.path.relpath(gif_path, data_dir)
        # Peek at the GIF to get frame count and dimensions without loading all frames
        try:
            img = Image.open(gif_path)
            w, h = img.size
            n_frames = 0
            try:
                while True:
                    n_frames += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            if n_frames < 2:
                continue
        except Exception:
            continue

        experiments[exp_id] = {
            'filename': rel_path,
            'gif_path': gif_path,
            'num_epochs': n_frames,
            'height': h,
            'width': w,
        }
        print(f'  Found {rel_path} ({n_frames} epochs, {w}x{h})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Game — Epoch Comparison Tool')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory with GIF files to pre-load')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()

    if args.data_dir:
        print(f'Scanning GIFs from {args.data_dir}...')
        load_data_dir(args.data_dir)
        print(f'Found {len(experiments)} experiment(s) (lazy loading — frames loaded on first use)')

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
