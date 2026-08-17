"""Build eval-launch exports from past tournament results.

Pure (no Flask) so it can be unit-tested directly. The Flask endpoints in
app.py wrap these with access control and HTTP plumbing.

A finetune experiment directory encodes everything we need:
    <metadata>_rec__from_<checkpoint_metadata>_rec__<finetune_epoch>
and the tournament winner string adds the selected epoch:
    <gif_path> Ep.<selected_epoch>
"""

import csv
import io
import os
import re
from collections import namedtuple

EvalRow = namedtuple('EvalRow', [
    'tile',                      # short target tile, e.g. T001
    'full_metadata',             # e.g. NM0029_HT_100nm_T001_0001_rec_
    'finetune_tile',             # short source/checkpoint tile, e.g. T000
    'selected_epoch',            # winner epoch (int)
    'full_checkpoint_metadata',  # e.g. NM0029_HT_100nm_T000_0001_rec_
    'finetune_epoch',            # checkpoint epoch finetuning started from (int)
])

# "<gif_path> Ep.142" — the winner is the first entry of the Top 3 line.
_WINNER_RE = re.compile(r'^(?P<gif>.+?)\s+Ep\.(?P<ep>\d+)\s*$')
# Finetune dir: <meta>_rec_ _from_ <ckpt>_rec_ _ <finetune_epoch>
_FT_DIR_RE = re.compile(r'^(?P<meta>.+?_rec_)_from_(?P<ckpt>.+?_rec_)_(?P<fe>\d+)$')
# Short tile token (e.g. T01, T002) as an underscore-delimited segment. The tile
# isn't always right before `_rec_` — names vary: `..._T01_rec_`,
# `..._T002_0001_rec_`, `..._T03_050nm_rec_`. The leading delimiter keeps the `T`
# in tokens like `HT_100nm` from matching (its `T` isn't digit-led anyway).
_TILE_RE = re.compile(r'(?:^|_)(T\d+)(?:_|$)')
# A plain training-run directory: the whole dir name IS the metadata, ending in
# `_rec_`/`_rec` (no `_from_` checkpoint). Tile names vary a lot across datasets
# — some carry no T<number> token at all — so match only the _rec suffix.
_TRAIN_DIR_RE = re.compile(r'^(?!.*_from_).+_rec_?$')


def _parse_winner(top3):
    """Return (gif_path, epoch) for the winner, or None if unparseable."""
    if not top3:
        return None
    first = top3.split(',')[0].strip()
    m = _WINNER_RE.match(first)
    if not m:
        return None
    return m.group('gif').strip(), int(m.group('ep'))


def _short_tile(full_metadata):
    m = _TILE_RE.search(full_metadata)
    return m.group(1) if m else ''


def collect_eval_rows(results, selected_filenames=None):
    """Turn past-result dicts into finetune EvalRows.

    results: iterable of dicts with 'top3' (and optionally 'gifs').
    selected_filenames: if not None, keep only winners whose GIF path is in it.
    Training-tile / unrecognized winners are skipped. Sorted by
    (full_metadata, selected_epoch) for deterministic output.
    """
    rows = []
    for res in results:
        parsed = _parse_winner(res.get('top3', ''))
        if not parsed:
            continue
        gif_path, epoch = parsed
        if selected_filenames is not None and gif_path not in selected_filenames:
            continue
        dirname = os.path.basename(os.path.dirname(gif_path))
        dm = _FT_DIR_RE.match(dirname)
        if not dm:
            continue  # training tile or unrecognized layout
        meta, ckpt, fe = dm.group('meta'), dm.group('ckpt'), int(dm.group('fe'))
        rows.append(EvalRow(
            tile=_short_tile(meta),
            full_metadata=meta,
            finetune_tile=_short_tile(ckpt),
            selected_epoch=epoch,
            full_checkpoint_metadata=ckpt,
            finetune_epoch=fe,
        ))
    rows.sort(key=lambda r: (r.full_metadata, r.selected_epoch))
    return rows


def build_csv(rows):
    """CSV with the four user-facing columns."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['tile', 'full_metadata_name',
                'selected_finetuning_tile', 'selected_epoch'])
    for r in rows:
        w.writerow([r.tile, r.full_metadata, r.finetune_tile, r.selected_epoch])
    return buf.getvalue()


# Shown in the workdir prompt — the dataset's denoising/ directory. batch_eval.sh
# takes it as the optional 5th arg; the user pastes their own path at runtime.
_WORKDIR_EXAMPLE = ('/data/visitor/ls3639/id16a/20260211/'
                    'PROCESSED_DATA/NM0029/denoising/')


def build_command(row):
    """One copy-pastable sbatch line for a single EvalRow.

    The workdir (batch_eval.sh's optional 5th arg) is deliberately omitted so
    the line can be pasted and run as-is; batch_eval.sh then falls back to its
    own DEFAULT_DATASET_DIR. Append a workdir argument to override it.
    """
    return (f'sbatch batch_eval.sh "{row.full_metadata}" "{row.selected_epoch}" '
            f'"{row.full_checkpoint_metadata}" "{row.finetune_epoch}"')


def command_for_winner(full_label):
    """Eval command for a tournament winner, or None if one can't be built.

    full_label is the app's '<gif path> Ep.<N>' candidate label — the same shape
    the saved-results 'Top 3:' line uses, so finetune parsing is shared with
    collect_eval_rows rather than duplicated.

    Unlike the CSV/.sh batch export (finetune-only by design), this also covers
    **training tiles**: batch_eval.sh evaluates those through its
    `metadata == checkpoint_metadata` branch, reading
    results/checkpoints/<metadata>/weights_epoch_<epoch>.torch. So the tile's own
    metadata is repeated as the checkpoint and the epoch as finetune_epoch (that
    argument is unused on that branch). Returns None when the path is not a
    recognizable experiment directory.
    """
    if not full_label:
        return None

    rows = collect_eval_rows([{'top3': full_label}])
    if rows:
        return build_command(rows[0])

    # Not a finetune experiment — try the training-tile form.
    parsed = _parse_winner(full_label)
    if not parsed:
        return None
    gif_path, epoch = parsed
    dirname = os.path.basename(os.path.dirname(gif_path))
    if not _TRAIN_DIR_RE.match(dirname):
        return None
    return (f'sbatch batch_eval.sh "{dirname}" "{epoch}" "{dirname}" "{epoch}"')


def build_sh(rows):
    """Argument-driven batch script: prompt once for the workdir, then one
    `sbatch batch_eval.sh <metadata> <epoch> <checkpoint_metadata>
    <finetune_epoch> <workdir>` line per row. Jobs are independent (no shared
    file to rewrite), so the lines aren't chained — a bad submit won't abort
    the rest. Returns just the shebang when there are no rows."""
    if not rows:
        return '#!/bin/bash\n'
    lines = [
        '#!/bin/bash',
        f'read -rp "Enter workdir (e.g. {_WORKDIR_EXAMPLE}): " workdir',
        '',
    ]
    for r in rows:
        lines.append(
            f'sbatch batch_eval.sh "{r.full_metadata}" "{r.selected_epoch}" '
            f'"{r.full_checkpoint_metadata}" "{r.finetune_epoch}" "$workdir"')
    return '\n'.join(lines) + '\n'
