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


# Shown in the generated script's prompt — the dataset's denoising/metadata/
# directory, which submit_eval.sh takes a path into as its first argument.
_METADATA_DIR_EXAMPLE = ('/data/visitor/ls3639/id16a/20260211/'
                         'PROCESSED_DATA/NM0029/denoising/metadata')


# The cluster path to the dataset's denoising/metadata/ directory can't be
# derived from a GIF path (the beamtime date isn't in it), so the command is
# emitted with this token and the UI substitutes the dir the user has saved.
METADATA_DIR_PLACEHOLDER = '<METADATA_DIR>'

# How every generated eval line starts. One constant so the results-screen
# one-liner and the batch export can't drift apart.
_SUBMIT = 'sbatch submit_eval.sh'


def build_command(row, metadata_dir=None):
    """The `submit_eval.sh` line for one EvalRow (always transfer learning —
    collect_eval_rows only yields finetune rows, so the 4-arg form applies).

    metadata_dir may be a shell expression such as '$metadir' for use inside a
    generated script.
    """
    base = (metadata_dir or METADATA_DIR_PLACEHOLDER).rstrip('/')
    return (f'{_SUBMIT} "{base}/{row.full_metadata}.json" '
            f'"{row.selected_epoch}" "{row.full_checkpoint_metadata}" '
            f'"{row.finetune_epoch}"')


def command_for_winner(full_label, metadata_dir=None):
    """One copy-pastable `submit_eval.sh` line for a tournament winner.

    Follows submit_eval.sh's argument contract, which takes a *path* to the
    metadata file and decides the mode purely from the positional count:

      training run (no transfer)  ->  sbatch submit_eval.sh "<meta>.json" "<epoch>"
          the script then sets checkpoint_metadata=metadata, finetune_epoch=epoch
      transfer learning (finetune) ->  ... "<epoch>" "<ckpt_metadata>" "<ft_epoch>"
          where ckpt_metadata is a bare *name*, not a path

    (1 or 3 arguments are rejected by that script, so we only ever emit 2 or 4.)

    full_label is the app's '<gif path> Ep.<N>' candidate label — the same shape
    the saved-results 'Top 3:' line uses, so the parsing is shared with
    collect_eval_rows. metadata_dir, when given, is prefixed onto the metadata
    filename; otherwise METADATA_DIR_PLACEHOLDER is emitted for the UI to fill.

    Returns None when the path is not a recognizable experiment directory.
    """
    if not full_label:
        return None
    parsed = _parse_winner(full_label)
    if not parsed:
        return None
    gif_path, epoch = parsed
    dirname = os.path.basename(os.path.dirname(gif_path))
    base = (metadata_dir or METADATA_DIR_PLACEHOLDER).rstrip('/')

    ft = _FT_DIR_RE.match(dirname)
    if ft:
        meta, ckpt, fe = ft.group('meta'), ft.group('ckpt'), int(ft.group('fe'))
        return (f'{_SUBMIT} "{base}/{meta}.json" "{epoch}" '
                f'"{ckpt}" "{fe}"')
    if _TRAIN_DIR_RE.match(dirname):
        # The whole dir name is the metadata; two args means "no transfer".
        return f'{_SUBMIT} "{base}/{dirname}.json" "{epoch}"'
    return None


def build_sh(rows):
    """Batch script of `submit_eval.sh` calls — the same front-end the results
    screen emits, so there is only one submission script to maintain.

    Prompts once for the dataset's denoising/metadata directory (submit_eval.sh
    takes a *path* to the metadata .json) and reuses it for every line. Jobs are
    independent, so the lines aren't chained — a bad submit won't abort the rest.
    Returns just the shebang when there are no rows.
    """
    if not rows:
        return '#!/bin/bash\n'
    lines = [
        '#!/bin/bash',
        f'read -rp "Enter metadata dir (e.g. {_METADATA_DIR_EXAMPLE}): " metadir',
        'metadir="${metadir%/}"',
        '',
    ]
    for r in rows:
        lines.append(build_command(r, '$metadir'))
    return '\n'.join(lines) + '\n'
