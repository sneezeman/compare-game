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


def build_sh(rows):
    """One long &&-chained script: rewrite eval_manual.sh's four vars and
    sbatch it, once per row. Returns just the shebang when there are no rows."""
    blocks = []
    for r in rows:
        blocks.append(
            f'sed -i \'s/^metadata=.*/metadata="{r.full_metadata}"/\' eval_manual.sh && \\\n'
            f'sed -i \'s/^epoch=.*/epoch="{r.selected_epoch}"/\' eval_manual.sh && \\\n'
            f'sed -i \'s/^checkpoint_metadata=.*/checkpoint_metadata="{r.full_checkpoint_metadata}"/\' eval_manual.sh && \\\n'
            f'sed -i \'s/^finetune_epoch=.*/finetune_epoch="{r.finetune_epoch}"/\' eval_manual.sh && \\\n'
            f'sbatch eval_manual.sh'
        )
    if not blocks:
        return '#!/bin/bash\n'
    return '#!/bin/bash\n' + ' && \\\n'.join(blocks) + '\n'
