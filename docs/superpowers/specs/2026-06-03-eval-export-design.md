# Eval export buttons (CSV + sbatch .sh) — design

**Date:** 2026-06-03
**Status:** approved (design)

## Problem

After running tournaments, the selected best epoch for each finetuning tile is
buried in saved TSVs. To launch SLURM eval jobs the user hand-edits
`eval_manual.sh` (four variables) and `sbatch`es it, once per tile. Tedious and
error-prone across many tiles.

## Goal

Two buttons on the main page (`phase-upload`) that export the selected results of
**finetuning** tournaments:

1. **Export eval sheet** — a CSV with columns
   `tile, full_metadata_name, selected_finetuning_tile, selected_epoch`.
   Example row: `T01, S1M_RH_rot_HTb_100nm_T01_rec_, T35, 198`.
2. **Export eval script** — one `.sh` that, for every row, rewrites the four
   `eval_manual.sh` variables and `sbatch`es it, all blocks joined with `&&`
   into a single long chain.

## Scope of rows

- Source: the already-scanned `past_results` (all saved TSVs, access-filtered by
  the requesting user — reuse the existing filter from `list_past_results`).
- One row per result = its **winner** (first entry of the `Top 3:` line).
- **Finetune results only.** Training-tile results have no `_from_` checkpoint, so
  they can't populate the finetune columns or the eval script; skip them.
- **Subset by selection:** if the request carries selected experiment ids
  (`state.selected` from the main-page `.exp-checkbox`es), export only results
  whose winner GIF belongs to a selected experiment. Empty/absent ⇒ all.
- Deterministic order: sort rows by `full_metadata_name` then `selected_epoch`.

## Parsing (the winner string)

Winner example:
```
.../NM0029_HT_100nm_T001_0001_rec__from_NM0029_HT_100nm_T000_0001_rec__140/all_epochs_view2_141-150.gif Ep.142
```
- `Ep.(\d+)` at the end → **selected_epoch** (142).
- The GIF's parent dir matches
  `^(?P<meta>.+?_rec_)_from_(?P<ckpt>.+?_rec_)_(?P<fe>\d+)$`:
  - `meta` → **full_metadata_name** (`NM0029_HT_100nm_T001_0001_rec_`)
  - `ckpt` → **full_checkpoint_metadata** (`NM0029_HT_100nm_T000_0001_rec_`)
  - `fe` → **finetune_epoch** (140)
- short tiles via `_?(T\d+)(?:_\d+)?_rec_` on meta/ckpt → **tile** (T001),
  **selected_finetuning_tile** (T000).

A result whose winner dir doesn't match the finetune pattern is skipped.

## Architecture

Shared, unit-tested helper (no Flask, pure):

```python
def collect_eval_rows(results, selected_filenames=None):
    """results: list of past_results dicts (with 'gifs', 'top3').
    selected_filenames: optional set of GIF filenames to keep.
    Returns list of EvalRow(tile, full_metadata, finetune_tile, selected_epoch,
                            full_checkpoint_metadata, finetune_epoch),
    finetune-only, sorted by (full_metadata, selected_epoch)."""
```

Two thin endpoints build files from the helper and return attachments:

- `POST /api/export/eval-sheet` → `text/csv`, `eval_selection.csv`.
- `POST /api/export/eval-script` → `text/x-shellscript`, `eval_selection.sh`.

Body: `{exp_ids: [...]}` (optional). The endpoint maps `exp_ids` →
their GIF filenames to build `selected_filenames`, applies the same
per-user access filter as `list_past_results`, then calls the helper.

### CSV output

```
tile,full_metadata_name,selected_finetuning_tile,selected_epoch
T001,NM0029_HT_100nm_T001_0001_rec_,T000,142
```

### .sh output

Updated 2026-06-04: the eval script is now the argument-driven `batch_eval.sh`
(`sbatch batch_eval.sh <metadata> <epoch> <checkpoint_metadata> <finetune_epoch>
[workdir]`), replacing the sed-rewrite of `eval_manual.sh`. The script prompts
once for the dataset workdir, then emits one independent `sbatch` line per row
(no `&&` chain — the jobs no longer share a file to rewrite, so a bad submit
must not abort the rest):

```bash
#!/bin/bash
read -rp "Enter workdir (e.g. /data/visitor/ls3639/id16a/20260211/PROCESSED_DATA/NM0029/denoising/): " workdir

sbatch batch_eval.sh "NM0029_HT_100nm_T001_0001_rec_" "142" "NM0029_HT_100nm_T000_0001_rec_" "140" "$workdir"
sbatch batch_eval.sh "<next metadata>" "<epoch>" "<next checkpoint>" "<finetune_epoch>" "$workdir"
...
```

## Frontend

Two buttons in `phase-upload` near the Past Results section. Each POSTs the
currently-checked `exp_id`s (from `state.selected`) — empty array means "all" —
and downloads the returned blob (`eval_selection.csv` / `.sh`). A short note: "no
selection ⇒ exports all processed finetuning results."

## Testing

`compare_tool/test_eval_export.py` (new) for `collect_eval_rows`:
- finetune winner → correct row fields;
- training-tile winner (no `_from_`) → skipped;
- multi-GIF result → winner's own GIF drives the row;
- subset filter keeps only selected filenames;
- ordering is deterministic.

## Files touched

- `compare_tool/app.py` — `collect_eval_rows` helper + two endpoints.
- `compare_tool/static/compare.js` — two button handlers (POST + blob download).
- `compare_tool/templates/index.html` + `static/style.css` — the two buttons.
- `compare_tool/test_eval_export.py` — new unit tests.

## Non-goals

- Editing/owning `eval_manual.sh` itself (the generated script edits it on the
  cluster).
- Real `.xlsx` (CSV chosen; opens in Excel, zero new dependency).
- Exporting training-tile or non-winner epochs.
