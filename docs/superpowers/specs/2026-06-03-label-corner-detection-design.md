# Label-corner detection + manual override — design

**Date:** 2026-06-03
**Status:** approved (design)

## Problem

The GIF generator burns an epoch/RAW label into the **top-right corner** of every
frame. `compute_label_exclusion(W, H)` in `gif_loader.py` returns that corner box
from the known burn-in geometry, and the ROI-selection screen uses it to:

1. zero the corner in the variability heatmap (`compute_variability_map`), and
2. forbid the user's drag-selected ROI from entering the corner
   (`clampRect()` in `compare.js`, fed by `state.roiExclusion`).

The exclusion is **unconditional** — it assumes the label is always present. GIFs
produced **without** the burn-in (a per-GIF generator flag; all-or-nothing across
frames) needlessly lose that corner for ROI selection.

## Goal

Detect whether the label is actually present in a GIF and only restrict the corner
when it is. Provide a manual checkbox so the user can override a wrong detection.
Default the checkbox to the detection result ("auto default + override").

## Non-goals

- Per-frame detection (it's all-or-nothing per GIF).
- Locating labels placed somewhere other than the known top-right box.
- OCR / reading the label text.

## Approach A — bright-outlined-text signature

The burn-in is near-white text (brightness > ~0.85) wrapped in a dark 5×5 outline
(brightness < ~0.2). Inside the known label box, count bright pixels that have at
least one dark pixel among their 4-neighbors. Tomography content in that corner
almost never places max-white directly adjacent to near-black, so a count above a
small threshold is a reliable "label present".

- Runs on **one** frame: `frames[-1]` (highest epoch — guaranteed a numeric label
  if any are present). Tiny box, no new dependencies.
- Rejected alternatives: temporal-variance (confused by genuinely busy corners +
  the RAW→number transition); OCR (heavy dep, slow, overkill for a yes/no).

### Function

```python
def detect_label_present(frame, box, bright=0.85, dark=0.20, min_pixels=12):
    """frame: 2D float32 [0,1]; box: (x, y, w, h) from compute_label_exclusion.
    Returns True if the box contains a bright-text-on-dark-outline signature."""
```

`min_pixels` and the thresholds are tuned against real labelled GIFs and the
manual override backstops any misfire.

## Data flow

1. **Load time** (`app.py` `_ensure_loaded`, beside `compute_variability_map`):
   compute `box = compute_label_exclusion(W, H)`, then
   `exp['label_present'] = detect_label_present(frames[-1], box)`. Stored as a
   persistent scalar (cheap to recompute on reload; not deleted on LRU eviction).
2. **`/api/roi-suggestions/<exp>`**: return
   - `exclusion: {x,y,w,h}` only when `label_present` is true, else `null`;
   - `label_detected: <bool>` so the frontend knows the verdict.
   `suggest_rois` keeps excluding the box only when the label is present.
3. **Frontend (`compare.js`, ROI screen)**: a checkbox
   *"Top-right corner has burned-in labels (exclude from ROI)"* whose initial
   `checked` = `label_detected`. Toggling it sets `state.roiExclusion` to the box
   (checked) or `null` (unchecked), then redraws the overlay and re-clamps any
   current selection.

## Testing

- Unit-test `detect_label_present` on synthetic frames: (a) a frame with burned-in
  white-on-dark text in the box → True; (b) a plain/noisy frame with no such
  signature → False; (c) a bright-but-no-dark-outline blob → False.
- Tests live in `compare_tool/test_gif_loader.py` (new; first test file in repo).

## Files touched

- `compare_tool/gif_loader.py` — add `detect_label_present`.
- `compare_tool/app.py` — compute/store `label_present`; branch in
  `/api/roi-suggestions`.
- `compare_tool/static/compare.js` — checkbox + toggle handler on ROI screen.
- `compare_tool/templates/index.html` + `static/style.css` — the checkbox element.
- `compare_tool/test_gif_loader.py` — new unit tests.
