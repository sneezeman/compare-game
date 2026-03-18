# Compare Game — Epoch Comparison Tool

A web-based tool for visually ranking 3D reconstruction epochs. Upload GIF animations (one frame per epoch), compare pairs side-by-side in a tournament, and find the best epoch using human judgment backed by 20 image quality metrics.

## Quick Start

```bash
git clone https://github.com/sneezeman/compare-game/
cd compare-game/compare_tool
pip install -r requirements.txt
python app.py /path/to/gifs
```

Then open http://127.0.0.1:5000 in your browser.

## Setup

**Requirements:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/sneezeman/compare-game/
cd compare-game

# 2. Create a virtual environment (recommended — outside Google Drive)
python -m venv ~/venvs/compare-game
source ~/venvs/compare-game/bin/activate

# 3. Install dependencies
cd compare_tool
pip install -r requirements.txt
```

## Usage

### Starting the server

```bash
# Basic — point to a directory with GIF files
python app.py /path/to/gifs

# With epoch numbering
python app.py /path/to/gifs --epoch-start 100 --epoch-end 110
python app.py /path/to/gifs --epoch-start 0 --epoch-step 5 --raw-first

# Custom port, accessible on network
python app.py /path/to/gifs --port 8080 --host 0.0.0.0

# Custom results directory
python app.py /path/to/gifs --results-dir ./my_results
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `data_dir` | **(required)** Directory with GIF files to load |
| `--epoch-start N` | First epoch number (e.g. 100) |
| `--epoch-end N` | Last epoch number (used to derive step) |
| `--epoch-step N` | Step between epochs (default: 1) |
| `--raw-first` | First GIF frame is RAW (no denoising) — excluded from tournament |
| `--results-dir DIR` | Where to save results (default: `DATA_DIR/results`) |
| `--port N` | Server port (default: 5000) |
| `--host ADDR` | Server host (default: 127.0.0.1) |

### Epoch Configuration

The tool needs to know how GIF frame numbers map to actual training epochs. It resolves this in order:

1. **CLI flags** — `--epoch-start`, `--epoch-end`, `--epoch-step`, `--raw-first` (applies to all GIFs)
2. **Filename patterns** — automatically detected from GIF filenames:
   - `all_epochs_view0_0-45_raw.gif` — epochs 0 to 45, first frame is RAW
   - `all_epochs_view0_5-180.gif` — epochs 5 to 180
   - `E100-110` or `E0-180-5` — explicit range with optional step
3. **UI prompt** — if neither CLI nor filename provides epoch info, the app shows a configuration form with live preview
4. **Skip** — use frame numbers as-is (Ep.1, Ep.2, ...)

When `raw_first` is set, the RAW frame is automatically **excluded from the tournament** — it serves as an unprocessed reference, not a candidate.

### Workflow (step by step)

**1. Select GIFs**

When the app opens, you'll see pre-loaded experiments from the data directory, plus a drag-and-drop zone for additional uploads. Each GIF should be a grayscale animation where each frame is one training epoch.

Select the GIFs you want to compare and click **"Select GIFs to compare"**.

**2. Configure Epochs**

Set how frame numbers map to epoch numbers (first epoch, step, whether frame 1 is RAW). A live label preview shows the result. Click **"Skip"** to use plain frame numbers.

**3. Pick a Region of Interest (ROI)**

You'll see the first frame with a variability heatmap overlay (red/yellow = areas that change most across epochs). Either:

- **Click a suggested ROI** — the tool auto-detects up to 5 high-variability regions (256x256 minimum)
- **Draw manually** — click and drag to select a custom region

Options:
- Toggle the heatmap on/off with the checkbox
- Adjust **OTF radius (r_o)** if needed (default 0.5 works for most cases)
- Click **"Start Comparison"**

**4. Tournament**

You'll be shown two epoch crops. For each pair:

- Press **Left arrow** or **Right arrow** to toggle between the two images
- Press **Space twice** to select the currently visible image as the winner
- Press **Enter** to declare a tie (metrics break it using z-score-normalized voting, with an explanation shown)
- Press **Ctrl+Z** to undo your last choice
- Press **Esc** to finish early with current rankings

The metrics table uses **color-coded intensity** — big differences between candidates produce vivid green/gray, small differences stay muted.

A progress bar shows comparisons remaining (~14 for 10 candidates, ~31 for 20, ~40 for 26).

**5. Results**

Once done, you'll see:
- A podium with the **top epochs**
- A full ranking table with all image quality metrics (color-coded)
- Results are **auto-saved** to `DATA_DIR/results/` as TSV
- Options to **export results**, **re-run** with the same ROI, pick a **new ROI**, or **start over**

### Keyboard Shortcuts (Tournament Phase)

| Key | Action |
|-----|--------|
| Left/Right arrow | Switch between left and right images |
| Space (x2) | Select visible image as winner |
| Enter | Tie (z-score metrics decide, shows explanation) |
| Ctrl+Z | Undo last choice |
| Esc | Finish early with current rankings |

## Auto-Saved Results

Tournament results are automatically saved to `DATA_DIR/results/` (or `--results-dir`). Each file includes:

- Full ranking with all 20 metric values per candidate
- **Spearman correlation** (rho + p-value) between your manual ranking and each metric's ranking
- **Full pairwise choice history** — every comparison with timestamps, for building learned quality models
- Ranking confidence information

## Image Quality Metrics

20 metrics are used for tie-breaking and analysis:

| Category | Metrics |
|----------|---------|
| Differential | Brenner, Tenengrad, Laplacian Var., SML |
| Correlative | Vollath F4, Vollath F5 |
| Statistical | Normalized Var., Histogram Entropy, Kurtosis, Local Std |
| Spectral (DCT) | DCTS, NDCTBE, HF Energy, Spectral Flatness, Spectral Slope |
| Wavelet | NHWTSE, Wavelet Energy Ratio |
| Cryo-ET specific | LoG Response, Noise Estimate, Contrast Consistency |

For all metrics, higher = better. On a tie, each metric "votes" only when the z-score difference exceeds a threshold (0.5), preventing noise from influencing results. The tie-breaking explanation is shown to the user.

## Input Format

- **GIF files** with 2+ frames
- Each frame = one epoch of a reconstruction
- Grayscale (color GIFs are auto-converted)
- Any resolution (larger images = slower metrics)
- Filenames can encode epoch info (see Epoch Configuration above)

## Data Directory Layout

The tool recursively scans for `.gif` files. A typical layout:

```
data/
├── sample_A_from_ref_B_140/
│   ├── all_epochs_view0_0-45_raw.gif
│   ├── all_epochs_view1_0-45_raw.gif
│   └── all_epochs_view2_0-45_raw.gif
├── sample_A_from_ref_C_170/
│   ├── all_epochs_view0_5-180.gif
│   ├── all_epochs_view1_5-180.gif
│   └── all_epochs_view2_5-180.gif
└── results/
    └── tournament_2026-03-18_143022.tsv   (auto-saved)
```
