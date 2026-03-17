# Compare Game — Epoch Comparison Tool

A web-based tool for visually ranking 3D reconstruction epochs. Upload GIF animations (one frame per epoch), compare pairs side-by-side in a tournament, and find the best epoch using human judgment backed by image quality metrics.

## Quick Start

```bash
cd compare_tool
pip install -r requirements.txt
python app.py --data-dir ..
```

Then open http://127.0.0.1:5000 in your browser.

## Setup

**Requirements:** Python 3.8+

```bash
# 1. Clone the repo
git clone <repo-url>
cd Compare_game

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
cd compare_tool
pip install -r requirements.txt
```

## Usage

### Starting the server

```bash
# Pre-load GIFs from a directory
python app.py --data-dir /path/to/gifs

# Or start empty and upload GIFs through the browser
python app.py

# Options
python app.py --port 8080 --host 0.0.0.0  # custom port, accessible on network
```

### Workflow (step by step)

**1. Select GIFs**

When the app opens, you'll see either pre-loaded experiments (if you used `--data-dir`) or a drag-and-drop zone to upload GIF files. Each GIF should be a grayscale animation where each frame is one training epoch.

Select the GIFs you want to compare and click **"Select GIFs to compare"**.

**2. Pick a Region of Interest (ROI)**

You'll see the first frame with a variability heatmap overlay (red/yellow = areas that change most across epochs). Click and drag to draw a rectangle around the region you care about.

- Toggle the heatmap on/off with the checkbox
- Adjust **OTF radius (r_o)** if needed (default 0.5 works for most cases)
- Click **"Start Comparison"**

**3. Tournament**

You'll be shown two epoch crops side-by-side. For each pair:

- Press **Left arrow** or **Right arrow** to toggle between the two images
- Press **Space twice** to select the currently visible image as the winner
- Press **Enter** to declare a tie (metrics will break it automatically)
- Press **Ctrl+Z** to undo your last choice

A progress bar shows how many comparisons remain. The tournament uses a merge-sort algorithm, so it needs roughly `n * log(n) * 0.65` comparisons to rank all epochs.

**4. Results**

Once done, you'll see:
- A podium with the **top 3 epochs**
- A full ranking table with all image quality metrics
- Options to **export results** (TSV), **re-run** with the same ROI, pick a **new ROI**, or **start over**

### Keyboard Shortcuts (Tournament Phase)

| Key | Action |
|-----|--------|
| Left/Right arrow | Switch between left and right images |
| Space (×2) | Select visible image as winner |
| Enter | Tie (metrics decide) |
| Ctrl+Z | Undo last choice |

## Image Quality Metrics

The tie-breaking system uses 17 metrics from [Royer et al. 2016](https://doi.org/10.1038/nbt.3708):

| Category | Metrics |
|----------|---------|
| Differential | Brenner, Tenengrad, Laplacian Var., SML |
| Correlative | Vollath F4, Vollath F5 |
| Statistical | Normalized Var., Histogram Entropy, Kurtosis, Local Std |
| Spectral (DCT) | DCTS, NDCTBE, HF Energy, Spectral Flatness, Spectral Slope |
| Wavelet | NHWTSE, Wavelet Energy Ratio |

For all metrics, higher = better. On a tie, each metric "votes" for one side, and the majority wins.

## Input Format

- **GIF files** with 2+ frames
- Each frame = one epoch of a reconstruction
- Grayscale (color GIFs are auto-converted)
- Any resolution (larger images = slower metrics)

## Data Directory Layout

If using `--data-dir`, the tool recursively scans for `.gif` files. A typical layout:

```
data/
├── sample_A_from_ref_B_140/
│   ├── all_epochs_view0.gif
│   ├── all_epochs_view1.gif
│   └── all_epochs_view2.gif
└── sample_A_from_ref_C_170/
    ├── all_epochs_view0.gif
    ├── all_epochs_view1.gif
    └── all_epochs_view2.gif
```
