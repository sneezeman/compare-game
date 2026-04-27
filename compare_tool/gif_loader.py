"""
GIF frame extraction and variability map computation.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Cache the font + measured text box at each width — the burn-in script uses
# font_size = W//20 with DejaVu Sans Bold and labels of up to 3 characters
# ("RAW", "180", etc.), positioned 10 px from the top/right edges with a
# 5x5 outline. We mirror that exactly so the mask matches the actual label.
_LABEL_BOX_CACHE = {}


def _measure_label_box(W):
    """Return the textbbox tuple for the worst-case label rendered at this
    image width. PIL's textbbox returns the *ink* bounds relative to the
    draw anchor (bbox[1] is non-zero — the y-offset from anchor to the
    top of the ink). Callers must use the bbox as-is and add the anchor
    position to get absolute pixel coordinates."""
    if W in _LABEL_BOX_CACHE:
        return _LABEL_BOX_CACHE[W]
    font_size = max(10, W // 20)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    tmp = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(tmp)
    # Worst-case label: "RAW" is wider in DejaVu Sans Bold than any
    # 3-digit epoch number, and the mask has to cover every frame.
    bbox = draw.textbbox((0, 0), "RAW", font=font)
    _LABEL_BOX_CACHE[W] = bbox
    return bbox


def extract_frames(gif_path):
    """Extract all frames from a GIF as grayscale float32 arrays in [0, 1].
    float32 instead of float64 halves memory: a 38-frame 2000x2000 GIF needs
    ~580 MB instead of ~1.16 GB, which is the difference between staying alive
    and getting OOM-killed on a 4 GB box."""
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.convert('L')
            arr = (np.array(frame, dtype=np.float32) / np.float32(255.0))
            frames.append(arr)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    finally:
        img.close()
    return frames


def compute_label_exclusion(W, H):
    """Top-right corner where the GIF generator burns in epoch/RAW labels.

    The generator does:
        font_size = W // 20
        text_width = textbbox[2] - textbbox[0]
        x_anchor = W - text_width - 10
        y_anchor = 10
        draw.text((x_anchor, y_anchor), label, ...)
        for dx,dy in -2..+2: outline pass

    The ink ends up at:
        x: from (x_anchor + bbox[0]) to (x_anchor + bbox[2])
        y: from (y_anchor + bbox[1]) to (y_anchor + bbox[3])
    expanded by ±2 px from the outline. PIL's bbox[1] is non-zero (font
    ascender area), so the actual top of the ink is well below y=10."""
    bbox = _measure_label_box(W)
    bx0, by0, bx1, by1 = bbox
    OUTLINE = 2
    SAFETY = 4
    # x_anchor uses bbox[2]-bbox[0] as text_width, ink left = x_anchor + bbox[0]
    text_width = bx1 - bx0
    ink_left = (W - text_width - 10) + bx0
    mask_x = max(0, ink_left - OUTLINE - SAFETY)
    mask_w = W - mask_x
    # ink bottom = y_anchor (10) + bbox[3]
    ink_bottom = 10 + by1
    mask_h = min(H, ink_bottom + OUTLINE + SAFETY)
    return (mask_x, 0, mask_w, mask_h)


def compute_variability_map(frames, mask_label_corner=True):
    """Compute pixel-wise standard deviation across frames.
    Returns a 2D array normalized to [0, 1]. The top-right label corner is
    zeroed so it doesn't dominate the heatmap.

    Implemented with Welford's online algorithm so we don't allocate the
    N×H×W stack np.std would build internally — for a 38×2000×2000 GIF
    that stack is ~600 MB on top of the frames already in memory."""
    n = len(frames)
    H, W = frames[0].shape
    if n < 2:
        return np.zeros((H, W), dtype=np.float32)

    mean = np.zeros((H, W), dtype=np.float32)
    m2 = np.zeros((H, W), dtype=np.float32)
    for i, f in enumerate(frames, 1):
        delta = f - mean
        mean += delta / np.float32(i)
        m2 += delta * (f - mean)
    std_map = np.sqrt(m2 / np.float32(n), dtype=np.float32)

    if mask_label_corner:
        x, y, w, h = compute_label_exclusion(W, H)
        std_map[y:y + h, x:x + w] = 0.0
    max_val = float(std_map.max())
    if max_val > 0:
        std_map = std_map / np.float32(max_val)
    return std_map


def variability_to_heatmap_rgba(std_map):
    """Convert a [0,1] variability map to an RGBA heatmap image.
    Uses a red-yellow colormap with alpha proportional to intensity."""
    h, w = std_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Red channel: always high in hot regions
    rgba[:, :, 0] = (255 * np.clip(std_map * 2, 0, 1)).astype(np.uint8)
    # Green channel: rises for very high values (yellow)
    rgba[:, :, 1] = (255 * np.clip(std_map * 2 - 0.5, 0, 1)).astype(np.uint8)
    # Blue: 0
    # Alpha: proportional to variability
    rgba[:, :, 3] = (180 * np.clip(std_map, 0, 1)).astype(np.uint8)
    return rgba


def _compute_iou(a, b):
    """Compute Intersection over Union between two rectangles.

    Each rectangle is a dict with keys x, y, w, h.

    Args:
        a: first rectangle dict {x, y, w, h}
        b: second rectangle dict {x, y, w, h}

    Returns:
        float: IoU value in [0, 1]
    """
    x1 = max(a['x'], b['x'])
    y1 = max(a['y'], b['y'])
    x2 = min(a['x'] + a['w'], b['x'] + b['w'])
    y2 = min(a['y'] + a['h'], b['y'] + b['h'])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def suggest_rois(variability_map, k=5, min_size=256, max_size_frac=0.4,
                 exclusion_box=None):
    """Suggest top-K ROI regions based on variability.

    Uses a sliding window approach to find rectangular regions
    with highest mean variability. Returns non-overlapping regions.

    Args:
        variability_map: 2D array from compute_variability_map(), values in [0,1]
        k: number of suggestions to return
        min_size: minimum ROI dimension in pixels
        max_size_frac: maximum ROI size as fraction of image dimension

    Returns:
        list of dicts: [{x, y, w, h, score}, ...] sorted by score descending
    """
    if variability_map.size == 0:
        return []

    h, w = variability_map.shape

    # If the image is too small for even the minimum window, return the
    # entire image as a single ROI (if it has any variability).
    if h < min_size or w < min_size:
        score = float(variability_map.mean())
        if score == 0.0:
            return []
        return [{'x': 0, 'y': 0, 'w': w, 'h': h, 'score': score}]

    # If the entire map is zero there is nothing interesting to suggest.
    if variability_map.max() == 0.0:
        return []

    # ------------------------------------------------------------------
    # Step 1: Integral image for O(1) rectangle summation.
    # integral[i, j] = sum of variability_map[0:i, 0:j]
    # We pad with a row/col of zeros so that the rectangle formula is clean.
    # ------------------------------------------------------------------
    integral = np.zeros((h + 1, w + 1), dtype=np.float64)
    np.cumsum(variability_map, axis=0, out=integral[1:, 1:])
    np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

    # ------------------------------------------------------------------
    # Step 2: Determine window sizes to try.
    # We pick 4 sizes linearly spaced between min_size and max_dim.
    # ------------------------------------------------------------------
    max_dim_h = max(min_size, int(h * max_size_frac))
    max_dim_w = max(min_size, int(w * max_size_frac))

    num_scales = 4
    sizes_h = np.linspace(min_size, max_dim_h, num_scales, dtype=int)
    sizes_w = np.linspace(min_size, max_dim_w, num_scales, dtype=int)
    # Remove duplicate sizes
    window_sizes = list(dict.fromkeys(zip(sizes_h.tolist(), sizes_w.tolist())))

    # ------------------------------------------------------------------
    # Steps 3-4: Slide each window across the image and score by mean
    # variability.
    # ------------------------------------------------------------------
    candidates = []  # list of (score, x, y, w_win, h_win)

    for win_h, win_w in window_sizes:
        if win_h > h or win_w > w:
            continue
        stride_y = max(1, win_h // 2)
        stride_x = max(1, win_w // 2)
        area = win_h * win_w

        for y in range(0, h - win_h + 1, stride_y):
            for x in range(0, w - win_w + 1, stride_x):
                # Skip windows that overlap the label-exclusion box at all.
                # Even partial overlap means the metric will be biased.
                if exclusion_box is not None:
                    ex, ey, ew, eh = exclusion_box
                    if (x < ex + ew and x + win_w > ex and
                            y < ey + eh and y + win_h > ey):
                        continue
                # Sum via integral image: sum = I[y2,x2] - I[y1,x2] - I[y2,x1] + I[y1,x1]
                y2 = y + win_h
                x2 = x + win_w
                region_sum = (
                    integral[y2, x2]
                    - integral[y, x2]
                    - integral[y2, x]
                    + integral[y, x]
                )
                mean_score = region_sum / area
                candidates.append((mean_score, x, y, win_w, win_h))

    if not candidates:
        return []

    # ------------------------------------------------------------------
    # Step 5: Sort candidates by score descending.
    # ------------------------------------------------------------------
    candidates.sort(key=lambda c: c[0], reverse=True)

    # ------------------------------------------------------------------
    # Step 6: Greedy non-maximum suppression (IoU < 0.3).
    # ------------------------------------------------------------------
    selected = []
    for score, cx, cy, cw, ch in candidates:
        rect = {'x': cx, 'y': cy, 'w': cw, 'h': ch}
        overlaps = False
        for sel in selected:
            if _compute_iou(rect, sel) >= 0.3:
                overlaps = True
                break
        if not overlaps:
            selected.append({**rect, 'score': float(score)})
        if len(selected) >= k:
            break

    return selected
