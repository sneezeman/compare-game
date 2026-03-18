"""
GIF frame extraction and variability map computation.
"""

import numpy as np
from PIL import Image


def extract_frames(gif_path):
    """Extract all frames from a GIF as grayscale float64 arrays in [0, 1].
    Returns list of numpy arrays."""
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.convert('L')
            arr = np.array(frame, dtype=np.float64) / 255.0
            frames.append(arr)
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def compute_variability_map(frames):
    """Compute pixel-wise standard deviation across frames.
    Returns a 2D array normalized to [0, 1]."""
    stack = np.stack(frames, axis=0)
    std_map = np.std(stack, axis=0)
    max_val = std_map.max()
    if max_val > 0:
        std_map = std_map / max_val
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


def suggest_rois(variability_map, k=5, min_size=256, max_size_frac=0.4):
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
