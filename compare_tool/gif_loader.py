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
