"""Unit tests for label-corner detection (Approach A: bright text + dark outline)."""

import os

import numpy as np

from gif_loader import compute_label_exclusion, detect_label_present

SAMPLE_GIF = ('../NM0029_HT_100nm_T001_0001_rec__from_'
              'NM0029_HT_100nm_T000_0001_rec__140/all_epochs_view0.gif')


def _blank(h=120, w=300, val=0.5):
    return np.full((h, w), val, dtype=np.float32)


def _draw_outlined_text(f, r0=10, r1=40, cols=range(20, 200, 20)):
    """White vertical strokes flanked by black outline — a label stand-in."""
    for cx in cols:
        f[r0:r1, cx] = 1.0
        f[r0:r1, cx - 1] = 0.0
        f[r0:r1, cx + 1] = 0.0


def test_uniform_frame_has_no_label():
    assert detect_label_present(_blank(), (0, 0, 300, 120)) is False


def test_white_text_with_dark_outline_is_detected():
    f = _blank()
    _draw_outlined_text(f)
    assert detect_label_present(f, (0, 0, 300, 120)) is True


def test_bright_blob_without_dark_outline_is_not_label():
    f = _blank()
    f[10:60, 10:120] = 1.0  # white block, nothing dark nearby
    assert detect_label_present(f, (0, 0, 300, 120)) is False


def test_dark_region_without_bright_is_not_label():
    f = _blank()
    f[10:60, 10:120] = 0.0
    assert detect_label_present(f, (0, 0, 300, 120)) is False


def test_box_out_of_bounds_or_empty_returns_false():
    f = _blank()
    assert detect_label_present(f, (10_000, 10_000, 50, 50)) is False
    assert detect_label_present(f, (0, 0, 0, 0)) is False


def test_only_pixels_inside_box_count():
    f = _blank()
    _draw_outlined_text(f, r0=90, r1=110)  # label drawn below the box
    assert detect_label_present(f, (0, 0, 300, 60)) is False


def test_real_labeled_frame_detected_corner_only():
    if not os.path.exists(SAMPLE_GIF):
        import pytest
        pytest.skip('sample GIF not present')
    from gif_loader import extract_frames
    f = extract_frames(SAMPLE_GIF)[-1]
    H, W = f.shape
    box = compute_label_exclusion(W, H)
    assert detect_label_present(f, box) is True
    # The same-size center patch is real content, not a label.
    x, y, w, h = box
    center = f[H // 2:H // 2 + h, W // 2 - w:W // 2]
    assert detect_label_present(center, (0, 0, w, h)) is False
