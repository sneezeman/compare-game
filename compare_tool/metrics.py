"""
Image quality metrics for epoch comparison.
Includes metrics from Royer et al. 2016 plus additional sharpness,
texture, and spectral measures.
All metrics operate on 2D grayscale numpy arrays (float64, values in [0,1]).
"""

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import convolve


# ---------------------------------------------------------------------------
# Differential metrics
# ---------------------------------------------------------------------------

def brenner(img):
    """Brenner's focus measure (Eq. 10): sum of squared horizontal differences."""
    d = img[:, 2:] - img[:, :-2]
    return float(np.mean(d ** 2))


def tenengrad(img):
    """Tenengrad / Sobel gradient magnitude (Eq. 15)."""
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sy = sx.T
    gx = convolve(img, sx, mode='reflect')
    gy = convolve(img, sy, mode='reflect')
    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_variance(img):
    """Variance of the Laplacian — popular focus measure (e.g. OpenCV).
    Higher = sharper edges and more texture."""
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = convolve(img, lap_kernel, mode='reflect')
    return float(np.var(lap))


def sum_modified_laplacian(img):
    """Sum Modified Laplacian (SML, Eq. 12 in Royer et al.).
    Uses absolute second derivatives, more robust than standard Laplacian."""
    # Second derivative in x and y separately
    d2x = np.abs(img[:, 2:] - 2 * img[:, 1:-1] + img[:, :-2])
    d2y = np.abs(img[2:, :] - 2 * img[1:-1, :] + img[:-2, :])
    # Align sizes
    h = min(d2x.shape[0], d2y.shape[0])
    w = min(d2x.shape[1], d2y.shape[1])
    return float(np.mean(d2x[:h, :w] + d2y[:h, :w]))


# ---------------------------------------------------------------------------
# Correlative metrics
# ---------------------------------------------------------------------------

def vollath_f4(img):
    """Vollath's F4 autocorrelation measure (Eq. 18)."""
    a1 = np.mean(img[:, :-1] * img[:, 1:])
    a2 = np.mean(img[:, :-2] * img[:, 2:])
    return float(a1 - a2)


def vollath_f5(img):
    """Vollath's F5 autocorrelation measure (Eq. 19).
    F5 = autocorrelation(1) - mean^2. More noise-robust than F4."""
    a1 = np.mean(img[:, :-1] * img[:, 1:])
    mu = np.mean(img)
    return float(a1 - mu ** 2)


# ---------------------------------------------------------------------------
# Statistical metrics
# ---------------------------------------------------------------------------

def normalized_variance(img):
    """Normalized variance (Eq. 24)."""
    mu = np.mean(img)
    if mu == 0:
        return 0.0
    return float(np.var(img) / mu)


def histogram_entropy(img):
    """Histogram entropy (Eq. 27)."""
    hist, _ = np.histogram(img, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    bin_width = 1.0 / 256
    p = hist * bin_width  # normalize to probabilities
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def kurtosis(img):
    """Excess kurtosis of intensity distribution.
    Natural images have positive kurtosis (heavy tails).
    Higher kurtosis often indicates more structured content vs smooth/noisy."""
    mu = np.mean(img)
    var = np.var(img)
    if var < 1e-15:
        return 0.0
    m4 = np.mean((img - mu) ** 4)
    return float(m4 / (var ** 2) - 3.0)


def local_std_mean(img, block_size=7):
    """Mean of local standard deviations (texture richness measure).
    Computed in non-overlapping blocks. Higher = more local texture detail."""
    h, w = img.shape
    bh = h // block_size
    bw = w // block_size
    if bh == 0 or bw == 0:
        return float(np.std(img))
    cropped = img[:bh * block_size, :bw * block_size]
    blocks = cropped.reshape(bh, block_size, bw, block_size)
    local_stds = np.std(blocks, axis=(1, 3))
    return float(np.mean(local_stds))


# ---------------------------------------------------------------------------
# Spectral (DCT-based) metrics
# ---------------------------------------------------------------------------

def _dct_spectrum(img, r_o):
    """Compute the DCT power spectrum within OTF support radius r_o."""
    M, N = img.shape
    C = dctn(img, norm='ortho')
    # frequency grid
    u = np.arange(M).reshape(-1, 1) / M
    v = np.arange(N).reshape(1, -1) / N
    r = np.sqrt(u ** 2 + v ** 2)
    mask = r <= r_o
    return C, mask


def dcts(img, r_o=0.5):
    """Normalized DCT Shannon entropy — DCTS (Eq. 32). Recommended metric."""
    C, mask = _dct_spectrum(img, r_o)
    power = C[mask] ** 2
    total = np.sum(power)
    if total == 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    n = np.sum(mask)
    if n <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(n))


def ndctbe(img, r_o=0.5):
    """Normalized DCT Bayes entropy — NDCTBE (Eq. 30)."""
    C, mask = _dct_spectrum(img, r_o)
    power = C[mask] ** 2
    total = np.sum(power)
    if total == 0:
        return 0.0
    p = power / total
    n = np.sum(mask)
    if n <= 1:
        return 0.0
    # Bayes entropy: uses (p_i + 1/n) / (1 + 1/n) ≈ adjusted probabilities
    p_adj = (p + 1.0 / n) / (1.0 + 1.0)
    p_adj = p_adj[p_adj > 0]
    entropy = -np.sum(p_adj * np.log(p_adj))
    return float(entropy / np.log(n))


def hf_energy_ratio(img, r_o=0.5):
    """High-frequency energy ratio within OTF support.
    Fraction of DCT energy above r_o/2 vs total within r_o.
    Higher = more high-frequency content = sharper."""
    M, N = img.shape
    C = dctn(img, norm='ortho')
    u = np.arange(M).reshape(-1, 1) / M
    v = np.arange(N).reshape(1, -1) / N
    r = np.sqrt(u ** 2 + v ** 2)
    mask_all = r <= r_o
    mask_hf = (r > r_o / 2) & (r <= r_o)
    total = np.sum(C[mask_all] ** 2)
    if total == 0:
        return 0.0
    hf = np.sum(C[mask_hf] ** 2)
    return float(hf / total)


def spectral_flatness(img, r_o=0.5):
    """Spectral flatness (Wiener entropy) of DCT power spectrum.
    Ratio of geometric mean to arithmetic mean of power.
    Higher = flatter spectrum (more noise-like). Lower = more structured.
    We return 1 - flatness so higher = better (more structured)."""
    C, mask = _dct_spectrum(img, r_o)
    power = C[mask] ** 2
    power = power[power > 0]
    if len(power) < 2:
        return 0.0
    log_geo_mean = np.mean(np.log(power))
    arith_mean = np.mean(power)
    if arith_mean == 0:
        return 0.0
    flatness = np.exp(log_geo_mean) / arith_mean
    return float(1.0 - flatness)


def spectral_slope(img):
    """Power spectral density slope (beta) via radial averaging.
    Natural images follow P(f) ~ f^(-beta). Typical beta ≈ 2.
    Well-denoised images preserve this slope; over-smoothed images have steeper slopes."""
    M, N = img.shape
    F = np.fft.fft2(img)
    P = np.abs(F) ** 2
    # Radial average
    u = np.fft.fftfreq(M).reshape(-1, 1)
    v = np.fft.fftfreq(N).reshape(1, -1)
    r = np.sqrt(u ** 2 + v ** 2)
    r_flat = r.ravel()
    P_flat = P.ravel()
    # Bin by frequency
    max_r = 0.5
    n_bins = min(M, N) // 2
    if n_bins < 3:
        return 0.0
    bins = np.linspace(0, max_r, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    radial_power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
        if np.any(mask):
            radial_power[i] = np.mean(P_flat[mask])
    # Fit log-log slope (skip DC and zero bins)
    valid = (radial_power > 0) & (bin_centers > 0)
    if np.sum(valid) < 3:
        return 0.0
    log_f = np.log(bin_centers[valid])
    log_p = np.log(radial_power[valid])
    # Linear regression
    A = np.vstack([log_f, np.ones_like(log_f)]).T
    slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
    return float(-slope)  # return positive beta


# ---------------------------------------------------------------------------
# Wavelet metrics
# ---------------------------------------------------------------------------

def _haar_wavelet_2d(img):
    """Single-level 2D Haar wavelet transform, return detail coefficients."""
    # Ensure even dimensions
    h, w = img.shape
    h = h - h % 2
    w = w - w % 2
    img = img[:h, :w]

    # Row transform
    lo = (img[:, 0::2] + img[:, 1::2]) / 2.0
    hi = (img[:, 0::2] - img[:, 1::2]) / 2.0

    # Column transform on lo
    ll = (lo[0::2, :] + lo[1::2, :]) / 2.0
    lh = (lo[0::2, :] - lo[1::2, :]) / 2.0

    # Column transform on hi
    hl = (hi[0::2, :] + hi[1::2, :]) / 2.0
    hh = (hi[0::2, :] - hi[1::2, :]) / 2.0

    return lh, hl, hh


def nhwtse(img):
    """Normalized Haar Wavelet Transform Shannon Entropy — NHWTSE (Eq. 35).
    Works well for denoising data despite weaker paper benchmarks."""
    lh, hl, hh = _haar_wavelet_2d(img)
    detail = np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()])
    power = detail ** 2
    total = np.sum(power)
    if total == 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    n = len(detail)
    if n <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(n))


def wavelet_energy_ratio(img, levels=3):
    """Multi-level wavelet energy ratio: detail energy / total energy.
    Measures how much energy is in detail (edges/texture) vs approximation (low-freq).
    Higher = more texture/edges preserved = better denoising."""
    current = img.copy()
    detail_energy = 0.0
    total_energy = float(np.sum(img ** 2))
    if total_energy == 0:
        return 0.0
    for _ in range(levels):
        if min(current.shape) < 4:
            break
        lh, hl, hh = _haar_wavelet_2d(current)
        detail_energy += np.sum(lh ** 2) + np.sum(hl ** 2) + np.sum(hh ** 2)
        # Next level: approximate coefficients
        h, w = current.shape
        h = h - h % 2
        w = w - w % 2
        current = current[:h, :w]
        current = (current[0::2, 0::2] + current[0::2, 1::2] +
                   current[1::2, 0::2] + current[1::2, 1::2]) / 4.0
    return float(detail_energy / total_energy)


# ---------------------------------------------------------------------------
# Convenience: compute all metrics
# ---------------------------------------------------------------------------

METRICS = [
    # Differential
    ("Brenner", brenner, True),
    ("Tenengrad", tenengrad, True),
    ("Laplacian Var", laplacian_variance, True),
    ("SML", sum_modified_laplacian, True),
    # Correlative
    ("Vollath F4", vollath_f4, True),
    ("Vollath F5", vollath_f5, True),
    # Statistical
    ("Norm. Var.", normalized_variance, True),
    ("Hist. Entropy", histogram_entropy, True),
    ("Kurtosis", kurtosis, True),
    ("Local Std", local_std_mean, True),
    # Spectral (DCT-based)
    ("DCTS", dcts, True),              # recommended by Royer et al.
    ("NDCTBE", ndctbe, True),
    ("HF Energy", hf_energy_ratio, True),
    ("Spec. Struct.", spectral_flatness, True),  # 1-flatness: higher = more structured
    ("Spec. Slope", spectral_slope, True),       # beta: higher = steeper falloff
    # Wavelet
    ("NHWTSE", nhwtse, True),
    ("Wavelet E.R.", wavelet_energy_ratio, True),
]

# Metrics that take r_o parameter
_RO_METRICS = {dcts, ndctbe, hf_energy_ratio, spectral_flatness}


def compute_all(img, r_o=0.5):
    """Compute all metrics on a grayscale image (values in [0, 1]).
    Returns list of (name, value, higher_is_better)."""
    results = []
    for name, func, hib in METRICS:
        if func in _RO_METRICS:
            val = func(img, r_o=r_o)
        else:
            val = func(img)
        results.append((name, val, hib))
    return results
