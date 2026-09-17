"""
noise_reduction.py
===================
TECHNICAL TASK 3 (previously missing): noise reduction and image
enhancement, applied before change-detection / segmentation inference.

Two different noise problems, two different filters -- using the wrong
one matters:

  SAR (Sentinel-1, VV/VH) images are corrupted by SPECKLE -- a
  multiplicative, grainy salt-and-pepper-like noise inherent to radar
  imaging, not the additive Gaussian noise regular photos have. The
  standard classical fix is a LEE FILTER: it adapts to local variance,
  smoothing flat/homogeneous areas heavily while preserving edges
  (important -- you don't want to blur the flood boundary you're trying
  to detect).

  Optical images (Sentinel-2, FloodNet RGB) have more ordinary sensor
  noise plus haze/cloud-edge softness. A BILATERAL FILTER is the
  standard choice: like Gaussian blur, but it also respects edges by
  weighting neighbors by intensity similarity, not just spatial
  distance -- so it denoises without smearing flood/burn boundaries.

Both are classical (no training needed), fast, and directly usable on
the numpy arrays already flowing through inference.py, unified_system.py,
and gee_pipeline.py.

Usage:
    from noise_reduction import lee_filter, denoise_optical, denoise_image

    # SAR, single band, shape (H, W)
    vv_clean = lee_filter(vv_band, window_size=5)

    # Optical, shape (C, H, W) or (H, W, C)
    rgb_clean = denoise_optical(rgb_array)

    # Auto-dispatch based on channel count / declared sensor type
    clean = denoise_image(img_array, sensor="sar")   # or sensor="optical"
"""

import numpy as np


def lee_filter(band, window_size=5):
    """
    Classical Lee filter for SAR speckle reduction on a single 2D band.

    Speckle is multiplicative: observed = true_signal * noise. The Lee
    filter estimates the true signal per-pixel using the local mean and
    variance in a sliding window:

        output = mean + k * (pixel - mean)

    where k = local_variance / (local_variance + noise_variance) shrinks
    toward the local mean in flat/noisy regions (k near 0) and toward the
    original pixel value near strong edges/high-variance regions (k near
    1) -- this is what preserves edges instead of just blurring uniformly
    like a plain moving-average filter would.

    band: 2D numpy array (single SAR channel, e.g. VV or VH)
    window_size: sliding window side length in pixels (odd number, e.g. 5 or 7)
    """
    band = band.astype(np.float32)
    pad = window_size // 2
    padded = np.pad(band, pad, mode="reflect")

    # Compute local mean and local variance via a sliding-window sum,
    # using cumulative sums for speed instead of a naive nested loop
    # over every pixel (same result, much faster on real image sizes).
    def local_mean_and_var(img, w):
        kernel = np.ones((w, w), dtype=np.float32) / (w * w)
        # Sliding window mean via 2D convolution-equivalent using cumsum trick.
        mean = _box_filter(img, w)
        mean_sq = _box_filter(img * img, w)
        var = mean_sq - mean ** 2
        var[var < 0] = 0  # guard against tiny negative values from float rounding
        return mean, var

    local_mean, local_var = local_mean_and_var(padded, window_size)
    local_mean = local_mean[pad:-pad, pad:-pad] if pad > 0 else local_mean
    local_var = local_var[pad:-pad, pad:-pad] if pad > 0 else local_var

    # Overall image noise variance, estimated from the whole image as a
    # standard approximation (used in the classical Lee filter formula).
    overall_var = np.var(band)
    noise_var = overall_var if overall_var > 0 else 1e-6

    k = local_var / (local_var + noise_var + 1e-6)
    output = local_mean + k * (band - local_mean)
    return output.astype(np.float32)


def _box_filter(img, w):
    """Fast sliding-window average using an integral image (summed-area table)."""
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, wid = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    half = w // 2
    for i in range(h):
        r0, r1 = max(0, i - half), min(h, i + half + 1)
        for j in range(wid):
            c0, c1 = max(0, j - half), min(wid, j + half + 1)
            total = (
                integral[r1, c1] - integral[r0, c1]
                - integral[r1, c0] + integral[r0, c0]
            )
            out[i, j] = total / ((r1 - r0) * (c1 - c0))
    return out


def denoise_sar(img):
    """
    Apply the Lee filter to every band of a multi-band SAR array.
    img: (bands, H, W), e.g. (2, H, W) for VV/VH from get_sentinel1_pair.
    """
    return np.stack([lee_filter(img[b]) for b in range(img.shape[0])], axis=0)


def denoise_optical(img, sigma_color=25, sigma_space=5):
    """
    Bilateral filter for optical imagery (RGB or multi-band). Requires
    OpenCV, which is already a project dependency (used in datasets.py).

    img: (C, H, W) or (H, W, C), float or uint8. Returns same shape/dtype
    family as input (float32 in, float32 out).
    """
    import cv2

    was_chw = img.ndim == 3 and img.shape[0] <= 12 and img.shape[0] < img.shape[-1]
    arr = np.transpose(img, (1, 2, 0)) if was_chw else img
    arr = arr.astype(np.float32)

    # cv2.bilateralFilter needs 1 or 3 channels; for >3-band imagery
    # (e.g. multi-band satellite tiles), filter each band independently.
    if arr.ndim == 2 or arr.shape[-1] in (1, 3):
        out = cv2.bilateralFilter(arr, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    else:
        bands = [
            cv2.bilateralFilter(arr[..., c], d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            for c in range(arr.shape[-1])
        ]
        out = np.stack(bands, axis=-1)

    return np.transpose(out, (2, 0, 1)) if was_chw else out


def denoise_image(img, sensor="optical"):
    """
    Convenience dispatcher: pick the right filter for the sensor type.
    sensor: "sar" (Sentinel-1, speckle -> Lee filter) or
            "optical" (Sentinel-2 / FloodNet RGB -> bilateral filter).
    """
    if sensor == "sar":
        return denoise_sar(img)
    elif sensor == "optical":
        return denoise_optical(img)
    else:
        raise ValueError(f"Unknown sensor type: {sensor!r} (expected 'sar' or 'optical')")
