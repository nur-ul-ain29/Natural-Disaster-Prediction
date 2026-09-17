"""
wildfire_detector.py
=====================
A lightweight, CLASSICAL (non-trained) wildfire detector, using the same
band-math approach real GEE wildfire tutorials use. This exists so the
overall system handles a SECOND disaster type without needing a full
training run (which wouldn't finish in time tonight).

The core idea, using bands you'd get from Sentinel-2 or similar:
  - NDVI (Normalized Difference Vegetation Index): healthy vegetation
    reflects strongly in near-infrared (NIR) and weakly in red light.
    NDVI = (NIR - Red) / (NIR + Red)
    Burned/dead vegetation has LOW NDVI.
  - NBR (Normalized Burn Ratio): uses NIR and short-wave infrared (SWIR).
    NBR = (NIR - SWIR) / (NIR + SWIR)
    Burned areas have LOW NBR. Comparing NBR before vs. after a fire
    (dNBR = NBR_before - NBR_after) is the standard remote-sensing
    technique for mapping burn severity -- this is a genuine, widely
    used method, not a toy simplification.

This is deliberately classical/threshold-based rather than a trained
model, matching the note in the project conversation that time-series/
tiering pieces can be design-level -- this one IS implemented, just with
simpler math than the flood U-Net, which is an honest and defensible
choice given time constraints (no labeled wildfire training data was
prepared this session).
"""

import numpy as np
import rasterio

from noise_reduction import denoise_optical


def compute_ndvi(nir_band, red_band, eps=1e-6):
    """NDVI: healthy vegetation is high, burned/dead vegetation is low."""
    return (nir_band - red_band) / (nir_band + red_band + eps)


def compute_nbr(nir_band, swir_band, eps=1e-6):
    """NBR: healthy vegetation is high, burned areas are low."""
    return (nir_band - swir_band) / (nir_band + swir_band + eps)


# USGS-convention dNBR severity tiers (the standard bucketing used in
# real burn-severity mapping, rather than a single burned/not-burned
# threshold). Values are the commonly cited breakpoints:
#   < 0.1            : unburned / very low
#   0.1  - 0.27       : low severity
#   0.27 - 0.44       : moderate-low severity
#   0.44 - 0.66       : moderate-high severity
#   >= 0.66           : high severity
SEVERITY_THRESHOLDS = [0.1, 0.27, 0.44, 0.66]
SEVERITY_LABELS = ["unburned", "low", "moderate_low", "moderate_high", "high"]


def classify_severity(dnbr):
    """
    Bucket a dNBR array into severity tiers (0=unburned .. 4=high), using
    the USGS thresholds above, instead of a single binary cutoff. Gives
    an actual damage-assessment output (severity map) rather than just a
    burned/not-burned flag.
    """
    severity = np.zeros_like(dnbr, dtype=np.uint8)
    for level, thresh in enumerate(SEVERITY_THRESHOLDS, start=1):
        severity[dnbr >= thresh] = level
    return severity


def detect_burn_severity(before_path, after_path, nir_idx, swir_idx, threshold=0.1,
                          denoise=False):
    """
    Compare NBR before and after a suspected fire event to produce a
    burn-severity mask, using two multi-band GeoTIFFs of the same area.

    nir_idx, swir_idx: which band index (0-based) in your GeoTIFF file
    corresponds to near-infrared and short-wave-infrared. This varies by
    satellite -- e.g. for Sentinel-2, NIR is band 8, SWIR is band 12,
    but exact index depends on which bands you exported.

    threshold: dNBR value above which a pixel is called "burned" for the
    simple binary mask. 0.1-0.27 is the commonly used range in
    remote-sensing literature for low-to-moderate burn severity.

    denoise: if True, apply a bilateral filter to each band before
    computing NBR (Technical Task 3 -- noise reduction). Off by default
    to match prior behavior / fast mode.

    Returns (burn_mask, dnbr, geo_meta, severity) -- severity is a 0-4
    tiered map from classify_severity(), giving real damage-assessment
    granularity beyond the single binary mask.
    """
    with rasterio.open(before_path) as src:
        before = src.read().astype(np.float32)
        geo_meta = src.meta.copy()
    with rasterio.open(after_path) as src:
        after = src.read().astype(np.float32)

    if denoise:
        before = denoise_optical(before)
        after = denoise_optical(after)

    nbr_before = compute_nbr(before[nir_idx], before[swir_idx])
    nbr_after = compute_nbr(after[nir_idx], after[swir_idx])

    dnbr = nbr_before - nbr_after  # positive = vegetation loss (burned)
    burn_mask = (dnbr > threshold).astype(np.uint8)
    severity = classify_severity(dnbr)

    return burn_mask, dnbr, geo_meta, severity


def save_result(burn_mask, geo_meta, out_path, severity=None):
    meta = geo_meta.copy()
    meta.update(count=1, dtype="uint8")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(burn_mask, 1)
    print(f"Saved burn mask: {out_path}")
    burned_pct = 100 * burn_mask.sum() / burn_mask.size
    print(f"Burned area: {burned_pct:.2f}% of image")

    if severity is not None:
        sev_path = out_path.replace(".tif", "_severity.tif")
        with rasterio.open(sev_path, "w", **meta) as dst:
            dst.write(severity, 1)
        print(f"Saved severity tiers (0-4): {sev_path}")
        for level, label in enumerate(SEVERITY_LABELS):
            pct = 100 * (severity == level).sum() / severity.size
            print(f"  {label:<15}{pct:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Pre-fire GeoTIFF")
    parser.add_argument("--after", required=True, help="Post-fire GeoTIFF")
    parser.add_argument("--nir_band", type=int, required=True, help="0-based NIR band index")
    parser.add_argument("--swir_band", type=int, required=True, help="0-based SWIR band index")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--denoise", action="store_true",
                         help="Apply bilateral-filter noise reduction before computing NBR")
    parser.add_argument("--out", default="burn_severity.tif")
    args = parser.parse_args()

    mask, dnbr, geo_meta, severity = detect_burn_severity(
        args.before, args.after, args.nir_band, args.swir_band, args.threshold,
        denoise=args.denoise,
    )
    save_result(mask, geo_meta, args.out, severity=severity)
