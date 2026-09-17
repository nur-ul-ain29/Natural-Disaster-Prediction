"""
unified_system.py
==================
ONE entry point handling both disaster types, satisfying "handling diverse
disasters with a single system" -- pick --disaster_type and this routes
to the right pipeline, at the right resolution/speed tradeoff, and can
handle images larger than the model was trained on via tiling.

Covers three requirements at once:
  - Handling diverse disasters with a single system (--disaster_type)
  - Time-sensitive predictions vs. data accuracy (--mode fast/accurate)
  - Geospatial accuracy vs. large-scale coverage (--tile_size)

Usage:
    python unified_system.py --disaster_type flood --input image.tif \
        --checkpoint floodnet_unet.pt --in_channels 3 --mode accurate

    python unified_system.py --disaster_type flood --before b.tif --after a.tif \
        --checkpoint floodnet_unet.pt --in_channels 3 --mode fast --tile_size 512

    python unified_system.py --disaster_type wildfire --before b.tif --after a.tif \
        --nir_band 3 --swir_band 5
"""

import argparse
import json
import os
import time

import numpy as np
import rasterio
import torch

from inference import load_image, predict_mask, save_mask
from model import build_model
from noise_reduction import denoise_optical, denoise_sar
from wildfire_detector import detect_burn_severity, save_result


def tile_image(img, tile_size):
    """Split a (bands, H, W) array into tiles, remembering source position for stitching."""
    _, h, w = img.shape
    tiles = []
    for row in range(0, h, tile_size):
        for col in range(0, w, tile_size):
            tile = img[:, row:row + tile_size, col:col + tile_size]
            tiles.append((tile, row, col))
    return tiles


def stitch_masks(tile_results, full_shape, tile_size):
    """Reassemble per-tile predicted masks into one full-size mask."""
    full_mask = np.zeros(full_shape, dtype=np.uint8)
    for mask, row, col in tile_results:
        h, w = mask.shape
        full_mask[row:row + h, col:col + w] = mask
    return full_mask


def run_flood_tiled(img, geo_meta, model, device, tile_size):
    """Run flood inference over an image, tiling if it's larger than tile_size."""
    _, h, w = img.shape
    if tile_size is None or (h <= tile_size and w <= tile_size):
        return predict_mask(model, img, device)

    tiles = tile_image(img, tile_size)
    results = [(predict_mask(model, tile, device), row, col) for tile, row, col in tiles]
    return stitch_masks(results, (h, w), tile_size)


def _save_timing(disaster_type, mode, tile_size, elapsed_seconds):
    """
    Append this run's real timing to a results file, so the speed-tradeoff
    chart can always be regenerated from actual measured runs instead of
    numbers transcribed out of the conversation.
    """
    path = "results_timing.json"
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    records.append({
        "disaster_type": disaster_type, "mode": mode,
        "tile_size": tile_size, "seconds": elapsed_seconds,
    })
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster_type", choices=["flood", "wildfire"], required=True)
    parser.add_argument("--input", help="Single image (flood mode only)")
    parser.add_argument("--before", help="Before-disaster image")
    parser.add_argument("--after", help="After-disaster image")
    parser.add_argument("--checkpoint", help="Trained model checkpoint (flood only)")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--mode", choices=["fast", "accurate"], default="accurate",
                         help="fast = downsample before inference, skip denoise; "
                              "accurate = full resolution, real denoising applied")
    parser.add_argument("--tile_size", type=int, default=None,
                         help="Split large images into tiles of this size before inference")
    parser.add_argument("--nir_band", type=int, help="Wildfire mode: NIR band index")
    parser.add_argument("--swir_band", type=int, help="Wildfire mode: SWIR band index")
    parser.add_argument("--sensor", choices=["optical", "sar"], default="optical",
                         help="Which denoiser to apply in --mode accurate")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    if args.disaster_type == "wildfire":
        if not (args.before and args.after and args.nir_band is not None and args.swir_band is not None):
            parser.error("wildfire mode requires --before, --after, --nir_band, --swir_band")

        mask, dnbr, geo_meta, severity = detect_burn_severity(
            args.before, args.after, args.nir_band, args.swir_band,
            denoise=(args.mode == "accurate"),
        )
        save_result(mask, geo_meta, os.path.join(args.out_dir, "burn_severity.tif"),
                    severity=severity)

    else:  # flood
        if not args.checkpoint:
            parser.error("flood mode requires --checkpoint")
        model = build_model(args.in_channels, 2).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()

        def load_for_mode(path):
            img, geo_meta = load_image(path)
            if args.mode == "fast":
                img = img[:, ::2, ::2]
            else:
                img = denoise_sar(img) if args.sensor == "sar" else denoise_optical(img)
            return img, geo_meta

        if args.input:
            img, geo_meta = load_for_mode(args.input)
            mask = run_flood_tiled(img, geo_meta, model, device, args.tile_size)
            save_mask(mask, args.out_dir, "flood_mask", geo_meta)
        else:
            if not (args.before and args.after):
                parser.error("flood mode requires --input, or both --before and --after")
            before_img, before_geo = load_for_mode(args.before)
            after_img, after_geo = load_for_mode(args.after)
            before_mask = run_flood_tiled(before_img, before_geo, model, device, args.tile_size)
            after_mask = run_flood_tiled(after_img, after_geo, model, device, args.tile_size)
            newly_flooded = ((after_mask == 1) & (before_mask == 0)).astype(np.uint8)
            geo_meta = after_geo if after_geo is not None else before_geo
            save_mask(before_mask, args.out_dir, "before_mask", geo_meta)
            save_mask(after_mask, args.out_dir, "after_mask", geo_meta)
            save_mask(newly_flooded, args.out_dir, "change_newly_flooded", geo_meta)

    elapsed = time.time() - start_time
    print(f"\nMode: {args.mode} | Tile size: {args.tile_size} | Total time: {elapsed:.2f}s")

    _save_timing(args.disaster_type, args.mode, args.tile_size, elapsed)


if __name__ == "__main__":
    main()
