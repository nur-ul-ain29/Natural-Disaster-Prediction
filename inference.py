"""
inference.py
============
Run a trained model on new images. Two modes:

  SINGLE IMAGE MODE (--input):
    Predict a flood mask for one image. This is the "use the finished
    model" step, as opposed to training or evaluating.

  CHANGE DETECTION MODE (--before and --after):
    Predict masks for a before-disaster image and an after-disaster
    image, then compute the DIFFERENCE between them -- pixels that
    became flooded that weren't before. This is technical task #2 from
    the project brief ("change detection algorithms").

Usage:
    # Single image
    python inference.py --input path/to/image.tif \
        --in_channels 8 --checkpoint baseline_unet.pt --out_dir results

    # Before/after change detection
    python inference.py --before path/to/before.tif --after path/to/after.tif \
        --in_channels 8 --checkpoint baseline_unet.pt --out_dir results
"""

import argparse
import os

import numpy as np
import rasterio
import torch
from PIL import Image

from model import build_model


def load_image(path):
    """
    Loads either a GeoTIFF (multi-band, has geographic metadata) or a
    regular image file (jpg/png), returning a normalized array plus
    the rasterio metadata if available (needed to write a georeferenced
    output later).
    """
    if path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            geo_meta = src.meta.copy()
    else:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))  # (H,W,3) -> (3,H,W)
        img /= 255.0
        geo_meta = None

    mean = img.mean(axis=(1, 2), keepdims=True)
    std = img.std(axis=(1, 2), keepdims=True) + 1e-6
    img = (img - mean) / std
    return img, geo_meta


def predict_mask(model, img, device):
    """
    Run the model on one already-loaded image array and return a plain
    numpy mask (0/1 per pixel). Pulled out as its own function so both
    single-image mode and change-detection mode can reuse it -- exactly
    like you'd factor out a repeated block in any OpenCV script.
    """
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def save_mask(mask, out_dir, name, geo_meta=None):
    """Save a mask as a PNG preview, and as a GeoTIFF if geo_meta is available."""
    png_path = os.path.join(out_dir, f"{name}.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(png_path)
    print(f"Saved: {png_path}")

    if geo_meta is not None:
        meta = geo_meta.copy()
        meta.update(count=1, dtype="uint8")
        tif_path = os.path.join(out_dir, f"{name}.tif")
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(mask.astype(np.uint8), 1)
        print(f"Saved: {tif_path}")


def run_single(args, model, device):
    img, geo_meta = load_image(args.input)
    mask = predict_mask(model, img, device)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    save_mask(mask, args.out_dir, f"{stem}_mask", geo_meta)


def run_change_detection(args, model, device):
    """
    Change detection: predict a mask for both the before and after image,
    then compute where NEW flooding appeared.

    The logic itself is simple, same as comparing two OpenCV masks:
      newly_flooded = after_mask AND NOT before_mask
    i.e. a pixel counts as "newly flooded" only if the after-image
    predicts flood there AND the before-image did NOT.
    """
    before_img, before_geo = load_image(args.before)
    after_img, after_geo = load_image(args.after)

    before_mask = predict_mask(model, before_img, device)
    after_mask = predict_mask(model, after_img, device)

    # Boolean logic, identical to combining two OpenCV masks with
    # bitwise operations -- here using numpy directly.
    newly_flooded = (after_mask == 1) & (before_mask == 0)
    newly_flooded = newly_flooded.astype(np.uint8)

    # Prefer geo metadata from whichever image had it (usually both do,
    # since before/after pairs are normally the same sensor/area).
    geo_meta = after_geo if after_geo is not None else before_geo

    save_mask(before_mask, args.out_dir, "before_mask", geo_meta)
    save_mask(after_mask, args.out_dir, "after_mask", geo_meta)
    save_mask(newly_flooded, args.out_dir, "change_newly_flooded", geo_meta)

    # A quick summary number: how much NEW area flooded, in pixels.
    # (Multiply by per-pixel ground resolution, if known, to convert to
    # real area -- e.g. Sentinel-1 is 10m/pixel, so pixels * 100 = m^2.)
    new_flood_pixels = int(newly_flooded.sum())
    total_pixels = newly_flooded.size
    print(
        f"\nChange summary: {new_flood_pixels} / {total_pixels} pixels "
        f"newly flooded ({100 * new_flood_pixels / total_pixels:.2f}% of image area)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Single image mode: path to one image")
    parser.add_argument("--before", help="Change detection mode: path to before-disaster image")
    parser.add_argument("--after", help="Change detection mode: path to after-disaster image")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--checkpoint", default="baseline_unet.pt")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    if not args.input and not (args.before and args.after):
        parser.error("Provide either --input, or both --before and --after")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.in_channels, args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.input:
        run_single(args, model, device)
    else:
        run_change_detection(args, model, device)


if __name__ == "__main__":
    main()
