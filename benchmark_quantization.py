"""
benchmark_quantization.py
==========================
Quantization shrinks a trained model's numbers from 32-bit floats to
8-bit integers, making it smaller and faster -- at some accuracy cost.
This measures that trade-off directly for the "limited computational
power" requirement.

Usage:
    python benchmark_quantization.py --checkpoint floodnet_unet.pt \
        --root_dir "data/floodnet/FloodNet Challenge - Track 1/Train/Labeled" \
        --in_channels 3
"""

import argparse
import json
import os
import time

import torch

from datasets import FloodNetDataset, Sen1FloodsDataset
from model import build_model
from train import iou_score


def get_model_size_mb(model):
    torch.save(model.state_dict(), "_tmp_size_check.pt")
    size_mb = os.path.getsize("_tmp_size_check.pt") / (1024 * 1024)
    os.remove("_tmp_size_check.pt")
    return size_mb


def benchmark(model, dataset, device, num_samples=20):
    model.eval()
    total_time = 0.0
    total_iou = 0.0
    n = min(num_samples, len(dataset))
    with torch.no_grad():
        for i in range(n):
            img, mask = dataset[i]
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            start = time.time()
            output = model(img)
            total_time += time.time() - start

            total_iou += iou_score(output, mask, num_classes=2)
    return total_time / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_dir", help="FloodNet root labeled folder")
    parser.add_argument("--image_dir", help="Sen1Floods11 image folder")
    parser.add_argument("--mask_dir", help="Sen1Floods11 mask folder")
    parser.add_argument("--in_channels", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cpu")

    if args.root_dir:
        dataset = FloodNetDataset(args.root_dir)
    else:
        dataset = Sen1FloodsDataset(args.image_dir, args.mask_dir)

    model = build_model(args.in_channels, 2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    orig_size = get_model_size_mb(model)
    orig_time, orig_iou = benchmark(model, dataset, device)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    quant_size = get_model_size_mb(quantized_model)
    quant_time, quant_iou = benchmark(quantized_model, dataset, device)

    print("\n--- Quantization Benchmark ---")
    print(f"{'Metric':<20}{'Original':<15}{'Quantized':<15}{'Change':<15}")
    print(f"{'Model size (MB)':<20}{orig_size:<15.2f}{quant_size:<15.2f}{100*(quant_size-orig_size)/orig_size:+.1f}%")
    print(f"{'Time/image (sec)':<20}{orig_time:<15.4f}{quant_time:<15.4f}{100*(quant_time-orig_time)/orig_time:+.1f}%")
    print(f"{'IoU':<20}{orig_iou:<15.4f}{quant_iou:<15.4f}{100*(quant_iou-orig_iou)/orig_iou:+.1f}%")

    # Save real results to a file, so the report chart can be regenerated
    # from actual saved data instead of transcribed numbers.
    with open("results_quantization.json", "w") as f:
        json.dump({
            "original": {"size_mb": orig_size, "time_sec": orig_time, "iou": orig_iou},
            "quantized": {"size_mb": quant_size, "time_sec": quant_time, "iou": quant_iou},
        }, f, indent=2)
    print("\nSaved: results_quantization.json")


if __name__ == "__main__":
    main()
