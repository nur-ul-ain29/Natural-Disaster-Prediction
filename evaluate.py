"""
evaluate.py
===========
Loads a trained checkpoint and reports its accuracy, plus saves a few
example predictions as images so you can SEE how well it's doing, not
just look at a number.

Usage:
    python evaluate.py --dataset floodnet \
        --image_dir data/floodnet/images --mask_dir data/floodnet/masks \
        --in_channels 3 --checkpoint baseline_unet.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import FloodNetDataset, Sen1FloodsDataset
from model import build_model
from train import iou_score, print_iou_report


def save_example_predictions(model, dataset, device, out_dir, num_examples=4, num_classes=2):
    """Save a handful of (input, ground truth, prediction) side-by-side images."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Binary (flood/no-flood) reads cleanly as blue-intensity. 10-class
    # semantic labels need a real categorical colormap -- shading 10
    # classes in one color makes them visually indistinguishable and
    # defeats the point of looking at the image at all.
    cmap = "Blues" if num_classes <= 2 else "tab10"
    vmin, vmax = (0, 1) if num_classes <= 2 else (0, num_classes - 1)

    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            img, mask = dataset[i]
            pred = model(img.unsqueeze(0).to(device))  # add batch dimension of size 1
            pred_mask = pred.argmax(dim=1).squeeze(0).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            # Only show the first 3 channels as an RGB-ish preview, since
            # some inputs (Sen1Floods11) have more than 3 bands.
            preview = img[:3].permute(1, 2, 0).numpy()
            preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-6)
            axes[0].imshow(preview)
            axes[0].set_title("Input (preview)")
            axes[1].imshow(mask.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title("Ground truth mask")
            axes[2].imshow(pred_mask, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2].set_title("Model prediction")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"example_{i}.png"))
            plt.close(fig)
    print(f"Saved {num_examples} example predictions to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sen1floods11", "floodnet"], required=True)
    parser.add_argument("--image_dir", help="Sen1Floods11: path to the image/ folder")
    parser.add_argument("--mask_dir", help="Sen1Floods11: path to the label/ folder")
    parser.add_argument("--root_dir", help="FloodNet: path to the Labeled/ folder")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--full_severity", action="store_true",
                         help="FloodNet only: evaluate against the full 10-class "
                              "semantic labels instead of collapsed binary flood mask. "
                              "Pass --num_classes 10 together with this flag.")
    parser.add_argument("--checkpoint", default="baseline_unet.pt")
    parser.add_argument("--out_dir", default="eval_examples")
    args = parser.parse_args()

    if args.full_severity and args.num_classes != 10:
        parser.error("--full_severity requires --num_classes 10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "sen1floods11":
        dataset = Sen1FloodsDataset(args.image_dir, args.mask_dir)
    else:
        dataset = FloodNetDataset(args.root_dir, binary=not args.full_severity)

    model = build_model(args.in_channels, args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Overall IoU across the whole dataset
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    class_ious = None
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            batch_ious = iou_score(outputs, masks, args.num_classes, per_class=True)
            if class_ious is None:
                class_ious = [0.0] * args.num_classes
            class_ious = [a + b for a, b in zip(class_ious, batch_ious)]
    class_ious = [v / len(loader) for v in class_ious]

    if args.num_classes > 2:
        print("Per-class IoU:")
        print_iou_report(class_ious, args.num_classes)
    else:
        print(f"Overall IoU: {sum(class_ious) / len(class_ious):.4f}")

    save_example_predictions(model, dataset, device, args.out_dir, num_classes=args.num_classes)


if __name__ == "__main__":
    main()
