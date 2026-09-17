"""
train.py
========
The actual training loop. Conceptually this is just:

    for many rounds (epochs):
        for each batch of images:
            1. run images through the model -> get predicted masks
            2. compare predicted masks to the real masks -> get a "loss"
               (a single number: how wrong was the model?)
            3. tell PyTorch to figure out how to adjust the model's
               internal numbers to reduce that wrongness (backward pass)
            4. actually apply that adjustment (optimizer step)

That five-line idea is the entire concept of "training a neural network."
Everything below is that loop, plus bookkeeping (progress bars, saving
the best model, measuring accuracy on held-out data).

Usage:
    python train.py --dataset sen1floods11 \
        --image_dir data/sen1floods11/images --mask_dir data/sen1floods11/masks \
        --in_channels 8 --epochs 10

    python train.py --dataset floodnet \
        --image_dir data/floodnet/images --mask_dir data/floodnet/masks \
        --in_channels 3 --epochs 10
"""

import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import FloodNetDataset, Sen1FloodsDataset
from model import build_model


FLOODNET_CLASS_NAMES = [
    "background", "building_flooded", "building_non_flooded",
    "road_flooded", "road_non_flooded", "water", "tree", "vehicle",
    "pool", "grass",
]


def iou_score(pred, target, num_classes=2, eps=1e-6, ignore_index=-1, per_class=False):
    """
    IoU (Intersection over Union): of all pixels either the prediction or
    the ground truth called a given class, what fraction did BOTH agree
    on? This is the standard accuracy metric for segmentation -- same
    idea as comparing two OpenCV masks with bitwise_and / bitwise_or and
    counting pixels, just formalized.

    ignore_index: pixels equal to this value in `target` are excluded
    entirely (not counted as a wrong prediction either) -- needed for
    Sen1Floods11, whose masks use -1 for no-data/cloud-obscured pixels
    that were never actually labeled.

    per_class: if True, return the list of per-class IoUs instead of the
    mean. With FloodNet's full 10-class labels, classes are heavily
    imbalanced (e.g. "vehicle"/"pool" cover very few pixels) -- a single
    averaged IoU can look artificially low even when the common classes
    (water, building, road) are segmented well. Per-class IoU is the
    honest way to report a multi-class run.
    """
    pred = pred.argmax(dim=1)  # model outputs a score per class; pick the highest
    valid = target != ignore_index
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid
        target_cls = (target == cls) & valid
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        ious.append((intersection / (union + eps)).item())
    return ious if per_class else sum(ious) / len(ious)


def print_iou_report(ious, num_classes):
    """
    Print per-class IoU with FloodNet's real class names when
    num_classes == 10, otherwise a generic per-class breakdown.
    """
    names = FLOODNET_CLASS_NAMES if num_classes == 10 else [f"class_{i}" for i in range(num_classes)]
    for name, iou in zip(names, ious):
        print(f"  {name:<22}{iou:.4f}")
    print(f"  {'mean':<22}{sum(ious) / len(ious):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sen1floods11", "floodnet"], required=True)
    parser.add_argument("--image_dir", help="Sen1Floods11: path to the image/ folder")
    parser.add_argument("--mask_dir", help="Sen1Floods11: path to the label/ folder")
    parser.add_argument("--root_dir", help="FloodNet: path to the Labeled/ folder (contains Flooded/, Non-Flooded/)")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--full_severity", action="store_true",
                         help="FloodNet only: use the full 10-class semantic labels "
                              "(building/road/water/etc, flooded vs non-flooded) "
                              "instead of collapsing to binary flood/no-flood. "
                              "Pass --num_classes 10 together with this flag.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--checkpoint", default="baseline_unet.pt")
    args = parser.parse_args()

    # Use the GPU if one's available (much faster), otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    if args.dataset == "sen1floods11":
        if not args.image_dir or not args.mask_dir:
            parser.error("sen1floods11 requires --image_dir and --mask_dir")
        full_dataset = Sen1FloodsDataset(args.image_dir, args.mask_dir)
    else:
        if not args.root_dir:
            parser.error("floodnet requires --root_dir (the Labeled/ folder)")
        if args.full_severity and args.num_classes != 10:
            parser.error("--full_severity requires --num_classes 10")
        full_dataset = FloodNetDataset(args.root_dir, binary=not args.full_severity)

    print(f"Loaded {len(full_dataset)} image/mask pairs.")

    # Split into training data (model learns from this) and validation data
    # (held out, used only to check progress -- never trained on).
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # DataLoader wraps a Dataset and automatically batches + shuffles it.
    # Instead of you writing "for i in range(0, len(data), batch_size): ...",
    # you just iterate over the DataLoader directly.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Build model, loss function, optimizer ---
    model = build_model(args.in_channels, args.num_classes).to(device)

    # CrossEntropyLoss: standard "how wrong was this classification"
    # measurement for multi-class problems (here, per pixel).
    # ignore_index=-1: Sen1Floods11's hand-labeled masks use -1 for
    # no-data/cloud-obscured pixels (not a real class) -- without this,
    # CrossEntropyLoss crashes with a CUDA assertion the first time it
    # hits a -1 label, since -1 isn't a valid class index. FloodNet's
    # masks never contain -1, so this is a no-op for that dataset.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Adam: the algorithm that decides HOW to adjust the model's numbers
    # given the loss. You don't need to understand its internals -- it's
    # essentially always a safe default choice.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- The training loop itself ---
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()  # tells PyTorch "we're training" (affects some layers' behavior)
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            imgs, masks = imgs.to(device), masks.to(device)  # move data to GPU/CPU

            optimizer.zero_grad()          # clear old adjustment info
            outputs = model(imgs)          # step 1: run the model forward
            loss = criterion(outputs, masks)  # step 2: how wrong were we?
            loss.backward()                # step 3: figure out the adjustment
            optimizer.step()               # step 4: apply the adjustment

            train_loss += loss.item()

        # --- Check progress on held-out validation data ---
        model.eval()  # tells PyTorch "we're evaluating, not training"
        val_iou = 0.0
        with torch.no_grad():  # don't bother tracking gradients -- we're not training here
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_iou += iou_score(outputs, masks, args.num_classes)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_iou={avg_val_iou:.4f}")

        # Only keep the checkpoint if it's the best one seen so far.
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  -> saved new best checkpoint (IoU={best_iou:.4f})")

    # Final per-class breakdown -- especially important for the 10-class
    # full_severity run, where classes are heavily imbalanced and a
    # single averaged IoU can be misleading (see iou_score docstring).
    if args.num_classes > 2:
        print("\nFinal per-class IoU on validation set:")
        model.eval()
        class_ious = None
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                batch_ious = iou_score(outputs, masks, args.num_classes, per_class=True)
                if class_ious is None:
                    class_ious = [0.0] * args.num_classes
                class_ious = [a + b for a, b in zip(class_ious, batch_ious)]
        class_ious = [v / len(val_loader) for v in class_ious]
        print_iou_report(class_ious, args.num_classes)


if __name__ == "__main__":
    main()
