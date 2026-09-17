"""
model.py
========
Defines the neural network. Notice how short this file is -- that's the
point of using segmentation_models_pytorch (smp): someone else already
designed and validated the U-Net architecture, so you just configure and
call it, similar to how you'd use cv2.CascadeClassifier without writing
face-detection math yourself.

What U-Net actually does, briefly: it shrinks the image down through
several layers (learning "what is in this image" at increasingly abstract
levels), then expands it back up to the original size (deciding "where
exactly is it"), with shortcut connections between matching shrink/expand
layers so fine spatial detail isn't lost. That's why it's the standard
choice for "turn an image into a same-sized mask" tasks.
"""

import segmentation_models_pytorch as smp


def build_model(in_channels, num_classes, encoder_name="resnet18"):
    """
    in_channels: how many bands the input image has
                 (3 for FloodNet's RGB, 8 for Sen1Floods11's multi-band tiles)
    num_classes: how many categories to predict per pixel
                 (2 for simple flood/no-flood; more if you add damage severity levels)
    encoder_name: which pretrained backbone to use inside the U-Net.
                  resnet18 is small and fast -- a reasonable default given
                  the "limited computational power" constraint in the brief.

    Pretrained ImageNet weights only make sense for 3-channel (RGB) input,
    since ImageNet is RGB photos -- that's why we only load them when
    in_channels == 3 (i.e. for FloodNet, not the 8-channel satellite data).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if in_channels == 3 else None,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model
