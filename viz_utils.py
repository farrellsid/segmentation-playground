"""
viz_utils.py
------------
Visualization helpers for SAM2 segmentation outputs.
Sourced from Meta's example code for SAM2 Image mode!

Functions:
    show_mask(mask, ax, ...)        - Overlay a single binary mask on an axes.
    show_points(coords, labels, ax) - Scatter positive/negative prompt points.
    show_box(box, ax)               - Draw a bounding box prompt rectangle.
    show_masks(image, masks, ...)   - Plot all masks with scores in separate figures.

Notes:
    - Intended for interactive / exploratory use in Jupyter notebooks.
    - DO NOT call show_masks in batch pipeline loops — 9k×9k matplotlib figures
      cause massive pagefile spikes on Windows. Use it on cropped regions or
      small test images only.
    - np.random.seed(3) is set at import time to keep random mask colors
      consistent across calls within a session.

Usage:
    from viz_utils import show_mask, show_points, show_box, show_masks
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fix random seed so random_color masks are reproducible within a session.
np.random.seed(3)


def show_mask(
    mask,
    ax,
    random_color: bool = False,
    borders: bool = True,
) -> None:
    """
    Overlay a single binary mask on a matplotlib axes.

    Args:
        mask:         2-D boolean/uint8 array (H x W).
        ax:           Matplotlib axes to draw on.
        random_color: If True, use a random RGBA color; otherwise use dodger-blue.
        borders:      If True, draw contour outlines over the mask.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
            for contour in contours
        ]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size: int = 375) -> None:
    """
    Scatter positive (green) and negative (red) prompt points on axes.

    Args:
        coords:      (N, 2) array of (x, y) point coordinates.
        labels:      (N,) array; 1 = positive, 0 = negative.
        ax:          Matplotlib axes to draw on.
        marker_size: Scatter marker size (default 375).
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1],
        color="green", marker="o", s=marker_size,
        edgecolor="white", linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1],
        color="red", marker="o", s=marker_size,
        edgecolor="white", linewidth=1.25,
    )


def show_box(box, ax) -> None:
    """
    Draw a bounding box prompt as a green rectangle on axes.

    Args:
        box: (x0, y0, x1, y1) in pixel coordinates.
        ax:  Matplotlib axes to draw on.
    """
    x0, y0 = box[0], box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders: bool = True,
) -> None:
    """
    Plot each mask in its own figure alongside the source image.

    WARNING: Each figure allocates a full-resolution canvas. On 9k×9k images
    this will spike pagefile usage significantly. Use only for small images
    or cropped regions during debugging.

    Args:
        image:        (H, W, 3) RGB image array.
        masks:        (N, H, W) boolean mask array (N masks).
        scores:       (N,) float array of mask confidence scores.
        point_coords: Optional (M, 2) prompt point array.
        box_coords:   Optional (4,) bounding box array.
        input_labels: Required if point_coords is provided; (M,) label array.
        borders:      Whether to draw contour borders on masks.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None, "input_labels required when point_coords is given"
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()