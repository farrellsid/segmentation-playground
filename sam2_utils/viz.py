"""Visualization helpers for SAM2 prompts and outputs.

Covers static display (`show_mask`, `show_points`, `show_box`, `show_masks`)
and interactive point picking (`pick_point`, `pick_landmark`).

Static functions are direct ports of the helpers from Meta's image/video
example notebooks, unified so the same `show_mask` works for both single-mask
and multi-object (per-obj_id colored) cases.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

# Seed once at module import for reproducible "random_color" choices.
np.random.seed(3)


# =============================================================================
# Static display
# =============================================================================

def show_mask(mask, ax, random_color: bool = False, borders: bool = True,
              obj_id: Optional[int] = None):
    """Overlay a binary mask on an axes.

    Parameters
    ----------
    mask : array, shape (..., H, W)
        Boolean or 0/1 mask. Leading dims are squeezed via shape[-2:].
    ax : matplotlib axes
    random_color : bool
        If True, draw a random RGBA color. Overrides obj_id.
    borders : bool
        If True, draw the mask contour on top. Ignored for obj_id (video) mode
        to match Meta's video predictor look.
    obj_id : int, optional
        If given, color is chosen from matplotlib tab10 cmap by obj_id.
        Useful for multi-object video segmentation.
    """
    import cv2  # local import; viz module shouldn't fail to import without cv2

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        draw_borders = borders
    elif obj_id is not None:
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(obj_id % 10)[:3], 0.6])
        # video reference uses no borders for multi-object case
        draw_borders = False
    else:
        # default SAM 2 dodger-blue
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        draw_borders = borders

    h, w = mask.shape[-2:]
    m_u8 = mask.astype(np.uint8)
    mask_image = m_u8.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if draw_borders:
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size: int = 50, marker: str = "o"):
    """Scatter positive (green) and negative (red) prompt points.

    The image predictor uses `marker="o"` at size 50; the video predictor
    uses `marker="*"` at size 200. Both are supported via kwargs.
    """
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color="green", marker=marker,
               s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg[:, 0], neg[:, 1], color="red", marker=marker,
               s=marker_size, edgecolor="white", linewidth=1.25)


def show_box(box, ax):
    """Draw a [x0, y0, x1, y1] box as an unfilled green rectangle."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h,
                               edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores,
               point_coords=None, box_coords=None, input_labels=None,
               borders: bool = True, figsize: Tuple[int, int] = (10, 10)):
    """Plot each mask in its own figure with overlaid prompts and a score title.

    Mirrors the helper in Meta's image_predictor example, but figure size is
    configurable.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=figsize)
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None, "input_labels required when point_coords given"
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


# =============================================================================
# Interactive point picker
# =============================================================================
# NOTE: callers must run `%matplotlib widget` in a notebook cell before using
# these. It's a Jupyter magic and can't be invoked from a regular function.

def pick_point(
    image,
    center: Optional[Tuple[float, float]] = None,
    zoom: Optional[float] = None,
    on_click: Optional[Callable[[Any, Any, Any], None]] = None,
    ax_setup: Optional[Callable[[Any], None]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
):
    """Generic interactive point picker over a matplotlib widget axes.

    Parameters
    ----------
    image : array
        Image to display (RGB or grayscale).
    center : (cx, cy), optional
        If given, draws a red crosshair + lime dot at this point.
    zoom : float, optional
        Window half-width in pixels. Only used when `center` is set.
    on_click : callable(event, ax, fig), optional
        Called on each in-axes click. Default just prints the coords.
    ax_setup : callable(ax), optional
        Hook for adding extra scatter, titles, etc. before the figure shows.
    figsize, title : standard mpl params.

    Returns
    -------
    (fig, ax)
        Live figure handle. Keep a reference or the widget will gc.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    if center is not None:
        cx, cy = center
        ax.axhline(cy, color="red", alpha=0.5, linewidth=0.5)
        ax.axvline(cx, color="red", alpha=0.5, linewidth=0.5)
        ax.scatter([cx], [cy], c="lime", s=200, edgecolor="red",
                   linewidth=2, zorder=10)
        if zoom is not None:
            ax.set_xlim(cx - zoom, cx + zoom)
            ax.set_ylim(cy + zoom, cy - zoom)  # inverted: image coords

    if title:
        ax.set_title(title)

    if ax_setup is not None:
        ax_setup(ax)

    handler = on_click or (
        lambda event, ax_, fig_: print(f"x={event.xdata:.1f}, y={event.ydata:.1f}")
    )

    def _dispatch(event):
        if event.xdata is None or event.inaxes != ax:
            return
        handler(event, ax, fig)

    fig.canvas.mpl_connect("button_release_event", _dispatch)
    return fig, ax


def pick_landmark(image, target_row, zoom: float = 1000,
                  collected: Optional[list] = None,
                  figsize: Tuple[int, int] = (10, 10)):
    """Pre-baked picker for the affine-alignment landmark workflow.

    Centers the view on `target_row['x'], target_row['y']`, and on each
    click prints the landmark tuple in
    `("cell_name", catmaid_x, catmaid_y, tif_x, tif_y)` form, drops a cyan
    dot, and (if `collected` is provided) appends the tuple to that list.

    Parameters
    ----------
    image : array
    target_row : pandas.Series
        Must have keys 'x', 'y', 'cell_name', 'node_id'.
    zoom : float
        Window half-width in px.
    collected : list, optional
        If given, appended-to on each click.
    """
    cx, cy = float(target_row["x"]), float(target_row["y"])
    name = str(target_row["cell_name"])
    node_id = int(target_row["node_id"])

    print(f"{name} (node {node_id})")
    print(f"  CATMAID coords: ({cx:.1f}, {cy:.1f})")
    print(f"  Click on the true location in the tif.\n")

    def _on_click(event, ax, fig):
        tx, ty = event.xdata, event.ydata
        tup = (name, round(cx, 3), round(cy, 3), round(tx, 1), round(ty, 1))
        print(f'  ("{name}", {cx:.3f}, {cy:.3f}, {tx:.1f}, {ty:.1f}),')
        if collected is not None:
            collected.append(tup)
        ax.scatter([tx], [ty], c="cyan", s=100,
                   edgecolor="black", linewidth=1, zorder=11)
        fig.canvas.draw_idle()

    return pick_point(
        image,
        center=(cx, cy),
        zoom=zoom,
        on_click=_on_click,
        title=f"{name} (node {node_id}), CATMAID coords ({cx:.1f}, {cy:.1f})",
        figsize=figsize,
    )
