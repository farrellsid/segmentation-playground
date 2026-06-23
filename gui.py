"""
gui.py: napari review / triage / correction GUI.

The one human-facing tool: one triage queue, one review tool. It reads the
batch's flagged chains, lets a human scrub a chain, inspect
why frames flagged, edit the SAM2 prompts (positive **and** negative points, plus a
drawn bounding box), paint an anchor mask, re-run the image phase, and resume
propagation over the interruptible ``PropagationSession``, then writes the corrected
masks + QC back to disk and logs every decision as a training label.

It is a *thin driver*, like ``run_aval.py`` / ``batch.py``: all real work lives in
the library it composes:

    sam2_utils.review        rebuild a finished chain's overlay from disk (load_chain)
    sam2_utils.review_queue  which chains need a human; the GUI-owned review ledger
    sam2_utils.labels        the per-frame label store (the "label engine")
    pipeline                 the phase functions + PropagationSession (re-segmentation)
    sam2_utils.setup         lazy predictor construction (GPU only when needed)

Two-tier loading (so review/labeling works WITHOUT a GPU)
--------------------------------------------------------
  * **light**: annotate_df (cached CSV + affine) + chains.json + the on-disk
    chain artifacts. Enough to browse, scrub, inspect flags, paint, and *label*.
    No torch, no predictors.
  * **heavy**: the SAM2 image + video predictors, built lazily the first time the
    human triggers a re-segmentation (re-run image phase / resume propagation).
    This enables *parallel review*: a reviewer can clear the labeling/approve
    queue while the GPU is busy with the background batch.

Coordinate spaces
-----------------
Everything the GUI shows shares one grid: the EM JPEG frames, the saved masks,
and therefore the napari Image/Labels/Points layers are all in **_sam** space
(``save_downscale == scale``, the canonical rule). So a point the human clicks on
the canvas is already an _sam coordinate: it feeds straight into ``image_predict``
on the same-resolution anchor frame and into the video seed, no transform. (The
default *crop* anchor path runs image mode in _crop; the GUI's re-predict instead
uses the legacy full-frame _sam path precisely because the displayed frame is the
_sam frame the human is clicking on. Crop re-predict is a not-implemented refinement,
see "not implemented" below.)

Not implemented this pass (placeholders marked ``# not implemented`` in code)
-----------------------------------------------------------------------------
  * **Crop-space anchor re-predict.** The re-run uses the legacy scale-8 full-frame
    image path (matches the displayed frame). High-res crop re-predict (the default
    for the *batch*) would sharpen a thin-neurite re-seed but needs the
    clicked points remapped _sam->_tif->_crop and a full-res tif read; left for later.
  * **Confidence-gated mask-vs-box video seed.** The GUI always seeds *propagation* with
    the *mask* (``add_mask`` of the re-predicted/painted mask on the frame), not a box,
    as the more-informative human-curated path (box vs mask). A drawn box IS supported,
    but only as an *image-phase* prompt: it shapes the re-predicted mask (``R``)
    alongside points, and that mask is what propagation then seeds from. The *automatic*
    confidence gate that chooses mask-vs-box in the headless pipeline is still label-gated.
  * **Marking/intervention GUI split & strict-by-default flagging.** Review-testing
    follow-ups (a two-mode UI, and an aggressive-recall
    QC posture): not built this pass.
  * **Cross-process file lock / live auto-poll / GPU arbitration**: see
    ``review_queue`` (not implemented); the GUI exposes a manual "refresh queue" button.
  * **micro_sam napari-plugin build-vs-adopt eval**: this module is the
    build path; the adopt evaluation is a separate spike, not code here.

Launch
------
    py -3 gui.py                              # opens the queue picker on config.OUTPUT_ROOT
    py -3 gui.py --neuron AIAL --chain 0      # opens straight onto one chain
    # or, from a notebook / REPL:
    from gui import launch
    launch(neuron="AIAL", chain_idx=0)        # napari.run() blocks until the window closes
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Silence an upstream deprecation surfaced by the lazy frame loader: dask_image.imread
# -> pims.ImageSequence still calls skimage.io with the deprecated `plugin` parameter
# (removed in skimage 0.27). It's harmless and not our code; suppress just that message
# so it doesn't spam the GUI console. The eager fallback (_load_frame_stack) is unaffected.
warnings.filterwarnings(
    "ignore", message="The plugin infrastructure in", category=FutureWarning)

from sam2_utils import config, alignment, review, review_queue, labels as labels_mod
import pipeline


# Stable colours matching video_viz's palette intent. Prompt points: green = +, red = -.
_POS_COLOR = "#2ca02c"
_NEG_COLOR = "#d62728"
_PROMPT_LABELS = ["positive", "negative"]
_LABEL_TO_SAM = {"positive": 1, "negative": 0}

# Picker modes (the "show" toggle). flagged: the review queue, today's behaviour;
# everything: every chain on disk, so a reviewer can open un-flagged chains.
_MODE_FLAGGED = "flagged only"
_MODE_EVERYTHING = "everything"
_MODE_CHOICES = [_MODE_FLAGGED, _MODE_EVERYTHING]

# Box-prompt geometry: convert between an xyxy box (the pipeline.Prompts.box_sam
# format) and a napari Shapes rectangle's (N, 3) vertices in (t, y, x). Pure and
# torch/napari-free so they unit-test without a GPU or a viewer.
_BOX_EDGE_COLOR = "#1f77b4"   # blue, distinct from the green/red prompt points


def _rect_to_xyxy(verts) -> np.ndarray:
    """A rectangle's vertices -> (x0, y0, x1, y1) in data coords. The last two columns
    are (y, x), so this handles both the 3D prompt-box layer ((N, 3) = t, y, x) and the
    2D recrop-picker layer ((N, 2) = y, x). Axis-aligned: takes the min/max of the y and
    x columns, correct for any corner order (and a rotated box's bounding extent)."""
    v = np.asarray(verts, dtype=float)
    ys, xs = v[:, -2], v[:, -1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)


def _xyxy_to_rect(xyxy, t: int) -> np.ndarray:
    """(x0, y0, x1, y1) + frame index t -> a (4, 3) rectangle vertex array (t, y, x),
    the inverse of _rect_to_xyxy, for seeding a saved box into the Shapes layer."""
    x0, y0, x1, y1 = (float(c) for c in xyxy)
    return np.array([[t, y0, x0], [t, y0, x1], [t, y1, x1], [t, y1, x0]], dtype=float)


def _box_on_frame(shapes_data, frame_idx: int) -> Optional[np.ndarray]:
    """The xyxy of the LAST rectangle on ``frame_idx`` in a list of napari shape vertex
    arrays, or None. A shape's frame is the rounded t of its first vertex (every vertex
    shares one t for a box drawn on a single slice). Last-wins matches "redraw replaces"."""
    box = None
    for verts in shapes_data:
        v = np.asarray(verts, dtype=float)
        if len(v) and int(round(v[0, 0])) == int(frame_idx):
            box = _rect_to_xyxy(v)
    return box


# =============================================================================
# Light context: annotate_df + chains + lazy predictors
# =============================================================================

class ReviewContext:
    """Shared, build-once handles for a review session.

    The *light* members (annotate_df, chains, cfg) load from the cached CSV +
    chains.json + an output tree (no torch). The *heavy* predictors are built
    lazily by ``ensure_predictors`` only when a re-segmentation action needs them,
    so a browse/label-only session never touches the GPU.
    """

    def __init__(self, output_root: Path, cfg: Optional[pipeline.PipelineConfig] = None,
                 *, annotate_df: Optional[pd.DataFrame] = None, chains: Optional[list] = None):
        self.output_root = Path(output_root)
        self.cfg = cfg or pipeline.PipelineConfig(
            model_size="large", scale=8, save_downscale=8,
            output_root=self.output_root, frames_root=config.FRAMES_ROOT)
        # ensure the re-segmentation/QC paths write where we're reading
        self.cfg.output_root = self.output_root
        self._annotate_df = annotate_df
        self._chains = chains
        self.image_predictor = None
        self.video_predictor = None

    # -- light -----------------------------------------------------------------
    @property
    def annotate_df(self) -> pd.DataFrame:
        """Cached CATMAID node table with the stack→tif affine applied (x_tif/y_tif).
        Same construction as batch._build_session, minus the predictors."""
        if self._annotate_df is None:
            df = pd.read_csv(config.CSV_PATH)
            xy = alignment.catmaid_to_tif(df["x"].values, df["y"].values)
            df["x_tif"], df["y_tif"] = xy[:, 0], xy[:, 1]
            self._annotate_df = df
        return self._annotate_df

    @property
    def chains(self) -> list:
        if self._chains is None:
            with open(config.CHAINS_PATH) as f:
                self._chains = json.load(f)
        return self._chains

    def find_chain(self, neuron: str, chain_idx: int) -> Optional[dict]:
        """The chain dict for (neuron, chain_idx): the position within that
        neuron's chain list, matching batch.enumerate_chains / on-disk chain_NN."""
        chs = [c for c in self.chains if c.get("cell_name") == neuron]
        return chs[chain_idx] if 0 <= chain_idx < len(chs) else None

    # -- heavy (lazy) ----------------------------------------------------------
    def ensure_predictors(self, *, need_image: bool = True, need_video: bool = True) -> None:
        """Build the SAM2 predictors on first use (heavy: VRAM + model load).
        Idempotent: built once, reused for the session."""
        from sam2_utils import setup
        if need_image and self.image_predictor is None:
            print("[gui] building image predictor (first use)...")
            self.image_predictor, _ = setup.build_predictor(size=self.cfg.model_size, kind="image")
        if need_video and self.video_predictor is None:
            print("[gui] building video predictor (first use)...")
            # correct_as_cond=True: a human paint/click on an already-tracked frame must
            # become a CONDITIONING frame so its mask is preserved verbatim on the next
            # propagate. Without it, SAM2 demotes a re-correction to non-conditioning and
            # re-infers that frame from memory on resume, silently reverting the paint
            # (the iterative paint->resume->repaint revert; box vs mask).
            self.video_predictor, _ = setup.build_predictor(
                size=self.cfg.model_size, kind="video", correct_as_cond=True)


# =============================================================================
# Frame stack loading (EM JPEGs -> a sliceable (T, H, W, 3) array)
# =============================================================================

def _load_frame_stack(frames_dir: str, n_frames: int):
    """Return an array-like (T, H, W, 3) uint8 over the chain's 0-indexed JPEGs.

    Prefers a lazy dask stack (dask_image, per napari's big-data tutorial) so long
    ~340-frame chains don't eagerly load ~1 GB; falls back to an eager np.stack via
    the same single-frame reader review/video_viz use, so it works without dask.
    """
    frames_dir = Path(frames_dir)
    try:
        from dask_image.imread import imread as _dimread  # lazy, optional
        stack = _dimread(str(frames_dir / "*.jpg"))        # (T, H, W, 3), BGR
        return stack[..., ::-1]                            # -> RGB, still lazy
    except Exception:
        from sam2_utils.video_viz import _load_frame
        frames = [_load_frame(frames_dir, i) for i in range(n_frames)]
        return np.stack(frames, axis=0)


def _label_stack_from_segments(video_segments: dict, frame_to_z: dict, obj_id: int,
                               t: int, hw: tuple) -> np.ndarray:
    """(T, H, W) uint8 label volume: obj_id where this chain's mask is set, else 0.
    Paintable in a napari Labels layer (the human-painted-anchor surface)."""
    H, W = hw
    lbl = np.zeros((t, H, W), dtype=np.uint8)
    for fi, seg in video_segments.items():
        if obj_id in seg and 0 <= fi < t:
            m = np.asarray(seg[obj_id])
            m = m[0] if m.ndim == 3 else m
            if m.shape == (H, W):
                lbl[fi][m.astype(bool)] = obj_id
    return lbl


# =============================================================================
# The GUI
# =============================================================================

class ReviewGUI:
    """A napari window bound to one output tree, opening one chain at a time.

    Build with a ReviewContext, then ``open_chain(neuron, chain_idx)``. The layer
    stack per chain:
        Image  'EM'        the _sam JPEG frames (T, H, W, 3)
        Labels 'mask'      editable obj_id volume (paint here for a mask correction)
        Points 'skeleton'  the chain's CATMAID nodes per z (read-only context)
        Points 'prompts'   the human's positive/negative click prompts (editable)
    """

    def __init__(self, ctx: ReviewContext, *, reviewer: str = "", viewer=None,
                 point_size: float = 4.0, auto_zoom: bool = True, zoom_pad: float = 3.0,
                 hires_em: bool = False):
        """
        point_size : default diameter of prompt/skeleton points, in _sam px (was 10;
                     4 is unobtrusive at scale-8). Tune live via the dock spinbox.
        auto_zoom  : on open and on jump-to-flagged, zoom the camera to the mask's
                     bounding box (+ zoom_pad× margin) so you land on the object,
                     not the whole frame.
        hires_em   : load the *full-resolution* EM tifs as the background instead of
                     the scale-8 JPEGs (lazy, opt-in; see _load_hires_stack and the
                     "Why low-res" note in the header). The MASK stays scale-8 (that
                     is the only resolution it was propagated/saved at; sharper masks
                     need the tier-2 per-chain crop), but it is scaled
                     to overlay the full-res image so the EM context is crisp.
        """
        import napari
        self.ctx = ctx
        self.reviewer = reviewer
        self.point_size = float(point_size)
        self.auto_zoom = bool(auto_zoom)
        self.zoom_pad = float(zoom_pad)
        self.hires_em = bool(hires_em)
        self._em_world = 1.0   # world units (EM px) per _sam px; set per chain in open_chain
        self.viewer = viewer if viewer is not None else napari.Viewer(title="SAM2 review (M4)")
        self.queue = review_queue.ReviewQueue(ctx.output_root)
        self.labels = labels_mod.LabelStore(ctx.output_root)

        # per-open-chain state
        self.neuron: Optional[str] = None
        self.chain_idx: Optional[int] = None
        self.data: Optional[review.ReviewData] = None      # ReviewData from review.load_chain
        self.chain: Optional[dict] = None                  # chain dict (nodes)
        self._state: Optional[pipeline.ChainState] = None  # the chain's serialized state (seed prompts)
        self._cw = None                                    # alignment.CropWindow for tier-2 chains, else None
        self.qc_df: Optional[pd.DataFrame] = None          # z-indexed
        self.session: Optional[pipeline.PropagationSession] = None  # built lazily on resume

        # layers (set in open_chain)
        self._img = self._mask = self._skel = self._prompts = self._box = None
        self._lscale = (1.0, 1.0, 1.0)   # _sam->EM world scale of the current chain's layers
        self._populating = False         # re-entrancy guard for the neuron->chain cascade
        self._recrop_picking = False     # True while the full-frame recrop region picker is open
        self._recrop_full_hw = None      # (H, W) _tif of the frame shown in the picker

        self._build_widgets()
        self._bind_keys()

    # -- public: open a chain --------------------------------------------------
    def open_chain(self, neuron: str, chain_idx: int) -> None:
        """Load (neuron, chain_idx) into the viewer from its on-disk artifacts."""
        chain_dir = self.ctx.output_root / neuron / f"chain_{chain_idx:02d}"
        if not chain_dir.exists():
            raise FileNotFoundError(f"no chain dir at {chain_dir}")

        self._close_session()
        self.neuron, self.chain_idx = neuron, chain_idx
        self.chain = self.ctx.find_chain(neuron, chain_idx)

        # rebuild the overlay from disk (one definition of "how a mask is read")
        self.data = review.load_chain(chain_dir, verbose=True)
        self.qc_df = self.data.qc if isinstance(self.data.qc, pd.DataFrame) else None
        # the chain's serialized state carries the ORIGINAL seed (prompts.points_sam
        # / labels / box_sam), loaded so we can pre-populate the prompts layer with
        # it rather than starting empty (else "re-run image phase" has no positive
        # point). Also reused by _anchor_dict.
        sp = chain_dir / "state.json"
        self._state = pipeline.load_state(sp) if sp.exists() else None
        # tier-2 chains were propagated/saved in a per-chain crop space (_pcrop). The
        # CropWindow (persisted in state.json) is what maps _tif skeleton nodes + drives
        # crop-aware QC. The displayed EM/mask/prompts are ALL _pcrop already (frames_dir
        # points at the crop view, masks are crop-sized), so a click is a _pcrop coord and
        # re-predict/resume need no transform: only skeleton/QC/hires consult the window.
        self._cw = None
        if self._state is not None and getattr(self._state, "crop_window", None):
            self._cw = alignment.CropWindow.from_dict(self._state.crop_window)
            print(f"[gui] tier-2 crop chain: _pcrop window {self._cw.size_tif} "
                  f"@ crop_scale {self._cw.crop_scale}")

        # frame stack + dims
        t = max(self.data.video_segments) + 1 if self.data.video_segments else 0
        any_mask = next(iter(self.data.video_segments.values()))[self.data.obj_id]
        any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
        H, W = np.asarray(any_mask).shape
        lbl = _label_stack_from_segments(self.data.video_segments, self.data.frame_to_z,
                                         self.data.obj_id, t, (H, W))

        # EM background. hires_em reads the full-res tifs (lazy) for crisp context;
        # otherwise the scale-8 JPEGs SAM2 actually saw. Either way the masks/points
        # stay in _sam data coords and are *scaled* to overlay the EM, so the click
        # round-trip (_prompts_for_frame) is unchanged: with scale=(1,s,s), a click's
        # data coordinate is world/s = _sam (see "Why low-res" in the header).
        if self.hires_em:
            em = self._load_hires_stack(self.data.frame_to_z, t)
        else:
            em = _load_frame_stack(self.data.frames_dir, t)
        # world units (EM px) per mask px. Both EM axes are the mask scaled uniformly
        # (by `scale` for the full frame, by crop_scale for a tier-2 crop window), so
        # use the WIDTH ratio: correct for non-square crop windows, identical to before
        # for the near-square full frame. ~8 / crop_scale hires, 1 else. NB em is
        # (T, H, W, 3): the WIDTH axis is shape[2] (shape[1] is H; using it stretched
        # the mask/skeleton/prompt layers by H/W on non-square tier-2 _pcrop windows).
        self._em_world = float(em.shape[2]) / float(W) if W else 1.0
        s = self._em_world
        lscale = self._lscale = (1.0, s, s)                            # mask -> EM world

        # (re)build layers
        self.viewer.layers.clear()
        self._img = self.viewer.add_image(em, name="EM", rgb=True)
        self._mask = self.viewer.add_labels(lbl, name="mask", opacity=0.5, scale=lscale)
        self._skel = self.viewer.add_points(
            self._skeleton_points(), name="skeleton", ndim=3, size=self.point_size,
            scale=lscale, face_color="yellow", border_color="black", opacity=0.7)
        self._skel.editable = False
        self._prompts = self._new_prompts_layer(scale=lscale)
        self._box = self._new_box_layer(scale=lscale)
        # pre-load the chain's original seed (points + box) at the anchor frame
        self._seed_prompts_from_state()

        # land on the anchor frame and update the info panel
        if self.data.anchor_idx is not None:
            self.viewer.dims.set_current_step(0, int(self.data.anchor_idx))
        self._zoom_to_mask(self.data.anchor_idx if self.data.anchor_idx is not None
                           else self._current_frame())
        self.queue.claim(neuron, chain_idx, reviewer=self.reviewer)
        self._sync_selectors()        # point the picker at this chain, refresh its badge
        self._refresh_info()
        print(f"[gui] opened {neuron} chain {chain_idx:02d}: {t} frames, "
              f"{len(self.data.triage_frames)} queued, anchor frame {self.data.anchor_idx}")

    # -- layer builders --------------------------------------------------------
    def _new_prompts_layer(self, scale=(1.0, 1.0, 1.0)):
        """An empty editable Points layer for human click-prompts, coloured by the
        'label' feature (green=positive, red=negative), in add mode. ndim=3 so a
        point binds to a specific frame (t, y, x). napari 0.5+ uses border_*. ``scale``
        matches the mask/EM layers so a click's data coord stays _sam (see header)."""
        features = pd.DataFrame({"label": pd.Categorical([], categories=_PROMPT_LABELS)})
        layer = self.viewer.add_points(
            np.empty((0, 3)), name="prompts", ndim=3, size=self.point_size, scale=scale,
            features=features,
            border_color="label", border_color_cycle=[_POS_COLOR, _NEG_COLOR],
            face_color="transparent", border_width=0.4, symbol="o",
        )
        layer.border_color_mode = "cycle"
        # default new points to positive; the '+/-' keys + the dropdown flip it
        layer.feature_defaults = {"label": "positive"}
        layer.mode = "add"
        return layer

    def _new_box_layer(self, scale=(1.0, 1.0, 1.0)):
        """An empty editable Shapes layer for one human-drawn bounding box per frame,
        a seed for the image-phase re-predict (R). ndim=3 so a rectangle binds to a
        frame (t, y, x); ``scale`` matches the other layers so the box's data coords
        stay _sam (or _pcrop), the space image_predict reads (see header). The 'B' key
        / 'draw box' button puts it in add_rectangle mode."""
        layer = self.viewer.add_shapes(
            name="box", ndim=3, scale=scale, edge_color=_BOX_EDGE_COLOR,
            face_color="transparent", edge_width=1.0, opacity=0.8)
        return layer

    def _seed_prompts_from_state(self) -> None:
        """Pre-load the chain's ORIGINAL seed (state.prompts) at the anchor frame, so
        re-run starts from what the batch used, not an empty layer: points into the
        prompts layer and the saved box (state.prompts.box_sam) into the box layer.
        Coords are _sam (or _pcrop), the layer scale handles overlay, matching the
        round-trips in _prompts_for_frame / _box_for_frame. The box seeds only the
        image phase; resume still propagates the mask, not the box. Best-effort: a
        chain with no serialized prompts (legacy) just leaves the layers empty."""
        st = self._state
        if st is None or st.prompts is None or st.anchor_frame_idx is None:
            return
        pts = np.asarray(st.prompts.points_sam, dtype=float)
        labs = np.asarray(st.prompts.labels, dtype=int)
        af = int(st.anchor_frame_idx)
        if len(pts):
            data = np.column_stack([np.full(len(pts), af, float), pts[:, 1], pts[:, 0]])  # (t,y,x)
            label_strs = ["positive" if int(l) == 1 else "negative" for l in labs]
            self._prompts.data = data
            self._prompts.features = pd.DataFrame(
                {"label": pd.Categorical(label_strs, categories=_PROMPT_LABELS)})
            self._prompts.feature_defaults = {"label": "positive"}
            self._prompts.mode = "add"
            n_pos = int((labs == 1).sum())
            print(f"[gui] loaded original seed: {n_pos} positive + {len(labs) - n_pos} "
                  f"negative point(s) at anchor frame {af} (edit these, then 'R')")
        if self._box is not None and st.prompts.box_sam is not None:
            rect = _xyxy_to_rect(np.asarray(st.prompts.box_sam, dtype=float), af)
            self._box.add_rectangles(rect)
            print(f"[gui] loaded original box (xyxy) "
                  f"{np.asarray(st.prompts.box_sam).astype(int).tolist()} at frame {af}")

    def _skeleton_points(self) -> np.ndarray:
        """This chain's CATMAID nodes as (frame_idx, y_sam, x_sam) for context.
        Empty (0,3) if annotate_df/chain unavailable."""
        if self.chain is None or self.data is None:
            return np.empty((0, 3))
        z_to_frame = {z: i for i, z in self.data.frame_to_z.items()}
        df = self.ctx.annotate_df
        ids = {str(n) for n in self.chain["nodes"]}
        sub = df[df["node_id"].astype(str).isin(ids)]
        pts = []
        for _, r in sub.iterrows():
            fi = z_to_frame.get(int(r["z"]))
            if fi is None:
                continue
            # tier-2: _tif -> _pcrop (the displayed grid); else _tif -> _sam.
            if self._cw is not None:
                xy = np.asarray(self._cw.tif_to_crop([r["x_tif"], r["y_tif"]]), float).ravel()
            else:
                xy = alignment.tif_to_sam(np.array([[r["x_tif"], r["y_tif"]]], float),
                                          self.ctx.cfg.scale)[0]
            pts.append([fi, xy[1], xy[0]])                  # (t, y, x)
        return np.array(pts, dtype=float) if pts else np.empty((0, 3))

    # -- reading the human's prompts ------------------------------------------
    def _prompts_for_frame(self, frame_idx: int) -> pipeline.Prompts:
        """Collect the prompt-layer points on ``frame_idx`` into a pipeline.Prompts
        in _sam space (points are (x, y); labels 1=pos/0=neg)."""
        layer = self._prompts
        data = np.asarray(layer.data, dtype=float)
        if not len(data):
            return pipeline.Prompts(points_sam=np.empty((0, 2)), labels=np.empty((0,), int))
        on = np.round(data[:, 0]).astype(int) == int(frame_idx)
        rows = data[on]
        feats = layer.features.get("label", pd.Series(["positive"] * len(data)))
        lab_strs = np.asarray(feats)[on]
        pts_xy = np.stack([rows[:, 2], rows[:, 1]], axis=1)        # (x, y)
        labs = np.array([_LABEL_TO_SAM.get(str(s), 1) for s in lab_strs], dtype=int)
        return pipeline.Prompts(points_sam=pts_xy, labels=labs)

    def _box_for_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """The human-drawn box on ``frame_idx`` as xyxy in _sam (or _pcrop) data coords,
        or None. Last rectangle wins (a redraw replaces). The coords match the prompt
        round-trip, so it feeds straight into image_predict on the displayed frame."""
        if self._box is None:
            return None
        return _box_on_frame(self._box.data, int(frame_idx))

    def _current_frame(self) -> int:
        return int(self.viewer.dims.current_step[0])

    def activate_box_draw(self, *_) -> None:
        """Make the box layer the active layer in add_rectangle mode, so the next drag
        draws a bounding box on the current frame (the 'B' key / 'draw box' button)."""
        if self._box is None:
            print("[gui] no chain open; nothing to draw a box on")
            return
        self.viewer.layers.selection.active = self._box
        self._box.mode = "add_rectangle"
        print("[gui] box draw mode: drag a rectangle on this frame, then 'R' to re-predict")

    # -- view helpers ----------------------------------------------------------
    def _load_hires_stack(self, frame_to_z: dict, t: int):
        """Lazy full-res EM stack (T, H_full, W_full, 3) over the chain's frames,
        read from the original WORM_PATH tifs (NOT the scale-8 JPEGs). Opt-in via
        ``hires_em``; see the "Why low-res" note in the header: this sharpens only
        the *underlying image*; the saved masks remain scale-8 (sharper masks need
        the tier-2 per-chain crop). Falls back to the scale-8 stack if dask is
        unavailable, so a missing optional dep degrades, not crashes."""
        order = [frame_to_z[i] for i in range(t) if i in frame_to_z]
        if not order:
            return _load_frame_stack(self.data.frames_dir, t)
        try:
            import dask.array as da
            from dask import delayed

            def _read(z):
                img, _ = pipeline.load_frame_sam(int(z), scale=1)   # full-res RGB
                if self._cw is not None:
                    img = img[self._cw.slice_tif()]   # tier-2: crop to the chain window
                return img

            sample = _read(order[0])                                # one eager read for shape/dtype
            lazy = [da.from_delayed(delayed(_read)(z), shape=sample.shape, dtype=sample.dtype)
                    for z in order]
            print(f"[gui] hires_em: lazy full-res EM {sample.shape[1]}x{sample.shape[0]} per frame")
            return da.stack(lazy, axis=0)
        except Exception as e:
            print(f"[gui] hires_em unavailable ({e}); using scale-{self.ctx.cfg.scale} frames")
            return _load_frame_stack(self.data.frames_dir, t)

    def _zoom_to_mask(self, frame_idx: Optional[int], *, pad: Optional[float] = None) -> None:
        """Center + zoom the camera on the mask's bounding box at ``frame_idx`` (with
        a ``pad``× margin), so opening/jumping lands on the object rather than the
        whole frame. Best-effort: any camera/canvas quirk is caught and logged, never
        breaking navigation. Works in EM-world coords (×_em_world), so it's correct
        whether the EM is scale-8 or hires."""
        if not self.auto_zoom or frame_idx is None:
            return
        try:
            m = self._painted_mask(int(frame_idx))
            if m is None or not m.any():
                return
            ys, xs = np.where(m)
            s = self._em_world
            cy = 0.5 * (ys.min() + ys.max()) * s
            cx = 0.5 * (xs.min() + xs.max()) * s
            h = max(1, (ys.max() - ys.min() + 1)) * s
            w = max(1, (xs.max() - xs.min() + 1)) * s
            self.viewer.camera.center = (0.0, float(cy), float(cx))   # (z, y, x) world
            cw = ch = 800.0
            try:                                                       # canvas px, if available
                size = self.viewer.window.qt_viewer.canvas.size
                cw, ch = float(size[0]), float(size[1])
            except Exception:
                pass
            p = pad if pad is not None else self.zoom_pad
            self.viewer.camera.zoom = float(min(ch / (h * p), cw / (w * p)))
        except Exception as e:
            print(f"[gui] auto-zoom skipped: {e}")

    # =====================================================================
    # Actions (wired to buttons + keys). GPU-touching ones build predictors lazily.
    # =====================================================================

    def next_flagged(self, *_) -> None:
        self._step_flagged(+1)

    def prev_flagged(self, *_) -> None:
        self._step_flagged(-1)

    def _step_flagged(self, direction: int) -> None:
        if not self.data or not self.data.triage_frames:
            print("[gui] no queued frames in this chain")
            return
        flagged = sorted(self.data.triage_frames)
        cur = self._current_frame()
        nxt = [f for f in flagged if (f > cur if direction > 0 else f < cur)]
        target = (nxt[0] if direction > 0 else nxt[-1]) if nxt else (
            flagged[0] if direction > 0 else flagged[-1])
        self.viewer.dims.set_current_step(0, int(target))
        self._zoom_to_mask(int(target))
        self._refresh_info()

    def rerun_image_phase(self, *_) -> None:
        """Re-run SAM2 image mode on the CURRENT frame from the human's prompt points
        and/or drawn box, writing the result into the **mask** layer. This is a *preview*
        step: it turns your clicks/box into a mask you can eyeball (and tweak by painting)
        before committing. ``resume propagation`` then seeds that mask directly: the box
        shapes only this image-phase mask, it is NOT itself the propagation seed (SAM2
        propagates the mask, the more-informative seed, box vs mask).

        Uses the legacy full-frame _sam image path (the displayed frame IS the _sam
        frame the human clicked on). Crop re-predict is not implemented (see header).
        """
        if self.data is None or self._recrop_picking:
            return
        frame_idx = self._current_frame()
        prompts = self._prompts_for_frame(frame_idx)
        prompts.box_sam = self._box_for_frame(frame_idx)       # human-drawn box, or None
        if not (prompts.labels == 1).any() and prompts.box_sam is None:
            print("[gui] need at least one positive point or a box on this frame to re-predict")
            return
        self.ctx.ensure_predictors(need_image=True, need_video=False)

        em_sam = self._frame_image_sam(frame_idx)              # (H, W, 3) RGB uint8
        mask, score, _ = pipeline.image_predict(self.ctx.image_predictor, em_sam, prompts)
        self.ctx.image_predictor.reset_predictor()
        self._set_frame_mask(frame_idx, mask)                  # into the mask layer + segments
        seed = (f"{int((prompts.labels == 1).sum())}+/{int((prompts.labels == 0).sum())}- pts"
                + ("" if prompts.box_sam is None else " + box"))
        print(f"[gui] re-predicted frame {frame_idx} ({seed}): {int(mask.sum())} px, "
              f"score {score:.3f}, tweak by painting if needed, then 'resume propagation'")
        self._zoom_to_mask(frame_idx)
        self._refresh_info()

    def resume_propagation(self, *_) -> None:
        """Re-propagate from the CURRENT frame over a PropagationSession, seeding with
        the **mask** on this frame (re-predicted via ``R`` and/or hand-painted), never
        a box. Falls back to point-prompts only if the mask layer is empty here.

        Direction is **away from the anchor**, so an already-corrected frame is never
        clobbered (the front/back fix):
          * correcting the anchor frame  -> propagate both ways (the whole chain);
          * a frame AFTER the anchor      -> forward only (anchor..here stays as-is);
          * a frame BEFORE the anchor     -> reverse only.
        The corrected frame itself is a conditioning frame in inference_state, and the
        anchor's original prompt is still in memory, so SAM2 tracks the degraded tail
        without re-touching the good segment.

        Then save masks, re-run QC, persist state, and mark the chain CORRECTED.
        """
        if self.data is None or self._recrop_picking:
            return
        self.ctx.ensure_predictors(need_image=False, need_video=True)
        frame_idx = self._current_frame()
        sess = self._ensure_session()

        # seed: prefer the mask on this frame (curated boundary); else fall back to points
        mask = self._painted_mask(frame_idx)
        prompts = self._prompts_for_frame(frame_idx)
        if mask is not None and mask.any():
            sess.add_mask(frame_idx, mask)
            self._set_frame_mask(frame_idx, mask)   # lock the corrected frame into segments
            seed_desc = f"mask-seed ({int(mask.sum())} px)"
        elif len(prompts.labels):
            sess.add_points(frame_idx, prompts.points_sam, prompts.labels)
            seed_desc = (f"point-seed ({int((prompts.labels == 1).sum())}+/"
                         f"{int((prompts.labels == 0).sum())}-)")
        else:
            print("[gui] no correction on this frame (re-predict 'R', paint the mask, "
                  "or add points first)")
            return

        anchor = self.data.anchor_idx
        if anchor is None or frame_idx == anchor:
            dirs, where = [False, True], "both ways (whole chain)"
        elif frame_idx > anchor:
            dirs, where = [False], f"forward only (frames {anchor}..{frame_idx} preserved)"
        else:
            dirs, where = [True], f"reverse only (frames {frame_idx}..{anchor} preserved)"
        print(f"[gui] resume from frame {frame_idx}: {seed_desc}, propagating {where}")
        for rev in dirs:
            for _ in sess.propagate(reverse=rev, start_frame_idx=frame_idx):
                pass

        # pull the session's masks into our segments + the labels layer
        self._merge_segments(sess.video_segments)
        self._save_and_qc(sess.frame_conf, sess.pred_iou)
        self.queue.set_status(self.neuron, self.chain_idx, review_queue.CORRECTED,
                              reviewer=self.reviewer)
        print("[gui] resume complete, masks + qc.csv + state.json updated, chain marked corrected")
        self._refresh_info()

    def recrop_chain(self, *_) -> None:
        """Re-run this tier-2 chain in a WIDER crop window. Grows the current crop_window
        by the 'grow crop' amount (_tif px per side, clipped to the frame) and re-runs the
        whole chain in it via the standard run path (override_crop_window), for a window
        that auto-sized too small and still clips the cell. Heavy and blocking, like resume:
        it re-preps the _pcrop frames and re-propagates. Tier-2 only, a _sam chain has no
        crop window to grow (re-run it with chain_crop=True in the batch first)."""
        if self.data is None or self._recrop_picking:
            return
        if self._cw is None:
            print("[gui] recrop is for a tier-2 (_pcrop) chain; this chain is _sam. Re-run "
                  "it with chain_crop=True in the batch to make it a crop chain first, or use "
                  "'pick recrop region' to draw a window on the full frame")
            return
        grow = int(self._grow_spin.value)
        if grow <= 0:
            print("[gui] set 'grow crop (tif px)' > 0 to widen the window")
            return
        full_hw = self._full_frame_hw()
        cw_new = pipeline.grow_crop_window(self._cw, grow_tif=grow, image_hw_tif=full_hw,
                                           max_px=self.ctx.cfg.chain_crop_max_px)
        self._recrop_to_window(cw_new, f"grow {grow}/side")

    def _full_frame_hw(self) -> tuple:
        """(H, W) of the full-res _tif frame at the anchor z (one scale-1 read; cheap
        relative to the re-run that follows)."""
        anchor_z = self.data.frame_to_z.get(self.data.anchor_idx)
        _img, full_hw = pipeline.load_frame_sam(int(anchor_z), scale=1)
        return full_hw

    def _recrop_to_window(self, cw_new, label: str) -> None:
        """Re-run the open chain in ``cw_new`` (an alignment.CropWindow) via the standard
        run path, then reopen it. Shared by grow-recrop and the region picker. Heavy and
        blocking: re-preps the _pcrop frames + re-propagates."""
        from dataclasses import replace
        old = self._cw.size_tif if self._cw is not None else "(_sam)"
        print(f"[gui] recrop {self.neuron} chain {self.chain_idx:02d}: {old} -> "
              f"{cw_new.size_tif} _tif ({label}), re-running the chain (this is slow)...")
        self.ctx.ensure_predictors(need_image=True, need_video=True)
        cfg = replace(self.ctx.cfg, chain_crop=True, chain_crop_from_mask=False)
        state = pipeline.ChainState(neuron=self.neuron, chain_idx=self.chain_idx, config=cfg)
        self._close_session()                  # the old _pcrop session is stale
        pipeline.run_chain(
            state, image_predictor=self.ctx.image_predictor,
            video_predictor=self.ctx.video_predictor, annotate_df=self.ctx.annotate_df,
            chain=self.chain, override_crop_window=cw_new)
        self.queue.set_status(self.neuron, self.chain_idx, review_queue.CORRECTED,
                              reviewer=self.reviewer)
        print("[gui] recrop complete; reopening the chain in the new window")
        self.open_chain(self.neuron, self.chain_idx)

    # -- recrop region picker (Phase 2: draw the window on the full frame) -----
    def enter_recrop_picker(self, *_) -> None:
        """Show the full _sam anchor frame and a draggable rectangle for a NEW crop window,
        so the reviewer can re-centre (not just grow) the tier-2 crop. The rectangle starts
        at the chain's current window (or a node-centred default for a _sam chain). Adjust
        it, then 'confirm recrop' re-runs the chain in that window; 'cancel recrop' reopens
        the chain unchanged. Everything here is _sam, so there is no crop/full-frame mixing."""
        if self.data is None or self._recrop_picking:
            return
        anchor_z = self.data.frame_to_z.get(self.data.anchor_idx)
        em_sam, full_hw = pipeline.load_frame_sam(int(anchor_z), scale=self.ctx.cfg.scale)
        self._recrop_full_hw = full_hw
        s = float(self.ctx.cfg.scale)
        if self._cw is not None:                       # current window -> _sam rectangle
            ox, oy = self._cw.origin_tif
            w, h = self._cw.size_tif
            x0, y0, x1, y1 = ox / s, oy / s, (ox + w) / s, (oy + h) / s
        else:                                          # _sam chain: default to the skeleton bbox
            sx0, sy0, sx1, sy1 = pipeline._chain_skeleton_box_tif(self.chain, self.ctx.annotate_df)
            x0, y0, x1, y1 = sx0 / s, sy0 / s, sx1 / s, sy1 / s
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)   # (y, x) 2D
        self.viewer.layers.clear()
        self.viewer.add_image(em_sam, name="full frame (_sam)", rgb=em_sam.ndim == 3)
        layer = self.viewer.add_shapes(
            [rect], shape_type="rectangle", name="recrop region", ndim=2,
            edge_color=_BOX_EDGE_COLOR, face_color="transparent", edge_width=2.0, opacity=0.9)
        layer.mode = "select"
        self._recrop_picking = True
        self.viewer.reset_view()
        print("[gui] recrop picker: drag/resize the blue box to the new window, then "
              "'confirm recrop'. 'cancel recrop' leaves the chain unchanged.")
        self._refresh_info()

    def confirm_recrop(self, *_) -> None:
        """Read the picker rectangle (_sam), build a window, and re-run the chain in it."""
        if not self._recrop_picking:
            print("[gui] not in the recrop picker (use 'pick recrop region' first)")
            return
        layer = next((ly for ly in self.viewer.layers if ly.name == "recrop region"), None)
        if layer is None or not len(layer.data):
            print("[gui] no recrop rectangle drawn; cancel or draw one")
            return
        box_sam = _rect_to_xyxy(layer.data[-1])        # last rectangle, (x0,y0,x1,y1) _sam
        cw_new = pipeline.window_from_sam_box(
            box_sam, sam_scale=self.ctx.cfg.scale, image_hw_tif=self._recrop_full_hw,
            crop_scale=self.ctx.cfg.chain_crop_scale, max_px=self.ctx.cfg.chain_crop_max_px)
        self._recrop_picking = False                    # leave picker; _recrop_to_window reopens
        self._recrop_to_window(cw_new, "picked region")

    def cancel_recrop(self, *_) -> None:
        """Leave the recrop picker without changing the chain (reopen it as it was)."""
        if not self._recrop_picking:
            return
        self._recrop_picking = False
        print("[gui] recrop cancelled")
        self.open_chain(self.neuron, self.chain_idx)

    def approve_chain(self, *_) -> None:
        """Mark the chain's auto masks acceptable as-is. Logs the queued frames as
        verdict='ok' + a uniform sample of un-flagged frames (the silent-error
        window) so the label set isn't censored to flagged-only."""
        if self.data is None:
            return
        self._log_frames(verdict="ok", source="approve")
        self.queue.set_status(self.neuron, self.chain_idx, review_queue.APPROVED,
                              reviewer=self.reviewer)
        print(f"[gui] {self.neuron} chain {self.chain_idx:02d} approved")
        self._refresh_info()

    def reject_chain(self, *_) -> None:
        """Mark the chain unfixable / to-be-redone (e.g. bad anchor). Logs the queued
        frames as verdict='wrong' with the error type selected in the dock picker."""
        if self.data is None:
            return
        self._log_frames(verdict="wrong", source="reject", error_type=self._error_type())
        self.queue.set_status(self.neuron, self.chain_idx, review_queue.REJECTED,
                              reviewer=self.reviewer)
        print(f"[gui] {self.neuron} chain {self.chain_idx:02d} rejected ({self._error_type()})")
        self._refresh_info()

    def reset_prompts(self, *_) -> None:
        """Discard the human's prompt edits and restore the chain's ORIGINAL saved
        seed (state.prompts) at the anchor frame. Does NOT undo mask paints (the Labels
        layer has its own Ctrl+Z): this is prompt-only, as asked."""
        if self.data is None:
            return
        if self._prompts is not None:                       # clear, then re-seed from disk
            self._prompts.data = np.empty((0, 3))
            self._prompts.features = pd.DataFrame(
                {"label": pd.Categorical([], categories=_PROMPT_LABELS)})
        self._seed_prompts_from_state()
        print("[gui] prompts reset to the original saved seed")
        self._refresh_info()

    def mark_frame_wrong(self, *_) -> None:
        """Label the CURRENT frame verdict='wrong' with the picker's error type. Use
        while scrubbing to flag a frame the rule missed (a silent error) or to record
        a specific failure mode before correcting it."""
        self._label_current_frame(verdict="wrong", error_type=self._error_type(), source="mark")

    def mark_frame_ok(self, *_) -> None:
        """Label the CURRENT frame verdict='ok': confirm a frame is fine (incl. a
        flagged frame you judge a false alarm)."""
        self._label_current_frame(verdict="ok", error_type="", source="mark")

    def _label_current_frame(self, *, verdict: str, error_type: str, source: str) -> None:
        if self.data is None:
            return
        cur = self._current_frame()
        z = self.data.frame_to_z.get(cur)
        if z is None:
            print("[gui] current frame has no z mapping; not logged")
            return
        anchor_z = (self.data.frame_to_z.get(self.data.anchor_idx)
                    if self.data.anchor_idx is not None else None)
        role = ("anchor" if z == anchor_z
                else "flagged" if cur in self.data.triage_frames else "sampled")
        qc_row = (self.qc_df.loc[z] if self.qc_df is not None and z in self.qc_df.index else None)
        self.labels.record(self.neuron, self.chain_idx, z, verdict=verdict,
                           role=role, error_type=error_type, source=source,
                           reviewer=self.reviewer, qc_row=qc_row, anchor=self._anchor_dict())
        print(f"[gui] frame {cur} (z={z}, role={role}) labelled {verdict}"
              f"{(' / ' + error_type) if error_type else ''}")

    def _error_type(self) -> str:
        """The error type currently selected in the dock picker (or 'other')."""
        w = getattr(self, "_err_mode", None)
        return str(w.value) if w is not None and w.value else "other"

    def open_next_in_queue(self, *_) -> None:
        self._step_chain(+1)

    def open_prev_in_queue(self, *_) -> None:
        self._step_chain(-1)

    def _step_chain(self, direction: int) -> None:
        """Cycle to the next/prev CHAIN that still needs a human (different chain, vs
        next/prev *flagged FRAME*, which moves between frames within the open chain).

        Cycles the SAME list the picker shows, per the mode toggle: the pending
        review queue in 'flagged only' mode, or every on-disk chain in 'everything'
        mode. In flagged mode this keeps ``in_review`` chains visible (the fix for
        "can't return to an unfinished chain"): opening a chain marks it in_review, so
        excluding those made the queue look empty as soon as you'd visited each once.
        Only terminal dispositions (approved / rejected / corrected) drop a chain out
        of the flagged list. Wraps around, relative to the chain currently open."""
        self.queue.refresh()
        chains = self._mode_chains()                        # mode-aware (see _mode_chains)
        if not chains:
            print("[gui] no chains to cycle "
                  f"({self._mode_combo.value}); switch mode or refresh")
            return
        cur = (self.neuron, self.chain_idx)
        if cur in chains:
            i = (chains.index(cur) + direction) % len(chains)
        else:
            i = 0 if direction > 0 else len(chains) - 1
        if chains[i] == cur and len(chains) == 1:
            print(f"[gui] {cur[0]} chain {cur[1]:02d} is the only chain in this list")
            return
        self.open_chain(*chains[i])

    # =====================================================================
    # Label logging
    # =====================================================================
    def _anchor_dict(self) -> Optional[dict]:
        """The chain's anchor verdict from state.json (the anchor-contamination
        guard feature). Read fresh so a re-seg's new state is picked up."""
        sp = self.ctx.output_root / self.neuron / f"chain_{self.chain_idx:02d}" / "state.json"
        if sp.exists():
            return json.loads(sp.read_text()).get("anchor_score")
        return None

    def _log_frames(self, *, verdict: str, source: str, error_type: str = "") -> None:
        """Log every queued frame of the open chain (role='flagged' or 'anchor'),
        plus a uniform un-flagged sample (role='sampled'), into the label store."""
        anchor = self._anchor_dict()
        anchor_z = self.data.frame_to_z.get(self.data.anchor_idx) if self.data.anchor_idx is not None else None
        queued_z = [self.data.frame_to_z[f] for f in self.data.triage_frames
                    if f in self.data.frame_to_z]
        for z in queued_z:
            qc_row = self.qc_df.loc[z] if self.qc_df is not None and z in self.qc_df.index else None
            role = "anchor" if z == anchor_z else "flagged"
            self.labels.record(self.neuron, self.chain_idx, z, verdict=verdict,
                               role=role, error_type=(error_type if verdict == "wrong" else ""),
                               source=source, reviewer=self.reviewer,
                               qc_row=qc_row, anchor=anchor)
        # the silent-error window: a uniform sample of un-flagged frames, logged 'ok'
        if self.qc_df is not None:
            self.labels.sample_unflagged(self.neuron, self.chain_idx, self.qc_df,
                                         n=8, reviewer=self.reviewer, anchor=anchor,
                                         exclude_z=queued_z)
        print(f"[gui] logged {len(queued_z)} queued + sampled frames to {self.labels.path.name}")

    # =====================================================================
    # Mask/segment plumbing
    # =====================================================================
    def _frame_image_sam(self, frame_idx: int) -> np.ndarray:
        """The _sam EM frame as (H, W, 3) RGB uint8, read from the same JPEG the
        Image layer shows (so re-predict sees exactly what the human clicked on)."""
        from sam2_utils.video_viz import _load_frame
        return _load_frame(self.data.frames_dir, frame_idx)

    def _set_frame_mask(self, frame_idx: int, mask: np.ndarray) -> None:
        """Write a bool mask into the labels layer + segments for one frame."""
        m = np.asarray(mask).astype(bool)
        self.data.video_segments.setdefault(frame_idx, {})[self.data.obj_id] = m
        if self._mask is not None and 0 <= frame_idx < self._mask.data.shape[0]:
            vol = self._mask.data
            vol[frame_idx] = 0
            vol[frame_idx][m] = self.data.obj_id
            self._mask.data = vol
            self._mask.refresh()

    def _painted_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """The human-edited mask for ``frame_idx`` from the Labels layer (obj_id
        pixels), or None if the layer is gone."""
        if self._mask is None or not (0 <= frame_idx < self._mask.data.shape[0]):
            return None
        return self._mask.data[frame_idx] == self.data.obj_id

    def _merge_segments(self, new_segments: dict) -> None:
        """Overwrite our segments with re-propagated frames (last write wins) and
        refresh the labels layer to match."""
        for fi, seg in new_segments.items():
            if self.data.obj_id in seg:
                m = np.asarray(seg[self.data.obj_id])
                m = m[0] if m.ndim == 3 else m
                self._set_frame_mask(fi, m.astype(bool))

    def _save_and_qc(self, frame_conf: dict, pred_iou: dict) -> None:
        """Persist corrected masks, re-run QC, rewrite qc.csv + state.json. Mirrors
        run_chain's save+QC tail so the corrected chain is indistinguishable on disk
        from a fresh batch run (review.load_chain re-reads it the same way)."""
        cfg = self.ctx.cfg
        chain_dir = self.ctx.output_root / self.neuron / f"chain_{self.chain_idx:02d}"
        masks_dir = chain_dir / "masks"
        pipeline.save_masks(self.data.video_segments, self.data.frame_to_z, masks_dir,
                            obj_id=self.data.obj_id, mask_space_downscale=cfg.save_downscale)
        # this chain's own skeleton (NOT the whole neuron; see run_qc docstring)
        skel_chain = None
        if self.chain is not None:
            ids = {str(n) for n in self.chain["nodes"]}
            skel_chain = self.ctx.annotate_df[
                self.ctx.annotate_df["node_id"].astype(str).isin(ids)
            ][["z", "x_tif", "y_tif"]]
        try:
            summary, triage_z, status = pipeline.run_qc(
                masks_dir, skel_chain, frame_to_z=self.data.frame_to_z,
                frame_conf=frame_conf, pred_iou=pred_iou, cfg=cfg,
                qc_csv_path=chain_dir / "qc.csv",
                crop_window=self._cw)            # tier-2: re-score in _pcrop
        except Exception as e:                                  # QC must never lose the masks
            print(f"[gui] QC skipped after resume ({e}); masks were saved")
            return
        # refresh in-memory qc + triage so the panel + next/prev reflect the re-seg
        self.qc_df = pd.read_csv(chain_dir / "qc.csv").set_index("z")
        self._update_state_after_qc(summary, triage_z, status)
        z_to_frame = {z: i for i, z in self.data.frame_to_z.items()}
        self.data.triage_frames = [z_to_frame[z] for z in triage_z if z in z_to_frame]

    def _update_state_after_qc(self, summary, triage_z, status) -> None:
        """Reload state.json, patch the QC fields, and re-save (keeps anchor_score,
        config, timing, etc. intact). status becomes 'flagged'/'done' from QC; the
        *review* disposition lives separately in _review.csv."""
        sp = self.ctx.output_root / self.neuron / f"chain_{self.chain_idx:02d}" / "state.json"
        if not sp.exists():
            return
        st = pipeline.load_state(sp)
        st.qc_summary, st.triage_frames, st.status = summary, list(triage_z), status
        pipeline.save_state(st, sp)

    def _ensure_session(self) -> pipeline.PropagationSession:
        """Build (once) a PropagationSession over this chain's frames. Heavy:
        init_state loads every frame into the predictor."""
        if self.session is None:
            self.session = pipeline.PropagationSession(
                self.ctx.video_predictor, self.data.frames_dir, obj_id=self.data.obj_id)
        return self.session

    def _close_session(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None

    # =====================================================================
    # Widgets / keybindings / info panel
    # =====================================================================
    def _build_widgets(self) -> None:
        from magicgui.widgets import (Container, PushButton, ComboBox, Label, LineEdit,
                                      FloatSpinBox, SpinBox, CheckBox)

        self._info = Label(value="(no chain open)")

        # chain picker: a mode toggle + two cascading selectors (neuron, then chain).
        # 'flagged only' lists the review queue (today's behaviour); 'everything' lists
        # every chain on disk so the reviewer can open un-flagged chains while proofing.
        # The mode is the single source of truth: it also drives next/prev CHAIN below.
        self._mode_combo = ComboBox(label="show", choices=_MODE_CHOICES, value=_MODE_FLAGGED)
        self._mode_combo.changed.connect(lambda *_: self._on_mode_change())
        self._neuron_combo = ComboBox(label="neuron", choices=[])
        self._neuron_combo.changed.connect(lambda *_: self._on_neuron_change())
        # chain choices are (label, idx) tuples, so .value is the int chain_idx directly
        self._chain_combo = ComboBox(label="chain", choices=[])
        open_btn = PushButton(text="open selected chain")
        open_btn.changed.connect(lambda *_: self._open_from_picker())
        prev_q = PushButton(text="⇇ prev CHAIN")                    # cycle, incl. unfinished
        prev_q.changed.connect(self.open_prev_in_queue)
        next_q = PushButton(text="next CHAIN ⇉")                    # cycle, incl. unfinished
        next_q.changed.connect(self.open_next_in_queue)
        refresh = PushButton(text="↻ refresh queue")
        refresh.changed.connect(lambda *_: self._refresh_picker())
        self._populate_selectors()                                  # fill from the default mode

        # within-chain frame nav (different from next CHAIN above)
        prevf = PushButton(text="◀ prev flagged FRAME ( , )")      # same chain, prev queued frame
        prevf.changed.connect(self.prev_flagged)
        nextf = PushButton(text="next flagged FRAME ( . ) ▶")      # same chain, next queued frame
        nextf.changed.connect(self.next_flagged)

        # prompt label toggle (positive / negative) + box draw + reset to saved seed
        self._prompt_mode = ComboBox(label="new point", choices=_PROMPT_LABELS, value="positive")
        self._prompt_mode.changed.connect(self._set_prompt_label)
        box_btn = PushButton(text="▭ draw box (B)")
        box_btn.changed.connect(self.activate_box_draw)
        reset_btn = PushButton(text="⟲ reset prompts to original")
        reset_btn.changed.connect(self.reset_prompts)

        # view controls (point size / auto-zoom)
        self._size_spin = FloatSpinBox(label="point size", value=self.point_size,
                                       min=1.0, max=40.0, step=1.0)
        self._size_spin.changed.connect(self._set_point_size)
        self._zoom_chk = CheckBox(text="auto-zoom to mask", value=self.auto_zoom)
        self._zoom_chk.changed.connect(
            lambda *_: setattr(self, "auto_zoom", bool(self._zoom_chk.value)))
        zoom_btn = PushButton(text="zoom to mask (Z)")
        zoom_btn.changed.connect(lambda *_: self._zoom_to_mask(self._current_frame(),
                                                               pad=self.zoom_pad))

        # correction actions
        rerun = PushButton(text="re-run image phase (R)")
        rerun.changed.connect(self.rerun_image_phase)
        resume = PushButton(text="resume propagation (G)")
        resume.changed.connect(self.resume_propagation)
        # tier-2 recrop: grow this chain's crop window by N _tif px/side and re-run it
        # (for a window that auto-sized too small). Only meaningful on a tier-2 chain.
        self._grow_spin = SpinBox(label="grow crop (tif px)", value=512, min=0, max=8192, step=64)
        recrop = PushButton(text="⤢ recrop chain (C)")
        recrop.changed.connect(self.recrop_chain)
        # recrop region picker (Phase 2): draw the new window on the full _sam frame
        pick_region = PushButton(text="▣ pick recrop region (F)")
        pick_region.changed.connect(self.enter_recrop_picker)
        confirm_recrop = PushButton(text="✓ confirm recrop")
        confirm_recrop.changed.connect(self.confirm_recrop)
        cancel_recrop = PushButton(text="✗ cancel recrop")
        cancel_recrop.changed.connect(self.cancel_recrop)

        # the error type used by 'mark wrong' (W key) and 'reject'. The per-frame
        # mark ok/wrong buttons were dropped to declutter the dock; the W/O keys remain.
        self._err_mode = ComboBox(label="error type", choices=list(labels_mod.ERROR_TYPES),
                                  value="other")

        # chain dispositions
        approve = PushButton(text="✓ approve CHAIN (A)")
        approve.changed.connect(self.approve_chain)
        reject = PushButton(text="✗ reject CHAIN (X)")
        reject.changed.connect(self.reject_chain)

        self._reviewer_edit = LineEdit(label="reviewer", value=self.reviewer)
        self._reviewer_edit.changed.connect(
            lambda *_: setattr(self, "reviewer", self._reviewer_edit.value))

        panel = Container(widgets=[
            self._reviewer_edit,
            Label(value=", chains, "), self._mode_combo, self._neuron_combo, self._chain_combo,
            open_btn, prev_q, next_q, refresh,
            Label(value=", frames (this chain), "), prevf, nextf,
            Label(value=", prompts, "), self._prompt_mode, box_btn, reset_btn,
            Label(value=", view, "), self._size_spin, self._zoom_chk, zoom_btn,
            Label(value=", correct, "), rerun, resume,
            Label(value=", recrop, "), self._grow_spin, recrop,
            pick_region, confirm_recrop, cancel_recrop,
            Label(value=", disposition, "), self._err_mode, approve, reject,
            self._info,
        ], labels=True)
        # Wrap in a scroll area so a tall panel never buries buttons below the fold
        # (napari's dock does not scroll on its own); see the no-dock-scroll note in docs.
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel.native)
        self.viewer.window.add_dock_widget(scroll, area="right", name="review")

    def _bind_keys(self) -> None:
        v = self.viewer

        @v.bind_key(".", overwrite=True)
        def _nf(_v): self.next_flagged()

        @v.bind_key(",", overwrite=True)
        def _pf(_v): self.prev_flagged()

        @v.bind_key("p", overwrite=True)
        def _pos(_v): self._set_prompt_label("positive")

        @v.bind_key("n", overwrite=True)
        def _neg(_v): self._set_prompt_label("negative")

        @v.bind_key("b", overwrite=True)
        def _box(_v): self.activate_box_draw()

        @v.bind_key("r", overwrite=True)
        def _rerun(_v): self.rerun_image_phase()

        @v.bind_key("g", overwrite=True)
        def _resume(_v): self.resume_propagation()

        @v.bind_key("c", overwrite=True)
        def _recrop(_v): self.recrop_chain()

        @v.bind_key("f", overwrite=True)
        def _pickregion(_v): self.enter_recrop_picker()

        @v.bind_key("z", overwrite=True)
        def _zoom(_v): self._zoom_to_mask(self._current_frame(), pad=self.zoom_pad)

        @v.bind_key("w", overwrite=True)
        def _markw(_v): self.mark_frame_wrong()

        @v.bind_key("o", overwrite=True)
        def _marko(_v): self.mark_frame_ok()

        @v.bind_key("a", overwrite=True)
        def _approve(_v): self.approve_chain()

        @v.bind_key("x", overwrite=True)
        def _reject(_v): self.reject_chain()

    def _set_prompt_label(self, value=None) -> None:
        """Flip what a freshly-clicked prompt point is labelled (positive/negative).
        Drives both the dropdown and the points layer's feature default."""
        if value is None:
            value = self._prompt_mode.value
        if isinstance(value, str) and value in _PROMPT_LABELS:
            if self._prompt_mode.value != value:
                self._prompt_mode.value = value
            if self._prompts is not None:
                self._prompts.feature_defaults = {"label": value}
                self._prompts.mode = "add"

    def _set_point_size(self, value=None) -> None:
        """Live-resize the prompt + skeleton points (in _sam data units)."""
        self.point_size = float(value if value is not None else self._size_spin.value)
        for layer in (self._prompts, self._skel):
            if layer is not None:
                try:
                    layer.size = self.point_size
                    if len(layer.data):
                        layer.current_size = self.point_size
                except Exception as e:
                    print(f"[gui] point-size set skipped: {e}")

    # -- picker: mode + cascading (neuron, chain) selectors --------------------
    @property
    def _show_all(self) -> bool:
        """True when the picker is in 'everything' mode (every on-disk chain), False
        in 'flagged only' mode (the review queue). Drives both selectors and nav."""
        return getattr(self, "_mode_combo", None) is not None \
            and self._mode_combo.value == _MODE_EVERYTHING

    def _mode_chains(self) -> list:
        """The chain list for the current mode, as (neuron, chain_idx): the full
        on-disk set in 'everything' mode, else the pending review queue (incl.
        in_review, so an unfinished chain stays reachable). The single source the
        selectors and next/prev CHAIN both read."""
        if self._show_all:
            return self.queue.all_chains()
        return self.queue.pending(include_in_review=True)

    def _populate_selectors(self, *, select: Optional[tuple] = None) -> None:
        """(Re)fill the neuron + chain selectors from the current mode. ``select``
        is an optional (neuron, chain_idx) to land on; otherwise the open chain, else
        the first available. The ``_populating`` guard stops the neuron->chain cascade
        from re-entering while we set choices programmatically."""
        if getattr(self, "_neuron_combo", None) is None:
            return
        self._populating = True
        try:
            chains = self._mode_chains()
            neurons = sorted({n for (n, _) in chains})
            want = select or ((self.neuron, self.chain_idx)
                              if self.neuron is not None else None)
            self._neuron_combo.choices = neurons
            if neurons:
                self._neuron_combo.value = (want[0] if want and want[0] in neurons
                                            else neurons[0])
        finally:
            self._populating = False
        self._populate_chains(select_idx=want[1] if want else None)

    def _populate_chains(self, *, select_idx: Optional[int] = None) -> None:
        """Fill the chain selector for the neuron now selected, as (label, idx) tuples
        with a status badge (e.g. 'chain_03 [flagged]'). ``select_idx`` lands on a
        chain when present."""
        if getattr(self, "_chain_combo", None) is None:
            return
        neuron = self._neuron_combo.value
        idxs = sorted(i for (n, i) in self._mode_chains() if n == neuron)
        choices = [(f"chain_{i:02d} [{self.queue.chain_status(neuron, i)}]", i) for i in idxs]
        self._populating = True
        try:
            self._chain_combo.choices = choices
            if idxs:
                self._chain_combo.value = select_idx if select_idx in idxs else idxs[0]
        finally:
            self._populating = False

    def _on_mode_change(self) -> None:
        if getattr(self, "_populating", False):
            return
        self.queue.refresh()                       # pick up newly-flagged chains too
        self._populate_selectors()
        print(f"[gui] picker mode: {self._mode_combo.value} "
              f"({len(self._mode_chains())} chains)")

    def _on_neuron_change(self) -> None:
        if getattr(self, "_populating", False):
            return
        self._populate_chains()

    def _sync_selectors(self) -> None:
        """Point the selectors at the currently-open chain and re-read its badge
        (it just became in_review via queue.claim in open_chain). Best-effort."""
        if self.neuron is None or getattr(self, "_neuron_combo", None) is None:
            return
        self._populate_selectors(select=(self.neuron, self.chain_idx))

    def _open_from_picker(self) -> None:
        neuron = self._neuron_combo.value if self._neuron_combo is not None else None
        idx = self._chain_combo.value if self._chain_combo is not None else None
        if not neuron or idx is None:
            print("[gui] nothing to open (no chain selected)")
            return
        self.open_chain(str(neuron), int(idx))

    def _refresh_picker(self) -> None:
        self.queue.refresh()
        self._populate_selectors()
        print(f"[gui] queue refreshed: {len(self._mode_chains())} chains "
              f"({self._mode_combo.value})")

    def _refresh_info(self) -> None:
        if self.data is None:
            self._info.value = "(no chain open)"
            return
        rs = self.queue.status_of(self.neuron, self.chain_idx)
        cur = self._current_frame()
        z = self.data.frame_to_z.get(cur, "?")
        reasons = ""
        if self.qc_df is not None and z in getattr(self.qc_df, "index", []):
            row = self.qc_df.loc[z]
            r = []
            if row.get("skeleton_contained") is False:
                r.append("noskel")
            for k, tag in (("area_ratio", "area"), ("temporal_iou", "tIoU"), ("pred_iou", "pIoU")):
                v = row.get(k)
                if pd.notna(v):
                    r.append(f"{tag} {v:.2f}")
            reasons = "  ".join(r)
        self._info.value = (
            f"{self.neuron} chain {self.chain_idx:02d}  [{rs}]\n"
            f"frame {cur}  z={z}{'  ANCHOR' if cur == self.data.anchor_idx else ''}\n"
            f"queued frames: {len(self.data.triage_frames)}\n"
            f"this frame: {reasons or 'ok'}")


# =============================================================================
# Entry points
# =============================================================================

def launch(output_root: Optional[Path] = None, *, neuron: Optional[str] = None,
           chain_idx: Optional[int] = None, reviewer: str = "",
           cfg: Optional[pipeline.PipelineConfig] = None, block: bool = True,
           point_size: float = 4.0, auto_zoom: bool = True, hires_em: bool = False) -> ReviewGUI:
    """Open the review GUI. With ``neuron``/``chain_idx`` it opens straight onto a
    chain; otherwise it opens on the first pending chain (or an empty viewer if the
    queue is empty). ``block=True`` runs napari's event loop (call from a script);
    pass False from an interactive napari/IPython session that already has one.

    ``point_size`` / ``auto_zoom`` / ``hires_em`` forward to ReviewGUI (smaller
    prompt points, zoom-to-mask on open, full-res EM background; see ReviewGUI)."""
    import napari
    ctx = ReviewContext(Path(output_root) if output_root else config.OUTPUT_ROOT, cfg)
    gui = ReviewGUI(ctx, reviewer=reviewer, point_size=point_size,
                    auto_zoom=auto_zoom, hires_em=hires_em)
    if neuron is not None and chain_idx is not None:
        gui.open_chain(neuron, int(chain_idx))
    else:
        pend = gui.queue.pending(include_in_review=True)
        if pend:
            gui.open_chain(*pend[0])
        else:
            print("[gui] review queue is empty (no chains with manifest status 'flagged' "
                  "left undisposed). Open one manually via the queue picker.")
    if block:
        napari.run()
    return gui


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM2 napari review/triage GUI (M4)")
    ap.add_argument("--output-root", type=str, default=None,
                    help="output tree to review (default: config.OUTPUT_ROOT)")
    ap.add_argument("--neuron", type=str, default=None, help="open this neuron directly")
    ap.add_argument("--chain", type=int, default=None, help="open this chain_idx directly")
    ap.add_argument("--reviewer", type=str, default="", help="reviewer name stamped on labels")
    ap.add_argument("--point-size", type=float, default=4.0, help="prompt/skeleton point diameter (_sam px)")
    ap.add_argument("--no-auto-zoom", action="store_true", help="don't zoom to the mask on open/jump")
    ap.add_argument("--hires-em", action="store_true",
                    help="full-res EM background (lazy; mask stays scale-8, see gui.py header)")
    args = ap.parse_args()
    launch(Path(args.output_root) if args.output_root else None,
           neuron=args.neuron, chain_idx=args.chain, reviewer=args.reviewer,
           point_size=args.point_size, auto_zoom=not args.no_auto_zoom, hires_em=args.hires_em)


if __name__ == "__main__":
    main()
