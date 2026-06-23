"""gui_neuron.py: napari NEURON-level review GUI (the second review paradigm).

The per-chain tool (gui.py) opens one chain at a time. This one opens a whole NEURON:
all its chains (branches) on a single per-neuron crop canvas (_ncrop), shown as one
multi-color object. Branches stay separate SAM2 objects; the neuron is a presentation +
union layer. See docs/superpowers/specs/2026-06-23-neuron-review-gui-design.md and the
plan docs/superpowers/plans/2026-06-23-neuron-review-gui.md.

gui.py is untouched; this driver imports its shared pieces (ReviewContext, helpers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sam2_utils import review_queue


def neurons_on_disk(output_root) -> list[tuple[str, list[int]]]:
    """Every neuron with on-disk chains under output_root, as (neuron, [chain_idx,...]),
    sorted by neuron then chain. Built on ReviewQueue.all_chains so the openable set
    matches exactly what review.load_chain can read."""
    q = review_queue.ReviewQueue(Path(output_root))
    by_neuron: dict[str, list[int]] = {}
    for neuron, idx in q.all_chains():
        by_neuron.setdefault(neuron, []).append(idx)
    return [(n, sorted(by_neuron[n])) for n in sorted(by_neuron)]


def build_neuron_label_volume(branch_masks: dict, t: int,
                              hw: tuple[int, int]) -> np.ndarray:
    """(t, H, W) uint16 volume: for each branch label L and frame fi,
    branch_masks[L][fi] (bool, shape hw) is written as L. Ascending label order, so a
    higher label wins on overlap (deterministic). Labels are the per-branch editing
    integers; the saved neuron identity is independent of them."""
    H, W = hw
    vol = np.zeros((t, H, W), dtype=np.uint16)
    for label in sorted(branch_masks):
        for fi, m in branch_masks[label].items():
            if 0 <= fi < t:
                vol[fi][np.asarray(m, bool)] = label
    return vol


# =============================================================================
# The neuron-review GUI
# =============================================================================

class NeuronReviewGUI:
    """A napari window that opens one NEURON at a time onto a per-neuron crop canvas.

    Layers per neuron:
        Image  'EM'      the _ncrop frames over the neuron's union z-range
        Labels 'neuron'  one integer per branch (label = chain_idx + 1; selected_label
                         is the active branch)
        Points 'prompts' the active branch's click prompts (built in Task 5)
        Shapes 'box'     the active branch's bounding box (built in Task 5)
    """

    def __init__(self, ctx, *, reviewer: str = "", viewer=None):
        import napari
        self.ctx = ctx
        self.reviewer = reviewer
        self.viewer = viewer if viewer is not None else napari.Viewer(title="SAM2 neuron review")
        self.queue = review_queue.ReviewQueue(ctx.output_root)
        self.neuron: Optional[str] = None
        self.cw = None                            # the neuron CropWindow (_ncrop)
        self.chain_idxs: list[int] = []           # branch chain indices; label = idx + 1
        self.frame_to_z: dict = {}
        self.frames_dir: Optional[str] = None
        self._img = self._neuron = None
        self._prompts = self._box = None
        self._active_session = None
        self._session_label = None
        self._build_widgets()
        self._bind_keys()

    # -- open a neuron ---------------------------------------------------------
    def open_neuron(self, neuron: str) -> None:
        import pipeline
        from sam2_utils import review
        from sam2_utils.alignment import CropWindow
        self.neuron = neuron
        chains = [c for c in self.ctx.chains if c.get("cell_name") == neuron]
        self.chain_idxs = sorted(i for (n, i) in self.queue.all_chains() if n == neuron)
        if not self.chain_idxs:
            print(f"[gui_neuron] no on-disk chains for {neuron}")
            return

        # 1. load every branch's ReviewData; union z-range over their saved frames
        reviews = {i: review.load_chain(self.ctx.output_root / neuron / f"chain_{i:02d}")
                   for i in self.chain_idxs}
        all_z = sorted({z for rd in reviews.values() for z in rd.frame_to_z.values()})
        anchor_z = all_z[0]
        _img_full, full_hw = pipeline.load_frame_sam(int(anchor_z), scale=1)

        # 2. the per-neuron crop window (_ncrop)
        self.cw = pipeline.neuron_crop_window(chains, self.ctx.annotate_df,
                                              cfg=self.ctx.cfg, image_hw_tif=full_hw)
        print(f"[gui_neuron] {neuron}: {len(self.chain_idxs)} branches, {len(all_z)} slices, "
              f"_ncrop {self.cw.size_tif[0]}x{self.cw.size_tif[1]}px @ crop_scale {self.cw.crop_scale}")

        # 3. _ncrop frames over the union z-range. Reuse the chain-crop frame writer with a
        #    synthetic "chain" spanning all the neuron's nodes (it reads node z extent + the
        #    window slice). chain_idx=999 namespaces the neuron's frame cache.
        merged = {"cell_name": neuron, "nodes": [n for c in chains for n in c["nodes"]]}
        self.frames_dir, frame_to_z2, _af, _n = pipeline.prepare_chain_crop_frames(
            merged, self.ctx.annotate_df, self.cw, frames_root=self.ctx.cfg.frames_root,
            anchor_catmaid_z=int(anchor_z), neuron=neuron, chain_idx=999)
        self.frame_to_z = frame_to_z2
        z_to_frame = {z: fi for fi, z in frame_to_z2.items()}

        # 4. remap each branch's saved masks into _ncrop; label = chain_idx + 1
        H, W = self.cw.crop_hw
        t = len(frame_to_z2)
        branch_masks: dict[int, dict[int, np.ndarray]] = {}
        for i in self.chain_idxs:
            rd = reviews[i]
            sp = self.ctx.output_root / neuron / f"chain_{i:02d}" / "state.json"
            st = pipeline.load_state(sp) if sp.exists() else None
            src_cw = (CropWindow.from_dict(st.crop_window)
                      if st is not None and getattr(st, "crop_window", None) else None)
            label = i + 1
            branch_masks[label] = {}
            for fi_src, z in rd.frame_to_z.items():
                fi = z_to_frame.get(z)
                if fi is None or fi_src not in rd.video_segments:
                    continue
                seg = rd.video_segments[fi_src].get(rd.obj_id)
                if seg is None:
                    continue
                m = np.asarray(seg)
                m = m[0] if m.ndim == 3 else m
                if src_cw is not None:                       # tier-2 branch: _pcrop footprint
                    so, ss = src_cw.origin_tif, src_cw.size_tif
                else:                                        # legacy _sam branch: full frame
                    so, ss = (0.0, 0.0), (full_hw[1], full_hw[0])
                branch_masks[label][fi] = pipeline.remap_mask_to_window(
                    m, src_origin_tif=so, src_size_tif=ss, dst_cw=self.cw)

        vol = build_neuron_label_volume(branch_masks, t, (H, W))

        # 5. (re)build layers
        em = self._load_ncrop_stack(t)
        self.viewer.layers.clear()
        self._img = self.viewer.add_image(em, name="EM", rgb=True)
        self._neuron = self.viewer.add_labels(vol, name="neuron", opacity=0.5)
        self._neuron.selected_label = (self.chain_idxs[0] + 1)
        self._build_prompt_layers()
        self.viewer.reset_view()
        self._refresh_info()

    def _load_ncrop_stack(self, t: int):
        from gui import _load_frame_stack
        return _load_frame_stack(self.frames_dir, t)

    # -- prompt / box layers (scoped to the active branch) ---------------------
    def _build_prompt_layers(self) -> None:
        import pandas as pd
        from gui import _POS_COLOR, _NEG_COLOR, _PROMPT_LABELS, _BOX_EDGE_COLOR
        feats = pd.DataFrame({"label": pd.Categorical([], categories=_PROMPT_LABELS)})
        self._prompts = self.viewer.add_points(
            np.empty((0, 3)), name="prompts", ndim=3, size=6.0, features=feats,
            border_color="label", border_color_cycle=[_POS_COLOR, _NEG_COLOR],
            face_color="transparent", border_width=0.4, symbol="o")
        self._prompts.border_color_mode = "cycle"
        self._prompts.feature_defaults = {"label": "positive"}
        self._prompts.mode = "add"
        self._box = self.viewer.add_shapes(
            name="box", ndim=3, edge_color=_BOX_EDGE_COLOR, face_color="transparent",
            edge_width=1.0, opacity=0.8)

    # -- widgets / keys / info -------------------------------------------------
    def _build_widgets(self) -> None:
        from magicgui.widgets import Container, ComboBox, PushButton, Label
        self._info = Label(value="(no neuron open)")
        neurons = [n for (n, _idxs) in neurons_on_disk(self.ctx.output_root)] or ["(none)"]
        self._neuron_combo = ComboBox(label="neuron", choices=neurons)
        open_btn = PushButton(text="open neuron")
        open_btn.changed.connect(lambda *_: self.open_neuron(str(self._neuron_combo.value)))
        rerun = PushButton(text="re-run image phase (R)")
        rerun.changed.connect(self.rerun_image_phase)
        resume = PushButton(text="resume propagation (G)")
        resume.changed.connect(self.resume_propagation)
        approve = PushButton(text="✓ approve NEURON")
        approve.changed.connect(self.approve_neuron)
        reject = PushButton(text="✗ reject NEURON")
        reject.changed.connect(self.reject_neuron)
        panel = Container(widgets=[
            self._neuron_combo, open_btn,
            Label(value=", correct active branch, "), rerun, resume,
            Label(value=", neuron, "), approve, reject,
            self._info,
        ], labels=True)
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel.native)
        self.viewer.window.add_dock_widget(scroll, area="right", name="neuron review")

    def _bind_keys(self) -> None:
        v = self.viewer

        @v.bind_key("r", overwrite=True)
        def _r(_v): self.rerun_image_phase()

        @v.bind_key("g", overwrite=True)
        def _g(_v): self.resume_propagation()

        @v.bind_key("b", overwrite=True)
        def _b(_v):
            if self._box is not None:
                self.viewer.layers.selection.active = self._box
                self._box.mode = "add_rectangle"

    def _refresh_info(self) -> None:
        if self.neuron is None:
            self._info.value = "(no neuron open)"
            return
        self._info.value = (f"{self.neuron}\n{len(self.chain_idxs)} branches\n"
                            f"active branch (label) = {getattr(self._neuron, 'selected_label', '?')}")

    # -- active branch ---------------------------------------------------------
    def _active_label(self) -> int:
        return int(getattr(self._neuron, "selected_label", 0) or 0)

    def _active_obj_id(self) -> int:
        """The branch's saved obj_id (label = chain_idx + 1)."""
        from sam2_utils import review
        idx = self._active_label() - 1
        rd = review.load_chain(self.ctx.output_root / self.neuron / f"chain_{idx:02d}")
        return int(rd.obj_id)

    def _current_frame(self) -> int:
        return int(self.viewer.dims.current_step[0])

    def _prompts_for_frame(self, fi: int):
        """The prompt-layer points on frame ``fi`` as a pipeline.Prompts, in _ncrop coords
        (the data grid, so points are (x, y) directly; labels 1=pos/0=neg)."""
        import pipeline
        from gui import _LABEL_TO_SAM
        data = np.asarray(self._prompts.data, dtype=float)
        if not len(data):
            return pipeline.Prompts(points_sam=np.empty((0, 2)), labels=np.empty((0,), int))
        on = np.round(data[:, 0]).astype(int) == int(fi)
        rows = data[on]
        feats = self._prompts.features.get("label", None)
        labs_src = np.asarray(feats)[on] if feats is not None else ["positive"] * len(rows)
        pts_xy = np.stack([rows[:, 2], rows[:, 1]], axis=1)
        labs = np.array([_LABEL_TO_SAM.get(str(s), 1) for s in labs_src], dtype=int)
        return pipeline.Prompts(points_sam=pts_xy, labels=labs)

    def _box_for_frame(self, fi: int):
        from gui import _box_on_frame
        return None if self._box is None else _box_on_frame(self._box.data, int(fi))

    def _set_branch_mask(self, fi: int, mask) -> None:
        """Write a bool mask for the ACTIVE branch on one frame, into the neuron layer."""
        lbl = self._active_label()
        vol = self._neuron.data
        m = np.asarray(mask, bool)
        frame = vol[fi]
        frame[frame == lbl] = 0          # clear this branch's old pixels on this frame
        frame[m] = lbl
        self._neuron.data = vol
        self._neuron.refresh()

    # -- corrections (active branch, in _ncrop) --------------------------------
    def rerun_image_phase(self, *_) -> None:
        """Re-predict the ACTIVE branch on the current frame from its points/box, in the
        shared _ncrop space, and write the result into the neuron layer (that branch only)."""
        import pipeline
        from sam2_utils.video_viz import _load_frame
        if self.neuron is None:
            return
        fi = self._current_frame()
        prompts = self._prompts_for_frame(fi)
        prompts.box_sam = self._box_for_frame(fi)
        if not (prompts.labels == 1).any() and prompts.box_sam is None:
            print("[gui_neuron] need a positive point or a box on this frame")
            return
        self.ctx.ensure_predictors(need_image=True, need_video=False)
        em = _load_frame(self.frames_dir, fi)            # the _ncrop frame
        mask, score, _ = pipeline.image_predict(self.ctx.image_predictor, em, prompts)
        self.ctx.image_predictor.reset_predictor()
        self._set_branch_mask(fi, mask)
        print(f"[gui_neuron] re-predicted branch {self._active_label()} frame {fi}: "
              f"{int(mask.sum())} px, score {score:.3f}")
        self._refresh_info()

    def _session(self):
        """A PropagationSession over the shared _ncrop frames for the ACTIVE branch's
        obj_id, rebuilt (and the previous closed) when the active branch changes."""
        import pipeline
        lbl = self._active_label()
        if self._session_label != lbl:
            if self._active_session is not None:
                self._active_session.close()
            self._active_session = pipeline.PropagationSession(
                self.ctx.video_predictor, self.frames_dir, obj_id=self._active_obj_id())
            self._session_label = lbl
        return self._active_session

    def resume_propagation(self, *_) -> None:
        """Re-track the ACTIVE branch over the _ncrop frames, seeded by its mask on the
        current frame (both directions), then save just that branch."""
        if self.neuron is None:
            return
        self.ctx.ensure_predictors(need_image=False, need_video=True)
        fi = self._current_frame()
        lbl = self._active_label()
        mask = (self._neuron.data[fi] == lbl)
        if not mask.any():
            print("[gui_neuron] no mask for the active branch on this frame; re-predict first")
            return
        sess = self._session()
        sess.add_mask(fi, mask)
        for rev in (False, True):
            for _ in sess.propagate(reverse=rev, start_frame_idx=fi):
                pass
        for f2, seg in sess.video_segments.items():
            if sess.obj_id in seg:
                mm = np.asarray(seg[sess.obj_id])
                mm = mm[0] if mm.ndim == 3 else mm
                self._set_branch_mask(f2, mm.astype(bool))
        self._save_branch(lbl)
        print(f"[gui_neuron] branch {lbl} re-propagated and saved")
        self._refresh_info()

    def _save_branch(self, label: int) -> None:
        """Persist the ACTIVE branch's _ncrop masks + state (its crop_window becomes the
        neuron window) and mark it corrected. Writes to the chain's on-disk masks dir."""
        import pipeline
        idx = label - 1
        chain_dir = self.ctx.output_root / self.neuron / f"chain_{idx:02d}"
        obj_id = self._active_obj_id()
        vol = self._neuron.data
        segments = {fi: {obj_id: (vol[fi] == label)}
                    for fi in range(vol.shape[0]) if (vol[fi] == label).any()}
        pipeline.save_masks(segments, self.frame_to_z, chain_dir / "masks",
                            obj_id=obj_id, mask_space_downscale=self.ctx.cfg.save_downscale)
        sp = chain_dir / "state.json"
        if sp.exists():
            st = pipeline.load_state(sp)
            st.crop_window = self.cw.to_dict()           # branch now lives in _ncrop
            pipeline.save_state(st, sp)
        self.queue.set_status(self.neuron, idx, review_queue.CORRECTED, reviewer=self.reviewer)

    # -- neuron-level disposition ----------------------------------------------
    def approve_neuron(self, *_) -> None:
        if self.neuron is None:
            return
        for i in self.chain_idxs:
            self.queue.set_status(self.neuron, i, review_queue.APPROVED, reviewer=self.reviewer)
        print(f"[gui_neuron] {self.neuron} approved ({len(self.chain_idxs)} branches)")
        self._refresh_info()

    def reject_neuron(self, *_) -> None:
        if self.neuron is None:
            return
        for i in self.chain_idxs:
            self.queue.set_status(self.neuron, i, review_queue.REJECTED, reviewer=self.reviewer)
        print(f"[gui_neuron] {self.neuron} rejected ({len(self.chain_idxs)} branches)")
        self._refresh_info()


def launch(output_root: Optional[Path] = None, *, neuron: Optional[str] = None,
           reviewer: str = "", block: bool = True):
    """Open the neuron-review GUI. With ``neuron`` it opens straight onto one neuron."""
    import napari
    from gui import ReviewContext
    from sam2_utils import config
    ctx = ReviewContext(Path(output_root) if output_root else config.OUTPUT_ROOT)
    gui = NeuronReviewGUI(ctx, reviewer=reviewer)
    if neuron is not None:
        gui.open_neuron(neuron)
    if block:
        napari.run()
    return gui


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="SAM2 napari NEURON-level review GUI")
    ap.add_argument("--output-root", type=str, default=None)
    ap.add_argument("--neuron", type=str, default=None)
    ap.add_argument("--reviewer", type=str, default="")
    args = ap.parse_args()
    launch(Path(args.output_root) if args.output_root else None,
           neuron=args.neuron, reviewer=args.reviewer)


if __name__ == "__main__":
    main()

