"""Co-propagation lab: a standalone, disposable napari test for the neighbor-competition
hypothesis. Not part of the pipeline; saves nothing, scores nothing. See
docs/superpowers/specs/2026-06-30-coprop-lab-design.md.

torch and napari are imported lazily inside the functions that need them, so importing
this module for the pure helpers stays CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def label_stack(segments, obj_ids, t, hw):
    """(t, H, W) uint8: each obj_id in `obj_ids` painted with its own id, background 0.

    `segments` is {frame_idx: {obj_id: mask}} where a mask is bool HxW or the SAM2
    video shape (1, H, W); the leading dim is squeezed. Used for the target-alone, the
    target-with-neighbors, and the neighbors layers, by passing the relevant id list.
    """
    H, W = hw
    out = np.zeros((t, H, W), dtype=np.uint8)
    for fi, seg in segments.items():
        if not (0 <= fi < t):
            continue
        for oid in obj_ids:
            if oid in seg:
                m = np.asarray(seg[oid])
                m = m[0] if m.ndim == 3 else m   # SAM2 logits carry a leading (1,H,W) dim
                if m.shape == (H, W):
                    out[fi][m.astype(bool)] = oid
    return out


def build_diff_stack(alone, withn, obj_id):
    """(t, H, W) uint8: 1 where the target had a pixel ALONE but lost it with neighbors
    (bleed carved out), 2 where it GAINED one. The visual read of the test. Under the
    output-only variant the target can only lose pixels, so a correct Test 1 diff has no 2s.
    """
    a = (alone == obj_id)
    b = (withn == obj_id)
    out = np.zeros_like(alone, dtype=np.uint8)
    out[a & ~b] = 1     # lost (carved-out bleed)
    out[b & ~a] = 2     # gained
    return out


def load_em_stack(frames_dir, n_frames):
    """(n, H, W, 3) uint8 RGB over the chain's 0-indexed frames, eager. Reuses the same
    single-frame reader review/video_viz use, so naming and color match the rest of the repo."""
    from sam2_utils.video_viz import _load_frame
    frames = [_load_frame(Path(frames_dir), i) for i in range(n_frames)]
    return np.stack(frames, axis=0)


@dataclass
class LabChain:
    em: np.ndarray                  # (T, H, W, 3) uint8 RGB, the display stack
    frames_dir: str                 # path SAM2 init_state needs
    anchor_idx: int
    obj_id: int                     # the target object id (== 1 in production)
    frame_to_z: dict
    target_prompts: object          # pipeline.Prompts (box + points in propagation space)
    anchor_mask: np.ndarray         # bool (H, W), the saved target mask at the anchor frame
    n_frames: int
    hw: tuple
    crop_window: Optional[object] = None


def load_lab_chain(output_root, neuron, chain_idx):
    """Build a LabChain from an on-disk, already-run chain. Pure I/O, no torch."""
    import pipeline
    from sam2_utils import review, alignment

    chain_dir = Path(output_root) / neuron / f"chain_{int(chain_idx):02d}"
    if not chain_dir.exists():
        raise FileNotFoundError(f"no chain dir at {chain_dir}")

    data = review.load_chain(chain_dir, verbose=True)        # ReviewData
    t = (max(data.video_segments) + 1) if data.video_segments else 0

    any_mask = next(iter(data.video_segments.values()))[data.obj_id]
    any_mask = any_mask[0] if np.asarray(any_mask).ndim == 3 else any_mask
    H, W = np.asarray(any_mask).shape

    anchor_seg = data.video_segments.get(data.anchor_idx, {})
    am = anchor_seg.get(data.obj_id)
    if am is None:
        raise ValueError(f"no target mask at anchor frame {data.anchor_idx}")
    am = np.asarray(am)
    anchor_mask = (am[0] if am.ndim == 3 else am).astype(bool)

    state_path = chain_dir / "state.json"
    state = pipeline.load_state(state_path) if state_path.exists() else None
    prompts = getattr(state, "prompts", None)
    cw = None
    if state is not None and getattr(state, "crop_window", None):
        cw = alignment.CropWindow.from_dict(state.crop_window)

    em = load_em_stack(data.frames_dir, t)
    return LabChain(em=em, frames_dir=str(data.frames_dir), anchor_idx=int(data.anchor_idx),
                    obj_id=int(data.obj_id), frame_to_z=data.frame_to_z,
                    target_prompts=prompts, anchor_mask=anchor_mask,
                    n_frames=t, hw=(H, W), crop_window=cw)


class MultiObjectCopropSession:
    """Co-propagate several objects (target + neighbors) in ONE inference_state.

    SAM2 tracks each object's memory independently; the only coupling is the per-pixel
    non-overlap argmax. This session seeds N objects and toggles that argmax via the two
    predictor flags (set at runtime, RESTORED on close so a shared predictor is never left
    mutated):
      non_overlap          -> non_overlap_masks (OUTPUT masks only; Test 1 cleanup)
      non_overlap_mem_enc  -> non_overlap_masks_for_mem_enc (fed back into memory, changes
                              the trajectory; Test 2 propagation).
    A no-op with a single object: the constraint needs at least two objects to do anything.
    """

    def __init__(self, video_predictor, frames_dir, *, non_overlap=False,
                 non_overlap_mem_enc=False, offload_video_to_cpu=True):
        self.vp = video_predictor
        self.obj_ids = []
        self._orig_no = getattr(video_predictor, "non_overlap_masks", False)
        self._orig_no_mem = getattr(video_predictor, "non_overlap_masks_for_mem_enc", False)
        video_predictor.non_overlap_masks = bool(non_overlap)
        video_predictor.non_overlap_masks_for_mem_enc = bool(non_overlap_mem_enc)
        self.inference_state = video_predictor.init_state(
            video_path=str(frames_dir), offload_video_to_cpu=offload_video_to_cpu)
        video_predictor.reset_state(self.inference_state)
        self.video_segments = {}
        self._closed = False

    def seed_points_box(self, obj_id, prompts, anchor_frame_idx, *, seed_box=True,
                        seed_points=True, seed_negatives=False):
        """Seed one object from a Prompts (box + positive point by default)."""
        pts = np.asarray(prompts.points_sam, dtype=np.float32)
        labels = np.asarray(prompts.labels, dtype=np.int32)
        keep = np.ones(len(labels), dtype=bool)
        if not seed_points:
            keep &= labels != 1
        if not seed_negatives:
            keep &= labels != 0
        pts, labels = pts[keep], labels[keep]
        box = (np.asarray(prompts.box_sam, dtype=np.float32)
               if (seed_box and prompts.box_sam is not None) else None)
        if box is None and len(pts) == 0:
            raise ValueError("empty seed: enable at least one of box / points")
        # SAM2 requires clear_old_points=True when a box is present (box must precede points).
        self.vp.add_new_points_or_box(
            inference_state=self.inference_state, frame_idx=int(anchor_frame_idx),
            obj_id=int(obj_id), box=box, points=(pts if len(pts) else None),
            labels=(labels if len(labels) else None))
        if int(obj_id) not in self.obj_ids:
            self.obj_ids.append(int(obj_id))

    def seed_mask(self, obj_id, mask, anchor_frame_idx):
        """Seed one object from a 2D bool mask (the correct-seed path, Test 2)."""
        self.vp.add_new_mask(
            inference_state=self.inference_state, frame_idx=int(anchor_frame_idx),
            obj_id=int(obj_id), mask=np.asarray(mask, dtype=bool))
        if int(obj_id) not in self.obj_ids:
            self.obj_ids.append(int(obj_id))

    def _drain(self, *, reverse):
        for f, obj_ids, mask_logits in self.vp.propagate_in_video(
                self.inference_state, reverse=reverse):
            fi = int(f)
            per_obj = self.video_segments.setdefault(fi, {})
            for i, oid in enumerate(obj_ids):
                per_obj[int(oid)] = (mask_logits[i].cpu().numpy() > 0.0)

    def run_bidirectional(self):
        """Forward then reverse over all seeded objects (one shared memory)."""
        self._drain(reverse=False)
        self._drain(reverse=True)

    def close(self):
        if not self._closed:
            self.vp.non_overlap_masks = self._orig_no
            self.vp.non_overlap_masks_for_mem_enc = self._orig_no_mem
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def predict_neighbor_at(image_predictor, frame_img, xy, *, box_margin=6):
    """Image-mode predict a neighbor mask from a single positive click in propagation space.

    Returns (mask bool HxW, Prompts with box + the positive point), or None if SAM2 returns
    an empty mask. The returned Prompts is ready for MultiObjectCopropSession.seed_points_box.
    """
    import pipeline
    x, y = float(xy[0]), float(xy[1])
    point_prompts = pipeline.Prompts(
        points_sam=np.asarray([[x, y]], dtype=float),
        labels=np.asarray([1], dtype=int), box_sam=None)
    mask, _score, _logits = pipeline.image_predict(image_predictor, frame_img, point_prompts)
    if mask is None or not mask.any():
        return None
    box = pipeline.box_from_mask(mask, margin=box_margin, image_hw_sam=mask.shape[:2])
    if box is None:
        return None
    seed = pipeline.Prompts(
        points_sam=np.asarray([[x, y]], dtype=float),
        labels=np.asarray([1], dtype=int),
        box_sam=np.asarray(box, dtype=np.float32))
    return mask.astype(bool), seed


class CopropLab:
    """Standalone napari app for the co-propagation A/B. Saves nothing.

    Controls (in the dock):
      - target seed:  'auto-prompts' (the saved box+point) or 'current mask' (the paint layer)
      - variant:      'output-only' (Test 1 cleanup) or 'memory' (Test 2 propagation)
      - presets:      Test 1 sets (auto-prompts, output-only); Test 2 sets (current mask, memory)
      - Alt+click the EM to seed a neighbor (image-mode preview shown immediately). Alt keeps
        seeding separate from painting the target mask.
      - 'remove last neighbor', 'run A/B'
    Result layers: target (alone), target (w/ neighbors), neighbors, diff (1=lost, 2=gained).
    """

    NEIGHBOR_BASE_ID = 2     # target is obj 1; neighbors are 2, 3, ...

    def __init__(self, output_root, neuron, chain_idx):
        self.lc = load_lab_chain(output_root, neuron, chain_idx)
        self.neuron, self.chain_idx = neuron, int(chain_idx)
        self.target_seed = "auto-prompts"
        self.variant = "output-only"
        self.neighbors = []         # list of dicts: {"obj_id", "prompts", "mask"}
        self.image_predictor = None
        self.video_predictor = None
        self.viewer = None
        self._em_world = 1.0        # EM px per mask px (width ratio; 1.0 since EM == propagation space)

    # -- predictors (lazy; built once) ----------------------------------------
    def _ensure_predictors(self):
        from sam2_utils.setup import build_predictor
        if self.video_predictor is None:
            self.video_predictor, dev = build_predictor(kind="video")
            self.image_predictor, _ = build_predictor(kind="image", device=dev)

    # -- viewer ---------------------------------------------------------------
    def run(self):
        import napari
        lc = self.lc
        H, W = lc.hw
        self._em_world = float(lc.em.shape[2]) / float(W) if W else 1.0
        s = self._em_world
        lscale = (1.0, s, s)

        self.viewer = napari.Viewer(title=f"coprop lab: {self.neuron} chain {self.chain_idx:02d}")
        self._em_layer = self.viewer.add_image(lc.em, name="EM", rgb=True)

        # target paint layer, preloaded with the saved anchor mask at the anchor frame
        paint = np.zeros((lc.n_frames, H, W), dtype=np.uint8)
        paint[lc.anchor_idx][lc.anchor_mask] = lc.obj_id
        self._paint = self.viewer.add_labels(paint, name="target seed (paint)",
                                             opacity=0.5, scale=lscale)
        self._neighbors_layer = self.viewer.add_labels(
            np.zeros((lc.n_frames, H, W), dtype=np.uint8), name="neighbor seeds",
            opacity=0.5, scale=lscale)

        # global click handler (fires regardless of the selected layer); Alt+click seeds.
        self.viewer.mouse_drag_callbacks.append(self._on_click)

        self._build_dock()
        self.viewer.dims.set_current_step(0, int(lc.anchor_idx))
        napari.run()

    def _build_dock(self):
        from magicgui.widgets import ComboBox, PushButton, Label, Container
        seed_cb = ComboBox(label="target seed", choices=["auto-prompts", "current mask"],
                           value=self.target_seed)
        var_cb = ComboBox(label="variant", choices=["output-only", "memory"],
                          value=self.variant)
        seed_cb.changed.connect(lambda v: setattr(self, "target_seed", v))
        var_cb.changed.connect(lambda v: setattr(self, "variant", v))

        t1 = PushButton(text="preset: Test 1 (cleanup)")
        t2 = PushButton(text="preset: Test 2 (propagation)")
        t1.clicked.connect(lambda: self._set_preset(seed_cb, var_cb, "auto-prompts", "output-only"))
        t2.clicked.connect(lambda: self._set_preset(seed_cb, var_cb, "current mask", "memory"))

        rm = PushButton(text="remove last neighbor")
        run = PushButton(text="run A/B")
        rm.clicked.connect(self._remove_last_neighbor)
        run.clicked.connect(lambda: self.run_ab())
        self._status = Label(label="neighbors", value="0 seeded (Alt+click to add)")

        box = Container(widgets=[seed_cb, var_cb, t1, t2, rm, run, self._status])
        self.viewer.window.add_dock_widget(box, name="coprop", area="right")

    def _set_preset(self, seed_cb, var_cb, seed, variant):
        seed_cb.value = seed
        var_cb.value = variant
        self.target_seed, self.variant = seed, variant

    # -- neighbor seeding by Alt+click ----------------------------------------
    def _on_click(self, viewer, event):
        if "Alt" not in getattr(event, "modifiers", ()):
            return                                   # only Alt+click seeds (leaves paint free)
        if int(self.viewer.dims.current_step[0]) != int(self.lc.anchor_idx):
            print("[coprop] Alt+click on the anchor frame to seed a neighbor")
            return
        self._ensure_predictors()
        pos = self._em_layer.world_to_data(event.position)   # (z, y, x) in EM world
        y, x = float(pos[1]) / self._em_world, float(pos[2]) / self._em_world
        res = predict_neighbor_at(self.image_predictor, self.lc.em[self.lc.anchor_idx], (x, y))
        if res is None:
            print(f"[coprop] empty neighbor mask at ({x:.0f},{y:.0f}); try another spot")
            return
        mask, prompts = res
        obj_id = self.NEIGHBOR_BASE_ID + len(self.neighbors)
        self.neighbors.append({"obj_id": obj_id, "prompts": prompts, "mask": mask})
        self._refresh_neighbor_preview()
        print(f"[coprop] neighbor {obj_id} seeded ({int(mask.sum())} px)")

    def _remove_last_neighbor(self):
        if self.neighbors:
            dropped = self.neighbors.pop()
            self._refresh_neighbor_preview()
            print(f"[coprop] removed neighbor {dropped['obj_id']}")

    def _refresh_neighbor_preview(self):
        H, W = self.lc.hw
        vol = np.zeros((self.lc.n_frames, H, W), dtype=np.uint8)
        for nb in self.neighbors:
            vol[self.lc.anchor_idx][nb["mask"]] = nb["obj_id"]
        self._neighbors_layer.data = vol
        self._status.value = f"{len(self.neighbors)} seeded (Alt+click to add)"

    # -- the A/B run ----------------------------------------------------------
    def _seed_all(self, session):
        lc = self.lc
        if self.target_seed == "current mask":
            tgt = (self._paint.data[lc.anchor_idx] == lc.obj_id)
            if not tgt.any():
                raise ValueError("target paint layer is empty at the anchor frame")
            session.seed_mask(lc.obj_id, tgt, lc.anchor_idx)
        else:
            # auto-prompts: the CATMAID skeleton prompts only (positive node + negative
            # neighbor nodes from state.prompts), NOT the box derived later by box_from_mask.
            if lc.target_prompts is None:
                raise ValueError("no saved CATMAID prompts; use the 'current mask' seed instead")
            session.seed_points_box(lc.obj_id, lc.target_prompts, lc.anchor_idx,
                                    seed_box=False, seed_points=True, seed_negatives=True)
        for nb in self.neighbors:
            session.seed_points_box(nb["obj_id"], nb["prompts"], lc.anchor_idx)

    def run_ab(self):
        if not self.neighbors:
            print("[coprop] seed at least one neighbor first (the constraint is a no-op "
                  "with a single object)")
            return
        self._ensure_predictors()
        lc = self.lc
        mem = (self.variant == "memory")
        neigh_ids = [nb["obj_id"] for nb in self.neighbors]

        print(f"[coprop] baseline pass (neighbors off), seed={self.target_seed}")
        with MultiObjectCopropSession(self.video_predictor, lc.frames_dir) as sa:
            self._seed_all(sa)
            sa.run_bidirectional()
            seg_a = sa.video_segments

        # Diagnostic: the only way a neighbor can change the target is by contesting (overlapping)
        # its pixels. If this is 0, the target is unchanged under EITHER variant by construction,
        # so a 100% IoU A/B is the architecture (separated neighbors), not a coding error.
        overlap = self._target_neighbor_overlap(seg_a, lc.obj_id, neigh_ids)
        print(f"[coprop] target<->neighbor overlap across frames (baseline): {overlap} px "
              f"(0 => neighbors never contest the target => no variant can change it)")

        print(f"[coprop] treatment pass (variant={self.variant})")
        with MultiObjectCopropSession(self.video_predictor, lc.frames_dir,
                                      non_overlap=not mem, non_overlap_mem_enc=mem) as sb:
            self._seed_all(sb)
            sb.run_bidirectional()
            seg_b = sb.video_segments

        alone = label_stack(seg_a, [lc.obj_id], lc.n_frames, lc.hw)
        withn = label_stack(seg_b, [lc.obj_id], lc.n_frames, lc.hw)
        neighbors = label_stack(seg_b, neigh_ids, lc.n_frames, lc.hw)
        diff = build_diff_stack(alone, withn, lc.obj_id)

        s = self._em_world
        lscale = (1.0, s, s)
        self._upsert_labels("target (alone)", alone, lscale)
        self._upsert_labels("target (w/ neighbors)", withn, lscale)
        self._upsert_labels("neighbors (propagated)", neighbors, lscale)
        self._upsert_labels("diff (1=lost, 2=gained)", diff, lscale)

        lost, gained = int((diff == 1).sum()), int((diff == 2).sum())
        print(f"[coprop] diff: lost {lost} px, gained {gained} px "
              f"(variant={self.variant}; output-only should gain 0)")

    @staticmethod
    def _target_neighbor_overlap(segments, obj_id, neigh_ids):
        """Total pixels where the target and any neighbor both fire, summed over frames.
        Computed on the baseline (flags-off) masks, where objects may freely overlap."""
        def _sq(m):
            m = np.asarray(m)
            return m[0] if m.ndim == 3 else m
        total = 0
        for seg in segments.values():
            if obj_id not in seg:
                continue
            tg = _sq(seg[obj_id]).astype(bool)
            for nid in neigh_ids:
                if nid in seg:
                    total += int((tg & _sq(seg[nid]).astype(bool)).sum())
        return total

    def _upsert_labels(self, name, data, scale):
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name, opacity=0.5, scale=scale)


def main():
    import argparse
    from sam2_utils import config
    # default: the sensory-ablated TARGET worm output (the experiment's real subject).
    # pass --root .../batch_masks_multichain to run against the cross-worm GT chains instead.
    default_root = str(config.OUTPUT_ROOT)
    p = argparse.ArgumentParser(description="Standalone co-propagation A/B lab (no saving).")
    p.add_argument("--neuron", required=True)
    p.add_argument("--chain", type=int, required=True)
    p.add_argument("--root", default=default_root,
                   help="output root holding <neuron>/chain_NN (default: the target-worm OUTPUT_ROOT)")
    args = p.parse_args()
    CopropLab(args.root, args.neuron, args.chain).run()


if __name__ == "__main__":
    main()
