"""SAM3 (HuggingFace transformers) tracker API characterization spike.

Runs tiny real SAM3 calls to record the exact working call sequences, output
shapes, and the reverse-propagation verdict, so the adapter code in
sam2_utils/sam3_backend.py wraps a verified API instead of a guessed one.

Not a test and not library code: a throwaway probe. Run:  py -3 experiments/sam3_probe.py
Every stage is guarded so a failure late in the script still leaves the earlier
findings printed. Introspects real signatures with inspect.signature rather than
trusting the README snippets.
"""
from __future__ import annotations

import gc
import inspect
import traceback

import numpy as np
import torch

CKPT = r"F:\sam3\huggingface"


def _free_gpu():
    """Best-effort GPU memory reclaim between probe stages.

    Each stage below loads its own SAM3 model instance (none are shared, on
    purpose, so each smoke is a clean end-to-end construction); on a 6GB card
    the four sequential loads (image tracker x2, video tracker x2) are tight
    enough that a stale reference from a prior stage can push a later `.to(cuda)`
    over budget. `gc.collect()` drops any Python-side refs to the previous
    stage's model/processor before `empty_cache()` returns their CUDA blocks
    to the allocator.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def banner(msg):
    print("\n" + "=" * 8 + " " + msg + " " + "=" * 8, flush=True)


def sig(obj, name):
    fn = getattr(obj, name, None)
    if fn is None:
        print(f"  {name}: MISSING")
        return
    try:
        print(f"  {name}{inspect.signature(fn)}")
    except (ValueError, TypeError):
        print(f"  {name}: (signature unavailable)")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} cuda_avail={torch.cuda.is_available()} device={dev}")
    import transformers
    print(f"transformers {transformers.__version__}")

    # ---------------- Image tracker (SAM1-style PVS) ----------------
    banner("IMAGE tracker load")
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        imodel = Sam3TrackerModel.from_pretrained(CKPT).to(dev)
        iproc = Sam3TrackerProcessor.from_pretrained(CKPT)
        print("image model + processor loaded")
        print("processor signature:")
        sig(iproc, "__call__")
        sig(iproc, "post_process_masks")
    except Exception:
        traceback.print_exc()
        return

    banner("IMAGE predict (single positive point)")
    try:
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
        inputs = iproc(images=img, input_points=[[[[128, 128]]]],
                       input_labels=[[[1]]], return_tensors="pt").to(dev)
        print("processor output keys:", list(inputs.keys()))
        with torch.no_grad():
            out = imodel(**inputs, multimask_output=True)
        print("model output keys:", list(out.keys()))
        print("pred_masks:", tuple(out.pred_masks.shape),
              "| iou_scores:", tuple(out.iou_scores.shape))
        pp = iproc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
        arr0 = np.asarray(pp[0])
        print("post_process_masks: type", type(pp).__name__, "len", len(pp),
              "elem0 shape", arr0.shape, "dtype", arr0.dtype)
        if dev == "cuda":
            print("image peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    except Exception:
        traceback.print_exc()

    # free the image model before loading the video model (6GB card)
    try:
        del imodel, iproc
    except Exception:
        pass
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---------------- Video tracker (SAM2-style PVS) ----------------
    banner("VIDEO tracker load (bfloat16)")
    try:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        vmodel = Sam3TrackerVideoModel.from_pretrained(CKPT).to(dev, dtype=torch.bfloat16)
        vproc = Sam3TrackerVideoProcessor.from_pretrained(CKPT)
        print("video model + processor loaded")
        print("video processor / session signatures:")
        sig(vproc, "init_video_session")
        sig(vproc, "add_inputs_to_inference_session")
        sig(vproc, "post_process_masks")
        sig(vmodel, "propagate_in_video_iterator")
    except Exception:
        traceback.print_exc()
        return

    banner("VIDEO init session + box prompt on a MID-stack frame (idx 4 of 8)")
    sess = None
    try:
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        frames = [(np.random.rand(128, 128, 3) * 255).astype("uint8") for _ in range(8)]
        sess = vproc.init_video_session(video=frames, inference_device=dev,
                                        dtype=torch.bfloat16)
        print("session type:", type(sess).__name__)
        vproc.add_inputs_to_inference_session(
            inference_session=sess, frame_idx=4, obj_ids=1,
            input_boxes=[[[30.0, 30.0, 90.0, 90.0]]])
        print("added box on frame 4 for obj 1")
    except Exception:
        traceback.print_exc()

    banner("VIDEO forward propagate")
    try:
        n = 0
        for out in vmodel.propagate_in_video_iterator(sess):
            if n == 0:
                print("yield attrs:", [a for a in dir(out) if not a.startswith("_")][:20])
            print("  fwd frame_idx", int(out.frame_idx), "pred_masks", tuple(out.pred_masks.shape))
            n += 1
        print("forward frames:", n)
    except Exception:
        traceback.print_exc()

    banner("VIDEO reverse propagate probe")
    try:
        params = inspect.signature(vmodel.propagate_in_video_iterator).parameters
        print("propagate params:", list(params))
        tried = {}
        for kw in ("reverse", "start_frame_idx", "max_frame_num_to_track"):
            tried[kw] = kw in params
        print("kwargs present:", tried)
        if tried.get("reverse"):
            m = 0
            for out in vmodel.propagate_in_video_iterator(sess, start_frame_idx=4, reverse=True):
                m += 1
            print("REVERSE SUPPORTED via reverse=True kwarg, yielded", m, "frames")
        else:
            print("REVERSE: no 'reverse' kwarg on propagate_in_video_iterator; workaround needed")
    except Exception:
        traceback.print_exc()

    if dev == "cuda":
        print("\nvideo peak VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    banner("PROBE DONE")


def probe_image_predictor_adapter():
    """Smoke-test Sam3ImagePredictor through pipeline.predict.image_predict.

    Uses a synthetic 256x256 RGB image on purpose (no data dependency): this
    checks the adapter's plumbing (set_image -> predict -> select_image_masks
    round-trips into image_predict's expected return shape), not mask quality.
    Real-frame quality is judged later in the bake-off.
    """
    banner("IMAGE adapter smoke (Sam3ImagePredictor via pipeline.predict.image_predict)")
    try:
        import sys
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

        from pipeline import Prompts, image_predict
        from sam2_utils.sam3_backend import Sam3ImagePredictor

        adapter = Sam3ImagePredictor(CKPT)
        image = (np.random.rand(256, 256, 3) * 255).astype("uint8")
        prompts = Prompts(
            points_sam=np.array([[128.0, 128.0]]),
            labels=np.array([1]),
        )
        mask, score, logits = image_predict(
            adapter, image, prompts, multimask=True, select_area_bounds=(1e-5, 0.4))
        assert mask.dtype == bool, f"mask dtype {mask.dtype}, expected bool"
        assert mask.shape == (256, 256), f"mask shape {mask.shape}, expected (256, 256)"
        assert np.isfinite(score), f"score {score} is not finite"
        print(f"OK: mask shape {mask.shape} dtype {mask.dtype}, score {float(score):.4f}")
    except Exception:
        traceback.print_exc()
    banner("IMAGE adapter smoke DONE")


def probe_video_predictor_adapter():
    """Smoke-test Sam3VideoPredictor through pipeline.propagate.propagate.

    Writes ~10 synthetic 128x128 RGB frames to a temp dir (no data dependency: this
    checks the adapter's plumbing, not mask quality, same spirit as the image-adapter
    smoke above), seeds a box at the MIDDLE frame, and asserts propagate() ran BOTH
    directions from that anchor (frames below AND above the anchor appear in
    video_segments). Confirms the anchor-frame default (SAM3 doesn't auto-infer
    start_frame_idx, see docs/explanation/sam3-bakeoff-findings.md) actually lets
    PropagationSession.run_bidirectional's two no-start_frame_idx calls work.
    """
    banner("VIDEO adapter smoke (Sam3VideoPredictor via pipeline.propagate.propagate)")
    try:
        import shutil
        import sys
        import pathlib
        import tempfile

        import cv2

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

        from pipeline import Prompts, propagate
        from sam2_utils.sam3_backend import Sam3VideoPredictor

        n_frames = 10
        anchor = 5
        tmpdir = tempfile.mkdtemp(prefix="sam3_probe_video_")
        try:
            for i in range(n_frames):
                frame = (np.random.rand(128, 128, 3) * 255).astype("uint8")
                cv2.imwrite(str(pathlib.Path(tmpdir) / f"{i:05d}.jpg"), frame)

            adapter = Sam3VideoPredictor(CKPT)
            prompts = Prompts(
                points_sam=np.empty((0, 2)),
                labels=np.empty((0,), dtype=int),
                box_sam=np.array([30.0, 30.0, 90.0, 90.0]),
            )
            video_segments, frame_conf, pred_iou = propagate(
                adapter, tmpdir, prompts, anchor_frame_idx=anchor, obj_id=1,
                seed_points=False, seed_box=True)

            assert isinstance(video_segments, dict) and len(video_segments) > 1, (
                f"expected a multi-frame video_segments dict, got {video_segments!r}")
            below = [fi for fi in video_segments if fi < anchor]
            above = [fi for fi in video_segments if fi > anchor]
            assert below, f"no frames BELOW the anchor in {sorted(video_segments)}"
            assert above, f"no frames ABOVE the anchor in {sorted(video_segments)}"
            for fi, segs in video_segments.items():
                assert isinstance(segs, dict) and 1 in segs, f"frame {fi} missing obj 1: {segs}"
                m = segs[1]
                assert m.dtype == bool and m.shape == (128, 128), (
                    f"frame {fi} mask shape {m.shape} dtype {m.dtype}")
            print(f"OK: {len(video_segments)} frames, "
                  f"below-anchor={sorted(below)} above-anchor={sorted(above)}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        traceback.print_exc()
    banner("VIDEO adapter smoke DONE")


if __name__ == "__main__":
    main()
    _free_gpu()
    probe_image_predictor_adapter()
    _free_gpu()
    probe_video_predictor_adapter()
