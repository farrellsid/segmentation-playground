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

import inspect
import traceback

import numpy as np
import torch

CKPT = r"F:\sam3\huggingface"


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


if __name__ == "__main__":
    main()
