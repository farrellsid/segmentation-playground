"""
metrics.py — region-overlap + split/merge metrics for eval/ Stage 0.

The buildable-now half of the Stage-0 ruler (FUTURE_DIRECTIONS §4.1): per-object
overlap (IoU / Dice / precision / recall) and **Variation of Information split into
VOI_split + VOI_merge**, computed straight off the GT labelmaps — no skeletons
required. ERL (the skeleton-based half) is deferred until SEM-Dauer 1's skeletons
are wired in (see eval/README.md); when it lands it joins these on (neuron, slice).

Pure numpy, torch-free, no IO — every function takes arrays and returns plain
Python floats/dicts, so it unit-tests on synthetic masks with known answers.

Conventions
-----------
- Binary metrics treat inputs as boolean. The empty/empty case (both masks empty)
  is defined as perfect agreement (iou = dice = precision = recall = 1.0); a
  denominator of zero in any other case yields ``nan`` (documented per metric).
- VOI follows the connectomics split (Nunez-Iglesias / gala):
    VOI_split = H(SEG | GT)  — over-segmentation: one GT object cut across many
                               predicted labels.
    VOI_merge = H(GT | SEG)  — under-segmentation: one predicted label spanning
                               many GT objects (the costly error; weight it higher).
    VOI       = VOI_split + VOI_merge.
  Lower is better; 0 = identical partitions. Reported in bits (``base=2``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# =============================================================================
# Binary region overlap
# =============================================================================

def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Overlap metrics for one predicted vs one GT boolean mask.

    Returns a dict with ``iou``, ``dice``, ``precision``, ``recall``, the raw
    ``tp`` / ``fp`` / ``fn`` counts, and ``pred_area`` / ``gt_area``.

    Empty/empty -> all ratios 1.0. Otherwise a zero denominator -> ``nan`` for
    that ratio (e.g. recall is nan when GT is empty but pred is not).
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    union = tp + fp + fn

    if union == 0:                          # both empty: perfect (trivial) agreement
        iou = dice = precision = recall = 1.0
    else:
        iou = tp / union
        dice = (2 * tp) / (2 * tp + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")

    return {
        "iou": float(iou), "dice": float(dice),
        "precision": float(precision), "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn,
        "pred_area": tp + fp, "gt_area": tp + fn,
    }


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks (empty/empty -> 1.0)."""
    return binary_metrics(pred, gt)["iou"]


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice (F1) of two boolean masks (empty/empty -> 1.0)."""
    return binary_metrics(pred, gt)["dice"]


def metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Micro-aggregate overlap ratios from summed tp/fp/fn (volume rollup).

    Lets score.py sum per-frame counts across a chain/neuron, then derive a single
    volume IoU/Dice without re-touching pixels. Same empty/empty convention.
    """
    tp, fp, fn = int(tp), int(fp), int(fn)
    union = tp + fp + fn
    if union == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0,
                "tp": 0, "fp": 0, "fn": 0}
    return {
        "iou": tp / union,
        "dice": (2 * tp) / (2 * tp + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }


# =============================================================================
# Variation of Information (split / merge)
# =============================================================================

def variation_of_information(
    seg: np.ndarray,
    gt: np.ndarray,
    *,
    ignore_gt: Sequence[int] = (),
    base: float = 2.0,
) -> Dict[str, float]:
    """Split/merge-decomposed VOI between two integer labelings.

    Parameters
    ----------
    seg, gt : integer label arrays of identical shape (any dimensionality;
        flattened internally). ``seg`` is the proposed/predicted segmentation,
        ``gt`` the ground truth.
    ignore_gt : labels in ``gt`` to exclude (drop those pixels entirely). Pass
        ``(0,)`` to score only the GT foreground, the usual connectomics choice
        when background dominates the field.
    base : log base for the entropies (2 -> bits).

    Returns
    -------
    dict with ``voi_split`` (= H(seg|gt)), ``voi_merge`` (= H(gt|seg)), ``voi``
    (their sum), and ``n`` (pixels scored). All-one-label / empty inputs -> zeros.
    """
    seg = np.asarray(seg).ravel()
    gt = np.asarray(gt).ravel()
    if seg.shape != gt.shape:
        raise ValueError(f"shape mismatch: seg {seg.size} vs gt {gt.size}")

    if ignore_gt:
        keep = ~np.isin(gt, np.asarray(list(ignore_gt)))
        seg, gt = seg[keep], gt[keep]

    n = seg.size
    if n == 0:
        return {"voi_split": 0.0, "voi_merge": 0.0, "voi": 0.0, "n": 0}

    # Dense-index both labelings, then a single-pass joint histogram.
    _, si = np.unique(seg, return_inverse=True)
    _, gi = np.unique(gt, return_inverse=True)
    cont = np.zeros((si.max() + 1, gi.max() + 1), dtype=np.float64)
    np.add.at(cont, (si, gi), 1.0)
    p = cont / n                       # joint p(seg=i, gt=j)
    p_i = p.sum(axis=1)                # marginal over seg
    q_j = p.sum(axis=0)               # marginal over gt

    ii, jj = np.nonzero(p)
    pij = p[ii, jj]
    log = np.log(pij) / np.log(base)
    # H(seg|gt) = -Σ p_ij log(p_ij / q_j) ;  H(gt|seg) = -Σ p_ij log(p_ij / p_i)
    voi_split = float(-(pij * (log - np.log(q_j[jj]) / np.log(base))).sum())
    voi_merge = float(-(pij * (log - np.log(p_i[ii]) / np.log(base))).sum())
    # clamp tiny negative fp noise
    voi_split = max(0.0, voi_split)
    voi_merge = max(0.0, voi_merge)
    return {"voi_split": voi_split, "voi_merge": voi_merge,
            "voi": voi_split + voi_merge, "n": int(n)}


# =============================================================================
# Adapted Rand (SNEMI3D / Arganda-Carreras et al. 2015) — split / merge
# =============================================================================

def adapted_rand(
    seg: np.ndarray,
    gt: np.ndarray,
    *,
    ignore_gt: Sequence[int] = (),
) -> Dict[str, float]:
    """Adapted-Rand F-score + error between two integer labelings (FUTURE_DIRECTIONS §4.1).

    The SNEMI3D connectomics metric, complementary to VOI: it scores agreement over
    *pairs of pixels grouped into the same segment* via the contingency matrix.
    Convention matches ``skimage.metrics.adapted_rand_error`` (Arganda-Carreras 2015):
    the Rand index counts agreeing *pairs of distinct pixels*, so we use pair counts
    ``Σ c(c-1)`` (NOT ``Σ c²`` — that would include self-pairs and disagree with the
    SNEMI3D reference). With contingency counts ``c[i,j]`` (seg label i, gt label j):
    ``sum_p2 = Σ c_ij(c_ij-1)``, ``sum_seg2 = Σ a_i(a_i-1)`` over seg sizes ``a_i``,
    ``sum_gt2 = Σ b_j(b_j-1)`` over gt sizes ``b_j``,

        precision = sum_p2 / sum_seg2     (seg/pred marginals -> MERGE sensitivity)
        recall    = sum_p2 / sum_gt2      (gt marginals       -> SPLIT sensitivity)
        f_score   = 2·precision·recall / (precision + recall) = 2·sum_p2/(sum_seg2+sum_gt2)
        are       = 1 - f_score           (0 = perfect)

    Low precision => the prediction MERGES distinct GT objects (under-segmentation);
    low recall => it SPLITS one GT object (over-segmentation). So ``merge_error =
    1 - precision`` and ``split_error = 1 - recall`` give the same split/merge split
    VOI does, on the Rand scale.

    Parameters mirror :func:`variation_of_information`: ``seg`` = prediction, ``gt`` =
    ground truth; ``ignore_gt`` drops those GT labels' pixels (pass ``(0,)`` to score
    only GT foreground, the usual connectomics choice). All-empty -> a perfect score.
    """
    seg = np.asarray(seg).ravel()
    gt = np.asarray(gt).ravel()
    if seg.shape != gt.shape:
        raise ValueError(f"shape mismatch: seg {seg.size} vs gt {gt.size}")

    if ignore_gt:
        keep = ~np.isin(gt, np.asarray(list(ignore_gt)))
        seg, gt = seg[keep], gt[keep]

    n = seg.size
    if n == 0:
        return {"are": 0.0, "precision": 1.0, "recall": 1.0, "f_score": 1.0,
                "merge_error": 0.0, "split_error": 0.0, "n": 0}

    _, si = np.unique(seg, return_inverse=True)
    _, gi = np.unique(gt, return_inverse=True)
    cont = np.zeros((si.max() + 1, gi.max() + 1), dtype=np.float64)
    np.add.at(cont, (si, gi), 1.0)

    seg_marg = cont.sum(axis=1)
    gt_marg = cont.sum(axis=0)
    sum_p2 = float((cont * (cont - 1.0)).sum())          # Σ c(c-1)  (agreeing pairs)
    sum_seg2 = float((seg_marg * (seg_marg - 1.0)).sum())   # pred-pair count
    sum_gt2 = float((gt_marg * (gt_marg - 1.0)).sum())      # gt-pair count

    precision = sum_p2 / sum_seg2 if sum_seg2 else 1.0
    recall = sum_p2 / sum_gt2 if sum_gt2 else 1.0
    denom = precision + recall
    f_score = (2.0 * precision * recall / denom) if denom else 1.0
    return {
        "are": 1.0 - f_score,
        "precision": precision, "recall": recall, "f_score": f_score,
        "merge_error": 1.0 - precision, "split_error": 1.0 - recall,
        "n": int(n),
    }


def voi_arand(
    seg: np.ndarray,
    gt: np.ndarray,
    *,
    ignore_labels: Sequence[int] = (),
    prefer_skimage: bool = True,
) -> Dict[str, float]:
    """Unified VOI + ARAND, defaulting to **scikit-image's reference implementations** —
    the connectomics-standard methodology used by the CAD/FGNet papers
    (`skimage.metrics.variation_of_information` + `adapted_rand_error`, with
    `voi = voi_split + voi_merge`). Falls back to the pure-numpy `variation_of_information`
    + `adapted_rand` in this module when skimage is unavailable (so it still runs on a
    torch/skimage-free box; the two agree on ARE and on VOI orientation, see tests).

    Args mirror the other metrics here: ``seg`` = prediction, ``gt`` = ground truth.
    skimage is called as the papers do — ``f(gt, seg)`` (im_true=gt, im_test=seg) — so
    ``voi_split = H(seg|gt)`` (over-seg), ``voi_merge = H(gt|seg)`` (under-seg). Pass
    ``ignore_labels=(0,)`` to drop GT background, or pre-restrict the inputs to the scored
    region (equivalent — that's what `score_labelmap` does, hence the ``()`` default).
    """
    seg = np.asarray(seg)
    gt = np.asarray(gt)
    if prefer_skimage:
        try:
            from skimage.metrics import variation_of_information as _sk_voi
            from skimage.metrics import adapted_rand_error as _sk_are
        except Exception:
            prefer_skimage = False

    if prefer_skimage:
        vs = _sk_voi(gt, seg, ignore_labels=tuple(ignore_labels))   # [H(seg|gt), H(gt|seg)]
        voi_split, voi_merge = float(vs[0]), float(vs[1])
        are, prec, rec = _sk_are(gt, seg, ignore_labels=tuple(ignore_labels))
        return {"voi_split": voi_split, "voi_merge": voi_merge,
                "voi": voi_split + voi_merge,
                "are": float(are), "arand_precision": float(prec),
                "arand_recall": float(rec), "backend": "skimage"}

    ig = tuple(ignore_labels)
    v = variation_of_information(seg, gt, ignore_gt=ig)
    a = adapted_rand(seg, gt, ignore_gt=ig)
    return {"voi_split": v["voi_split"], "voi_merge": v["voi_merge"], "voi": v["voi"],
            "are": a["are"], "arand_precision": a["precision"],
            "arand_recall": a["recall"], "backend": "numpy"}


def weighted_voi(voi: Dict[str, float], merge_split_ratio: float = 5.0) -> float:
    """Single cost number weighting merges over splits (FUTURE_DIRECTIONS §4.1).

    Mergers are far costlier to fix by hand than splits, so the roadmap proposes a
    merge:split cost ratio of ~5:1 or higher. ``cost = ratio*VOI_merge + VOI_split``.
    """
    return float(merge_split_ratio) * voi["voi_merge"] + voi["voi_split"]
