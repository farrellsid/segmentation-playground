"""eval/ — evaluation harness against the cross-worm VAST ground truth.

See README.md. Region-overlap + VOI
metrics (`metrics`) and skeleton-based ERL (`erl`) are implemented, reading the
VAST GT (`groundtruth`) and scoring a prediction source (`score`). ERL's metric
core is pure; only sampling real per-node labels needs the prediction pixels.
"""

from __future__ import annotations

from . import erl, groundtruth, metrics, score
from .erl import (Skeletons, expected_run_length, load_skeletons, merge_labels,
                  per_neuron_erl, sample_node_labels, summarize)
from .groundtruth import GroundTruth, parse_metadata, parse_name
from .metrics import (binary_metrics, variation_of_information, weighted_voi,
                      adapted_rand, voi_arand)
from .score import DirPredictionSource, score_region

__all__ = [
    "erl", "groundtruth", "metrics", "score",
    "Skeletons", "expected_run_length", "load_skeletons", "merge_labels",
    "per_neuron_erl", "sample_node_labels", "summarize",
    "GroundTruth", "parse_metadata", "parse_name",
    "binary_metrics", "variation_of_information", "weighted_voi", "adapted_rand", "voi_arand",
    "DirPredictionSource", "score_region",
]
