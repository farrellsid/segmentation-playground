"""CATMAID API client and annotation fetch helpers.

The `Catmaid` class wraps the minimum REST endpoints needed to pull skeleton
node coordinates from the Zhen Lab project. `fetch_all_annotations` runs the
full pull-and-convert pipeline (skeleton list -> node overview -> nm-to-px
conversion -> single DataFrame).
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd
import requests
from tqdm import tqdm

from . import config, alignment


class Catmaid:
    """Thin REST wrapper for the Zhen Lab CATMAID instance.

    Parameters
    ----------
    url, project_id, api_token
        Defaults are pulled from sam2_utils.config when not provided.
    """

    def __init__(self, url: str | None = None,
                 project_id: int | None = None,
                 api_token: str | None = None):
        self.url = url or config.CATMAID_URL
        self.pid = project_id if project_id is not None else config.CATMAID_PROJECT_ID
        self.api_token = api_token or config.get_catmaid_token()

    # ---- low-level fetch ----

    def fetch(self, endpoint: str, method: str = "get", data: dict | None = None):
        """Perform a GET or POST against the configured CATMAID base URL."""
        headers = {"X-Authorization": f"Token {self.api_token}"}
        data = data or {}
        full_url = self.url + endpoint
        if method == "get":
            return requests.get(full_url, params=data, headers=headers)
        if method == "post":
            return requests.post(full_url, data=data, headers=headers)
        raise ValueError(f"Unsupported method: {method}")

    # ---- endpoint helpers ----

    def get_skeletons(self) -> List[int]:
        """Return all skeleton IDs in the project."""
        return self.fetch(
            f"{self.pid}/skeletons/", "get", {"project_id": self.pid}
        ).json()

    def load_skeleton_names(self, skeletons: List[int]) -> Dict[str, str]:
        """Resolve skeleton IDs -> human-readable neuron names."""
        data = {"neuronnames": "1", "metaannotations": "0"}
        for i, sk in enumerate(skeletons):
            data[f"skeleton_ids['{i}']"] = sk
        return self.fetch(
            f"{self.pid}/skeleton/annotationlist", "post", data
        ).json()["neuronnames"]

    def node_overview(self, skid):
        """Per-skeleton node overview: [[node_id, parent_id, conf, x, y, z, r, ...], ...]"""
        return self.fetch(
            f"{self.pid}/skeletons/{skid}/node-overview",
            "get",
            {"project_id": self.pid, "skeleton_id": skid},
        ).json()

    def stack_info(self, stack_id: int | None = None) -> dict:
        """Get stack dimensions/resolution. Defaults to the first stack in the project."""
        if stack_id is None:
            stacks = self.fetch(f"{self.pid}/stacks", "get").json()
            stack_id = stacks[0]["id"]
        return self.fetch(f"{self.pid}/stack/{stack_id}/info", "get").json()


# =============================================================================
# Full-pull convenience
# =============================================================================

_NODE_COLS = [
    "node_id", "parent_id", "confidence",
    "x", "y", "z",
    "radius", "creator_id", "edition_time",
]


def fetch_all_annotations(catmaid: Catmaid | None = None,
                          to_stack_px: bool = True) -> pd.DataFrame:
    """Pull every skeleton's nodes into one DataFrame.

    Parameters
    ----------
    catmaid : Catmaid, optional
        Reuse an existing client. If None, builds one from config defaults.
    to_stack_px : bool
        If True (default), convert x/y/z from nm to stack-pixel coords using
        STACK_RESOLUTION_NM. Set False to keep raw nm.

    Returns
    -------
    DataFrame with columns: node_id, parent_id, confidence, x, y, z, radius,
    creator_id, edition_time, cell_name
    """
    if catmaid is None:
        catmaid = Catmaid()

    skeletons = catmaid.get_skeletons()
    skeleton_names = catmaid.load_skeleton_names(skeletons)
    print(f"Skeletons: {len(skeleton_names)}")

    frames = []
    for cell_code in tqdm(skeleton_names, desc="Pulling node overviews"):
        cell_data = pd.DataFrame(
            catmaid.node_overview(cell_code)[0],
            columns=_NODE_COLS,
        )
        cell_data["cell_name"] = skeleton_names[cell_code]
        if to_stack_px:
            # nm -> stack-px voxel divide lives in alignment (the one transform home).
            # Coerce to numeric first so bad cells become NaN rather than raising.
            x = pd.to_numeric(cell_data["x"], errors="coerce")
            y = pd.to_numeric(cell_data["y"], errors="coerce")
            z = pd.to_numeric(cell_data["z"], errors="coerce")
            cell_data["x"], cell_data["y"], cell_data["z"] = alignment.nm_to_stack_px(x, y, z)
        frames.append(cell_data)

    aggregate = pd.concat(frames, ignore_index=True)

    for col in ["x", "y", "z", "node_id", "parent_id", "confidence", "radius"]:
        aggregate[col] = pd.to_numeric(aggregate[col], errors="coerce")

    print(f"Total nodes: {len(aggregate)}")
    return aggregate
