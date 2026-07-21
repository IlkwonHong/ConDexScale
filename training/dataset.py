"""
GraspContractionDataset

Loads three sources per sample:
  1. result JSON       (from run_partial)            → pseudo-label anchors + metadata
  2. graph JSON        (from precompute_sample_graphs) → contact nodes + edges + obj_rotation
  3. offline_cache JSON (from precompute_object_patches) → canonical patch positions/normals/edges

All tensors are in object canonical frame (YCB object's own coordinate system).
The rotation R stored in graph JSON transforms canonical → camera if needed at inference.

Usage:
    from training.dataset import GraspContractionDataset
    from torch.utils.data import DataLoader

    ds = GraspContractionDataset(
        results_dir="optimisation/results_train",
        graph_dir="graph_cache",
        robot="Robotiq",
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    for batch in loader:
        patch_pos   = batch["patch_positions"]    # [B, P, 3]  canonical
        contact_pos = batch["contact_positions"]  # [B, 5, 3]  canonical
        anchor_gt   = batch["human_anchor"]       # [B, M, 3]  canonical
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT         = Path(__file__).resolve().parents[1]
OFFLINE_CACHE_DIR = REPO_ROOT / "data" / "offline_cache"

_VALID_ROBOTS = {"Shadow", "Allegro", "Barrett", "Robotiq"}


class GraspContractionDataset(Dataset):
    """
    All tensors are in object canonical frame.

    Tensor shapes (P = num_patches, M = num_anchors):
      patch_positions    [P, 3]   object surface patches — canonical frame
      patch_normals      [P, 3]   canonical frame
      patch_curvature    [P]      normal-variance curvature proxy
      obj_edge_src       [E_oo]   Object-Object kNN edge indices
      obj_edge_dst       [E_oo]
      finger_ids         [5]      int64, 0=index 1=middle 2=ring 3=little 4=thumb
      contact_positions  [5, 3]   finger contact positions — canonical frame
      contact_normals    [5, 3]   nearest patch normal at contact — canonical frame
      hand_joints        [21, 3]  MANO joints — canonical frame
      fo_src             [E_fo]   Finger-Object edge: finger index
      fo_dst             [E_fo]   Finger-Object edge: patch index
      fo_dist            [E_fo]   Finger-Object edge: Euclidean distance
      human_anchor       [M, 3]   pseudo-label contracted anchors — canonical frame
      num_anchors        scalar   M
      ycb_id             scalar   int64
    """

    def __init__(
        self,
        results_dir:       str | Path = REPO_ROOT / "data" / "results_train",
        graph_dir:         str | Path = REPO_ROOT / "data" / "graph_cache",
        offline_cache_dir: str | Path = OFFLINE_CACHE_DIR,
        robot:             Optional[str] = None,
        subjects:          Optional[list[str]] = None,
        mu:                Optional[float] = None,
        max_anchors:       int = 3,
        preload:           bool = False,
    ):
        self.graph_dir         = Path(graph_dir)
        self.offline_cache_dir = Path(offline_cache_dir)
        self.max_anchors       = max_anchors
        self._patch_cache: dict[int, dict] = {}   # ycb_id → offline_cache dict
        self._preloaded:   dict[int, tuple] | None = None  # idx → (result, graph)

        if robot is not None and robot not in _VALID_ROBOTS:
            raise ValueError(f"robot must be one of {_VALID_ROBOTS}, got {robot!r}")

        self.samples: list[tuple[Path, Path]] = []
        self._scan(Path(results_dir), robot, subjects, mu)

        if preload:
            self._preload_all()

    # ── scanning ─────────────────────────────────────────────────────────────

    def _scan(self, results_dir: Path, robot: Optional[str],
              subjects: Optional[list[str]], mu: Optional[float]) -> None:
        mu_folder = f"mu{mu:.1f}" if mu is not None else None
        for result_path in sorted(results_dir.glob("**/*.json")):
            parts = result_path.parts
            robot_in_path  = parts[-3] if len(parts) >= 3 else None
            mu_in_path     = parts[-2] if len(parts) >= 2 else None
            subject_in_path = parts[-4] if len(parts) >= 4 else None
            if robot is not None and robot_in_path != robot:
                continue
            if mu_folder is not None and mu_in_path != mu_folder:
                continue
            if subjects is not None and subject_in_path not in subjects:
                continue
            graph_path = self._graph_path_for(result_path)
            if graph_path is None or not graph_path.exists():
                continue
            self.samples.append((result_path, graph_path))

    def _graph_path_for(self, result_path: Path) -> Optional[Path]:
        try:
            with open(result_path) as f:
                header = json.load(f)
            if "error" in header or "file_name" not in header or header["file_name"] is None:
                return None
            return self.graph_dir / f"{header['file_name']}_graph.json"
        except Exception:
            return None

    # ── preload (all result + graph JSONs into RAM) ───────────────────────────

    def _preload_all(self) -> None:
        import sys
        print(f"[dataset] preloading {len(self.samples)} samples into RAM ...", flush=True)
        self._preloaded = {}
        for idx, (result_path, graph_path) in enumerate(self.samples):
            with open(result_path) as f:
                result = json.load(f)
            with open(graph_path) as f:
                graph = json.load(f)
            self._preloaded[idx] = (result, graph)
            if (idx + 1) % 5000 == 0:
                mb = sys.getsizeof(self._preloaded) / 1e6
                print(f"  [{idx+1}/{len(self.samples)}]  ~{mb:.0f} MB", flush=True)
        print(f"[dataset] preload done — {len(self._preloaded)} samples in RAM", flush=True)

    # ── patch cache (per YCB object, shared across samples) ──────────────────

    def _get_patch_cache(self, ycb_id: int) -> dict:
        if ycb_id not in self._patch_cache:
            path = self.offline_cache_dir / f"ycb_{ycb_id}_patches.json"
            with open(path) as f:
                self._patch_cache[ycb_id] = json.load(f)
        return self._patch_cache[ycb_id]

    # ── item loading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._preloaded is not None:
            result, graph = self._preloaded[idx]
        else:
            result_path, graph_path = self.samples[idx]
            with open(result_path) as f:
                result = json.load(f)
            with open(graph_path) as f:
                graph = json.load(f)

        ycb_id = int(graph["ycb_id"])

        # ── Object patches — from offline_cache (canonical frame) ────────────
        patches = self._get_patch_cache(ycb_id)
        patch_positions = torch.tensor(patches["patch_positions"], dtype=torch.float32)  # [P, 3]
        patch_normals   = torch.tensor(patches["patch_normals"],   dtype=torch.float32)  # [P, 3]
        obj_edge_src    = torch.tensor(patches["edge_src"],        dtype=torch.long)
        obj_edge_dst    = torch.tensor(patches["edge_dst"],        dtype=torch.long)
        curv_raw = patches.get("patch_curvature")
        patch_curvature = (torch.tensor(curv_raw, dtype=torch.float32)
                           if curv_raw is not None
                           else torch.zeros(patch_positions.shape[0]))                   # [P]

        # ── Contact nodes — from graph_cache (canonical frame) ───────────────
        finger_ids        = torch.tensor(graph["finger_ids"],        dtype=torch.long)
        contact_positions = torch.tensor(graph["contact_positions"], dtype=torch.float32)
        contact_normals   = torch.tensor(graph["contact_normals"],   dtype=torch.float32)
        hand_joints       = torch.tensor(graph["hand_joints"],       dtype=torch.float32)

        # ── Finger-Object edges ───────────────────────────────────────────────
        fo_src  = torch.tensor(graph["finger_patch_src"],  dtype=torch.long)
        fo_dst  = torch.tensor(graph["finger_patch_dst"],  dtype=torch.long)
        fo_dist = torch.tensor(graph["finger_patch_dist"], dtype=torch.float32)

        # ── GT anchors — result JSON (object-centric, camera axes) → canonical
        # R (canonical→camera) stored in graph; R.T rotates back to canonical.
        R = np.array(graph["obj_rotation"], dtype=np.float32)   # [3, 3]
        raw_anchor = np.array(result["human_anchor"], dtype=np.float32)
        if raw_anchor.ndim == 3:
            raw_anchor = raw_anchor[0]                           # (M, 3)
        anchor_canonical = (R.T @ raw_anchor.T).T               # (M, 3) canonical
        M = anchor_canonical.shape[0]

        anchor = np.zeros((self.max_anchors, 3), dtype=np.float32)
        anchor[:M] = anchor_canonical
        anchor_mask = torch.zeros(self.max_anchors, dtype=torch.bool)
        anchor_mask[:M] = True

        return {
            "patch_positions":   patch_positions,
            "patch_normals":     patch_normals,
            "patch_curvature":   patch_curvature,
            "obj_edge_src":      obj_edge_src,
            "obj_edge_dst":      obj_edge_dst,
            "finger_ids":        finger_ids,
            "contact_positions": contact_positions,
            "contact_normals":   contact_normals,
            "hand_joints":       hand_joints,
            "fo_src":            fo_src,
            "fo_dst":            fo_dst,
            "fo_dist":           fo_dist,
            "human_anchor":      torch.tensor(anchor, dtype=torch.float32),
            "anchor_mask":       anchor_mask,
            "num_anchors":       torch.tensor(M, dtype=torch.long),
            "ycb_id":            torch.tensor(ycb_id, dtype=torch.long),
        }


ARCTIC_RESULTS_DIR = REPO_ROOT / "data" / "results_arctic"
ARCTIC_GRAPH_DIR   = REPO_ROOT / "data" / "arctic_cache" / "sample_graphs"


class ArcticContractionDataset(Dataset):
    """ARCTIC bimanual dataset (design B, 2026-06-18).

    Differences from the single-hand DexYCB ``GraspContractionDataset``:
      * Object patches are ALREADY baked into the per-sample graph (articulated by
        this sample's theta + centered), so there is NO separate offline_cache load.
      * Bimanual: 10 finger contacts (0-4 = right, 5-9 = left); ``hand_joints`` plus
        ``hand_joints_left``.
      * The graph stores model INPUT only — the GT ``human_anchor`` is read straight
        from the result JSON and rotated into the canonical frame with the stored
        ``obj_rotation`` (Rg):  anchor_canonical = Rg.T @ anchor  (verified 2026-06-18).
      * ``num_anchors`` varies (2 or 4) → padded to ``max_anchors`` (default 4) + mask.

    Optional contact gating: pass ``gate_max_dist`` (metres) to drop frames where
    either hand's nearest-patch distance exceeds the threshold (hand-off-object
    frames). Distances are precomputed in the graph (``min_dist_right/left``).

    Tensor shapes (P = num_patches=256, N = 10 contacts, M = max_anchors):
      patch_positions    [P, 3]   articulated + centered, canonical label frame
      patch_normals      [P, 3]
      patch_curvature    [P]
      patch_part_id      [P]       int64, 0 = movable part
      obj_edge_src/dst   [E_oo]    store-only (V2 GNN)
      finger_ids         [N]       int64, 0=index..4=thumb (repeats for both hands)
      finger_side        [N]       int64, 0=right 1=left
      contact_positions  [N, 3]    canonical
      contact_normals    [N, 3]
      hand_joints        [21, 3]   right, canonical (store-only; V2)
      hand_joints_left   [21, 3]   left,  canonical (store-only; V2)
      fo_src/dst/dist    [E_fo]    store-only (V2 GNN)
      human_anchor       [M, 3]    pseudo-label, canonical (Rg.T applied), padded
      anchor_mask        [M]       bool, True for valid anchors
      num_anchors        scalar    M (pre-padding)
      min_dist_right     scalar    nearest-patch dist of right hand (gating)
      min_dist_left      scalar
    """

    def __init__(
        self,
        results_dir:  str | Path = ARCTIC_RESULTS_DIR,
        graph_dir:    str | Path = ARCTIC_GRAPH_DIR,
        robot:        Optional[str] = None,
        subjects:     Optional[list[str]] = None,
        mu:           Optional[float] = None,
        max_anchors:  int = 4,
        gate_max_dist: Optional[float] = None,
        anchor_dir:   Optional[str | Path] = None,
        preload:      bool = False,
    ):
        self.graph_dir     = Path(graph_dir)
        self.max_anchors   = max_anchors
        self.gate_max_dist = gate_max_dist
        # Optional standalone re-contraction cache (recontract_arctic.py). When set,
        # human_anchor is read from here instead of the result JSON — lets us swap the
        # contraction algorithm without touching generation output (design B).
        self.anchor_dir    = Path(anchor_dir) if anchor_dir is not None else None
        self._preloaded: dict[int, tuple] | None = None

        if robot is not None and robot not in _VALID_ROBOTS:
            raise ValueError(f"robot must be one of {_VALID_ROBOTS}, got {robot!r}")

        self.samples: list[tuple[Path, Path]] = []
        self._scan(Path(results_dir), robot, subjects, mu)

        if preload:
            self._preload_all()

    # ── scanning ─────────────────────────────────────────────────────────────

    def _scan(self, results_dir: Path, robot: Optional[str],
              subjects: Optional[list[str]], mu: Optional[float]) -> None:
        # path: results_arctic/<subject>/<seq_name>/<robot>/mu<mu>/<file>.json
        mu_folder = f"mu{mu:.1f}" if mu is not None else None
        for result_path in sorted(results_dir.glob("**/*.json")):
            parts = result_path.parts
            robot_in_path   = parts[-3] if len(parts) >= 3 else None
            mu_in_path      = parts[-2] if len(parts) >= 2 else None
            subject_in_path = parts[-5] if len(parts) >= 5 else None
            if robot is not None and robot_in_path != robot:
                continue
            if mu_folder is not None and mu_in_path != mu_folder:
                continue
            if subjects is not None and subject_in_path not in subjects:
                continue
            graph_path = self._graph_path_for(result_path)
            if graph_path is None or not graph_path.exists():
                continue
            if self.gate_max_dist is not None and not self._passes_gate(graph_path):
                continue
            self.samples.append((result_path, graph_path))

    def _graph_path_for(self, result_path: Path) -> Optional[Path]:
        try:
            with open(result_path) as f:
                header = json.load(f)
            if "error" in header or header.get("file_name") is None:
                return None
            return self.graph_dir / f"{header['file_name']}_graph.json"
        except Exception:
            return None

    def _passes_gate(self, graph_path: Path) -> bool:
        try:
            with open(graph_path) as f:
                g = json.load(f)
            return max(g["min_dist_right"], g["min_dist_left"]) <= self.gate_max_dist
        except Exception:
            return False

    # ── preload ───────────────────────────────────────────────────────────────

    def _preload_all(self) -> None:
        print(f"[arctic-dataset] preloading {len(self.samples)} samples into RAM ...", flush=True)
        self._preloaded = {}
        for idx, (result_path, graph_path) in enumerate(self.samples):
            with open(result_path) as f:
                result = json.load(f)
            with open(graph_path) as f:
                graph = json.load(f)
            self._preloaded[idx] = (result, graph)
        print(f"[arctic-dataset] preload done — {len(self._preloaded)} samples", flush=True)

    # ── item loading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._preloaded is not None:
            result, graph = self._preloaded[idx]
        else:
            result_path, graph_path = self.samples[idx]
            with open(result_path) as f:
                result = json.load(f)
            with open(graph_path) as f:
                graph = json.load(f)

        # ── Object patches — already articulated + centered in the graph ─────
        patch_positions = torch.tensor(graph["patch_positions"], dtype=torch.float32)  # [P,3]
        patch_normals   = torch.tensor(graph["patch_normals"],   dtype=torch.float32)  # [P,3]
        patch_curvature = torch.tensor(graph["patch_curvature"], dtype=torch.float32)  # [P]
        patch_part_id   = torch.tensor(graph["patch_part_id"],   dtype=torch.long)     # [P]
        obj_edge_src    = torch.tensor(graph["obj_edge_src"],    dtype=torch.long)
        obj_edge_dst    = torch.tensor(graph["obj_edge_dst"],    dtype=torch.long)

        # ── Bimanual contact nodes (10 fingers) ──────────────────────────────
        finger_ids        = torch.tensor(graph["finger_ids"],        dtype=torch.long)   # [10]
        finger_side       = torch.tensor(graph["finger_side"],       dtype=torch.long)   # [10]
        contact_positions = torch.tensor(graph["contact_positions"], dtype=torch.float32)# [10,3]
        contact_normals   = torch.tensor(graph["contact_normals"],   dtype=torch.float32)# [10,3]
        hand_joints       = torch.tensor(graph["hand_joints"],       dtype=torch.float32)# [21,3]
        hand_joints_left  = torch.tensor(graph["hand_joints_left"],  dtype=torch.float32)# [21,3]

        fo_src  = torch.tensor(graph["finger_patch_src"],  dtype=torch.long)
        fo_dst  = torch.tensor(graph["finger_patch_dst"],  dtype=torch.long)
        fo_dist = torch.tensor(graph["finger_patch_dist"], dtype=torch.float32)

        # ── GT anchors — result JSON (object-centric world) → canonical ──────
        # Rg (world ← canonical) stored in graph; Rg.T rotates anchor into the
        # canonical label frame (same transform the contacts received).
        Rg = np.array(graph["obj_rotation"], dtype=np.float32)        # [3,3]
        anchor_src = result["human_anchor"]
        if self.anchor_dir is not None:
            apath = self.anchor_dir / f"{result['file_name']}_anchor.json"
            if apath.exists():
                with open(apath) as f:
                    anchor_src = json.load(f)["human_anchor"]
        raw_anchor = np.array(anchor_src, dtype=np.float32)
        if raw_anchor.ndim == 3:
            raw_anchor = raw_anchor[0]                                # (M,3)
        anchor_canonical = (Rg.T @ raw_anchor.T).T                    # (M,3) canonical
        M = anchor_canonical.shape[0]

        anchor = np.zeros((self.max_anchors, 3), dtype=np.float32)
        anchor[:M] = anchor_canonical
        anchor_mask = torch.zeros(self.max_anchors, dtype=torch.bool)
        anchor_mask[:M] = True

        return {
            "patch_positions":   patch_positions,
            "patch_normals":     patch_normals,
            "patch_curvature":   patch_curvature,
            "patch_part_id":     patch_part_id,
            "obj_edge_src":      obj_edge_src,
            "obj_edge_dst":      obj_edge_dst,
            "finger_ids":        finger_ids,
            "finger_side":       finger_side,
            "contact_positions": contact_positions,
            "contact_normals":   contact_normals,
            "hand_joints":       hand_joints,
            "hand_joints_left":  hand_joints_left,
            "fo_src":            fo_src,
            "fo_dst":            fo_dst,
            "fo_dist":           fo_dist,
            "human_anchor":      torch.tensor(anchor, dtype=torch.float32),
            "anchor_mask":       anchor_mask,
            "num_anchors":       torch.tensor(M, dtype=torch.long),
            "min_dist_right":    torch.tensor(graph["min_dist_right"], dtype=torch.float32),
            "min_dist_left":     torch.tensor(graph["min_dist_left"],  dtype=torch.float32),
        }


def build_loader(
    results_dir:       str | Path = REPO_ROOT / "data" / "results_train",
    graph_dir:         str | Path = REPO_ROOT / "data" / "graph_cache",
    offline_cache_dir: str | Path = OFFLINE_CACHE_DIR,
    robot:             Optional[str] = None,
    subjects:          Optional[list[str]] = None,
    mu:                Optional[float] = None,
    batch_size:        int = 16,
    shuffle:           bool = True,
    num_workers:       int = 4,
    max_anchors:       int = 3,
):
    ds = GraspContractionDataset(
        results_dir=results_dir,
        graph_dir=graph_dir,
        offline_cache_dir=offline_cache_dir,
        robot=robot,
        subjects=subjects,
        mu=mu,
        max_anchors=max_anchors,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, ds
