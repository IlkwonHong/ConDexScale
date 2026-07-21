#!/usr/bin/env python3
"""
Precompute per-sample hand-object graphs from run_partial result JSONs.

For each unique DexYCB sample (identified by file_name):
  1. Load result JSON  → ycb_id, obj_transform, human_finger_contacts, hand_joints
  2. Load hand_trans   → from dexycb_train_s0.json annotation
  3. Load object patch cache → canonical patch positions + normals + edges
  4. Transform patches to hand-local frame using obj_transform and hand_trans
  5. Build Finger-Object edges (each finger contact → k nearest patches)
  6. Save graph_cache/{file_name}_graph.json

The dataloader then reads:
  - results_train/.../idx_robot_mu.json  (contracted grasp + pseudo-label)
  - graph_cache/{file_name}_graph.json   (hand-object graph)

Run after precompute_object_patches.py and run_partial.py:
  cd /workspace/genhand && source venv/bin/activate
  python scripts/precompute_sample_graphs.py
  python scripts/precompute_sample_graphs.py --finger-knn 5 --results-dir optimisation/results_train
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "offline_cache"
GRAPH_DIR = REPO_ROOT / "data" / "graph_cache"
RESULTS_DIR = REPO_ROOT / "data" / "results_train"
TRAIN_META = REPO_ROOT / "dataset" / "data" / "dexycb_train_s0.json"

_FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]
_FINGER_ID = {name: i for i, name in enumerate(_FINGER_NAMES)}


def load_hand_trans_lookup(meta_path: Path) -> dict:
    """Build {file_name: hand_trans [3]} from dexycb_train_s0.json."""
    with open(meta_path) as f:
        meta = json.load(f)
    lookup = {}
    for img, ann in zip(meta["images"], meta["annotations"]):
        lookup[img["file_name"]] = np.array(ann["hand_trans"], dtype=np.float32)
    return lookup


def load_patch_cache(ycb_id: int) -> dict | None:
    path = CACHE_DIR / f"ycb_{ycb_id}_patches.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)



def build_finger_object_edges(contact_positions: np.ndarray, patch_positions: np.ndarray,
                               patch_normals: np.ndarray, k: int):
    """
    For each of N finger contacts, find k nearest patches.
    Returns:
      finger_patch_src: [E_fo]  finger index (0..N-1)
      finger_patch_dst: [E_fo]  patch index  (0..P-1)
      finger_patch_dist: [E_fo] Euclidean distance
      contact_normals: [N, 3]   normal of the nearest patch for each contact
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(patch_positions)
    dists, indices = tree.query(contact_positions, k=k)  # [N, k]

    N = len(contact_positions)
    src_list, dst_list, dist_list = [], [], []
    for i in range(N):
        for j_idx in range(k):
            src_list.append(i)
            dst_list.append(int(indices[i, j_idx]))
            dist_list.append(float(dists[i, j_idx]))

    # Nearest patch normal as contact normal
    nearest_patch_ids = indices[:, 0]  # [N]
    contact_normals = patch_normals[nearest_patch_ids]  # [N, 3]

    return (np.array(src_list, dtype=np.int32),
            np.array(dst_list, dtype=np.int32),
            np.array(dist_list, dtype=np.float32),
            contact_normals)


def process_sample(result_json: dict, hand_trans: np.ndarray,
                   patch_cache: dict, finger_knn: int) -> dict:
    file_name = result_json["file_name"]
    ycb_id = result_json["ycb_id"]

    # R: canonical → camera  (DexYCB GT obj_transform)
    T = np.array(result_json["obj_transform"], dtype=np.float32)  # [4, 4]
    R = T[:3, :3]                                                  # [3, 3]
    obj_centre = np.array(result_json["obj_center"], dtype=np.float32)  # [3] hand-local

    # Canonical patches — used as-is for building finger-object edges.
    # Patch data is NOT stored here; dataset.py reads offline_cache directly.
    canonical_pos = np.array(patch_cache["patch_positions"], dtype=np.float32)  # [P, 3]
    canonical_nor = np.array(patch_cache["patch_normals"],   dtype=np.float32)  # [P, 3]

    # Contact positions: Optimize.py saves in object-centric / camera-axes frame.
    # Apply R.T to rotate into object canonical frame.
    fc = result_json["human_finger_contacts"]
    contact_cam = np.array([fc[name] for name in _FINGER_NAMES], dtype=np.float32)  # [5, 3]
    contact_positions = (R.T @ contact_cam.T).T   # [5, 3] canonical frame

    finger_ids = list(range(len(_FINGER_NAMES)))

    # Hand joints: hand-local → subtract obj_centre → object-centric (camera axes)
    #              → apply R.T → canonical frame
    joints_handlocal  = np.array(result_json["hand_joints"], dtype=np.float32)  # [21, 3]
    joints_objcentric = joints_handlocal - obj_centre
    hand_joints       = (R.T @ joints_objcentric.T).T   # [21, 3] canonical frame

    # Finger-Object edges: computed in canonical frame (distances are rotation-invariant)
    fo_src, fo_dst, fo_dist, contact_normals = build_finger_object_edges(
        contact_positions, canonical_pos, canonical_nor, finger_knn
    )

    return {
        "file_name":         file_name,
        "ycb_id":            ycb_id,
        "obj_rotation":      R.tolist(),                     # [3, 3] canonical→camera
        # Contact nodes — canonical frame
        "finger_ids":        finger_ids,
        "finger_names":      _FINGER_NAMES,
        "contact_positions": contact_positions.tolist(),     # [5, 3]
        "contact_normals":   contact_normals.tolist(),       # [5, 3]
        "hand_joints":       hand_joints.tolist(),           # [21, 3]
        # Finger-Object edges
        "finger_patch_src":  fo_src.tolist(),
        "finger_patch_dst":  fo_dst.tolist(),
        "finger_patch_dist": fo_dist.tolist(),
        # Patch data (positions/normals/edges/curvature) intentionally omitted —
        # dataset.py loads offline_cache/ycb_{id}_patches.json directly.
    }


def collect_result_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="Precompute per-sample hand-object graphs.")
    parser.add_argument("--finger-knn", type=int, default=3,
                        help="k nearest patches per finger contact (default: 3)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Root dir of run_partial results (default: optimisation/results_train)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing graph cache files")
    args = parser.parse_args()

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("[precompute_sample_graphs] loading hand_trans lookup ...", flush=True)
    hand_trans_lookup = load_hand_trans_lookup(TRAIN_META)
    print(f"  {len(hand_trans_lookup)} samples in train meta")

    result_jsons = collect_result_jsons(args.results_dir)
    print(f"[precompute_sample_graphs] found {len(result_jsons)} result JSONs")

    patch_caches = {}   # ycb_id → cache dict
    processed_files = set()
    skipped = done = errors = 0

    for json_path in result_jsons:
        with open(json_path) as f:
            try:
                result = json.load(f)
            except Exception:
                errors += 1
                continue

        if "error" in result:
            errors += 1
            continue

        file_name = result.get("file_name")
        if file_name is None:
            errors += 1
            continue

        # One graph per DexYCB sample (robot-independent)
        if file_name in processed_files:
            skipped += 1
            continue

        out_path = GRAPH_DIR / f"{file_name}_graph.json"
        if out_path.exists() and not args.overwrite:
            processed_files.add(file_name)
            skipped += 1
            continue

        hand_trans = hand_trans_lookup.get(file_name)
        if hand_trans is None:
            print(f"[warn] hand_trans not found for {file_name}")
            errors += 1
            continue

        ycb_id = result.get("ycb_id")
        if ycb_id not in patch_caches:
            patch_caches[ycb_id] = load_patch_cache(ycb_id)
        cache = patch_caches[ycb_id]
        if cache is None:
            print(f"[warn] patch cache missing for ycb_id={ycb_id} — run precompute_object_patches.py first")
            errors += 1
            continue

        try:
            graph = process_sample(result, hand_trans, cache, args.finger_knn)
        except Exception as e:
            print(f"[error] {file_name}: {e}")
            errors += 1
            continue

        with open(out_path, "w") as f:
            json.dump(graph, f)

        processed_files.add(file_name)
        done += 1
        if done % 100 == 0:
            print(f"  [{done} done / {skipped} skip / {errors} err]", flush=True)

    print(f"\n[done] generated={done}  skipped={skipped}  errors={errors}")
    print(f"       graph cache → {GRAPH_DIR}")


if __name__ == "__main__":
    main()
