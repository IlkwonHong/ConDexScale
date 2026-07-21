#!/usr/bin/env python3
"""
Precompute object patch graphs for all YCB objects used in DexYCB.

For each YCB object:
  1. Sample dense points on canonical mesh surface
  2. FPS to select P patch centers
  3. Extract surface normals at patch centers
  4. Build kNN Object-Object edge graph
  5. Save to offline_cache/ycb_{id}_patches.json

Run once before training:
  cd /workspace/genhand && source venv/bin/activate
  python scripts/precompute_object_patches.py
  python scripts/precompute_object_patches.py --num-patches 128 --knn-k 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("DEX_YCB_DIR", str(REPO_ROOT.parent / "dataset"))) / "models"
CACHE_DIR = REPO_ROOT / "data" / "offline_cache"

_YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick',
}

_FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]


def fps(points: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling. Returns selected indices."""
    rng = np.random.default_rng(seed)
    N = len(points)
    n_samples = min(n_samples, N)
    selected = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf)
    selected[0] = rng.integers(0, N)
    for i in range(1, n_samples):
        last_pt = points[selected[i - 1]]
        dist = np.sum((points - last_pt) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)
    return selected


def build_knn_edges(positions: np.ndarray, k: int):
    """Build symmetric kNN edges. Returns (src, dst) arrays."""
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    # k+1 because query includes self
    dists, indices = tree.query(positions, k=k + 1)
    N = len(positions)
    src_list, dst_list = [], []
    for i in range(N):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            src_list.append(i)
            dst_list.append(int(j))
    return np.array(src_list, dtype=np.int32), np.array(dst_list, dtype=np.int32)


def compute_curvature(patch_normals: np.ndarray, edge_src: np.ndarray,
                      edge_dst: np.ndarray) -> np.ndarray:
    """Normal-variance curvature proxy per patch. flat≈0, edge/corner→large."""
    dots = np.sum(patch_normals[edge_src] * patch_normals[edge_dst], axis=1)  # [E]
    angular_diff = 1.0 - np.clip(dots, -1.0, 1.0)  # [E]

    N = len(patch_normals)
    curvature = np.zeros(N, dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)
    np.add.at(curvature, edge_src, angular_diff)
    np.add.at(counts, edge_src, 1.0)
    return curvature / np.maximum(counts, 1.0)


def process_object(ycb_id: int, class_name: str, num_patches: int, knn_k: int,
                   surface_samples: int) -> dict:
    mesh_path = MODEL_DIR / class_name / "textured_simple.obj"
    mesh = trimesh.load_mesh(str(mesh_path), process=False)

    # Dense surface sampling
    points, face_ids = trimesh.sample.sample_surface(mesh, surface_samples)
    normals = mesh.face_normals[face_ids]  # normal at each sampled point

    # FPS to select patch centers
    patch_ids = fps(points, num_patches)
    patch_positions = points[patch_ids]   # [P, 3]
    patch_normals = normals[patch_ids]    # [P, 3]

    # Normalize normals
    norms = np.linalg.norm(patch_normals, axis=1, keepdims=True)
    patch_normals = patch_normals / np.maximum(norms, 1e-8)

    # kNN Object-Object edges
    edge_src, edge_dst = build_knn_edges(patch_positions, knn_k)

    # Normal-variance curvature proxy (rotation-invariant scalar)
    patch_curvature = compute_curvature(patch_normals, edge_src, edge_dst)

    return {
        "ycb_id": ycb_id,
        "class_name": class_name,
        "num_patches": num_patches,
        "knn_k": knn_k,
        "patch_positions": patch_positions.tolist(),   # [P, 3] canonical frame
        "patch_normals": patch_normals.tolist(),       # [P, 3] canonical frame
        "patch_curvature": patch_curvature.tolist(),   # [P]    rotation-invariant
        "edge_src": edge_src.tolist(),                 # [E]
        "edge_dst": edge_dst.tolist(),                 # [E]
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute YCB object patch graphs.")
    parser.add_argument("--num-patches", type=int, default=256,
                        help="Number of FPS patch centers per object (default: 256)")
    parser.add_argument("--knn-k", type=int, default=6,
                        help="kNN neighbours for Object-Object edges (default: 6)")
    parser.add_argument("--surface-samples", type=int, default=8192,
                        help="Dense surface samples before FPS (default: 8192)")
    parser.add_argument("--ycb-ids", type=int, nargs="+", default=None,
                        help="Process only these ycb_ids (default: all 21)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ids_to_process = args.ycb_ids if args.ycb_ids else list(_YCB_CLASSES.keys())

    for ycb_id in ids_to_process:
        class_name = _YCB_CLASSES[ycb_id]
        out_path = CACHE_DIR / f"ycb_{ycb_id}_patches.json"

        if out_path.exists():
            print(f"[skip] ycb_id={ycb_id} ({class_name}) already cached")
            continue

        mesh_path = MODEL_DIR / class_name / "textured_simple.obj"
        if not mesh_path.exists():
            print(f"[warn] mesh not found: {mesh_path}")
            continue

        print(f"[process] ycb_id={ycb_id} ({class_name}) ...", end=" ", flush=True)
        data = process_object(ycb_id, class_name, args.num_patches, args.knn_k,
                              args.surface_samples)
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"patches={args.num_patches} edges={len(data['edge_src'])} → {out_path.name}")

    print(f"\n[done] cache → {CACHE_DIR}")


if __name__ == "__main__":
    main()
