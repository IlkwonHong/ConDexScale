#!/usr/bin/env python3
"""
Precompute object patch graphs for ARCTIC articulated objects.

Mirrors precompute_object_patches.py (DexYCB) but:
  - reads ARCTIC canonical meshes (meta/object_vtemplates/{obj}/mesh.obj, mm → m)
  - stores patch_part_id per patch (0 = movable/articulated, 1 = static),
    derived from parts.json via nearest mesh vertex.

The patch cache is theta-INDEPENDENT (canonical rest pose) and shared across all
samples of the same object — exactly like the DexYCB cache. The per-sample
articulation (rotating parts==0 patches by theta) is applied later, in ONE place
only (the dataloader). See CLAUDE.md §4.

  parts.json indexing note: Arctic's official object_tensors builds
  parts_ids = parts.json + 1 and articulates parts_ids == 1, i.e. parts.json == 0.
  So the MOVABLE part is parts.json == 0. (Same convention as
  optimisation/dataset_opt.py:_apply_articulation, fixed 2026-06-14.)

Run once before precompute_arctic_sample_graphs.py:
  cd /home/robin/a_workspace/condex && source venv/bin/activate
  python scripts/precompute_arctic_object_patches.py
  python scripts/precompute_arctic_object_patches.py --num-patches 128 --objects box laptop
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the DexYCB patch utilities — identical FPS / kNN / curvature logic.
from scripts.precompute_object_patches import fps, build_knn_edges, compute_curvature

ARCTIC_ROOT = REPO_ROOT.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data"
VTEMPLATE_DIR = ARCTIC_ROOT / "data" / "meta" / "object_vtemplates"
CACHE_DIR = REPO_ROOT / "data" / "arctic_cache" / "object_patches"

ARCTIC_OBJECTS = [
    "box", "capsulemachine", "espressomachine", "ketchup", "laptop",
    "microwave", "mixer", "notebook", "phone", "scissors", "waffleiron",
]


def process_object(obj_name: str, num_patches: int, knn_k: int,
                   surface_samples: int) -> dict:
    mesh_path = VTEMPLATE_DIR / obj_name / "mesh.obj"
    parts_path = VTEMPLATE_DIR / obj_name / "parts.json"

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    verts_m = np.array(mesh.vertices, dtype=np.float64) / 1000.0   # mm → m, canonical
    parts = np.array(json.load(open(parts_path)), dtype=np.int32)  # [V] per-vertex 0/1

    # Dense surface sampling on the canonical (rest-pose) mesh, in metres.
    mesh_m = mesh.copy()
    mesh_m.vertices = verts_m
    points, face_ids = trimesh.sample.sample_surface(mesh_m, surface_samples)
    normals = mesh_m.face_normals[face_ids]

    # FPS to select patch centers.
    patch_ids = fps(points, num_patches)
    patch_positions = np.asarray(points[patch_ids], dtype=np.float32)   # [P, 3]
    patch_normals = np.asarray(normals[patch_ids], dtype=np.float32)    # [P, 3]
    patch_normals = patch_normals / np.maximum(
        np.linalg.norm(patch_normals, axis=1, keepdims=True), 1e-8)

    # part_id per patch = parts.json label of the nearest mesh vertex.
    vtree = cKDTree(verts_m)
    _, nearest_v = vtree.query(patch_positions)
    patch_part_id = parts[nearest_v].astype(np.int32)                   # [P]

    # kNN Object-Object edges + curvature (canonical; connectivity/proxy only).
    edge_src, edge_dst = build_knn_edges(patch_positions, knn_k)
    patch_curvature = compute_curvature(patch_normals, edge_src, edge_dst)

    n_mov = int((patch_part_id == 0).sum())
    return {
        "obj_name": obj_name,
        "num_patches": int(len(patch_positions)),
        "knn_k": knn_k,
        "n_movable_patches": n_mov,
        "patch_positions": patch_positions.tolist(),   # [P, 3] canonical (m)
        "patch_normals": patch_normals.tolist(),       # [P, 3] canonical
        "patch_part_id": patch_part_id.tolist(),       # [P]    0=movable, 1=static
        "patch_curvature": patch_curvature.tolist(),   # [P]
        "edge_src": edge_src.tolist(),                 # [E]
        "edge_dst": edge_dst.tolist(),                 # [E]
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute ARCTIC object patch graphs.")
    parser.add_argument("--num-patches", type=int, default=256,
                        help="FPS patch centers per object (default: 256)")
    parser.add_argument("--knn-k", type=int, default=6,
                        help="kNN neighbours for Object-Object edges (default: 6)")
    parser.add_argument("--surface-samples", type=int, default=8192,
                        help="Dense surface samples before FPS (default: 8192)")
    parser.add_argument("--objects", nargs="+", default=None,
                        help="Process only these objects (default: all 11)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    objs = args.objects if args.objects else ARCTIC_OBJECTS

    for obj_name in objs:
        out_path = CACHE_DIR / f"arctic_{obj_name}_patches.json"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {obj_name} already cached")
            continue
        mesh_path = VTEMPLATE_DIR / obj_name / "mesh.obj"
        if not mesh_path.exists():
            print(f"[warn] mesh not found: {mesh_path}")
            continue
        print(f"[process] {obj_name} ...", end=" ", flush=True)
        data = process_object(obj_name, args.num_patches, args.knn_k,
                              args.surface_samples)
        with open(out_path, "w") as f:
            json.dump(data, f)
        print(f"patches={data['num_patches']} movable={data['n_movable_patches']} "
              f"edges={len(data['edge_src'])} → {out_path.name}")

    print(f"\n[done] cache → {CACHE_DIR}")


if __name__ == "__main__":
    main()
