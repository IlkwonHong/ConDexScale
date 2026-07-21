#!/usr/bin/env python3
"""
Precompute per-sample bimanual INPUT graphs from ARCTIC run_arctic result JSONs.

Design "B" (2026-06-18): the sample graph holds the model INPUT only — it does
NOT store the contracted anchor (the GT pseudo-label).  The dataloader reads
`human_anchor` straight from the result JSON (same as the DexYCB dataset).  This
keeps the FO cache INDEPENDENT of contraction: re-running contraction never
invalidates these graphs, only the result JSON it reads anchors from.

For each unique ARCTIC sample (by file_name):
  1. Load result JSON  → obj_name, obj_center(world), contacts(R+L), joints(R+L), theta
  2. Read raw object npy → global rotation Rg, translation t  (obj_transform in JSON is identity)
  3. Load the (theta-independent) object patch cache for obj_name.
  4. ARTICULATE the patches by this sample's theta (parts==0 rotate around
     canonical -Z), then center by canonical_centroid so patches and labels share
     one frame.  Articulation lives ONLY here (single place — CLAUDE.md §4).
  5. Normalize every label into the object CANONICAL frame (global rotation removed,
     origin at the canonical object centroid):
        contacts (object-centric world)  :  Rg.T @ x
        hand_joints (world)              :  Rg.T @ (x - obj_center)
     Rg itself is stored (`obj_rotation`) so the dataloader can bring the
     result-JSON anchor (object-centric world) into this frame: Rg.T @ anchor.
  6. Build contact_normals + Finger-Object edges against the ARTICULATED patches.
  7. Store per-hand min hand-object distance for downstream contact gating.

Frame derivation (verified 2026-06-14):
  obj_center               = mean of WORLD articulated verts
  hand_joints/_left        = WORLD
  human_finger_contacts/_left = object-centric WORLD (x - obj_center)
  obj_transform            = identity  → Rg taken from raw npy obj_data[t,1:4]
  canonical_centroid       = Rg.T @ (obj_center - t)
  patch (articulated canonical) - canonical_centroid  ==  Rg.T @ (x_world - obj_center)
    → patches and labels end up in the SAME frame (proof: x_world = Rg @ canon_vert + t).

Run after precompute_arctic_object_patches.py and (regenerated) run_arctic.py:
  cd /home/robin/a_workspace/condex && source venv/bin/activate
  python scripts/precompute_arctic_sample_graphs.py
  python scripts/precompute_arctic_sample_graphs.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the DexYCB finger-object edge / contact-normal builder (single source).
from scripts.precompute_sample_graphs import build_finger_object_edges

ARCTIC_ROOT = REPO_ROOT.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data"
RAW_SEQS = ARCTIC_ROOT / "data" / "raw_seqs"
RESULTS_DIR = REPO_ROOT / "data" / "results_arctic"
OBJ_PATCH_DIR = REPO_ROOT / "data" / "arctic_cache" / "object_patches"
GRAPH_DIR = REPO_ROOT / "data" / "arctic_cache" / "sample_graphs"

_FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]


def parse_file_name(file_name: str):
    """'s01_capsulemachine_use_01_00420' -> (subject, seq_name, t)."""
    parts = file_name.split("_")
    return parts[0], "_".join(parts[1:-1]), int(parts[-1])


def read_object_pose(subject: str, seq_name: str, t: int):
    """Return (Rg [3,3], trans_m [3], arti_angle) from the raw object npy."""
    obj_npy = RAW_SEQS / subject / (seq_name + ".object.npy")
    od = np.load(str(obj_npy))[t]
    arti = float(od[0])
    Rg = Rotation.from_rotvec(od[1:4].astype(np.float64)).as_matrix()
    trans_m = od[4:].astype(np.float64) / 1000.0   # mm → m
    return Rg, trans_m, arti


def articulate_patches(patch_pos, patch_nor, part_id, arti_angle):
    """Rotate movable (part_id==0) patches by arti_angle around canonical -Z.

    Port of dataset_opt._apply_articulation, CANONICAL part only: the global
    rotation Rg / translation t are intentionally NOT applied here because the
    labels are already brought to the canonical frame via Rg.T.  Part indexing
    matches the 2026-06-14 fix: movable == parts.json == 0.  (CLAUDE.md §4 —
    this is the single place articulation happens.)
    """
    R_arti = Rotation.from_rotvec([0.0, 0.0, -float(arti_angle)]).as_matrix()
    pos = np.asarray(patch_pos, dtype=np.float64).copy()
    nor = np.asarray(patch_nor, dtype=np.float64).copy()
    mov = np.asarray(part_id) == 0
    pos[mov] = (R_arti @ pos[mov].T).T
    nor[mov] = (R_arti @ nor[mov].T).T
    nor = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-8)
    return pos, nor


def process_sample(result: dict, patch_cache: dict, finger_knn: int) -> dict:
    file_name = result["file_name"]
    obj_name = result["obj_name"]
    subject, seq_name, t = parse_file_name(file_name)

    Rg, trans_m, arti = read_object_pose(subject, seq_name, t)
    RgT = Rg.T
    obj_center = np.array(result["obj_center"], dtype=np.float64)   # world

    canonical_centroid = RgT @ (obj_center - trans_m)              # [3]

    # ── Object patches: articulate by theta, then center to label frame ───────
    rest_pos = np.array(patch_cache["patch_positions"], dtype=np.float64)  # [P,3] canonical rest
    rest_nor = np.array(patch_cache["patch_normals"],   dtype=np.float64)  # [P,3]
    part_id  = np.array(patch_cache["patch_part_id"],   dtype=np.int32)    # [P]
    art_pos, art_nor = articulate_patches(rest_pos, rest_nor, part_id, arti)
    patch_pos = art_pos - canonical_centroid                       # [P,3] label frame
    patch_nor = art_nor                                            # rotation only

    # ── Human contacts (object-centric world) → canonical ─────────────────────
    cr = np.array([result["human_finger_contacts"][n] for n in _FINGER_NAMES])
    cl = np.array([result["human_finger_contacts_left"][n] for n in _FINGER_NAMES])
    contact_r = (RgT @ cr.T).T                                     # [5,3]
    contact_l = (RgT @ cl.T).T                                     # [5,3]
    contact_positions = np.concatenate([contact_r, contact_l], axis=0)  # [10,3]

    # ── Hand joints (WORLD) → canonical ───────────────────────────────────────
    hj_r = (RgT @ (np.array(result["hand_joints"],      dtype=np.float64) - obj_center).T).T
    hj_l = (RgT @ (np.array(result["hand_joints_left"], dtype=np.float64) - obj_center).T).T

    # ── contact_normals + Finger-Object edges (vs ARTICULATED patches) ────────
    fo_src, fo_dst, fo_dist, contact_normals = build_finger_object_edges(
        contact_positions, patch_pos, patch_nor, finger_knn)

    # ── Per-hand min hand-object distance (for contact gating) ────────────────
    nn_dist, _ = cKDTree(patch_pos).query(contact_positions, k=1)  # [10]
    min_dist_right = float(nn_dist[:5].min())
    min_dist_left  = float(nn_dist[5:].min())

    return {
        "file_name": file_name,
        "obj_name": obj_name,
        "arti_angle": arti,                                # theta (radians), record
        "canonical_centroid": canonical_centroid.tolist(), # [3], record
        # Global object rotation Rg (world ← canonical).  Contraction-independent
        # object pose, so storing it does NOT break design B.  The dataloader needs
        # it to bring the result-JSON anchor (object-centric world) into the canonical
        # label frame:  anchor_canonical = Rg.T @ anchor  (verified 2026-06-18).
        "obj_rotation": Rg.astype(np.float64).tolist(),    # [3,3]
        # ── object patch tokens (articulated, label frame) ──
        "patch_positions": patch_pos.astype(np.float32).tolist(),  # [P,3]
        "patch_normals":   patch_nor.astype(np.float32).tolist(),  # [P,3]
        "patch_curvature": patch_cache["patch_curvature"],         # [P]
        "patch_part_id":   part_id.tolist(),                       # [P] 0=movable
        "obj_edge_src":    patch_cache["edge_src"],                # [E] store-only
        "obj_edge_dst":    patch_cache["edge_dst"],                # [E] store-only
        # ── human contact tokens (canonical, 10 fingers: 0-4=R, 5-9=L) ──
        "finger_ids":   [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        "finger_side":  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "finger_names": _FINGER_NAMES + _FINGER_NAMES,
        "contact_positions": contact_positions.astype(np.float32).tolist(),  # [10,3]
        "contact_normals":   contact_normals.astype(np.float32).tolist(),    # [10,3]
        # ── Finger-Object edges (store-only; V2 GNN) ──
        "finger_patch_src":  fo_src.tolist(),
        "finger_patch_dst":  fo_dst.tolist(),
        "finger_patch_dist": fo_dist.tolist(),
        # ── hand joints (store-only; V2) ──
        "hand_joints":      hj_r.astype(np.float32).tolist(),      # [21,3]
        "hand_joints_left": hj_l.astype(np.float32).tolist(),      # [21,3]
        # ── contact gating ──
        "min_dist_right": min_dist_right,
        "min_dist_left":  min_dist_left,
        # NOTE (design B): human_anchor / num_anchors are NOT stored here.
        # The dataloader reads them from the result JSON directly.
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute ARCTIC per-sample bimanual input graphs (design B).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--patch-dir", type=Path, default=OBJ_PATCH_DIR)
    parser.add_argument("--finger-knn", type=int, default=5,
                        help="k nearest patches per finger contact (default: 5)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.graph_dir.mkdir(parents=True, exist_ok=True)
    result_jsons = sorted(args.results_dir.rglob("*.json"))
    print(f"[arctic_sample_graphs] found {len(result_jsons)} result JSONs")

    patch_caches: dict[str, dict] = {}   # obj_name → patch cache

    def get_patch_cache(obj_name: str) -> dict:
        if obj_name not in patch_caches:
            with open(args.patch_dir / f"arctic_{obj_name}_patches.json") as f:
                patch_caches[obj_name] = json.load(f)
        return patch_caches[obj_name]

    processed = set()
    done = skipped = errors = 0
    for json_path in result_jsons:
        try:
            result = json.load(open(json_path))
        except Exception:
            errors += 1; continue
        if "error" in result or result.get("file_name") is None:
            errors += 1; continue
        file_name = result["file_name"]
        if file_name in processed:
            skipped += 1; continue

        out_path = args.graph_dir / f"{file_name}_graph.json"
        if out_path.exists() and not args.overwrite:
            processed.add(file_name); skipped += 1; continue

        try:
            graph = process_sample(result, get_patch_cache(result["obj_name"]), args.finger_knn)
        except Exception as e:
            print(f"[error] {file_name}: {e}")
            errors += 1; continue

        json.dump(graph, open(out_path, "w"))
        processed.add(file_name)
        done += 1
        if done % 100 == 0:
            print(f"  [{done} done / {skipped} skip / {errors} err]", flush=True)

    print(f"\n[done] generated={done} skipped={skipped} errors={errors}")
    print(f"       graph cache → {args.graph_dir}")


if __name__ == "__main__":
    main()
