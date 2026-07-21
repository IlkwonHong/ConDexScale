#!/usr/bin/env python3
"""Verify exported contraction anchors lie on the object surface (CLAUDE.md §4).

Reprojects world-frame anchors into the object's canonical frame using the
same pose trajectory the training environment consumes (obj_pos/obj_quat/
obj_arti from the processed npy), un-articulates top-part anchors (box.urdf:
revolute joint, origin 0, axis (0,0,-1)), then measures distance to the part
meshes ({top,bottom}_watertight_tiny.obj).

Pass criterion: median surface distance < 1 cm on contact frames.

Usage:
  python scripts/verify_dexmachina_guidance.py --obj box --hand point4_hand
"""

import argparse
import os.path as osp

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEXM_ASSETS = osp.join(REPO_ROOT, "extlib", "dexmachina", "dexmachina", "assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="box")
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--clip", default="01")
    ap.add_argument("--hand", default="point4_hand")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    stem = f"{args.obj}_use_{args.clip}"
    proc = np.load(osp.join(DEXM_ASSETS, "arctic", "processed", args.subject,
                            stem + ".npy"), allow_pickle=True).item()
    params = proc["params"]
    obj_pos = np.asarray(params["obj_trans"], dtype=np.float64)
    obj_quat = np.asarray(params["obj_quat"], dtype=np.float64)   # wxyz
    obj_arti = np.asarray(params["obj_arti"], dtype=np.float64)

    cr = np.load(osp.join(DEXM_ASSETS, "contact_retarget", args.hand,
                          args.subject, stem + ".npy"), allow_pickle=True).item()

    meshes = {
        1: trimesh.load_mesh(osp.join(DEXM_ASSETS, "arctic", args.obj,
                                      "top_watertight_tiny.obj"), process=False),
        2: trimesh.load_mesh(osp.join(DEXM_ASSETS, "arctic", args.obj,
                                      "bottom_watertight_tiny.obj"), process=False),
    }

    R_obj = Rotation.from_quat(obj_quat[:, [1, 2, 3, 0]])  # scipy wants xyzw

    all_d = []
    for side in ["left", "right"]:
        c = cr[side]["dexlink_contacts"]      # (T,2,k,4)
        v = cr[side]["dexlink_valid_contacts"]  # (T,2,k)
        T, _, k, _ = c.shape
        dists = []
        for t in range(0, T, args.stride):
            for j, pid in [(0, 1), (1, 2)]:
                for i in range(k):
                    if not v[t, j, i]:
                        continue
                    p_world = c[t, j, i, :3].astype(np.float64)
                    p_obj = R_obj[t].inv().apply(p_world - obj_pos[t])
                    if pid == 1:  # top part: undo articulation about (0,0,-1)
                        p_obj = Rotation.from_rotvec(
                            [0, 0, obj_arti[t]]).apply(p_obj)
                    _, d, _ = trimesh.proximity.closest_point(
                        meshes[pid], p_obj[None])
                    dists.append(d[0])
        dists = np.array(dists)
        all_d.append(dists)
        print(f"{side:5s}: n={len(dists)}  median {np.median(dists)*1000:.1f}mm  "
              f"p90 {np.percentile(dists,90)*1000:.1f}mm  "
              f"max {dists.max()*1000:.1f}mm")

    med = np.median(np.concatenate(all_d))
    print(f"\nOVERALL median {med*1000:.1f}mm  ->  "
          f"{'PASS (<10mm)' if med < 0.010 else 'FAIL'}")


if __name__ == "__main__":
    main()
