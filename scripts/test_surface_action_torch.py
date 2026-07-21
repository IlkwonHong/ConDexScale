#!/usr/bin/env python3
"""Unit-verify SurfaceActionHelper (torch, batched) against a numpy reference.

Same inputs -> same outputs, plus semantic checks:
  A. zero action  -> center = nn(anchor) surface point + radius * normal
  B. random actions/poses, batch vs numpy per-env loop  -> max diff < 1e-4
  C. press semantics: d=+1 -> radius-press_max offset; d=-1 -> radius+hover_max
  D. batch consistency: full batch == stacked single-env calls

Run: /home/robin/miniforge3/envs/dexmachina/bin/python scripts/test_surface_action_torch.py
"""

import os.path as osp
import sys

import numpy as np
import torch

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO, "extlib", "dexmachina"))
from dexmachina.envs.surface_action import (SurfaceActionHelper,  # noqa: E402
                                            tangent_basis_np)
from scipy.spatial.transform import Rotation  # noqa: E402

BOX = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets", "arctic", "box")
RET = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets", "retargeted",
               "point4_hand", "s01", "box_use_01_vector_genesis.npy")


def numpy_reference(helper, actions, part_poses, t_idx):
    """Same pipeline in pure numpy, using the helper's own clouds (so we test
    the torch ops, with tangent bases recomputed independently)."""
    B, S = actions.shape[:2]
    out = np.zeros((B, S, 3))
    for b in range(B):
        for s in range(S):
            pid = helper.part_ids[s]
            pts = helper.clouds[pid]["pts"].cpu().numpy()
            nrm = helper.clouds[pid]["nrm"].cpu().numpy()
            t1_all, t2_all = tangent_basis_np(nrm)   # independent recompute
            a_can = helper.anchors_can[s][min(t_idx[b], helper.num_frames - 1)].cpu().numpy()
            j0 = np.argmin(np.linalg.norm(pts - a_can, axis=1))
            du, dv = np.clip(actions[b, s, 0], -1, 1), np.clip(actions[b, s, 1], -1, 1)
            p_can = a_can + helper.slide_scale * (du * t1_all[j0] + dv * t2_all[j0])
            j1 = np.argmin(np.linalg.norm(pts - p_can, axis=1))
            s_pt, n = pts[j1], nrm[j1]
            d_act = np.clip(actions[b, s, 2], -1, 1)
            sp = helper.press_split
            if d_act >= sp:
                d = helper.press_max * (d_act - sp) / (1.0 - sp)
            else:
                d = helper.hover_max * (d_act - sp) / (sp + 1.0)
            center_can = s_pt + (helper.radius - d) * n
            pos, quat = part_poses[pid]
            R = Rotation.from_quat(np.asarray(quat[b])[[1, 2, 3, 0]]).as_matrix()
            out[b, s] = R @ center_can + np.asarray(pos[b])
    return out


def main():
    dev = "cpu"
    ret = np.load(RET, allow_pickle=True).item()["retarget_data"]
    specs = []
    for side in ["left", "right"]:
        sp = ret[side]["surface_prior"]
        for i in range(sp["anchor_can"].shape[1]):
            specs.append(dict(part_id=int(sp["part_id"][i]),
                              anchor_can=sp["anchor_can"][:, i]))
    helper = SurfaceActionHelper(
        mesh_paths={1: osp.join(BOX, "top_watertight_tiny.obj"),
                    2: osp.join(BOX, "bottom_watertight_tiny.obj")},
        sphere_specs=specs, device=dev)

    rng = np.random.RandomState(0)
    B, S = 8, helper.n_spheres
    # random-ish poses per env (rotation + translation + per-part difference)
    poses = {}
    for pid in [1, 2]:
        q = Rotation.random(B, random_state=rng).as_quat()[:, [3, 0, 1, 2]]  # wxyz
        p = rng.uniform(-0.5, 0.5, (B, 3)) + np.array([0, 0, 1.0])
        poses[pid] = (torch.tensor(p, dtype=torch.float32),
                      torch.tensor(q, dtype=torch.float32))
    t_idx = torch.tensor(rng.randint(0, helper.num_frames, B))

    # --- Test A: zero action -> DEFAULT ENGAGED (press = press_max/3 with
    # split=-0.5): signed offset along normal == radius - press_max/3 ---
    a0 = torch.zeros(B, S, 3)
    outA = helper.compute_targets(a0, poses, t_idx)
    d0 = helper.press_max * (0.0 - helper.press_split) / (1.0 - helper.press_split)
    off0 = ((outA["centers"] - outA["surface_points"]) * outA["normals"]).sum(-1)
    gap = off0 - (helper.radius - d0)
    print(f"A. zero-action default-press offset (기대 {helper.radius-d0:.4f}): "
          f"측정 {off0.mean():.4f}, err max {gap.abs().max():.2e}  "
          f"{'PASS' if gap.abs().max() < 1e-5 else 'FAIL'}")

    # --- Test B: random actions vs numpy reference ---
    acts = torch.tensor(rng.uniform(-1, 1, (B, S, 3)), dtype=torch.float32)
    outB = helper.compute_targets(acts, poses, t_idx)
    ref = numpy_reference(helper, acts.numpy(),
                          {k: (v[0].numpy(), v[1].numpy()) for k, v in poses.items()},
                          t_idx.numpy())
    diff = np.abs(outB["centers"].numpy() - ref).max()
    print(f"B. torch vs numpy reference: max diff {diff:.2e} m  "
          f"{'PASS' if diff < 1e-4 else 'FAIL'}")

    # --- Test C: press semantics ---
    ap = torch.zeros(B, S, 3); ap[..., 2] = 1.0
    am = torch.zeros(B, S, 3); am[..., 2] = -1.0
    op = helper.compute_targets(ap, poses, t_idx)
    om = helper.compute_targets(am, poses, t_idx)
    # SIGNED offset along outward normal: radius - d (can be negative when
    # press_max > radius: PD target sits past the surface -> force ~ kp*d)
    off_p = ((op["centers"] - op["surface_points"]) * op["normals"]).sum(-1)
    off_m = ((om["centers"] - om["surface_points"]) * om["normals"]).sum(-1)
    okC = (torch.allclose(off_p, torch.tensor(helper.radius - helper.press_max), atol=1e-5)
           and torch.allclose(off_m, torch.tensor(helper.radius + helper.hover_max), atol=1e-5))
    print(f"C. press offsets: d=+1 -> {off_p.mean():.4f} (기대 {helper.radius-helper.press_max:.4f}), "
          f"d=-1 -> {off_m.mean():.4f} (기대 {helper.radius+helper.hover_max:.4f})  "
          f"{'PASS' if okC else 'FAIL'}")

    # --- Test D: batch == per-env single calls ---
    singles = []
    for b in range(B):
        pb = {k: (v[0][b:b+1], v[1][b:b+1]) for k, v in poses.items()}
        singles.append(helper.compute_targets(acts[b:b+1], pb, t_idx[b:b+1])["centers"])
    diffD = (torch.cat(singles) - outB["centers"]).abs().max()
    print(f"D. batch vs single-env: max diff {diffD:.2e}  "
          f"{'PASS' if diffD < 1e-5 else 'FAIL'}")


if __name__ == "__main__":
    main()
