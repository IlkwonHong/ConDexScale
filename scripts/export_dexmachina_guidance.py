#!/usr/bin/env python3
"""Export contraction-based guidance data for DexMachina point-contact robots.

Replaces DexMachina's kinematic-retargeting preprocessing (impossible for
non-anthropomorphic multi-arm embodiments) with contraction anchors: per
frame, each hand's ARCTIC contact points are contracted (k-means) into k
anchors; each anchor becomes one floating sphere's reference trajectory.

Inputs (their shipped processed ARCTIC data — frames/coordinates guaranteed
consistent with the training environment):
  assets/arctic/processed/{subj}/{obj}_use_{clip}.npy
    params: obj_trans/obj_quat/obj_rot (T,*), obj_arti (T,)
    world_coord: contacts.{side} (T,50,4) = xyz world metres + part_id,
                 valid_contacts.{side} (T,50)
    NOTE part_id convention: 1='top' (movable), 2='bottom' (map_contacts.py:250)

Outputs (formats replicated from shipped inspire_hand assets):
  A) assets/retargeted/{hand}/{subj}/{obj}_use_{clip}_vector_genesis.npy
     {'demo_data': {obj_pos, obj_rot, obj_quat, obj_arti, contact_links_*},
      'retarget_data': {side: {joint_qpos: {name: (T,)}, kpt_pos (T,k,3),
                               kpt_names, wrist_pose (T,7), wrist_link_name}}}
  B) assets/contact_retarget/{hand}/{subj}/{obj}_use_{clip}.npy
     {side: {dexlink_contacts (T,2,k,4), dexlink_valid_contacts (T,2,k),
             collision_link_names, collision_link_local_idxs}}
     part index j corresponds to part_id j+1 (map_contacts.py:272)

Also saves an analysis log (anchor azimuth around object centre, per frame)
for the deferred arm-placement discussion:
  data/arctic_cache/dexmachina_guidance_logs/{hand}_{subj}_{obj}_use_{clip}.npz

Usage:
  python scripts/export_dexmachina_guidance.py --obj box --spheres-per-side 2
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEXM_ASSETS = osp.join(REPO_ROOT, "extlib", "dexmachina", "dexmachina", "assets")
sys.path.insert(0, osp.join(REPO_ROOT, "scripts"))
from test_surface_projection import part_cloud  # noqa: E402  (outward-normal-fixed sampler)


def kmeans_anchors(pts, k, init=None, iters=10):
    """Plain Lloyd k-means on (n,3) points. Warm-startable. Returns (k,3)."""
    n = len(pts)
    if n == 1:
        return np.repeat(pts, k, axis=0)
    if init is None or len(init) != k:
        # farthest-point init
        centers = [pts[0]]
        for _ in range(k - 1):
            d = np.min(np.linalg.norm(pts[:, None] - np.array(centers)[None], axis=-1), axis=1)
            centers.append(pts[np.argmax(d)])
        centers = np.array(centers)
    else:
        centers = init.copy()
    for _ in range(iters):
        assign = np.argmin(np.linalg.norm(pts[:, None] - centers[None], axis=-1), axis=1)
        for j in range(k):
            if (assign == j).any():
                centers[j] = pts[assign == j].mean(axis=0)
            else:
                # empty cluster (e.g. warm-start center left behind after the
                # hand's contacts migrated to another part): re-seed at the
                # point farthest from the other centers, otherwise the center
                # freezes at its stale warm-start position forever
                d = np.min(np.linalg.norm(pts[:, None] - centers[None], axis=-1), axis=1)
                centers[j] = pts[np.argmax(d)]
    return centers


def match_to_prev(centers, prev):
    """Permute centers (k<=2 here) to minimise distance to prev anchors."""
    k = len(centers)
    if prev is None or k == 1:
        return centers
    if k == 2:
        d_keep = np.linalg.norm(centers - prev, axis=1).sum()
        d_swap = np.linalg.norm(centers[::-1] - prev, axis=1).sum()
        return centers if d_keep <= d_swap else centers[::-1].copy()
    # general (unused for k<=2): greedy
    out = np.zeros_like(centers)
    used = set()
    for i in range(k):
        d = np.linalg.norm(centers - prev[i], axis=1)
        for j in np.argsort(d):
            if j not in used:
                out[i] = centers[j]
                used.add(j)
                break
    return out


def interp_gaps(arr, defined):
    """Linear-interpolate (T,...) over frames where defined==False; hold ends."""
    T = arr.shape[0]
    idx = np.flatnonzero(defined)
    if len(idx) == 0:
        raise ValueError("no defined frames at all")
    out = arr.copy()
    flat = out.reshape(T, -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(np.arange(T), idx, flat[idx, c])
    return flat.reshape(arr.shape)


def moving_average(arr, win=5):
    if win <= 1:
        return arr
    k = np.ones(win) / win
    flat = arr.reshape(arr.shape[0], -1).copy()
    for c in range(flat.shape[1]):
        flat[:, c] = np.convolve(flat[:, c], k, mode="same")
    # fix conv edge shrinkage by holding original ends
    half = win // 2
    flat[:half] = arr.reshape(arr.shape[0], -1)[:half]
    flat[-half:] = arr.reshape(arr.shape[0], -1)[-half:]
    return flat.reshape(arr.shape)


def process_side(wc, params, side, k, press_depth, hover_dist, smooth_win):
    """Returns dict with anchors (T,k,3) raw, refs (T,k,3) pressed/hovered,
    valid (T,k) bool, part (T,k) int in {0,1,2} (0=unassigned)."""
    contacts = np.asarray(wc[f"contacts.{side}"], dtype=np.float64)      # (T,50,4)
    valids = np.asarray(wc[f"valid_contacts.{side}"], dtype=bool)        # (T,50)
    obj_pos = np.asarray(params["obj_trans"], dtype=np.float64)          # (T,3)
    T = contacts.shape[0]

    anchors = np.zeros((T, k, 3))
    valid = np.zeros((T, k), dtype=bool)
    part = np.zeros((T, k), dtype=np.int64)
    defined = np.zeros(T, dtype=bool)
    prev = None
    for t in range(T):
        pts = contacts[t, valids[t], :3]
        pids = contacts[t, valids[t], 3].astype(int)
        if len(pts) == 0:
            continue
        centers = kmeans_anchors(pts, k, init=prev)
        centers = match_to_prev(centers, prev)
        # tiny separation if all centers collapsed (single contact point)
        if k > 1 and np.linalg.norm(centers[0] - centers[1]) < 1e-6:
            centers[1] += np.array([0.0, 0.0, 0.03])
        assign = np.argmin(
            np.linalg.norm(pts[:, None] - centers[None], axis=-1), axis=1)
        for j in range(k):
            sel = assign == j
            part[t, j] = np.bincount(pids[sel]).argmax() if sel.any() else 0
            # per-anchor validity: only clusters that actually received
            # contact points count as "in contact" this frame
            valid[t, j] = bool(sel.any())
        anchors[t] = centers
        defined[t] = True
        prev = centers

    anchors = interp_gaps(anchors, defined)
    # press inward during valid contact, hover outward otherwise
    outward = anchors - obj_pos[:, None, :]
    outward /= (np.linalg.norm(outward, axis=-1, keepdims=True) + 1e-9)
    offset = np.where(valid[..., None], -press_depth, hover_dist)
    refs = anchors + offset * outward
    refs = moving_average(refs, smooth_win)
    return dict(anchors=anchors, refs=refs, valid=valid, part=part,
                contact_any=valids.any(axis=1))


def part_pose_demo(params, t):
    """Demo-frame world pose (R, p) of each part. box.urdf: top link = root pose
    composed with revolute about axis (0,0,-1) at origin -> R_top = R_root @ Rz(-arti)."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(params["obj_quat"][t], dtype=np.float64)
    R_root = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    p = np.asarray(params["obj_trans"][t], dtype=np.float64)
    arti = float(params["obj_arti"][t])
    Rz = Rotation.from_rotvec([0, 0, -arti]).as_matrix()
    return {2: (R_root, p),                 # part_id 2 = bottom (root link)
            1: (R_root @ Rz, p)}            # part_id 1 = top (lid)


def build_surface_prior(res, params, clouds, T, k):
    """Per anchor: assigned part (majority over valid frames) + part-local
    on-surface anchor trajectory (world anchor -> assigned-part frame ->
    projected to that part's canonical surface cloud). Runtime transforms it
    by the CURRENT sim part pose, so the prior rides the actual object."""
    part_id = np.zeros(k, dtype=np.int64)
    anchor_can = np.zeros((T, k, 3), dtype=np.float32)
    for i in range(k):
        vals = res["part"][res["valid"][:, i], i]
        vals = vals[vals > 0]
        part_id[i] = np.bincount(vals).argmax() if len(vals) else 2  # default bottom
        pts_can, _ = clouds[int(part_id[i])]
        for t in range(T):
            R, p = part_pose_demo(params, t)[int(part_id[i])]
            local = R.T @ (res["anchors"][t, i] - p)
            j = np.argmin(np.linalg.norm(pts_can - local, axis=1))
            anchor_can[t, i] = pts_can[j]
        # light temporal smoothing of the canonical trajectory
        anchor_can[:, i] = moving_average(anchor_can[:, i], win=5)
    return part_id, anchor_can


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="box")
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--clip", default="01")
    ap.add_argument("--spheres-per-side", type=int, default=2)
    ap.add_argument("--press-depth", type=float, default=0.005)
    ap.add_argument("--hover-dist", type=float, default=0.02)
    ap.add_argument("--smooth-win", type=int, default=5)
    args = ap.parse_args()

    k = args.spheres_per_side
    hand = f"point{2*k}_hand"
    stem = f"{args.obj}_use_{args.clip}"
    proc = np.load(
        osp.join(DEXM_ASSETS, "arctic", "processed", args.subject, stem + ".npy"),
        allow_pickle=True).item()
    params, wc = proc["params"], proc["world_coord"]
    T = np.asarray(params["obj_trans"]).shape[0]
    print(f"[{stem}] T={T}  obj_trans range "
          f"{np.abs(np.asarray(params['obj_trans'])).max():.3f} m")

    # canonical surface clouds (outward normals) per part, for surface_prior
    import trimesh
    clouds = {}
    for pid, name in [(1, "top"), (2, "bottom")]:
        mesh = trimesh.load_mesh(
            osp.join(DEXM_ASSETS, "arctic", args.obj, f"{name}_watertight_tiny.obj"),
            process=False)
        clouds[pid] = part_cloud(mesh, n=1500)

    retarget_data, contact_retarget, logs = dict(), dict(), dict()
    for side in ["left", "right"]:
        prefix = "L" if side == "left" else "R"
        res = process_side(wc, params, side, k, args.press_depth,
                           args.hover_dist, args.smooth_win)
        refs = res["refs"]                                            # (T,k,3)
        joint_qpos = {}
        for i in range(k):
            p = f"{prefix}{i+1}_forearm"
            for ax, key in enumerate(["tx", "ty", "tz"]):
                joint_qpos[f"{p}_{key}_link_joint"] = refs[:, i, ax].astype(np.float32)
            for key in ["roll", "pitch", "yaw"]:
                joint_qpos[f"{p}_{key}_link_joint"] = np.zeros(T, dtype=np.float32)
        tip_names = [f"{prefix}{i+1}_tip_link" for i in range(k)]
        wrist_pose = np.concatenate(
            [refs[:, 0, :], np.tile([1.0, 0, 0, 0], (T, 1))], axis=1)  # pos + wxyz identity
        part_id, anchor_can = build_surface_prior(res, params, clouds, T, k)
        retarget_data[side] = dict(
            joint_qpos=joint_qpos,
            kpt_pos=refs.astype(np.float32),
            kpt_names=tip_names,
            wrist_pose=wrist_pose.astype(np.float32),
            wrist_link_name=tip_names[0],
            # surface action mode prior: assigned part per sphere (1=top, 2=bottom)
            # + part-local on-surface anchor trajectory (rides the sim object)
            surface_prior=dict(part_id=part_id, anchor_can=anchor_can),
        )
        # (B): part index j <-> part_id j+1 (map_contacts.py convention)
        dex_c = np.zeros((T, 2, k, 4), dtype=np.float32)
        dex_v = np.zeros((T, 2, k), dtype=np.float32)
        for j in range(2):
            pid = j + 1
            hit = res["valid"] & (res["part"] == pid)                  # (T,k)
            dex_c[:, j, :, :3] = np.where(hit[..., None], res["anchors"], 0.0)
            dex_c[:, j, :, 3] = np.where(hit, pid, 0.0)
            dex_v[:, j] = hit.astype(np.float32)
        contact_retarget[side] = dict(
            dexlink_contacts=dex_c,
            dexlink_valid_contacts=dex_v,
            collision_link_names=tip_names,
            collision_link_local_idxs=list(range(7, 7 + k * 7, 7)),   # tip after 6 dummies each
        )
        # placement-analysis log: azimuth of each anchor around object centre
        rel = res["anchors"] - np.asarray(params["obj_trans"])[:, None, :]
        logs[f"azimuth_{side}"] = np.arctan2(rel[..., 1], rel[..., 0]).astype(np.float32)
        logs[f"valid_{side}"] = res["valid"]
        logs[f"anchors_{side}"] = res["anchors"].astype(np.float32)

    demo_data = dict(
        obj_pos=np.asarray(params["obj_trans"], dtype=np.float32),
        obj_rot=np.asarray(params["obj_rot"], dtype=np.float32),
        obj_quat=np.asarray(params["obj_quat"], dtype=np.float32),
        obj_arti=np.asarray(params["obj_arti"], dtype=np.float32),
        contact_links_left=np.asarray(wc["contact_links_left"], dtype=np.float32),
        contact_links_right=np.asarray(wc["contact_links_right"], dtype=np.float32),
    )

    out_a = osp.join(DEXM_ASSETS, "retargeted", hand, args.subject)
    out_b = osp.join(DEXM_ASSETS, "contact_retarget", hand, args.subject)
    log_dir = osp.join(REPO_ROOT, "data", "arctic_cache", "dexmachina_guidance_logs")
    for d in [out_a, out_b, log_dir]:
        os.makedirs(d, exist_ok=True)

    np.save(osp.join(out_a, f"{stem}_vector_genesis.npy"),
            dict(demo_data=demo_data, retarget_data=retarget_data))
    np.save(osp.join(out_b, f"{stem}.npy"), contact_retarget)
    np.savez_compressed(osp.join(log_dir, f"{hand}_{args.subject}_{stem}.npz"), **logs)
    print(f"wrote {out_a}/{stem}_vector_genesis.npy")
    print(f"wrote {out_b}/{stem}.npy")

    # quick stats
    for side in ["left", "right"]:
        v = logs[f"valid_{side}"]
        print(f"  {side}: contact frames {v.any(axis=1).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
