#!/usr/bin/env python3
"""Measure per-frame hand-object contact events across ARCTIC sequences.

Purpose: keyframe-density feasibility study for the offline
"keyframe contraction + planning" idea. For every sequence in the p1
split (train + val), computes per-finger min distance to the articulated
object at full 30 fps, then counts contact make/break events per hand
and articulation motion segments.

Contact definition matches the pipeline: MANO contact-zone vertices
(5 fingers x 16 verts, optimisation/robot.py human.contact_zones),
distance to object mesh vertices, nominal threshold 1.5 cm
(2026-07-01 gating decision). Hysteresis on/off + min-segment debounce
to avoid counting flicker as events.

Outputs:
  data/arctic_cache/contact_dist/{subject}__{seq}.npz   (dist (T,10), theta (T,))
  md_claude/keyframes/contact_event_stats.json           (per-seq + aggregate)

Usage:
  ./venv/bin/python scripts/measure_arctic_contact_events.py --device cuda:0
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from manopth.manolayer import ManoLayer

_DEFAULT_ARCTIC_ROOT = osp.join(
    osp.dirname(REPO_ROOT), "ext_proj", "arctic", "unpack", "arctic_data"
)

# optimisation/robot.py human.contact_zones (right-hand topology; left uses
# the same indices — validated 2026-06-19, L≈R fingertip distances)
CONTACT_ZONES = {
    "index":  [350, 355, 329, 332, 349, 354, 343, 327, 351, 353, 348, 328, 347, 326, 337, 346],
    "middle": [462, 439, 467, 442, 461, 466, 455, 437, 459, 438, 465, 460, 436, 463, 449, 434],
    "ring":   [573, 550, 578, 577, 553, 572, 566, 570, 576, 549, 548, 547, 574, 571, 546, 569],
    "little": [690, 689, 667, 695, 670, 694, 664, 683, 688, 665, 693, 691, 666, 687, 661, 663],
    "thumb":  [743, 738, 768, 740, 763, 737, 766, 767, 764, 734, 735, 762, 745, 761, 717, 765],
}
FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]
ZONE_IDX = np.array([CONTACT_ZONES[f] for f in FINGER_NAMES])  # (5, 16)

FPS = 30.0


def load_split_sequences(arctic_root, protocol):
    split_path = osp.join(arctic_root, "data", "splits_json", f"protocol_{protocol}.json")
    with open(split_path) as f:
        splits = json.load(f)
    out = []
    for split in ("train", "val"):
        for seq_path in splits.get(split, []):
            subject, seq_name = seq_path.split("/")
            out.append((split, subject, seq_name))
    return out


def load_object_templates(arctic_root):
    import trimesh

    root = osp.join(arctic_root, "data", "meta", "object_vtemplates")
    templates = {}
    for obj_name in sorted(os.listdir(root)):
        mesh_path = osp.join(root, obj_name, "mesh.obj")
        parts_path = osp.join(root, obj_name, "parts.json")
        if not osp.exists(mesh_path) or not osp.exists(parts_path):
            continue
        mesh = trimesh.load_mesh(mesh_path, process=False)
        with open(parts_path) as f:
            parts = np.array(json.load(f), dtype=np.int32)
        templates[obj_name] = (
            np.array(mesh.vertices, dtype=np.float32) / 1000.0,  # mm -> m
            parts,
        )
    return templates


def object_verts_world(verts_m, parts, obj_data, device):
    """Vectorized port of ArcticFeed._apply_articulation for all frames.

    obj_data: (T, 7) = [arti_angle, rot_aa(3), trans_mm(3)].
    Movable part = parts == 0, rotated around canonical -Z by arti_angle.
    Returns (T, V, 3) float32 tensor on device, metres, world frame.
    """
    T = obj_data.shape[0]
    theta = obj_data[:, 0].astype(np.float64)
    rotvec_arti = np.stack([np.zeros(T), np.zeros(T), -theta], axis=1)
    R_arti = Rotation.from_rotvec(rotvec_arti).as_matrix()          # (T,3,3)
    R_glob = Rotation.from_rotvec(obj_data[:, 1:4].astype(np.float64)).as_matrix()
    trans = obj_data[:, 4:7].astype(np.float64) / 1000.0            # (T,3)

    v = torch.from_numpy(verts_m).to(device, torch.float32)          # (V,3)
    top = torch.from_numpy(parts == 0).to(device)
    Ra = torch.from_numpy(R_arti).to(device, torch.float32)
    Rg = torch.from_numpy(R_glob).to(device, torch.float32)
    tr = torch.from_numpy(trans).to(device, torch.float32)

    out = v.unsqueeze(0).expand(T, -1, -1).clone()                   # (T,V,3)
    out[:, top] = torch.einsum("tij,vj->tvi", Ra, v[top])
    out = torch.einsum("tij,tvj->tvi", Rg, out) + tr[:, None, :]
    return out


def mano_zone_verts(mano_data, side, layer, device, chunk=512):
    """Batched MANO forward for a whole sequence -> zone verts (T, 5, 16, 3)."""
    rot = torch.from_numpy(np.asarray(mano_data[side]["rot"], dtype=np.float32))
    pose = torch.from_numpy(np.asarray(mano_data[side]["pose"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(mano_data[side]["trans"], dtype=np.float32))
    shape = torch.from_numpy(np.asarray(mano_data[side]["shape"], dtype=np.float32)).view(1, 10)
    T = rot.shape[0]
    full_pose = torch.cat([rot.view(T, 3), pose.view(T, 45)], dim=1)

    # NOTE: the manopth fork crashes on batched th_trans (root_trans line
    # assumes B=1). th_trans is purely additive on verts, so pass zeros
    # (center_idx=None -> no recentering) and add trans here instead.
    zones = []
    zone_flat = torch.from_numpy(ZONE_IDX.reshape(-1)).long()
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        verts, _, _, _, _, _ = layer(
            th_pose_coeffs=full_pose[s:e],
            th_betas=shape.expand(e - s, -1),
            th_trans=torch.zeros(e - s, 3),
            root_palm=False,
        )
        zv = verts[:, zone_flat, :] + trans[s:e].unsqueeze(1)
        zones.append(zv.to(device))                                  # (b, 80, 3)
    return torch.cat(zones, dim=0).view(T, 5, 16, 3)


def per_finger_min_dist(zones_r, zones_l, obj_verts, chunk=128):
    """Min distance from each finger zone to object verts. Returns (T,10) np."""
    T = obj_verts.shape[0]
    hands = torch.cat([zones_r, zones_l], dim=1)                     # (T,10,16,3)
    out = torch.empty(T, 10)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        d = torch.cdist(hands[s:e].reshape(e - s, 160, 3), obj_verts[s:e])
        out[s:e] = d.min(dim=2).values.view(e - s, 10, 16).min(dim=2).values
    return out.numpy()


def debounce(state, min_len=3):
    """Remove state segments shorter than min_len frames (merge into prev)."""
    state = state.copy()
    changed = True
    while changed:
        changed = False
        # find segment boundaries
        idx = np.flatnonzero(np.diff(state.astype(np.int8))) + 1
        bounds = np.concatenate([[0], idx, [len(state)]])
        for i in range(1, len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_len:
                state[s:e] = state[bounds[i - 1]]
                changed = True
                break
    return state


def hysteresis_contact(dist, thr_on, thr_off):
    """Per-frame boolean contact with hysteresis (dist 1-D)."""
    state = np.zeros(len(dist), dtype=bool)
    cur = dist[0] < thr_on
    for t in range(len(dist)):
        if cur and dist[t] > thr_off:
            cur = False
        elif not cur and dist[t] < thr_on:
            cur = True
        state[t] = cur
    return state


def count_segments(state):
    """Number of True segments and list of (start, end) frame pairs."""
    idx = np.flatnonzero(np.diff(state.astype(np.int8)))
    bounds = np.concatenate([[0], idx + 1, [len(state)]])
    segs = []
    for i in range(len(bounds) - 1):
        if state[bounds[i]]:
            segs.append((int(bounds[i]), int(bounds[i + 1])))
    return segs


def analyze_sequence(dist, theta, thr_on, thr_off, min_len,
                     arti_omega_thr=0.10, smooth_win=5):
    """dist (T,10) [right 5, left 5], theta (T,) -> event stats dict."""
    T = dist.shape[0]

    # per-finger contact (debounced hysteresis)
    finger_contact = np.stack(
        [debounce(hysteresis_contact(dist[:, i], thr_on, thr_off), min_len)
         for i in range(10)], axis=1)                                # (T,10)
    right = finger_contact[:, :5].any(axis=1)
    left = finger_contact[:, 5:].any(axis=1)
    right = debounce(right, min_len)
    left = debounce(left, min_len)

    # raw (no hysteresis/debounce) hand transitions, for comparison
    raw_r = (dist[:, :5] < (thr_on + thr_off) / 2).any(axis=1)
    raw_l = (dist[:, 5:] < (thr_on + thr_off) / 2).any(axis=1)

    segs_r = count_segments(right)
    segs_l = count_segments(left)

    # hand-set state: 0 none / 1 R / 2 L / 3 both
    hand_set = right.astype(np.int8) + 2 * left.astype(np.int8)
    set_changes = np.flatnonzero(np.diff(hand_set)) + 1              # frame ids
    finger_set_changes = np.flatnonzero(
        (np.diff(finger_contact.astype(np.int8), axis=0) != 0).any(axis=1)) + 1

    # articulation motion segments
    omega = np.gradient(theta) * FPS                                 # rad/s
    if smooth_win > 1:
        k = np.ones(smooth_win) / smooth_win
        omega = np.convolve(omega, k, mode="same")
    moving = debounce(np.abs(omega) > arti_omega_thr, min_len * 2)
    arti_segs = count_segments(moving)

    # keyframe estimate: hand-set change frames + arti segment boundaries,
    # merged if within 15 frames (0.5 s)
    kf = sorted(set(
        list(set_changes)
        + [s for s, _ in arti_segs] + [e - 1 for _, e in arti_segs]))
    merged = []
    for f in kf:
        if not merged or f - merged[-1] > 15:
            merged.append(int(f))

    frac = lambda m: float(m.mean())
    return {
        "n_frames": int(T),
        "duration_s": round(T / FPS, 2),
        "right_touch_events": len(segs_r),
        "left_touch_events": len(segs_l),
        "right_regrasps": max(0, len(segs_r) - 1),
        "left_regrasps": max(0, len(segs_l) - 1),
        "right_transitions_raw": int((np.diff(raw_r.astype(np.int8)) != 0).sum()),
        "left_transitions_raw": int((np.diff(raw_l.astype(np.int8)) != 0).sum()),
        "hand_set_changes": int(len(set_changes)),
        "finger_set_changes": int(len(finger_set_changes)),
        "arti_motion_segments": len(arti_segs),
        "arti_range_rad": round(float(theta.max() - theta.min()), 3),
        "keyframes_est": len(merged),
        "keyframe_frames": merged,
        "frac_none": round(frac(hand_set == 0), 3),
        "frac_right_only": round(frac(hand_set == 1), 3),
        "frac_left_only": round(frac(hand_set == 2), 3),
        "frac_both": round(frac(hand_set == 3), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic-root", default=_DEFAULT_ARCTIC_ROOT)
    ap.add_argument("--protocol", default="p1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--thr-on", type=float, default=0.015)
    ap.add_argument("--thr-off", type=float, default=0.022)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N seqs")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mano_root = osp.join(REPO_ROOT, "network", "manopth", "mano", "models")
    layers = {
        "right": ManoLayer(ncomps=45, side="right", mano_root=mano_root,
                           use_pca=False, flat_hand_mean=False),
        "left": ManoLayer(ncomps=45, side="left", mano_root=mano_root,
                          use_pca=False, flat_hand_mean=False),
    }
    templates = load_object_templates(args.arctic_root)
    sequences = load_split_sequences(args.arctic_root, args.protocol)
    if args.limit:
        sequences = sequences[: args.limit]

    dist_dir = osp.join(REPO_ROOT, "data", "arctic_cache", "contact_dist")
    out_dir = osp.join(REPO_ROOT, "md_claude", "keyframes")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    t0 = time.time()
    for i, (split, subject, seq_name) in enumerate(sequences):
        obj_name = seq_name.split("_")[0]
        if obj_name not in templates:
            continue
        obj_npy = osp.join(args.arctic_root, "data", "raw_seqs", subject,
                           seq_name + ".object.npy")
        mano_npy = osp.join(args.arctic_root, "data", "raw_seqs", subject,
                            seq_name + ".mano.npy")
        if not (osp.exists(obj_npy) and osp.exists(mano_npy)):
            continue

        cache_path = osp.join(dist_dir, f"{subject}__{seq_name}.npz")
        if osp.exists(cache_path):
            npz = np.load(cache_path)
            dist, theta = npz["dist"], npz["theta"]
        else:
            obj_data = np.load(obj_npy)
            mano_data = np.load(mano_npy, allow_pickle=True).item()
            T = min(obj_data.shape[0],
                    np.asarray(mano_data["right"]["rot"]).shape[0])
            obj_data = obj_data[:T]

            verts_m, parts = templates[obj_name]
            with torch.no_grad():
                ov = object_verts_world(verts_m, parts, obj_data, device)
                zr = mano_zone_verts(mano_data, "right", layers["right"], device)[:T]
                zl = mano_zone_verts(mano_data, "left", layers["left"], device)[:T]
                dist = per_finger_min_dist(zr, zl, ov)
            theta = obj_data[:, 0].astype(np.float32)
            np.savez_compressed(cache_path, dist=dist.astype(np.float32),
                                theta=theta)

        stats = analyze_sequence(dist, theta, args.thr_on, args.thr_off,
                                 args.min_len)
        stats.update({"split": split, "subject": subject, "seq": seq_name,
                      "object": obj_name,
                      "verb": seq_name.split("_")[1]})
        results.append(stats)
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(sequences)}] {subject}/{seq_name} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(osp.join(out_dir, "contact_event_stats.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ---- aggregate report ----
    def agg(rows, key):
        vals = np.array([r[key] for r in rows])
        return (f"mean {vals.mean():.1f}  median {np.median(vals):.0f}  "
                f"p10 {np.percentile(vals,10):.0f}  p90 {np.percentile(vals,90):.0f}  "
                f"max {vals.max():.0f}")

    print(f"\n=== {len(results)} sequences, "
          f"{sum(r['n_frames'] for r in results)} frames ===")
    for key in ["right_touch_events", "left_touch_events", "right_regrasps",
                "left_regrasps", "hand_set_changes", "finger_set_changes",
                "arti_motion_segments", "keyframes_est"]:
        print(f"{key:24s} {agg(results, key)}")
    both = np.mean([r["frac_both"] for r in results])
    none = np.mean([r["frac_none"] for r in results])
    print(f"frame fraction: both {both:.2f}  none {none:.2f}  "
          f"R-only {np.mean([r['frac_right_only'] for r in results]):.2f}  "
          f"L-only {np.mean([r['frac_left_only'] for r in results]):.2f}")

    print("\n--- by verb ---")
    for verb in sorted(set(r["verb"] for r in results)):
        rows = [r for r in results if r["verb"] == verb]
        print(f"{verb:6s} n={len(rows):3d}  keyframes {agg(rows, 'keyframes_est')}")

    print("\n--- by object (keyframes_est) ---")
    for obj in sorted(set(r["object"] for r in results)):
        rows = [r for r in results if r["object"] == obj]
        print(f"{obj:16s} n={len(rows):3d}  {agg(rows, 'keyframes_est')}")


if __name__ == "__main__":
    main()
