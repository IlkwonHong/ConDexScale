#!/usr/bin/env python3
"""Fast in-place re-contraction of human_anchor with the hybrid genhand+kmeans
fix, WITHOUT re-running Stage 1/2 force-closure.

The fix only changes the LABEL (human_anchor): the robot-grasp seed (right-hand
genhand-primary) and human_finger_contacts (input) are unchanged, and
sample_graphs don't store human_anchor. So we just recompute human_anchor per
existing result JSON and patch it in place. ~1.5 s/sample vs ~10 s for the full
pipeline.

Replicates run_idx_json_bimanual's contraction block exactly (initialization →
per-hand _contract_single_hand → concat / fingertip fallback).

Usage:
  cd /home/robin/a_workspace/condex && ./venv/bin/python scripts/recontract_arctic_genhand.py
"""
import os, sys, json, glob, time, collections
import numpy as np, torch

REPO = "/home/robin/a_workspace/condex"
sys.path.insert(0, REPO)
from optimisation.Optimize import Optimization
from optimisation.dataset_opt import ArcticFeed
from optimisation.robot import Robotiq, human as HumanModel

ARCTIC_ROOT = os.path.join(REPO, "..", "ext_proj", "arctic", "unpack", "arctic_data")
RESULTS = os.path.join(REPO, "data", "results_arctic")
DEVICE = "cuda:0"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--contact-threshold", type=float, default=0.015)
    args = ap.parse_args()

    ds = ArcticFeed(arctic_root=ARCTIC_ROOT, split=args.split, protocol="p1", frame_stride=30)
    fi = {f"{s}_{sq}_{tt:05d}": i for i, (s, sq, tt) in enumerate(ds.frame_index)}
    robot = Robotiq(batch=1, device=DEVICE)
    opt = Optimization(robot=robot, dataset=ds, device=DEVICE, maximum_iter=[2000, 1, 1000],
                       mu=0.9, task="GH", source="ycb", contact_target_method="hdbscan",
                       contact_threshold=args.contact_threshold, sdf_backend="mesh")
    aph = opt.anchors_per_hand
    print(f"[recontract] split={args.split} ct={args.contact_threshold} dir={args.results_dir}", flush=True)

    files = sorted(glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True))
    print(f"[recontract] {len(files)} result JSONs  anchors_per_hand={aph}", flush=True)
    before = collections.Counter(); after = collections.Counter()
    done = skipped = errors = 0
    t0 = time.time()
    for n, jp in enumerate(files):
        try:
            d = json.load(open(jp))
        except Exception:
            errors += 1; continue
        if "error" in d or d.get("file_name") is None:
            skipped += 1; continue
        idx = fi.get(d["file_name"])
        if idx is None:
            skipped += 1; continue
        before[len(d.get("human_anchor", []))] += 1
        try:
            ip, op, mp = ds[idx]
            opt._prepare_sdf_context(ip)
            oc = op["centre"].to(DEVICE).unsqueeze(0)
            opt.initialization(
                op["verts"].to(DEVICE).unsqueeze(0), op["faces"].to(DEVICE).unsqueeze(0), oc,
                mp["verts"].to(DEVICE).unsqueeze(0), mp["faces"].to(DEVICE).unsqueeze(0),
                mp["joint"].to(DEVICE).unsqueeze(0), mp["transformation"].to(DEVICE).unsqueeze(0))
            right_pack = opt.obj_model.hand_model.get_contact_points_normals_packed()
            lh = HumanModel(device=DEVICE)
            lh.load_hand_mesh(mesh_v=mp["verts_left"].to(DEVICE).unsqueeze(0) - oc,
                              mesh_f=mp["faces"].to(DEVICE).unsqueeze(0),
                              joints=mp["joint_left"].to(DEVICE).unsqueeze(0) - oc)
            raw = lh.get_contact_points_normals_packed()
            left_pack = torch.cat([raw[:, :3], -raw[:, 3:]], dim=1)
            ar = opt._contract_single_hand(right_pack, aph)
            al = opt._contract_single_hand(left_pack, aph)
            parts = [a for a in (ar, al) if a is not None]
            ha = torch.cat(parts, 0) if parts else opt._fallback_anchor_from_fingertips()
            d["human_anchor"] = ha.cpu().tolist()
            after[len(d["human_anchor"])] += 1
            json.dump(d, open(jp, "w"))
            done += 1
        except Exception as e:
            print(f"[error] {d.get('file_name')}: {e}", flush=True); errors += 1
        if (n + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{n+1}/{len(files)}] done={done} skip={skipped} err={errors} "
                  f"{el/ (n+1):.2f}s/it ETA {el/(n+1)*(len(files)-n-1)/60:.0f}min", flush=True)
    print(f"\n[done] patched={done} skipped={skipped} errors={errors} in {(time.time()-t0)/60:.1f}min")
    print(f"  K before: {dict(sorted(before.items()))}")
    print(f"  K after : {dict(sorted(after.items()))}")


if __name__ == "__main__":
    main()
