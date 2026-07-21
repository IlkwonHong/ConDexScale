#!/usr/bin/env python3
"""Inference + GT-vs-prediction visualization on the ARCTIC validation set.

Loads a trained GraspContractionNet checkpoint, reproduces the EXACT train/val
split used by training/train.py (random.seed=42 → shuffle → first val_split), runs
inference on N validation samples, Hungarian-matches the predicted anchors to the
valid GT anchors, and renders a grid of 3D panels (canonical frame):

  - object patches      : grey dots (context)
  - finger contacts     : cyan (R) / orange (L) dots (model input)
  - GT anchors          : green diamonds
  - PRED anchors        : red X (Hungarian-matched to GT)
  - GT ↔ PRED           : yellow dotted error lines

Also reports the matched anchor Euclidean error (cm) over ALL val samples.

Usage:
  cd /home/robin/a_workspace/condex && ./venv/bin/python scripts/visualize_eval_predictions.py \
      --ckpt checkpoints/best.pt --n 10 --gate-max-dist 0.03
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from training.dataset import ArcticContractionDataset, ARCTIC_RESULTS_DIR, ARCTIC_GRAPH_DIR
from training.model import GraspContractionNet

MODEL_KEYS = ["patch_positions", "patch_normals", "patch_curvature",
              "contact_positions", "contact_normals", "finger_ids"]


def reproduce_val_indices(n, seed, val_split):
    """Replicate train.py make_loaders() split exactly."""
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, int(n * val_split))
    return idx[:n_val]                      # val_idx (rest is train)


def infer(model, sample, device):
    batch = {k: sample[k].unsqueeze(0).to(device) for k in MODEL_KEYS}
    with torch.no_grad():
        out = model(batch)
    return out["anchor_positions"][0].cpu().numpy()   # [num_anchors, 3]


def match(gt, pred):
    """Hungarian-match valid GT (M,3) to predicted slots (Q,3). Returns matched
    pred (M,3) and per-GT Euclidean distance (M,)."""
    cost = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)   # [M, Q]
    r, c = linear_sum_assignment(cost)
    return pred[c], cost[r, c]


def build_panel(fig, row, col, sample, gt, pred_matched, dists, title):
    from plotly import graph_objects as go
    patch = sample["patch_positions"].numpy()
    contacts = sample["contact_positions"].numpy()
    side = sample["finger_side"].numpy()
    R, L = side == 0, side == 1

    def add(tr): fig.add_trace(tr, row=row, col=col)
    add(go.Scatter3d(x=patch[:, 0], y=patch[:, 1], z=patch[:, 2], mode="markers",
                     marker=dict(size=1.6, color="#888", opacity=0.35),
                     showlegend=False, hoverinfo="skip"))
    add(go.Scatter3d(x=contacts[R, 0], y=contacts[R, 1], z=contacts[R, 2], mode="markers",
                     marker=dict(size=3.5, color="#00D4FF"), showlegend=False, hoverinfo="skip"))
    add(go.Scatter3d(x=contacts[L, 0], y=contacts[L, 1], z=contacts[L, 2], mode="markers",
                     marker=dict(size=3.5, color="#FF8C00"), showlegend=False, hoverinfo="skip"))
    # GT ↔ PRED error lines
    for g, p in zip(gt, pred_matched):
        add(go.Scatter3d(x=[g[0], p[0]], y=[g[1], p[1]], z=[g[2], p[2]], mode="lines",
                         line=dict(color="rgba(255,230,80,0.8)", width=3),
                         showlegend=False, hoverinfo="skip"))
    add(go.Scatter3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="markers",
                     marker=dict(size=7, color="#00FF88", symbol="diamond",
                                 line=dict(width=1, color="white")),
                     showlegend=False,
                     hovertemplate="GT (%{x:.3f},%{y:.3f},%{z:.3f})<extra></extra>"))
    add(go.Scatter3d(x=pred_matched[:, 0], y=pred_matched[:, 1], z=pred_matched[:, 2],
                     mode="markers",
                     marker=dict(size=7, color="#FF3B3B", symbol="x"),
                     showlegend=False,
                     hovertemplate="PRED (%{x:.3f},%{y:.3f},%{z:.3f})<extra></extra>"))
    # per-panel scene title via annotation handled by subplot_titles
    return title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--robot", type=str, default="Robotiq")
    ap.add_argument("--gate-max-dist", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=10, help="panels to render")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--num-anchors", type=int, default=4)
    ap.add_argument("--results-dir", type=str, default=str(ARCTIC_RESULTS_DIR))
    ap.add_argument("--graph-dir", type=str, default=str(ARCTIC_GRAPH_DIR))
    ap.add_argument("--all", action="store_true",
                    help="evaluate ALL loaded samples (held-out set, e.g. s05), not a val split")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--save", type=str,
                    default=str(REPO_ROOT / "md_claude" / "arctic" / "eval_pred_vs_gt.html"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = ArcticContractionDataset(robot=args.robot, gate_max_dist=args.gate_max_dist,
                                  max_anchors=args.num_anchors,
                                  results_dir=args.results_dir, graph_dir=args.graph_dir)
    N = len(ds)
    val_idx = list(range(N)) if args.all else reproduce_val_indices(N, args.seed, args.val_split)
    print(f"[dataset] {N} samples  eval={len(val_idx)}  "
          f"({'ALL (held-out)' if args.all else f'val split seed={args.seed}'})")

    model = GraspContractionNet(d_model=args.d_model, num_anchors=args.num_anchors).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] loaded {args.ckpt}  (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('metrics',{}).get('val',{}).get('loss')})")

    # ── metric over ALL val samples ──
    all_d = []
    per_sample = []   # (idx, file_name, M, mean_dist_cm)
    for vi in val_idx:
        s = ds[vi]
        m = int(s["anchor_mask"].sum())
        gt = s["human_anchor"].numpy()[:m]
        pred = infer(model, s, device)
        _, d = match(gt, pred)
        all_d.extend((d * 100).tolist())          # cm
        fn = json.load(open(ds.samples[vi][0]))["file_name"]
        per_sample.append((vi, fn, m, float(d.mean() * 100)))
    all_d = np.array(all_d)
    print(f"\n[val anchor error]  mean={all_d.mean():.2f}cm  median={np.median(all_d):.2f}cm  "
          f"p90={np.percentile(all_d,90):.2f}cm  max={all_d.max():.2f}cm  (n_anchors={len(all_d)})")

    # ── pick N panels: prefer a mix of K=4 (bimanual) and K=2 ──
    k4 = [t for t in per_sample if t[2] == 4]
    k2 = [t for t in per_sample if t[2] == 2]
    n4 = min(len(k4), max(1, args.n * 2 // 3))
    picks = k4[:n4] + k2[:args.n - n4]
    picks = picks[:args.n]
    print(f"[viz] {len(picks)} panels  (K=4: {sum(1 for p in picks if p[2]==4)}, "
          f"K=2: {sum(1 for p in picks if p[2]==2)})")

    # ── build grid ──
    from plotly.subplots import make_subplots
    cols = 2
    rows = (len(picks) + cols - 1) // cols
    titles = [f"{fn.replace('Robotiq','')[:30]}  K={m}  err={d:.1f}cm"
              for (_, fn, m, d) in picks]
    fig = make_subplots(rows=rows, cols=cols,
                        specs=[[{"type": "scene"}] * cols for _ in range(rows)],
                        subplot_titles=titles, horizontal_spacing=0.02,
                        vertical_spacing=0.04)
    for i, (vi, fn, m, dmean) in enumerate(picks):
        r, c = i // cols + 1, i % cols + 1
        s = ds[vi]
        gt = s["human_anchor"].numpy()[:m]
        pred = infer(model, s, device)
        pm, d = match(gt, pred)
        build_panel(fig, r, c, s, gt, pm, d, titles[i])

    fig.update_layout(
        paper_bgcolor="rgb(15,15,25)", font=dict(color="white", size=11),
        title=dict(text=("ARCTIC val: GT (green ◆) vs PRED (red ✕), error lines yellow | "
                         "grey=object patches, cyan/orange=R/L contacts | "
                         f"val mean err {all_d.mean():.2f}cm"), font=dict(size=13)),
        height=460 * rows, width=1200, showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10
    n_scenes = rows * cols
    for k in range(1, n_scenes + 1):
        key = "scene" if k == 1 else f"scene{k}"
        if key in fig.layout:
            fig.layout[key].update(bgcolor="rgb(20,20,30)", aspectmode="data",
                                   xaxis_title="", yaxis_title="", zaxis_title="")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"[saved] {out}")
    print("\nper-panel:")
    for vi, fn, m, d in picks:
        print(f"  idx {vi:5d}  K={m}  err={d:5.1f}cm  {fn}")


if __name__ == "__main__":
    main()
