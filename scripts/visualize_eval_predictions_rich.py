#!/usr/bin/env python3
"""Rich eval visualization: per val sample, render [ camera photo | 3D scene ]
where the 3D scene shows the object mesh + both hand meshes + GT anchors (green)
+ model-PRED anchors (red) + error lines.

Reuses the mesh/photo helpers from visualize_arctic_grasp.py. Everything is drawn
in the OBJECT-CENTRIC frame (world - obj_centre), matching that script. The model
predicts in the CANONICAL frame, so predictions are rotated back:
    pred_object_centric = Rg @ pred_canonical          (Rg = graph['obj_rotation'])

Usage:
  cd /home/robin/a_workspace/condex && ./venv/bin/python scripts/visualize_eval_predictions_rich.py \
      --ckpt checkpoints/best.pt --n 6 --gate-max-dist 0.03 --cam 0
"""
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.dataset import ArcticContractionDataset, ARCTIC_GRAPH_DIR, ARCTIC_RESULTS_DIR
from training.model import GraspContractionNet
from scripts.visualize_arctic_grasp import (
    traces_object_mesh, traces_hand_mesh, load_articulated_mesh,
    load_frame_data, mano_forward, load_rgb_image, _make_scene_style,
)

MODEL_KEYS = ["patch_positions", "patch_normals", "patch_curvature",
              "contact_positions", "contact_normals", "finger_ids"]
ARCTIC_ROOT = str(REPO.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data")
MANO_ROOT = str(REPO / "network" / "manopth" / "mano" / "models")


def reproduce_val(n, seed, vs):
    random.seed(seed); idx = list(range(n)); random.shuffle(idx)
    return idx[: max(1, int(n * vs))]


def infer(model, sample, device):
    batch = {k: sample[k].unsqueeze(0).to(device) for k in MODEL_KEYS}
    with torch.no_grad():
        return model(batch)["anchor_positions"][0].cpu().numpy()


def anchor_traces(gt, pred, dists):
    import plotly.graph_objects as go
    tr = []
    if pred is not None:
        for g, p in zip(gt, pred):
            tr.append(go.Scatter3d(x=[g[0], p[0]], y=[g[1], p[1]], z=[g[2], p[2]],
                                   mode="lines", line=dict(color="rgba(255,230,80,0.9)", width=4),
                                   showlegend=False, hoverinfo="skip"))
    tr.append(go.Scatter3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="markers",
                           marker=dict(size=8, color="#00FF88", symbol="diamond",
                                       line=dict(width=1, color="white")),
                           name="GT", showlegend=False,
                           hovertemplate="GT<extra></extra>"))
    if pred is not None:
        tr.append(go.Scatter3d(x=pred[:, 0], y=pred[:, 1], z=pred[:, 2], mode="markers",
                               marker=dict(size=8, color="#FF3B3B", symbol="x"),
                               name="PRED", showlegend=False,
                               hovertemplate="PRED<extra></extra>"))
    return tr


FINGERS = ["index", "middle", "ring", "little", "thumb"]


def contact_traces(cr, cl):
    """Actual per-finger hand contact points (5/hand): right cyan, left orange."""
    import plotly.graph_objects as go
    tr = []
    for c, col, nm in [(cr, "#00D4FF", "R contact"), (cl, "#FF8C00", "L contact")]:
        tr.append(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
            marker=dict(size=5, color=col, symbol="circle",
                        line=dict(width=1, color="white")),
            name=nm, showlegend=False, hovertemplate=f"{nm}<extra></extra>"))
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--robot", default="Robotiq")
    ap.add_argument("--gate-max-dist", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--num-anchors", type=int, default=4)
    ap.add_argument("--cam", type=int, default=0, help="ARCTIC camera id for the photo")
    ap.add_argument("--cam-preset", default="top")
    ap.add_argument("--require-image", action="store_true",
                    help="only pick samples whose ARCTIC photo exists on disk "
                         "(images are partially downloaded; may include train samples)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results-dir", default=str(ARCTIC_RESULTS_DIR))
    ap.add_argument("--graph-dir", default=str(ARCTIC_GRAPH_DIR))
    ap.add_argument("--all", action="store_true",
                    help="pick from ALL loaded samples (held-out set, e.g. s05), not a val split")
    ap.add_argument("--no-pred", action="store_true",
                    help="GT-only: show the contraction labels (no model inference). "
                         "Use when the checkpoint predates the current labels.")
    ap.add_argument("--save", default=str(REPO / "md_claude" / "arctic" / "eval_rich.html"))
    args = ap.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = ArcticContractionDataset(robot=args.robot, gate_max_dist=args.gate_max_dist,
                                  max_anchors=args.num_anchors,
                                  results_dir=args.results_dir, graph_dir=args.graph_dir)
    val_idx = list(range(len(ds))) if args.all else reproduce_val(len(ds), args.seed, args.val_split)
    model = None
    if not args.no_pred:
        model = GraspContractionNet(d_model=256, num_anchors=args.num_anchors).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"]); model.eval()
    print(f"[model] {'(none, GT-only)' if args.no_pred else args.ckpt}  val={len(val_idx)}")

    # pick: prefer K=4 (bimanual) for richer panels.
    def img_exists(fn):
        p = fn.split("_"); subj = p[0]; t = int(p[-1]); seq = "_".join(p[1:-1])
        base = Path(ARCTIC_ROOT) / "data" / "images" / subj / seq / str(args.cam)
        return (base / f"{t+1:05d}.jpg").exists() or (base / f"{t:05d}.jpg").exists()

    if args.require_image:
        # ARCTIC images are only partially downloaded (a few s01 seqs). To show
        # the photo column, draw from ANY gated sample whose image exists (may
        # include train samples — flagged in the title).
        pool = [vi for vi in range(len(ds))
                if img_exists(ds.samples[vi][1].stem.replace("_graph", ""))]
        val_set = set(val_idx)
        print(f"[require-image] {len(pool)} gated samples have a cam{args.cam} photo")
    else:
        pool, val_set = list(val_idx), set(val_idx)
    cand = [(vi, int(ds[vi]["anchor_mask"].sum())) for vi in pool]
    cand.sort(key=lambda x: -x[1])
    # diversify: at most 2 panels per sequence for object/grasp variety
    from collections import Counter
    _seen = Counter(); picks = []
    for vi, m in cand:
        seq = "_".join(ds.samples[vi][1].stem.replace("_graph", "").split("_")[1:-1])
        if _seen[seq] < 2:
            _seen[seq] += 1; picks.append((vi, m))
        if len(picks) >= args.n:
            break

    rows = len(picks)
    specs = [[{"type": "image"}, {"type": "scene"}] for _ in range(rows)]
    titles = []
    panel_data = []
    for vi, m in picks:
        s = ds[vi]
        result_path, graph_path = ds.samples[vi]
        result = json.load(open(result_path)); graph = json.load(open(graph_path))
        fn = result["file_name"]
        Rg = np.array(graph["obj_rotation"], dtype=np.float32)
        oc_gen = np.array(result["obj_center"], dtype=np.float32)
        gt = s["human_anchor"].numpy()[:m]                      # canonical
        gt_w = (Rg @ gt.T).T + oc_gen                           # world
        if args.no_pred:
            pred_w, dists, errtxt = None, None, "GT contraction"
        else:
            pred = infer(model, s, device)                      # canonical [Q,3]
            cost = np.linalg.norm(gt[:, None] - pred[None], axis=2)
            r, c = linear_sum_assignment(cost); pred_m = pred[c]; dists = cost[r, c]
            pred_w = (Rg @ pred_m.T).T + oc_gen                 # world
            errtxt = f"err={dists.mean()*100:.1f}cm"
        # actual per-finger contacts (object-centric in JSON -> world)
        cr_w = np.array([result["human_finger_contacts"][f] for f in FINGERS], np.float32) + oc_gen
        cl_w = np.array([result["human_finger_contacts_left"][f] for f in FINGERS], np.float32) + oc_gen
        tag = "" if vi in val_set else " [train]"
        titles += [f"{fn}{tag}", f"K={m}  {errtxt}"]
        panel_data.append((fn, gt_w, pred_w, dists, cr_w, cl_w))

    fig = make_subplots(rows=rows, cols=2, specs=specs, subplot_titles=titles,
                        column_widths=[0.30, 0.70], horizontal_spacing=0.01,
                        vertical_spacing=0.03)

    for i, (fn, gt_w, pred_w, dists, cr_w, cl_w) in enumerate(panel_data):
        rr = i + 1
        parts = fn.split("_"); subject = parts[0]; t = int(parts[-1])
        seq = "_".join(parts[1:-1]); obj_name = seq.split("_")[0]
        # meshes (world -> object-centric on recomputed centroid)
        obj_params, mano_data = load_frame_data(ARCTIC_ROOT, subject, seq, t)
        verts_w, faces = load_articulated_mesh(ARCTIC_ROOT, obj_name, obj_params[0],
                                               obj_params[1:4], obj_params[4:])
        oc = verts_w.mean(0)
        vr, fr, _ = mano_forward(mano_data, "right", t, MANO_ROOT)
        vl, fl, _ = mano_forward(mano_data, "left", t, MANO_ROOT)
        scene_tr = (traces_object_mesh(verts_w - oc, faces, opacity=0.25, color="#89CFF0", name="obj")
                    + traces_hand_mesh(vr - oc, fr, color="#00BFFF", name="R", opacity=0.4)
                    + traces_hand_mesh(vl - oc, fl, color="#FF8C00", name="L", opacity=0.4)
                    + contact_traces(cr_w - oc, cl_w - oc)
                    + anchor_traces(gt_w - oc, None if pred_w is None else pred_w - oc, dists))
        for tr in scene_tr:
            fig.add_trace(tr, row=rr, col=2)
        # photo
        rgb = load_rgb_image(ARCTIC_ROOT, subject, seq, t, args.cam)
        if rgb is not None:
            rgb = rgb[::4, ::4]                                  # downsample for HTML size
            fig.add_trace(go.Image(z=rgb), row=rr, col=1)

    # style scenes
    for k in range(1, rows + 1):
        key = "scene" if k == 1 else f"scene{k}"
        if key in fig.layout:
            fig.layout[key].update(**_make_scene_style(args.cam_preset))
    for k in range(1, rows + 1):
        ax = "xaxis" if k == 1 else f"xaxis{k}"
        ay = "yaxis" if k == 1 else f"yaxis{k}"
        if ax in fig.layout: fig.layout[ax].update(visible=False)
        if ay in fig.layout: fig.layout[ay].update(visible=False)
    for ann in fig.layout.annotations:
        ann.font.size = 10
    fig.update_layout(paper_bgcolor="rgb(15,15,25)", font=dict(color="white"),
                      title=dict(text="ARCTIC val — photo | object+hand mesh, GT(green ◆) vs PRED(red ✕)",
                                 font=dict(size=13)),
                      height=430 * rows, width=1300, showlegend=False)
    out = Path(args.save); out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"[saved] {out}")
    for fn, _, _, d, *_ in panel_data:
        print(f"  {fn}  " + ("GT-only" if d is None else f"err={d.mean()*100:.1f}cm"))


if __name__ == "__main__":
    main()
