"""
Visualize dexterous grasp (Panel A) and contracted grasp (Panel B) using a trained checkpoint.

Panel A — Dexterous Grasp (network input)
  - Object surface patches (grey)
  - MANO hand skeleton drawn from hand_joints [21, 3]  (no ManoLayer needed)
  - Finger contact markers, coloured per finger

Panel B — Contracted Grasp (GT vs Predicted)
  - Object surface patches coloured by max anchor attention heatmap
  - GT pseudo-label anchors (green diamonds)
  - Predicted anchors from network (red diamonds)
  - Assignment lines: predicted anchor → its strongest finger contact
  - (--show-robot) GT robot mesh via qr + robot FK

Usage:
    python scripts/visualize_grasp.py --idx 0 --robot Robotiq
    python scripts/visualize_grasp.py --idx 0 --robot Robotiq --save vis_0.html
    python scripts/visualize_grasp.py --idx 0 --robot Robotiq --show-robot
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── MANO skeleton bone list (21 joints) ──────────────────────────────────────
MANO_BONES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),   # index
    (0, 5),  (5, 6),  (6, 7),  (7, 8),   # middle
    (0, 9),  (9, 10), (10, 11),(11, 12),  # ring
    (0, 13),(13, 14),(14, 15),(15, 16),   # little
    (0, 17),(17, 18),(18, 19),(19, 20),   # thumb
]

FINGER_COLORS = [
    "#4A90D9",  # index  — blue
    "#27AE60",  # middle — green
    "#E67E22",  # ring   — orange
    "#9B59B6",  # little — purple
    "#E74C3C",  # thumb  — red
]
FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]


# ── trace builders ────────────────────────────────────────────────────────────

def traces_object_patches(patch_pos: np.ndarray,
                           attn_weights: np.ndarray | None = None,
                           name: str = "object patches") -> list:
    """Return a single Scatter3d trace for object patches."""
    from plotly import graph_objects as go

    if attn_weights is not None:
        w = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)
        marker = dict(size=3, color=w, colorscale="RdBu_r",
                      cmin=0, cmax=1, showscale=True,
                      colorbar=dict(title="Attn", thickness=10, len=0.5))
    else:
        marker = dict(size=3, color="lightgrey")

    return [go.Scatter3d(
        x=patch_pos[:, 0], y=patch_pos[:, 1], z=patch_pos[:, 2],
        mode="markers", marker=marker, name=name,
        hovertemplate="patch<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    )]


def traces_mano_skeleton(hand_joints: np.ndarray) -> list:
    """Draw MANO skeleton as lines + joint dots from hand_joints [21, 3]."""
    from plotly import graph_objects as go
    traces = []
    for p, c in MANO_BONES:
        traces.append(go.Scatter3d(
            x=[hand_joints[p, 0], hand_joints[c, 0]],
            y=[hand_joints[p, 1], hand_joints[c, 1]],
            z=[hand_joints[p, 2], hand_joints[c, 2]],
            mode="lines",
            line=dict(color="white", width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
    traces.append(go.Scatter3d(
        x=hand_joints[:, 0], y=hand_joints[:, 1], z=hand_joints[:, 2],
        mode="markers",
        marker=dict(size=3, color="white"),
        name="hand joints",
        hovertemplate="joint %{pointNumber}<extra></extra>",
    ))
    return traces


def traces_finger_contacts(contact_pos: np.ndarray,
                            finger_ids: np.ndarray) -> list:
    from plotly import graph_objects as go
    traces = []
    for pos, fid in zip(contact_pos, finger_ids):
        fid = int(fid)
        traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers",
            marker=dict(size=9, color=FINGER_COLORS[fid],
                        symbol="circle", line=dict(width=1, color="white")),
            name=FINGER_NAMES[fid],
            hovertemplate=f"{FINGER_NAMES[fid]}<br>({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})<extra></extra>",
        ))
    return traces


def traces_anchors(positions: np.ndarray, color: str,
                   label: str, num_valid: int) -> list:
    from plotly import graph_objects as go
    pos = positions[:num_valid]
    return [go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(size=14, color=color, symbol="diamond",
                    line=dict(width=2, color="white")),
        name=label,
        hovertemplate=f"{label}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
    )]


def traces_assignment_lines(pred_pos: np.ndarray,
                             contact_pos: np.ndarray,
                             assignment: np.ndarray,
                             num_valid: int) -> list:
    """Thin lines from each predicted anchor to its strongest finger contact."""
    from plotly import graph_objects as go
    traces = []
    for m in range(num_valid):
        j = int(assignment[m].argmax())
        traces.append(go.Scatter3d(
            x=[pred_pos[m, 0], contact_pos[j, 0]],
            y=[pred_pos[m, 1], contact_pos[j, 1]],
            z=[pred_pos[m, 2], contact_pos[j, 2]],
            mode="lines",
            line=dict(color="rgba(255,200,100,0.7)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


_YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick',
}


def traces_object_mesh(ycb_id: int, opacity: float = 0.35) -> list:
    """Load YCB canonical mesh and return a Mesh3d trace (canonical frame)."""
    import os, trimesh
    from plotly import graph_objects as go

    obj_name = _YCB_CLASSES.get(ycb_id)
    if obj_name is None:
        return []

    dex_ycb_dir = os.environ.get("DEX_YCB_DIR", str(REPO_ROOT.parent / "dataset"))
    mesh_path = Path(dex_ycb_dir) / "models" / obj_name / "textured_simple.obj"
    if not mesh_path.exists():
        print(f"[warn] mesh not found: {mesh_path}")
        return []

    mesh = trimesh.load(str(mesh_path), force="mesh")
    v = np.array(mesh.vertices)     # already in canonical (YCB model) frame, metres
    f = np.array(mesh.faces)

    return [go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="lightblue", opacity=opacity,
        name="object mesh",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        flatshading=False,
    )]


def solve_ik_for_anchors(robot_name: str, pred_pos_canonical: np.ndarray,
                         R: np.ndarray, gt_qr: np.ndarray,
                         device: str, n_iter: int = 300) -> list:
    """
    Gradient-descent IK: find joint angles so robot fingertips match predicted anchors.

    - Base position/rotation kept from GT qr (same physical setup)
    - Only joint angles (q[:, 6:]) optimized
    - pred_pos_canonical [M, 3]: predicted anchors in canonical frame
    - R [3,3]: canonical→camera  →  apply R to convert pred to camera-axes for FK
    - Returns Mesh3d traces in canonical frame (R.T applied)
    """
    import torch
    from scipy.optimize import linear_sum_assignment as lsa
    from optimisation.robot import Shadow, Allegro, Barrett, Robotiq

    cls = {"Shadow": Shadow, "Allegro": Allegro,
           "Barrett": Barrett, "Robotiq": Robotiq}[robot_name]
    robot = cls(batch=1, device=device)

    # pred anchors: canonical → object-centric camera-axes
    pred_cam = (R @ pred_pos_canonical.T).T                          # [M, 3]
    target = torch.tensor(pred_cam, dtype=torch.float32,
                          device=device).unsqueeze(0)                # [1, M, 3]
    M = target.shape[1]

    # Initialise from GT qr (warm start with same base pose)
    qr_t = torch.tensor(gt_qr, dtype=torch.float32, device=device)
    if qr_t.ndim == 1:
        qr_t = qr_t.unsqueeze(0)                                     # [1, D]

    q_base   = qr_t[:, :6].clone().detach()                         # fixed
    q_joints = qr_t[:, 6:].clone().detach().requires_grad_(True)    # optimised

    optim = torch.optim.Adam([q_joints], lr=0.005)
    relu  = torch.nn.ReLU()

    for _ in range(n_iter):
        q = torch.cat([q_base, q_joints], dim=-1)
        tips = robot.get_keypoints(q, subset=M).squeeze(0)          # [M, 3]
        dist = torch.cdist(tips, target.squeeze(0))                  # [M, M]
        row, col = lsa(dist.detach().cpu().numpy())
        dist_loss = dist[row, col].sum() / M
        jl = (relu(q_joints - robot.joint_upper) +
              relu(robot.joint_lower - q_joints)).sum()
        loss = dist_loss + 10.0 * jl
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Final FK → mesh in camera-axes frame → rotate to canonical
    q_final = torch.cat([q_base, q_joints.detach()], dim=-1)
    robot.forward_kinematics(q_final)
    raw_traces = robot.get_mesh_updated(color="orange", opacity=0.5)

    from plotly import graph_objects as go
    Rt = R.T
    rotated = []
    for t in raw_traces:
        v = np.stack([t.x, t.y, t.z], axis=-1)
        v_c = (Rt @ v.T).T
        rotated.append(go.Mesh3d(
            x=v_c[:, 0], y=v_c[:, 1], z=v_c[:, 2],
            i=t.i, j=t.j, k=t.k,
            color="orange", opacity=0.5, name="Pred robot",
        ))
    return rotated


def traces_gt_robot(robot_name: str, qr: np.ndarray, device: str,
                    R: np.ndarray | None = None) -> list:
    """Build GT robot mesh traces from qr vector using robot FK.

    qr is in object-centric camera-axes frame (from Optimize.py).
    R [3,3] is canonical→camera; apply R.T to convert mesh to canonical frame.
    """
    import torch
    from optimisation.robot import Shadow, Allegro, Barrett, Robotiq

    cls = {"Shadow": Shadow, "Allegro": Allegro,
           "Barrett": Barrett, "Robotiq": Robotiq}[robot_name]
    robot = cls(batch=1, device=device)
    qr_t = torch.tensor(qr, dtype=torch.float32).to(device)
    if qr_t.ndim == 1:
        qr_t = qr_t.unsqueeze(0)
    robot.forward_kinematics(qr_t)
    traces = robot.get_mesh_updated(color="lightblue", opacity=0.45)

    # rotate each mesh link from object-centric camera-axes → canonical frame
    if R is not None:
        Rt = R.T   # [3,3]  camera-axes → canonical
        from plotly import graph_objects as go
        rotated = []
        for t in traces:
            v = np.stack([t.x, t.y, t.z], axis=-1)   # [N, 3]
            v_canon = (Rt @ v.T).T
            rotated.append(go.Mesh3d(
                x=v_canon[:, 0], y=v_canon[:, 1], z=v_canon[:, 2],
                i=t.i, j=t.j, k=t.k,
                color="lightblue", opacity=0.45,
            ))
        return rotated
    return traces


# ── RGB image loader ─────────────────────────────────────────────────────────

def _find_rgb_image(result: dict, args) -> "np.ndarray | None":
    """
    Locate the original DexYCB RGB frame for this sample.

    file_name format: {subject_num}_{date}_{sequence}_{serial}_{frame:06d}
    e.g.  1_20200709_152843_932122061900_70
    """
    import os
    file_name = result.get("file_name", "")
    parts = file_name.split("_")
    if len(parts) < 5:
        return None

    # parts: [subj_num, date, seq_time, serial, frame_idx]
    subj_num = parts[0]
    date     = parts[1]
    seq_time = parts[2]
    serial   = parts[3]
    frame    = int(parts[4])

    _SUBJECT_MAP = {
        "1": "20200709-subject-01", "2": "20200813-subject-02",
        "3": "20200820-subject-03", "4": "20200903-subject-04",
        "5": "20200908-subject-05", "6": "20200918-subject-06",
        "7": "20200928-subject-07", "8": "20201002-subject-08",
        "9": "20201015-subject-09", "10": "20201022-subject-10",
    }
    subject_folder = _SUBJECT_MAP.get(subj_num)
    if subject_folder is None:
        return None

    dex_ycb_dir = os.environ.get("DEX_YCB_DIR", str(REPO_ROOT.parent / "dataset"))
    img_path = (Path(dex_ycb_dir) / subject_folder
                / f"{date}_{seq_time}" / serial
                / f"color_{frame:06d}.jpg")

    if not img_path.exists():
        print(f"[warn] RGB image not found: {img_path}")
        return None

    try:
        from PIL import Image
        img = Image.open(img_path)
        return np.array(img)
    except Exception as e:
        print(f"[warn] Failed to load RGB image: {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--idx",         type=int,  default=0,
                   help="Index within the selected split")
    p.add_argument("--checkpoint",  type=str,
                   default=str(REPO_ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--robot",       type=str,  default="Robotiq",
                   choices=["Shadow", "Allegro", "Barrett", "Robotiq"])
    p.add_argument("--mu",          type=float, default=None,
                   help="Filter by friction coefficient (e.g. 0.5, 0.9)")
    p.add_argument("--split",       type=str,  default="val",
                   choices=["train", "val", "all"],
                   help="Which split to index into (default: val)")
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--results-dir", type=str,
                   default=str(REPO_ROOT / "data" / "results_train"))
    p.add_argument("--graph-dir",   type=str,
                   default=str(REPO_ROOT / "data" / "graph_cache"))
    p.add_argument("--device",      type=str,  default="cuda:0")
    p.add_argument("--show-robot",  action="store_true",
                   help="Overlay GT robot mesh in Panel B (requires robot FK)")
    p.add_argument("--show-mesh",   action="store_true",
                   help="Show YCB object as mesh instead of point cloud")
    p.add_argument("--save",        type=str,  default=None,
                   help="Save interactive HTML to this path")
    return p.parse_args()


def main():
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly import graph_objects as go  # ensure go.Image is available

    args = parse_args()
    if torch.cuda.is_available():
        try:
            # Quick sanity-check: run a tiny kernel to catch sm-mismatch early
            torch.zeros(4, device=args.device).sum()
            device = torch.device(args.device)
        except RuntimeError:
            print("[warning] CUDA kernel failed (sm_120 / driver mismatch?), falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ── dataset + split ──
    import random
    from training.dataset import GraspContractionDataset
    ds = GraspContractionDataset(
        results_dir=args.results_dir,
        graph_dir=args.graph_dir,
        robot=args.robot,
        mu=args.mu,
    )
    N = len(ds)

    if args.split == "all":
        indices = list(range(N))
    else:
        all_idx = list(range(N))
        random.seed(args.seed)
        random.shuffle(all_idx)
        n_val = max(1, int(N * args.val_split))
        val_idx   = all_idx[:n_val]
        train_idx = all_idx[n_val:]
        indices = val_idx if args.split == "val" else train_idx

    mu_tag = f" mu={args.mu}" if args.mu is not None else ""
    print(f"[dataset] {N} total  split={args.split}({len(indices)}){mu_tag}  robot={args.robot}")

    if args.idx >= len(indices):
        sys.exit(f"[error] idx {args.idx} out of range for split '{args.split}' (0–{len(indices)-1})")

    ds_idx = indices[args.idx]
    sample = ds[ds_idx]
    result_path, _ = ds.samples[ds_idx]

    # ── model ──
    num_anchors = 2 if args.robot == "Robotiq" else 3
    from training.model import GraspContractionNet
    model = GraspContractionNet(num_anchors=num_anchors).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt["epoch"]
    print(f"[checkpoint] epoch={epoch}")

    # ── inference ──
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
             for k, v in sample.items()}
    with torch.no_grad():
        out = model(batch)

    pred_pos   = out["anchor_positions"][0].cpu().numpy()            # [M, 3]
    logits     = out["anchor_patch_logits"][0].cpu().numpy()         # [M, P]
    assignment = out["human_to_anchor_assignment"][0].cpu().numpy()  # [M, 5]

    # ── unpack sample ──
    patch_pos   = sample["patch_positions"].numpy()    # [P, 3]
    hand_joints = sample["hand_joints"].numpy()        # [21, 3]
    contact_pos = sample["contact_positions"].numpy()  # [5, 3]
    finger_ids  = sample["finger_ids"].numpy()         # [5]
    gt_anchors  = sample["human_anchor"].numpy()       # [M, 3]
    num_valid   = int(sample["num_anchors"].item())
    ycb_id      = int(sample["ycb_id"].item())

    # attention heatmap: max over anchors → [P]
    w = np.exp(logits - logits.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True) + 1e-8
    max_attn = w.max(axis=0)

    # ── print summary ──
    print(f"\n[sample idx={args.idx}]  ycb_id={ycb_id}")
    print(f"  GT anchors   : {gt_anchors[:num_valid].round(4)}")
    print(f"  Pred anchors : {pred_pos[:num_valid].round(4)}")
    dists = np.linalg.norm(gt_anchors[:num_valid] - pred_pos[:num_valid], axis=-1)
    print(f"  L2 per anchor: {dists.round(4)}  mean={dists.mean():.4f} m")

    # ── build traces ──
    tA, tB, tC = [], [], []

    # ── find original RGB image ──
    result_data = json.load(open(result_path))
    rgb_img = _find_rgb_image(result_data, args)

    # IK for predicted anchors (computed once, used in Panel C)
    tC_robot = []
    R_mat = None
    if args.show_robot:
        graph_data = json.load(open(ds.samples[ds_idx][1]))
        R_mat = np.array(graph_data["obj_rotation"], dtype=np.float32)
        qr    = np.array(result_data["qr"],          dtype=np.float32)
        print("[IK] solving for predicted anchors ...", flush=True)
        tC_robot = solve_ik_for_anchors(args.robot, pred_pos[:num_valid],
                                        R_mat, qr, str(device))

    # Panel A — Dexterous Grasp
    if args.show_mesh:
        tA += traces_object_mesh(ycb_id, opacity=0.3)
    tA += traces_object_patches(patch_pos, name="object patches")
    tA += traces_mano_skeleton(hand_joints)
    tA += traces_finger_contacts(contact_pos, finger_ids)

    # Panel B — GT Contracted Grasp (GT anchors + GT robot)
    if args.show_mesh:
        tB += traces_object_mesh(ycb_id, opacity=0.3)
    tB += traces_object_patches(patch_pos, attn_weights=max_attn, name="object (attn)")
    tB += traces_anchors(gt_anchors, "green", "GT anchors", num_valid)
    if args.show_robot and R_mat is not None:
        tB += traces_gt_robot(args.robot, qr, str(device), R=R_mat)

    # Panel C — Predicted Contracted Grasp (Pred anchors + IK robot)
    if args.show_mesh:
        tC += traces_object_mesh(ycb_id, opacity=0.3)
    tC += traces_object_patches(patch_pos, attn_weights=max_attn, name="object (attn)")
    tC += traces_anchors(pred_pos, "red", "Pred anchors", num_valid)
    tC += traces_assignment_lines(pred_pos, contact_pos, assignment, num_valid)
    tC += tC_robot

    # ── assemble figure ──
    from plotly.subplots import make_subplots as _msp

    if rgb_img is not None:
        specs = [[{"type": "image"}, {"type": "scene"},
                  {"type": "scene"}, {"type": "scene"}]]
        titles = ["Panel 0 — Original RGB",
                  "Panel A — Dexterous Grasp",
                  "Panel B — GT Contracted (blue=GT robot)",
                  "Panel C — Pred Contracted (orange=IK robot)"]
        col_w  = [0.18, 0.27, 0.27, 0.27]
        fig = _msp(rows=1, cols=4, specs=specs,
                   subplot_titles=titles,
                   column_widths=col_w, horizontal_spacing=0.02)
        fig.add_trace(go.Image(z=rgb_img), row=1, col=1)
        for t in tA: fig.add_trace(t, row=1, col=2)
        for t in tB: fig.add_trace(t, row=1, col=3)
        for t in tC: fig.add_trace(t, row=1, col=4)
        scene_keys = {"scene": "scene", "scene2": "scene2", "scene3": "scene3"}
    else:
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
        titles = ["Panel A — Dexterous Grasp",
                  "Panel B — GT Contracted (blue=GT robot)",
                  "Panel C — Pred Contracted (orange=IK robot)"]
        fig = _msp(rows=1, cols=3, specs=specs,
                   subplot_titles=titles, horizontal_spacing=0.02)
        for t in tA: fig.add_trace(t, row=1, col=1)
        for t in tB: fig.add_trace(t, row=1, col=2)
        for t in tC: fig.add_trace(t, row=1, col=3)
        scene_keys = {"scene": "scene", "scene2": "scene2", "scene3": "scene3"}

    scene_common = dict(
        bgcolor="rgb(20,20,30)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        zaxis=dict(showgrid=False, zeroline=False),
        aspectmode="data",
    )
    layout_scenes = {v: scene_common for v in scene_keys.values()}

    fig.update_layout(
        title=dict(
            text=f"idx={args.idx}  ycb_id={ycb_id}  robot={args.robot}  epoch={epoch}  "
                 f"mean_L2={dists.mean():.4f}m",
            font=dict(size=14),
        ),
        paper_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color="white")),
        width=2200, height=750,
        **layout_scenes,
    )

    if args.save:
        fig.write_html(args.save)
        print(f"[saved] {args.save}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
