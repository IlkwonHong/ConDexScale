"""
Visualize EXACTLY what ArcticContractionDataset.__getitem__ returns.

Unlike visualize_arctic_grasp.py (which re-derives the grasp from the raw result
JSON + MANO forward), this renders the dataloader's OWN output tensors — the
articulated+centered object patches, the canonical-frame contacts, and the
Rg.T-transformed anchors that the model actually trains on.  Use it to confirm
the dataloader's coordinate transforms are correct and everything lands in one
coherent canonical frame.

Single 3D panel (object canonical frame):
  - Object patches      : dots, RED = movable (part_id==0), GREY = static
  - Patch normals       : short outward lines (optional, --normals)
  - Right contacts (0-4): cyan dots, finger-labelled
  - Left contacts (5-9) : orange dots
  - human_anchor (valid): large green diamonds
  - anchor → nearest contact : dotted yellow lines
  - Hand joints (R/L)   : faint skeletons (optional, --joints)

Usage:
    cd /home/robin/a_workspace/condex && source venv/bin/activate
    python scripts/visualize_dataloader_sample.py --idx 0
    python scripts/visualize_dataloader_sample.py --file-name s01_box_grab_01_00300 --normals
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.dataset import ArcticContractionDataset

FINGER_NAMES  = ["index", "middle", "ring", "little", "thumb"]
FINGER_COLORS = ["#4A90D9", "#27AE60", "#E67E22", "#9B59B6", "#E74C3C"]


def find_index_by_name(ds: ArcticContractionDataset, file_name: str) -> int:
    import json
    for i, (result_path, _) in enumerate(ds.samples):
        try:
            if json.load(open(result_path)).get("file_name") == file_name:
                return i
        except Exception:
            continue
    raise SystemExit(f"file_name {file_name!r} not found among {len(ds)} samples")


def build_figure(sample: dict, file_name: str, show_normals: bool, show_joints: bool):
    from plotly import graph_objects as go

    patch_pos = sample["patch_positions"].numpy()      # [P,3]
    patch_nor = sample["patch_normals"].numpy()         # [P,3]
    part_id   = sample["patch_part_id"].numpy()         # [P]
    contacts  = sample["contact_positions"].numpy()     # [10,3]
    finger_id = sample["finger_ids"].numpy()            # [10]
    side      = sample["finger_side"].numpy()           # [10] 0=R 1=L
    anchors   = sample["human_anchor"].numpy()          # [M,3] padded
    mask      = sample["anchor_mask"].numpy()           # [M]
    anchors   = anchors[mask]                           # valid only

    traces = []

    # ── object patches, split by movable/static ──
    mov = part_id == 0
    traces.append(go.Scatter3d(
        x=patch_pos[mov, 0], y=patch_pos[mov, 1], z=patch_pos[mov, 2],
        mode="markers", marker=dict(size=3, color="#E74C3C", opacity=0.7),
        name=f"patch movable ({mov.sum()})",
    ))
    traces.append(go.Scatter3d(
        x=patch_pos[~mov, 0], y=patch_pos[~mov, 1], z=patch_pos[~mov, 2],
        mode="markers", marker=dict(size=3, color="#888888", opacity=0.5),
        name=f"patch static ({(~mov).sum()})",
    ))

    # ── patch normals (optional) ──
    if show_normals:
        L = 0.02  # 2 cm
        xs, ys, zs = [], [], []
        for p, n in zip(patch_pos, patch_nor):
            q = p + L * n
            xs += [p[0], q[0], None]; ys += [p[1], q[1], None]; zs += [p[2], q[2], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgba(120,200,255,0.4)", width=1),
            name="patch normals", hoverinfo="skip",
        ))

    # ── contacts (10 fingers) ──
    for i in range(contacts.shape[0]):
        fname = FINGER_NAMES[int(finger_id[i])]
        is_left = bool(side[i])
        ring_color = "#FF8C00" if is_left else "#00D4FF"
        suffix = " (L)" if is_left else " (R)"
        p = contacts[i]
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
            marker=dict(size=9, color=FINGER_COLORS[int(finger_id[i])],
                        symbol="circle", line=dict(width=2, color=ring_color)),
            name=f"{fname}{suffix}",
            hovertemplate=f"{fname}{suffix}<br>(%{{x:.3f}},%{{y:.3f}},%{{z:.3f}})<extra></extra>",
        ))

    # ── anchors ──
    if len(anchors):
        traces.append(go.Scatter3d(
            x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2], mode="markers",
            marker=dict(size=16, color="#00FF88", symbol="diamond",
                        line=dict(width=2, color="white")),
            name=f"human_anchor ({len(anchors)})",
            hovertemplate="anchor %{pointNumber}<br>(%{x:.3f},%{y:.3f},%{z:.3f})<extra></extra>",
        ))
        # anchor → nearest contact
        for a in anchors:
            j = int(np.argmin(np.linalg.norm(contacts - a, axis=1)))
            n = contacts[j]
            traces.append(go.Scatter3d(
                x=[a[0], n[0]], y=[a[1], n[1]], z=[a[2], n[2]], mode="lines",
                line=dict(color="rgba(255,230,80,0.7)", width=2, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    # ── hand joints (optional) ──
    if show_joints:
        for key, col, nm in [("hand_joints", "#00FFFF", "R-joints"),
                             ("hand_joints_left", "#FFAA00", "L-joints")]:
            j = sample[key].numpy()
            traces.append(go.Scatter3d(
                x=j[:, 0], y=j[:, 1], z=j[:, 2], mode="markers",
                marker=dict(size=2, color=col, opacity=0.6),
                name=nm,
            ))

    fig = go.Figure(data=traces)
    mind = (np.min([np.linalg.norm(contacts - a, axis=1).min() for a in anchors]) * 1000
            if len(anchors) else float("nan"))
    fig.update_layout(
        paper_bgcolor="rgb(15,15,25)", font=dict(color="white", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        width=1100, height=900,
        title=dict(text=(f"DATALOADER OUTPUT (canonical frame) | {file_name} | "
                         f"patches={len(patch_pos)} movable={mov.sum()} | "
                         f"anchors={len(anchors)} | "
                         f"min_dist R={sample['min_dist_right']*1000:.0f}mm "
                         f"L={sample['min_dist_left']*1000:.0f}mm"), font=dict(size=12)),
        scene=dict(
            bgcolor="rgb(20,20,30)", aspectmode="data",
            xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"),
        ),
    )
    return fig


def main():
    p = argparse.ArgumentParser(description="Visualize ArcticContractionDataset.__getitem__ output.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--idx", type=int, help="dataset index")
    g.add_argument("--file-name", type=str, help="ARCTIC file_name, e.g. s01_box_grab_01_00300")
    p.add_argument("--robot", type=str, default="Robotiq")
    p.add_argument("--gate-max-dist", type=float, default=None)
    p.add_argument("--anchor-dir", type=str, default=None,
                   help="re-contraction anchor cache (recontract_arctic.py output)")
    p.add_argument("--normals", action="store_true", help="draw patch normals")
    p.add_argument("--joints", action="store_true", help="draw hand joints")
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    ds = ArcticContractionDataset(robot=args.robot, gate_max_dist=args.gate_max_dist,
                                  anchor_dir=args.anchor_dir)
    print(f"[dataset] {len(ds)} samples")

    if args.file_name:
        idx = find_index_by_name(ds, args.file_name)
    else:
        idx = args.idx
    result_path, _ = ds.samples[idx]
    import json
    file_name = json.load(open(result_path))["file_name"]
    print(f"[idx {idx}] {file_name}")

    sample = ds[idx]
    fig = build_figure(sample, file_name, args.normals, args.joints)

    out = args.save or str(REPO_ROOT / "md_claude" / "arctic" / f"loader_{file_name}.html")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
