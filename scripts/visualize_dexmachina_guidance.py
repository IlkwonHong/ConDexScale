#!/usr/bin/env python3
"""Visualize exported contraction anchor trajectories (DexMachina guidance).

Interactive plotly HTML with a time slider. Per frame shows:
  - object part point clouds (posed by obj_pos/obj_quat/obj_arti, top part blue / bottom grey)
  - human contact points of that frame (small dark dots)
  - contraction anchors (big markers, one color per arm: L1 L2 R1 R2)
  - impedance reference targets (x markers: pressed 5mm inside during contact,
    hovering 2cm outside otherwise) = what the spheres actually track
  - full anchor trails (faint lines)

Usage:
  ./venv/bin/python scripts/visualize_dexmachina_guidance.py --obj box --hand point4_hand
"""

import argparse
import os
import os.path as osp

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEXM_ASSETS = osp.join(REPO_ROOT, "extlib", "dexmachina", "dexmachina", "assets")

ARM_COLORS = {"L1": "#d62728", "L2": "#ff9896", "R1": "#1f77b4", "R2": "#aec7e8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="box")
    ap.add_argument("--subject", default="s01")
    ap.add_argument("--clip", default="01")
    ap.add_argument("--hand", default="point4_hand")
    ap.add_argument("--frame-start", type=int, default=30)
    ap.add_argument("--frame-end", type=int, default=230)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--n-obj-pts", type=int, default=700)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import plotly.graph_objects as go

    stem = f"{args.obj}_use_{args.clip}"
    k = int(args.hand.replace("point", "").replace("_hand", "")) // 2

    proc = np.load(osp.join(DEXM_ASSETS, "arctic", "processed", args.subject,
                            stem + ".npy"), allow_pickle=True).item()
    params, wc = proc["params"], proc["world_coord"]
    obj_pos = np.asarray(params["obj_trans"], dtype=np.float64)
    obj_arti = np.asarray(params["obj_arti"], dtype=np.float64)
    R_obj = Rotation.from_quat(np.asarray(params["obj_quat"])[:, [1, 2, 3, 0]])

    logz = np.load(osp.join(REPO_ROOT, "data", "arctic_cache",
                            "dexmachina_guidance_logs",
                            f"{args.hand}_{args.subject}_{stem}.npz"))
    ret = np.load(osp.join(DEXM_ASSETS, "retargeted", args.hand, args.subject,
                           stem + "_vector_genesis.npy"), allow_pickle=True).item()

    anchors = {s: logz[f"anchors_{s}"] for s in ["left", "right"]}   # (T,k,3)
    valid = {s: logz[f"valid_{s}"] for s in ["left", "right"]}
    refs = {s: ret["retarget_data"][s]["kpt_pos"] for s in ["left", "right"]}

    # object part point clouds (canonical)
    pts_can = {}
    for pid, name in [(1, "top"), (2, "bottom")]:
        m = trimesh.load_mesh(osp.join(DEXM_ASSETS, "arctic", args.obj,
                                       f"{name}_watertight_tiny.obj"), process=False)
        v = np.asarray(m.vertices)
        sel = np.random.RandomState(0).choice(len(v), min(args.n_obj_pts, len(v)),
                                              replace=False)
        pts_can[pid] = v[sel]

    frames_t = list(range(args.frame_start, args.frame_end, args.stride))
    fig = go.Figure()
    n_traces_per_frame = []
    all_frames = []

    # static anchor trails (drawn once, always visible)
    trail_traces = []
    for side in ["left", "right"]:
        prefix = "L" if side == "left" else "R"
        for i in range(k):
            name = f"{prefix}{i+1}"
            a = anchors[side][args.frame_start:args.frame_end, i]
            trail_traces.append(go.Scatter3d(
                x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines",
                line=dict(color=ARM_COLORS[name], width=2), opacity=0.3,
                name=f"trail {name}", showlegend=False))

    def frame_traces(t):
        traces = []
        # object points
        for pid, color in [(1, "#4c78a8"), (2, "#9a9a9a")]:
            p = pts_can[pid].copy()
            if pid == 1:
                p = Rotation.from_rotvec([0, 0, -obj_arti[t]]).apply(p)
            p = R_obj[t].apply(p) + obj_pos[t]
            traces.append(go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                marker=dict(size=1.5, color=color), opacity=0.5,
                name=f"obj part{pid}", showlegend=(t == frames_t[0])))
        # human contacts of this frame
        for side in ["left", "right"]:
            c = np.asarray(wc[f"contacts.{side}"])[t]
            m = np.asarray(wc[f"valid_contacts.{side}"])[t].astype(bool)
            c = c[m, :3] if m.any() else np.zeros((0, 3))
            traces.append(go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
                marker=dict(size=2.5, color="#222222"), opacity=0.8,
                name=f"human contacts {side}", showlegend=(t == frames_t[0])))
        # anchors + refs
        for side in ["left", "right"]:
            prefix = "L" if side == "left" else "R"
            for i in range(k):
                name = f"{prefix}{i+1}"
                a = anchors[side][t, i]
                r = refs[side][t, i]
                in_contact = bool(valid[side][t, i])
                traces.append(go.Scatter3d(
                    x=[a[0]], y=[a[1]], z=[a[2]], mode="markers",
                    marker=dict(size=7, color=ARM_COLORS[name],
                                symbol="circle" if in_contact else "circle-open"),
                    name=f"anchor {name}" + (" (contact)" if in_contact else ""),
                    showlegend=(t == frames_t[0])))
                traces.append(go.Scatter3d(
                    x=[r[0]], y=[r[1]], z=[r[2]], mode="markers",
                    marker=dict(size=4, color=ARM_COLORS[name], symbol="x"),
                    name=f"ref {name}", showlegend=(t == frames_t[0])))
        return traces

    first = frame_traces(frames_t[0])
    for tr in trail_traces + first:
        fig.add_trace(tr)
    n_static = len(trail_traces)

    plot_frames = []
    for t in frames_t:
        plot_frames.append(go.Frame(
            data=frame_traces(t),
            traces=list(range(n_static, n_static + len(first))),
            name=str(t)))
    fig.frames = plot_frames

    steps = [dict(method="animate", label=str(t),
                  args=[[str(t)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for t in frames_t]
    fig.update_layout(
        title=f"{args.hand} contraction guidance — {args.subject}/{stem} "
              f"(frames {args.frame_start}-{args.frame_end})",
        sliders=[dict(steps=steps, currentvalue=dict(prefix="frame "))],
        scene=dict(aspectmode="data"),
        height=850, margin=dict(l=0, r=0, t=40, b=0),
    )

    out = args.out or osp.join(REPO_ROOT, "md_claude", "dexmachina",
                               f"guidance_{args.hand}_{args.subject}_{stem}.html")
    os.makedirs(osp.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}  ({osp.getsize(out)/1e6:.1f} MB, {len(frames_t)} frames)")


if __name__ == "__main__":
    main()
