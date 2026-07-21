#!/usr/bin/env python3
"""Overlay G1-policy vs kinematic(prior) rollouts in one 3D plotly (slider).

Shows per frame, in world frame:
  - box (top+bottom parts) posed by each run's object state (semi-transparent):
    policy run = orange, prior run = blue
  - 4 sphere contact points per run (solid markers, same colors)
  - dashed lines connecting each policy sphere to its prior counterpart
    (= the residual displacement the policy applied)

Confirms visually how far the policy moves the contacts off the human-anchor prior.

Inputs: two rollout npy (roll_g1.npy, roll_kin.npy) each with
  obj_state (T,E,8)=pos3+quat4+arti1, kpt_left (T,E,2,3), kpt_right (T,E,2,3).
Uses env index 0.

Usage:
  ./venv/bin/python scripts/visualize_rollout_overlay.py \
    --g1 <scratch>/roll_g1.npy --kin <scratch>/roll_kin.npy
"""

import argparse
import os.path as osp

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
BOX = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets", "arctic", "box")


def load_rollout(path, env=0):
    d = np.load(path, allow_pickle=True).item()
    obj = d["obj_state"][:, env, :]                       # (T,8)
    spheres = np.concatenate([d["kpt_left"][:, env], d["kpt_right"][:, env]], axis=1)  # (T,4,3)
    return dict(pos=obj[:, :3], quat=obj[:, 3:7], arti=obj[:, 7], spheres=spheres)


def box_verts(part_mesh, pos, quat_wxyz, arti, is_top):
    """World verts of a box part at one frame. Top part rotates by arti about -Z."""
    v = np.asarray(part_mesh.vertices, dtype=np.float64)
    if is_top:  # articulation about canonical -Z (box.urdf revolute axis 0,0,-1)
        v = Rotation.from_rotvec([0, 0, -float(arti)]).apply(v)
    R = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    return R.apply(v) + np.asarray(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", required=True)
    ap.add_argument("--kin", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default=osp.join(REPO, "md_claude", "dexmachina",
                                              "rollout_overlay_g1_vs_prior.html"))
    args = ap.parse_args()
    import plotly.graph_objects as go

    g1 = load_rollout(args.g1)
    kin = load_rollout(args.kin)
    T = len(g1["pos"])
    frames_t = list(range(0, T, args.stride))

    top = trimesh.load_mesh(osp.join(BOX, "top_watertight_tiny.obj"), process=False)
    bot = trimesh.load_mesh(osp.join(BOX, "bottom_watertight_tiny.obj"), process=False)

    C_G1, C_KIN = "#ff7f0e", "#1f77b4"  # policy orange, prior blue

    def box_trace(run, t, color, name, show):
        tris = []
        for mesh, is_top in [(bot, False), (top, True)]:
            v = box_verts(mesh, run["pos"][t], run["quat"][t], run["arti"][t], is_top)
            f = np.asarray(mesh.faces)
            tris.append((v, f))
        # merge two parts into one mesh3d
        vv = np.vstack([tris[0][0], tris[1][0]])
        f0 = tris[0][1]; f1 = tris[1][1] + len(tris[0][0])
        f = np.vstack([f0, f1])
        return go.Mesh3d(x=vv[:, 0], y=vv[:, 1], z=vv[:, 2],
                         i=f[:, 0], j=f[:, 1], k=f[:, 2],
                         color=color, opacity=0.25, name=name, showlegend=show)

    def sphere_trace(run, t, color, name, show):
        s = run["spheres"][t]
        return go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers",
                            marker=dict(size=7, color=color),
                            name=name, showlegend=show)

    def link_trace(t, show):
        xs, ys, zs = [], [], []
        for i in range(4):
            a, b = g1["spheres"][t, i], kin["spheres"][t, i]
            xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                            line=dict(color="#888", width=3, dash="dash"),
                            name="residual", showlegend=show)

    def frame_data(t, show=False):
        return [box_trace(kin, t, C_KIN, "prior box", show),
                box_trace(g1, t, C_G1, "policy box", show),
                sphere_trace(kin, t, C_KIN, "prior spheres", show),
                sphere_trace(g1, t, C_G1, "policy spheres", show),
                link_trace(t, show)]

    fig = go.Figure(data=frame_data(frames_t[0], show=True))
    fig.frames = [go.Frame(data=frame_data(t), name=str(t)) for t in frames_t]
    steps = [dict(method="animate", label=str(t),
                  args=[[str(t)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for t in frames_t]
    # per-frame residual magnitude annotation
    resid = np.linalg.norm(g1["spheres"] - kin["spheres"], axis=-1).mean(axis=1) * 100
    fig.update_layout(
        title=f"G1 policy(orange) vs kinematic prior(blue) — box_use_01. "
              f"mean sphere이탈 {resid.mean():.1f}cm",
        sliders=[dict(steps=steps, currentvalue=dict(prefix="frame "))],
        scene=dict(aspectmode="data"),
        height=850, margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"wrote {args.out}")
    print(f"sphere 이탈(정책 vs prior): mean {resid.mean():.1f}cm  max {resid.max():.1f}cm")
    print(f"물체 위치 차이(정책 vs prior box): "
          f"mean {np.linalg.norm(g1['pos']-kin['pos'],axis=1).mean()*100:.1f}cm")


if __name__ == "__main__":
    main()
