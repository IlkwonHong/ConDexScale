#!/usr/bin/env python3
"""Validate surface-constrained contact math BEFORE sim/RL integration.

Given a recorded object pose trajectory + the box part meshes, this checks that
a contact point parameterized on the object surface:
  - rides the moving/articulating box surface correctly across frames,
  - slides tangentially when we apply a (Δu, Δv) tangential command,
  - yields a sensible inward normal for press-depth application.

No Genesis, no policy. Pure geometry check with a plotly slider overlay so we
can eyeball it (lesson 07-07: verify the math visually before scaling).

Surface model per part: sample surface points + vertex normals from the mesh
(canonical/part-local frame), transform by that part's per-frame pose. Project
a query to nearest surface sample; normal = that sample's normal; tangent basis
= any 2 vecs ⊥ normal. Slide = move along tangent, re-project.

Usage:
  ./venv/bin/python scripts/test_surface_projection.py
"""

import os.path as osp
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
BOX = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets", "arctic", "box")
ROLL = "/tmp/claude-1000/-home-robin-a-workspace-condex/a6ff5b19-6141-4d72-8c04-9de596fc64d2/scratchpad/roll_kin_true.npy"


def part_cloud(mesh, n=1500):
    """Canonical surface samples + OUTWARD normals from a mesh.

    trimesh face_normals are winding-dependent and were only ~56% outward on
    the box parts (2026-07-07 check) -> fix winding for the watertight mesh so
    press-inward (-normal) is correct. Fallback: flip toward-center normals.
    """
    mesh = mesh.copy()
    mesh.fix_normals()  # consistent outward winding for watertight mesh
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    normals = np.asarray(mesh.face_normals[fidx], np.float64)
    pts = np.asarray(pts, np.float64)
    # safety net: any still-inward normal (n . (p - centroid) < 0) gets flipped
    centroid = np.asarray(mesh.vertices).mean(0)
    inward = np.sum(normals * (pts - centroid), axis=1) < 0
    normals[inward] *= -1.0
    return pts, normals


def pose_apply(pts, normals, pos, quat_wxyz, arti, is_top):
    """Transform canonical part cloud to world at one frame."""
    p = pts.copy(); nrm = normals.copy()
    if is_top:  # articulation about canonical -Z (box.urdf revolute axis 0,0,-1)
        Ra = Rotation.from_rotvec([0, 0, -float(arti)])
        p = Ra.apply(p); nrm = Ra.apply(nrm)
    R = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    return R.apply(p) + np.asarray(pos), R.apply(nrm)


def nearest(query, cloud, normals):
    """Nearest surface sample to query -> (point, unit normal)."""
    d = np.linalg.norm(cloud - query, axis=1)
    j = int(np.argmin(d))
    n = normals[j] / (np.linalg.norm(normals[j]) + 1e-9)
    return cloud[j], n


def tangent_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = a - n * (a @ n); t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(n, t1)
    return t1, t2


def main():
    import plotly.graph_objects as go
    meshes = {"top": trimesh.load_mesh(osp.join(BOX, "top_watertight_tiny.obj"), process=False),
              "bottom": trimesh.load_mesh(osp.join(BOX, "bottom_watertight_tiny.obj"), process=False)}
    clouds = {k: part_cloud(m) for k, m in meshes.items()}

    d = np.load(ROLL, allow_pickle=True).item()
    obj = d["obj_state"][:, 0, :]                     # (T,8) pos+quat+arti
    T = len(obj)

    # Two test contacts: one on 'bottom' (body), one on 'top' (lid), each with a
    # canonical anchor point on that part; we command a steady tangential slide.
    # canonical anchor = a fixed surface sample index (rides the part rigidly at Δ=0).
    slide_speed = 0.03  # m of tangential command per frame (test large slide)
    press_depth = 0.005
    radius = 0.015

    # init canonical contact = a chosen surface point per part (Δ accumulates)
    init = {"bottom": clouds["bottom"][0][100].copy(),
            "top": clouds["top"][0][100].copy()}
    # accumulated tangential offset in canonical frame (walk the surface)
    c_can = {k: init[k].copy() for k in init}

    stride = 4
    frames_t = list(range(0, T, stride))
    traj = {"bottom": [], "top": []}   # world contact + normal per frame
    box_frames = []

    for t in range(T):
        pos, quat, arti = obj[t, :3], obj[t, 3:7], obj[t, 7]
        world_clouds = {}
        for part, is_top in [("bottom", False), ("top", True)]:
            cw, nw = pose_apply(clouds[part][0], clouds[part][1], pos, quat, arti, is_top)
            world_clouds[part] = (cw, nw)
        # box mesh verts for viz
        if t in frames_t:
            vparts = []
            for part, is_top in [("bottom", False), ("top", True)]:
                vw, _ = pose_apply(np.asarray(meshes[part].vertices), meshes[part].face_normals[:len(meshes[part].vertices)] if False else np.zeros((len(meshes[part].vertices),3)), pos, quat, arti, is_top)
                vparts.append((vw, np.asarray(meshes[part].faces)))
            box_frames.append(vparts)

        for part in ["bottom", "top"]:
            cw, nw = world_clouds[part]
            # current world contact = nearest to the canonical-anchor transformed to world
            # transform canonical anchor to world via same pose:
            ca_w, _ = pose_apply(c_can[part][None], np.zeros((1, 3)), pos, quat, arti, part == "top")
            p_s, n = nearest(ca_w[0], cw, nw)
            # apply tangential slide command, reproject
            t1, t2 = tangent_basis(n)
            slid = p_s + slide_speed * t1            # steady slide along t1
            p_s2, n2 = nearest(slid, cw, nw)
            # update canonical anchor by projecting world slide back to canonical
            R = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
            can = R.inv().apply(p_s2 - np.asarray(pos))
            if part == "top":
                can = Rotation.from_rotvec([0, 0, float(arti)]).apply(can)
            c_can[part] = can
            target = p_s2 - (radius - press_depth) * n2   # press inward
            if t in frames_t:
                traj[part].append((p_s2, n2, target))

    # ---- plotly overlay ----
    fig = go.Figure()
    COL = {"bottom": "#1f77b4", "top": "#d62728"}

    def frame_data(fi, show):
        data = []
        bf = box_frames[fi]
        for (vw, f), c in zip(bf, ["#9a9a9a", "#4c78a8"]):
            data.append(go.Mesh3d(x=vw[:,0], y=vw[:,1], z=vw[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
                                  color=c, opacity=0.25, showlegend=False))
        for part in ["bottom", "top"]:
            p_s, n, target = traj[part][fi]
            data.append(go.Scatter3d(x=[p_s[0]], y=[p_s[1]], z=[p_s[2]], mode="markers",
                        marker=dict(size=7, color=COL[part]), name=f"contact {part}", showlegend=show))
            # normal arrow
            e = p_s + n * 0.04
            data.append(go.Scatter3d(x=[p_s[0], e[0]], y=[p_s[1], e[1]], z=[p_s[2], e[2]],
                        mode="lines", line=dict(color=COL[part], width=5), showlegend=False))
        return data

    fig.add_traces(frame_data(0, True))
    fig.frames = [go.Frame(data=frame_data(fi, False), name=str(t)) for fi, t in enumerate(frames_t)]
    steps = [dict(method="animate", label=str(t),
                  args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for t in frames_t]
    fig.update_layout(title="surface-constrained contact test: contact rides box surface + slides + normal",
                      sliders=[dict(steps=steps, currentvalue=dict(prefix="frame "))],
                      scene=dict(aspectmode="data"), height=850, margin=dict(l=0,r=0,t=40,b=0))
    out = osp.join(REPO, "md_claude", "dexmachina", "surface_projection_test.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")

    # sanity numbers: contact stays near surface? report mean dist of contact to nearest cloud pt
    print(f"frames {len(frames_t)}; slide_speed {slide_speed*100:.1f}cm/frame; press {press_depth*100:.1f}cm")


if __name__ == "__main__":
    main()
