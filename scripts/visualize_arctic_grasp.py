"""
Visualize Arctic bimanual grasp and its contracted representation.

Panel A — Original Bimanual Grasp
  - Articulated object mesh (grey, semi-transparent)
  - Right-hand MANO skeleton (cyan)
  - Left-hand MANO skeleton (orange)
  - Per-finger contact markers (both hands, colour-coded)

Panel B — Contracted Grasp (human_anchor)
  - Same articulated object mesh
  - human_anchor points as large diamonds (green)
  - Right finger contacts (cyan dots)
  - Left finger contacts (orange dots)
  - Lines from each anchor to its nearest finger contact

Usage:
    cd /workspace/genhand && source venv/bin/activate

    # By result-JSON path
    python scripts/visualize_arctic_grasp.py \\
        --result data/results_arctic/s01/box_grab_01/Robotiq/mu0.9/4_Robotiq_0_mu_0.9.json \\
        --save vis_arctic_4.html

    # By dataset index (uses ArcticFeed frame_stride=30)
    python scripts/visualize_arctic_grasp.py --idx 4 --save vis_arctic_4.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARCTIC_ROOT = str(REPO_ROOT.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data")

FINGER_NAMES  = ["index", "middle", "ring", "little", "thumb"]
FINGER_COLORS = ["#4A90D9", "#27AE60", "#E67E22", "#9B59B6", "#E74C3C"]


# ── trace builders ────────────────────────────────────────────────────────────

def traces_object_mesh(verts: np.ndarray, faces: np.ndarray,
                       opacity: float = 0.35, color: str = "lightblue",
                       name: str = "object") -> list:
    from plotly import graph_objects as go
    return [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        flatshading=False,
    )]


def traces_hand_mesh(verts: np.ndarray, faces: np.ndarray,
                     color: str, name: str, opacity: float = 0.55) -> list:
    """Render MANO hand as a semi-transparent mesh."""
    from plotly import graph_objects as go
    return [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1),
        flatshading=False,
    )]


def traces_finger_contacts(contacts: dict, suffix: str, base_color: str) -> list:
    """contacts: {finger_name: [x,y,z]}. suffix: '' or ' (L)'."""
    from plotly import graph_objects as go
    traces = []
    for i, fname in enumerate(FINGER_NAMES):
        if fname not in contacts:
            continue
        pos = np.array(contacts[fname])
        # inner dot = finger color, outer ring = base_color
        traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers",
            marker=dict(size=10, color=FINGER_COLORS[i],
                        symbol="circle",
                        line=dict(width=2, color=base_color)),
            name=f"{fname}{suffix}",
            hovertemplate=f"{fname}{suffix}<br>({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})<extra></extra>",
        ))
    return traces


def traces_anchors(anchors: np.ndarray, color: str = "#00FF88") -> list:
    from plotly import graph_objects as go
    return [go.Scatter3d(
        x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
        mode="markers",
        marker=dict(size=16, color=color, symbol="diamond",
                    line=dict(width=2, color="white")),
        name="human_anchor",
        hovertemplate="anchor %{pointNumber}<br>"
                      "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    )]


def traces_wrist_marker(wrist_pos: np.ndarray, color: str, name: str) -> list:
    """Large sphere marker at the wrist for orientation reference."""
    from plotly import graph_objects as go
    return [go.Scatter3d(
        x=[wrist_pos[0]], y=[wrist_pos[1]], z=[wrist_pos[2]],
        mode="markers+text",
        marker=dict(size=8, color=color, symbol="cross",
                    line=dict(width=2, color="white")),
        text=[name], textposition="top center",
        textfont=dict(color="white", size=9),
        name=name,
        hovertemplate=f"{name}<br>({wrist_pos[0]:.3f},{wrist_pos[1]:.3f},{wrist_pos[2]:.3f})<extra></extra>",
    )]


def traces_finger_skeleton(joints: np.ndarray, color: str, name: str) -> list:
    """Draw finger chains for the 5-finger skeleton.

    manopth joint order after reindex:
      0=wrist  1-4=ring  5-8=index  9-12=middle  13-16=little  17-20=thumb
    Chains: wrist → finger_base → tip for each finger.
    """
    from plotly import graph_objects as go
    # Each chain: (wrist, base, mid, dip, tip)
    chains = [
        [0, 5,  6,  7,  8],   # index
        [0, 9,  10, 11, 12],  # middle
        [0, 13, 14, 15, 16],  # little
        [0, 1,  2,  3,  4],   # ring
        [0, 17, 18, 19, 20],  # thumb
    ]
    traces = []
    for chain in chains:
        xs = [joints[i, 0] for i in chain] + [None]
        ys = [joints[i, 1] for i in chain] + [None]
        zs = [joints[i, 2] for i in chain] + [None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=color, width=3),
            showlegend=False, hoverinfo="skip",
        ))
    # Joint dots
    traces.append(go.Scatter3d(
        x=joints[:, 0], y=joints[:, 1], z=joints[:, 2],
        mode="markers",
        marker=dict(size=3, color=color, opacity=0.7),
        name=name, showlegend=True,
    ))
    return traces


def traces_anchor_to_contacts(anchors: np.ndarray, contacts_r: dict,
                               contacts_l: dict) -> list:
    """Dashed lines from each anchor to its nearest finger contact (both hands)."""
    from plotly import graph_objects as go

    all_pos = []
    for d in (contacts_r, contacts_l):
        for pos in d.values():
            all_pos.append(np.array(pos))
    if not all_pos:
        return []
    all_pos = np.stack(all_pos)  # (N, 3)

    traces = []
    for a in anchors:
        dists = np.linalg.norm(all_pos - a, axis=1)
        j = int(np.argmin(dists))
        nearest = all_pos[j]
        traces.append(go.Scatter3d(
            x=[a[0], nearest[0]], y=[a[1], nearest[1]], z=[a[2], nearest[2]],
            mode="lines",
            line=dict(color="rgba(255,230,80,0.7)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    return traces


# ── data loading ──────────────────────────────────────────────────────────────

def load_articulated_mesh(arctic_root: str, obj_name: str,
                          arti_angle: float, global_rot_aa: np.ndarray,
                          trans_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts_world_m, faces) for the articulated object."""
    import trimesh
    from scipy.spatial.transform import Rotation

    data_root = Path(arctic_root) / "data"
    mesh_path  = data_root / "meta" / "object_vtemplates" / obj_name / "mesh.obj"
    parts_path = data_root / "meta" / "object_vtemplates" / obj_name / "parts.json"

    mesh  = trimesh.load_mesh(str(mesh_path), process=False)
    with open(parts_path) as f:
        parts = np.array(json.load(f), dtype=np.int32)

    verts_m = np.array(mesh.vertices, dtype=np.float32) / 1000.0
    R_arti  = Rotation.from_rotvec([0.0, 0.0, -float(arti_angle)]).as_matrix()
    # Arctic articulates parts_ids==1 == (parts.json==0); the movable part is parts==0.
    top = parts == 0
    verts_m = verts_m.copy()
    verts_m[top] = (R_arti @ verts_m[top].T).T

    R_global   = Rotation.from_rotvec(global_rot_aa.astype(np.float64)).as_matrix()
    verts_world = (R_global @ verts_m.T).T + trans_mm.astype(np.float64) / 1000.0
    return verts_world.astype(np.float32), np.array(mesh.faces)


def load_frame_data(arctic_root: str, subject: str, seq_name: str, t: int):
    """Load raw object params and MANO data for a single frame."""
    data_root = Path(arctic_root) / "data" / "raw_seqs"
    obj_data  = np.load(str(data_root / subject / (seq_name + ".object.npy")))
    mano_data = np.load(str(data_root / subject / (seq_name + ".mano.npy")),
                        allow_pickle=True).item()
    return obj_data[t], mano_data


def mano_forward(mano_data: dict, side: str, t: int,
                 mano_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MANO forward pass.

    Returns (verts (778,3), faces (F,3), joints (21,3)) in world frame.
    No wrist correction — stays consistent with the optimization pipeline.
    """
    import torch
    from manopth.manolayer import ManoLayer

    layer = ManoLayer(ncomps=45, side=side, mano_root=mano_root,
                      use_pca=False, flat_hand_mean=False)
    rot   = torch.tensor(mano_data[side]["rot"][t],   dtype=torch.float32).view(1, 3)
    pose  = torch.tensor(mano_data[side]["pose"][t],  dtype=torch.float32).view(1, 45)
    shape = torch.tensor(mano_data[side]["shape"],    dtype=torch.float32).view(1, 10)
    trans = torch.tensor(mano_data[side]["trans"][t], dtype=torch.float32)

    full_pose = torch.cat([rot, pose], dim=1)
    verts, joints, _, _, _, _ = layer(
        th_pose_coeffs=full_pose, th_betas=shape,
        th_trans=trans.view(1, 3), root_palm=False,
    )
    verts_np  = verts.squeeze(0).detach().numpy()
    joints_np = joints.squeeze(0).detach().numpy()
    faces_np  = layer.th_faces.cpu().numpy().astype(np.int32)
    return verts_np, faces_np, joints_np


# ── scene layout helper ───────────────────────────────────────────────────────

def _make_scene_style(camera_preset: str = "top") -> dict:
    """Return a scene layout dict.

    camera_preset:
      "top"    — bird's-eye view (best for bimanual side-grasp; shows XY plane)
      "front"  — looking from +Y toward origin (hands in front, box behind)
      "left"   — looking from -X toward origin (right-hand palm visible)
      "iso"    — default Plotly isometric
    """
    cameras = {
        "top":   dict(eye=dict(x=0, y=0.05, z=2.2),
                      center=dict(x=0, y=0, z=0),
                      up=dict(x=0, y=1, z=0)),
        "front": dict(eye=dict(x=0, y=2.8, z=0.8),
                      center=dict(x=0, y=0, z=0),
                      up=dict(x=0, y=0, z=1)),
        "left":  dict(eye=dict(x=-2.5, y=0.3, z=0.6),
                      center=dict(x=0, y=0, z=0),
                      up=dict(x=0, y=0, z=1)),
        "iso":   dict(eye=dict(x=1.25, y=1.25, z=1.25),
                      center=dict(x=0, y=0, z=0),
                      up=dict(x=0, y=0, z=1)),
    }
    return dict(
        bgcolor="rgb(20,20,30)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text="X", font=dict(color="#ff5555"))),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text="Y", font=dict(color="#55ff55"))),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text="Z↑", font=dict(color="#5599ff"))),
        aspectmode="data",
        camera=cameras.get(camera_preset, cameras["top"]),
    )


_SCENE_STYLE = _make_scene_style("top")


# ── main ──────────────────────────────────────────────────────────────────────

def load_rgb_image(arctic_root: str, subject: str, seq_name: str,
                   t: int, cam_id: int = 0) -> "np.ndarray | None":
    """Load RGB frame. Tries multiple candidate paths."""
    from PIL import Image as _PIL
    data_root = Path(arctic_root) / "data"
    candidates = [
        # standard: images/{subject}/{seq}/{cam}/{t+1:05d}.jpg
        data_root / "images" / subject / seq_name / str(cam_id) / f"{t+1:05d}.jpg",
        # 0-indexed
        data_root / "images" / subject / seq_name / str(cam_id) / f"{t:05d}.jpg",
        # cropped_images fallback
        data_root / "cropped_images" / subject / seq_name / str(cam_id) / f"{t+1:05d}.jpg",
        data_root / "cropped_images" / subject / seq_name / str(cam_id) / f"{t:05d}.jpg",
    ]
    for p in candidates:
        if p.exists():
            try:
                return np.array(_PIL.open(p))
            except Exception:
                continue
    return None


def build_figure(result_path: str, arctic_root: str, mano_root: str,
                 cam_id: int = 0, cam_preset: str = "top"):
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    # ── load result JSON ──
    result = json.load(open(result_path))
    if "error" in result:
        raise ValueError(f"Result JSON has error: {result['error']}")

    file_name    = result["file_name"]       # "s01_box_grab_01_00120"
    obj_name     = result["obj_name"]
    arti_angle   = float(result.get("obj_arti_angle", 0.0))
    obj_center   = np.array(result["obj_center"], dtype=np.float32)  # world frame (3,)

    # human_anchor may be stored as either:
    #   [[x,y,z],[x,y,z]]       shape (K,3)   — new bimanual code
    #   [[[x,y,z],[x,y,z]]]     shape (1,K,3) — legacy wrapper
    ha_raw = result["human_anchor"]
    _ha    = np.array(ha_raw, dtype=np.float32)
    anchors = _ha[0] if _ha.ndim == 3 else _ha   # always (K, 3)

    contacts_r = result.get("human_finger_contacts", {})          # object-centric
    contacts_l = result.get("human_finger_contacts_left", {})

    joints_r_oc = np.array(result.get("hand_joints", []),
                            dtype=np.float32)                      # world frame (21,3)
    joints_l_oc = np.array(result.get("hand_joints_left", []),
                            dtype=np.float32)

    # ── parse file_name → (subject, seq_name, t) ──
    parts    = file_name.split("_")  # s01, box, grab, 01, 00120
    subject  = parts[0]
    t        = int(parts[-1])
    seq_name = "_".join(parts[1:-1])  # "box_grab_01"

    # ── load RGB image ──
    rgb = load_rgb_image(arctic_root, subject, seq_name, t, cam_id)

    # ── load original raw data ──
    obj_params, mano_data = load_frame_data(arctic_root, subject, seq_name, t)
    arti_aa   = obj_params[0]          # arti angle (radians)
    global_rot= obj_params[1:4]        # axis-angle
    trans_mm  = obj_params[4:]         # mm

    # ── build articulated object mesh ──
    verts_w, faces = load_articulated_mesh(
        arctic_root, obj_name, arti_aa, global_rot, trans_mm
    )
    # object-centric (subtract obj_center computed as mean of articulated verts)
    obj_center_computed = verts_w.mean(axis=0)
    verts_oc = verts_w - obj_center_computed

    # ── MANO mesh (same frame as contact zones stored in result JSON) ──
    verts_r_w, faces_r, joints_r_w = mano_forward(mano_data, 'right', t, mano_root)
    verts_l_w, faces_l, joints_l_w = mano_forward(mano_data, 'left',  t, mano_root)
    # Center on object (same centering as initialization() used)
    verts_r_oc  = verts_r_w  - obj_center_computed
    verts_l_oc  = verts_l_w  - obj_center_computed
    joints_r_oc = joints_r_w - obj_center_computed
    joints_l_oc = joints_l_w - obj_center_computed

    # finger contacts are already in object-centric frame from optimization
    # (obj_center used in initialization == obj_pack["centre"] == mean of verts)

    # ── Panel A: original bimanual grasp ──
    tA  = traces_object_mesh(verts_oc, faces, opacity=0.30, color="#89CFF0",
                              name="object")
    tA += traces_hand_mesh(verts_r_oc, faces_r, color="#00BFFF", name="right hand",
                           opacity=0.45)
    tA += traces_hand_mesh(verts_l_oc, faces_l, color="#FF8C00", name="left hand",
                           opacity=0.45)
    # Skeleton overlay: clarifies hand orientation
    tA += traces_finger_skeleton(joints_r_oc, color="#00FFFF", name="R-skeleton")
    tA += traces_finger_skeleton(joints_l_oc, color="#FFAA00", name="L-skeleton")
    # Wrist markers
    tA += traces_wrist_marker(joints_r_oc[0], color="#00FFFF", name="R-wrist")
    tA += traces_wrist_marker(joints_l_oc[0], color="#FFAA00", name="L-wrist")
    tA += traces_finger_contacts(contacts_r, "",     base_color="#00D4FF")
    tA += traces_finger_contacts(contacts_l, " (L)", base_color="#FF8C00")

    # ── Panel B: contracted grasp ──
    tB  = traces_object_mesh(verts_oc, faces, opacity=0.30, color="#89CFF0",
                              name="object")
    tB += traces_anchors(anchors)
    tB += traces_finger_contacts(contacts_r, "",    base_color="#00D4FF")
    tB += traces_finger_contacts(contacts_l, " (L)", base_color="#FF8C00")
    tB += traces_anchor_to_contacts(anchors, contacts_r, contacts_l)

    # ── assemble figure ──
    if rgb is not None:
        specs  = [[{"type": "image"}, {"type": "scene"}, {"type": "scene"}]]
        titles = [
            f"Camera {cam_id}  t={t}",
            f"Panel A — Original Bimanual Grasp",
            f"Panel B — Contracted ({len(anchors)}-anchor)",
        ]
        col_widths = [0.22, 0.39, 0.39]
        fig = make_subplots(
            rows=1, cols=3, specs=specs,
            subplot_titles=titles,
            column_widths=col_widths,
            horizontal_spacing=0.02,
        )
        # downsample image for HTML size (2800×2000 → 700×500)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb).resize((700, 500))
        rgb_small = np.array(pil)
        fig.add_trace(go.Image(z=rgb_small), row=1, col=1)
        for t_ in tA: fig.add_trace(t_, row=1, col=2)
        for t_ in tB: fig.add_trace(t_, row=1, col=3)
        scene_a, scene_b = "scene", "scene2"
    else:
        specs  = [[{"type": "scene"}, {"type": "scene"}]]
        titles = [
            f"Panel A — Original Bimanual Grasp  [{file_name}]",
            f"Panel B — Contracted ({len(anchors)}-anchor)  arti={arti_angle:.3f} rad",
        ]
        fig = make_subplots(
            rows=1, cols=2, specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.02,
        )
        for t_ in tA: fig.add_trace(t_, row=1, col=1)
        for t_ in tB: fig.add_trace(t_, row=1, col=2)
        scene_a, scene_b = "scene", "scene2"

    fig.update_layout(
        paper_bgcolor="rgb(15,15,25)",
        font=dict(color="white", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", font=dict(color="white")),
        width=1900, height=800,
        title=dict(
            text=(f"Arctic Grasp → Contraction  |  "
                  f"obj={obj_name}  subject={subject}  seq={seq_name}  t={t}  "
                  f"anchors={len(anchors)}  arti={arti_angle:.3f} rad"),
            font=dict(size=13),
        ),
        **{scene_a: _make_scene_style(cam_preset),
           scene_b: _make_scene_style(cam_preset)},
    )
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Arctic bimanual grasp and contracted representation."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--result", type=str,
                   help="Direct path to a result JSON file")
    g.add_argument("--idx", type=int,
                   help="Dataset index (uses ArcticFeed with frame_stride=30)")

    p.add_argument("--arctic-root", type=str, default=ARCTIC_ROOT)
    p.add_argument("--results-dir", type=str,
                   default=str(REPO_ROOT / "data" / "results_arctic"))
    p.add_argument("--robot",  type=str, default="Robotiq",
                   choices=["Barrett", "Robotiq"])
    p.add_argument("--mu",     type=float, default=0.9)
    p.add_argument("--cam", type=int, default=0,
                   help="Camera view to show (0-7: 3rd-person, 8: egocentric)")
    p.add_argument("--view", type=str, default="top",
                   choices=["top", "front", "left", "iso"],
                   help="3D camera preset: top=bird-eye(default), front, left, iso")
    p.add_argument("--mano-root", type=str,
                   default=str(REPO_ROOT / "network" / "manopth" / "mano" / "models"))
    p.add_argument("--save",   type=str, default=None,
                   help="Save HTML to this path (default: auto-named in md_claude/arctic/)")
    p.add_argument("--show",   action="store_true",
                   help="Open in browser instead of saving")
    return p.parse_args()


def main():
    args = parse_args()

    if args.result:
        result_path = args.result
    else:
        # Find by index via glob
        import glob
        pattern = f"{args.results_dir}/**/{args.robot}/mu{args.mu:.1f}/*.json"
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            sys.exit(f"No result JSONs found matching: {pattern}")
        if args.idx >= len(files):
            sys.exit(f"idx {args.idx} >= {len(files)} available files")
        result_path = files[args.idx]
        print(f"[auto] result = {result_path}")

    print(f"[loading] {result_path}")
    fig = build_figure(result_path, args.arctic_root, args.mano_root,
                       cam_id=args.cam, cam_preset=args.view)

    if args.show:
        fig.show()
        return

    # Default save path
    if args.save is None:
        stem = Path(result_path).stem           # "4_Robotiq_0_mu_0.9"
        parent_parts = Path(result_path).parts
        # extract subject/seq from path: .../s01/box_grab_01/Robotiq/mu0.9/...
        try:
            ri = list(parent_parts).index("results_arctic")
            subject = parent_parts[ri + 1]
            seq     = parent_parts[ri + 2]
            tag     = f"{subject}_{seq}_{stem}"
        except Exception:
            tag = stem
        out_dir  = REPO_ROOT / "md_claude" / "arctic"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / f"vis_{tag}.html")
    else:
        save_path = args.save

    fig.write_html(save_path)
    print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
