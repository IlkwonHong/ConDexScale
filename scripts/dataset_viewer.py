#!/usr/bin/env python3
"""Interactive Dash viewer for the ARCTIC contraction dataset.

Browse ALL samples in the rich style — [ camera photo | 3D: object mesh + both
hand meshes + finger contacts + GT(green ◆) / PRED(red ✕) ] — without
pre-generating an HTML per view. Filter by object, scrub a slider, toggle PRED.

Run:
  cd /home/robin/a_workspace/condex
  ./venv/bin/python scripts/dataset_viewer.py \
      --results-dir data/results_arctic --graph-dir data/arctic_cache/sample_graphs \
      --ckpt checkpoints/best.pt --gate-max-dist 0.03 --port 8050
  # held-out s05:
  ./venv/bin/python scripts/dataset_viewer.py \
      --results-dir data/results_arctic_val --graph-dir data/arctic_cache/sample_graphs_val ...
Then open http://localhost:8050
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.dataset import ArcticContractionDataset, ARCTIC_RESULTS_DIR, ARCTIC_GRAPH_DIR
from training.model import GraspContractionNet
from scripts.visualize_arctic_grasp import (
    traces_object_mesh, traces_hand_mesh, load_frame_data, load_rgb_image, _make_scene_style,
)

ARCTIC_ROOT = str(REPO.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data")
MANO_ROOT = str(REPO / "network" / "manopth" / "mano" / "models")
MODEL_KEYS = ["patch_positions", "patch_normals", "patch_curvature",
              "contact_positions", "contact_normals", "finger_ids"]
FINGERS = ["index", "middle", "ring", "little", "thumb"]

# ── caches (built once, reused across samples for snappy navigation) ──────────
_mano = {}
_objc = {}


def _mano_fwd(mano_data, side, t):
    if not _mano:
        from manopth.manolayer import ManoLayer
        for sd in ("right", "left"):
            _mano[sd] = ManoLayer(ncomps=45, side=sd, mano_root=MANO_ROOT,
                                  use_pca=False, flat_hand_mean=False)
    layer = _mano[side]
    rot = torch.tensor(mano_data[side]["rot"][t], dtype=torch.float32).view(1, 3)
    pose = torch.tensor(mano_data[side]["pose"][t], dtype=torch.float32).view(1, 45)
    shape = torch.tensor(mano_data[side]["shape"], dtype=torch.float32).view(1, 10)
    trans = torch.tensor(mano_data[side]["trans"][t], dtype=torch.float32).view(1, 3)
    v, _, _, _, _, _ = layer(th_pose_coeffs=torch.cat([rot, pose], 1), th_betas=shape,
                             th_trans=trans, root_palm=False)
    return v.squeeze(0).detach().numpy(), layer.th_faces.cpu().numpy().astype(np.int32)


def _articulate(obj_name, arti, grot, trans_mm):
    if obj_name not in _objc:
        import trimesh
        d = Path(ARCTIC_ROOT) / "data" / "meta" / "object_vtemplates" / obj_name
        mesh = trimesh.load_mesh(str(d / "mesh.obj"), process=False)
        parts = np.array(json.load(open(d / "parts.json")), np.int32)
        _objc[obj_name] = (np.array(mesh.vertices, np.float32) / 1000.0,
                           np.array(mesh.faces), parts)
    vm, faces, parts = _objc[obj_name]
    vm = vm.copy()
    vm[parts == 0] = (Rotation.from_rotvec([0, 0, -float(arti)]).as_matrix() @ vm[parts == 0].T).T
    vw = (Rotation.from_rotvec(np.asarray(grot, float)).as_matrix() @ vm.T).T + np.asarray(trans_mm, float) / 1000.0
    return vw.astype(np.float32), faces


def build_fig(idx, ds, model, device, show_pred):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    s = ds[idx]
    result = json.load(open(ds.samples[idx][0]))
    graph = json.load(open(ds.samples[idx][1]))
    fn = result["file_name"]
    Rg = np.array(graph["obj_rotation"], np.float32)
    oc_gen = np.array(result["obj_center"], np.float32)
    m = int(s["anchor_mask"].sum())
    gt = s["human_anchor"].numpy()[:m]
    gt_w = (Rg @ gt.T).T + oc_gen
    cr_w = np.array([result["human_finger_contacts"][f] for f in FINGERS], np.float32) + oc_gen
    cl_w = np.array([result["human_finger_contacts_left"][f] for f in FINGERS], np.float32) + oc_gen

    p = fn.split("_"); subj = p[0]; t = int(p[-1]); seq = "_".join(p[1:-1]); obj = seq.split("_")[0]
    obj_params, mano_data = load_frame_data(ARCTIC_ROOT, subj, seq, t)
    verts_w, faces = _articulate(obj, obj_params[0], obj_params[1:4], obj_params[4:])
    oc = verts_w.mean(0)
    vr, fr = _mano_fwd(mano_data, "right", t)
    vl, fl = _mano_fwd(mano_data, "left", t)

    pred_w, dists = None, None
    if show_pred and model is not None:
        batch = {k: s[k].unsqueeze(0).to(device) for k in MODEL_KEYS}
        with torch.no_grad():
            pred = model(batch)["anchor_positions"][0].cpu().numpy()
        cost = np.linalg.norm(gt[:, None] - pred[None], axis=2)
        r, c = linear_sum_assignment(cost)
        pred_w = (Rg @ pred[c].T).T + oc_gen
        dists = cost[r, c]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "image"}, {"type": "scene"}]],
                        column_widths=[0.32, 0.68], horizontal_spacing=0.01)
    rgb = load_rgb_image(ARCTIC_ROOT, subj, seq, t, 1) or load_rgb_image(ARCTIC_ROOT, subj, seq, t, 0)
    if rgb is not None:
        fig.add_trace(go.Image(z=rgb[::3, ::3]), row=1, col=1)
    tr = (traces_object_mesh(verts_w - oc, faces, opacity=0.25, color="#89CFF0", name="obj")
          + traces_hand_mesh(vr - oc, fr, color="#00BFFF", name="R", opacity=0.4)
          + traces_hand_mesh(vl - oc, fl, color="#FF8C00", name="L", opacity=0.4))
    for c2, col in [(cr_w, "#00D4FF"), (cl_w, "#FF8C00")]:
        c2 = c2 - oc
        tr.append(go.Scatter3d(x=c2[:, 0], y=c2[:, 1], z=c2[:, 2], mode="markers",
                               marker=dict(size=5, color=col, line=dict(width=1, color="white")),
                               showlegend=False, hoverinfo="skip"))
    g = gt_w - oc
    tr.append(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="markers",
                           marker=dict(size=9, color="#00FF88", symbol="diamond",
                                       line=dict(width=1, color="white")), name="GT"))
    if pred_w is not None:
        pp = pred_w - oc
        for a, b in zip(g, pp):
            tr.append(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                                   line=dict(color="rgba(255,230,80,0.9)", width=3),
                                   showlegend=False, hoverinfo="skip"))
        tr.append(go.Scatter3d(x=pp[:, 0], y=pp[:, 1], z=pp[:, 2], mode="markers",
                               marker=dict(size=9, color="#FF3B3B", symbol="x"), name="PRED"))
    for x in tr:
        fig.add_trace(x, row=1, col=2)
    fig.update_layout(scene=_make_scene_style("top"), paper_bgcolor="rgb(15,15,25)",
                      font=dict(color="white"), height=620, margin=dict(l=0, r=0, t=30, b=0),
                      showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0.3)"))
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_yaxes(visible=False, row=1, col=1)
    err = f"  err={dists.mean()*100:.1f}cm" if dists is not None else ""
    return fig, f"{fn}   K={m}   photo={'Y' if rgb is not None else 'N'}{err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ARCTIC_RESULTS_DIR))
    ap.add_argument("--graph-dir", default=str(ARCTIC_GRAPH_DIR))
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--gate-max-dist", type=float, default=0.03)
    ap.add_argument("--num-anchors", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    import dash
    from dash import dcc, html, Input, Output, State, ctx

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = ArcticContractionDataset(robot="Robotiq", gate_max_dist=args.gate_max_dist,
                                  max_anchors=args.num_anchors,
                                  results_dir=args.results_dir, graph_dir=args.graph_dir)
    # sample index -> (file_name, obj)
    meta = []
    for i in range(len(ds)):
        fn = ds.samples[i][1].stem.replace("_graph", "")
        meta.append((i, fn, fn.split("_")[1]))
    objs = sorted(set(o for _, _, o in meta))
    print(f"[viewer] {len(ds)} samples, objects={objs}")

    model = None
    if Path(args.ckpt).exists():
        model = GraspContractionNet(d_model=256, num_anchors=args.num_anchors).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"]); model.eval()

    app = dash.Dash(__name__)
    app.layout = html.Div(style={"backgroundColor": "rgb(15,15,25)", "color": "white",
                                 "fontFamily": "monospace", "padding": "8px"}, children=[
        html.Div([
            html.Span("object: "),
            dcc.Dropdown(id="obj", options=[{"label": "all", "value": "all"}]
                         + [{"label": o, "value": o} for o in objs], value="all",
                         style={"width": "200px", "color": "black", "display": "inline-block"}),
            html.Span("  ", style={"margin": "0 10px"}),
            dcc.Checklist(id="pred", options=[{"label": " show PRED", "value": "p"}],
                          value=["p"] if model is not None else [], style={"display": "inline-block"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        html.Div(id="info", style={"margin": "6px 0", "fontSize": "13px"}),
        html.Div([
            html.Button("◀ prev", id="prev", n_clicks=0,
                        style={"marginRight": "8px", "padding": "4px 14px", "fontFamily": "monospace"}),
            html.Button("next ▶", id="next", n_clicks=0,
                        style={"padding": "4px 14px", "fontFamily": "monospace"}),
        ], style={"margin": "6px 0"}),
        dcc.Slider(id="idx", min=0, max=1, step=1, value=0, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="fig", style={"height": "640px"}),
    ])

    @app.callback(Output("idx", "value"), Output("idx", "max"),
                  Input("obj", "value"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
                  State("idx", "value"))
    def _nav(obj, prev, nxt, cur):
        flt = [i for i, _, o in meta if obj == "all" or o == obj]
        mx = max(len(flt) - 1, 0)
        cur = cur or 0
        t = ctx.triggered_id
        if t == "prev":
            cur = max(0, cur - 1)
        elif t == "next":
            cur = min(mx, cur + 1)
        else:                       # obj change or initial load -> reset to first
            cur = 0
        return min(cur, mx), mx

    @app.callback(Output("fig", "figure"), Output("info", "children"),
                  Input("idx", "value"), Input("obj", "value"), Input("pred", "value"))
    def _update(pos, obj, pred):
        flt = [i for i, _, o in meta if obj == "all" or o == obj]
        if not flt:
            return {}, "no samples"
        idx = flt[min(pos or 0, len(flt) - 1)]
        fig, info = build_fig(idx, ds, model, device, "p" in (pred or []))
        return fig, f"[{(pos or 0)+1}/{len(flt)}]  {info}"

    print(f"[viewer] open http://localhost:{args.port}")
    app.run(debug=False, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
