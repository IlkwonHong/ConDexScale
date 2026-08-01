"""Dexonomy FK/접촉 파이프라인 검증.

V1 (--parity): pytorch_kinematics FK vs MuJoCo mj_kinematics (권위 소스) 전 링크 비교.
    합성 규약: world = T(qpos[:3], quat qpos[3:7]) ∘ model_pose  (hand_loader.py 확정).
    가설 판별용으로 "palm offset 제거" 대안 합성의 오차도 함께 출력한다.
V2 (--plausibility): 접촉 통계가 논문 주장(contact-rich, penetration-free)과 부합하는지.
V3 (--vis): 파지를 plotly HTML로 저장해 육안 확인.
"""

import argparse
import random
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_fk import DEFAULT_XML, LINK_NAMES, QPOS_JOINT_ORDER, DexonomyShadowFK  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SUCC_ROOT = REPO_ROOT / "data" / "dexonomy" / "Dexonomy_GRASP_shadow" / "succ_collect"


def sample_grasps(n_per_type: int = 7, seed: int = 0) -> list[dict]:
    """타입별로 파일을 뽑아 [{qpos, scene_path, scene_scale, file}...] 반환."""
    rng = random.Random(seed)
    samples = []
    for tdir in sorted(SUCC_ROOT.iterdir()):
        files = list(tdir.glob("*/floating/*.npy"))
        for f in rng.sample(files, min(n_per_type, len(files))):
            d = np.load(f, allow_pickle=True).item()
            i = rng.randrange(len(d["grasp_qpos"]))
            samples.append({
                "qpos": d["grasp_qpos"][i].astype(np.float64),
                "scene_path": str(d["scene_path"]),
                "scene_scale": float(d["scene_scale"][i]),
                "file": str(f.relative_to(SUCC_ROOT)),
            })
    return samples


def mujoco_link_poses(m: mujoco.MjModel, d: mujoco.MjData, qpos29: np.ndarray):
    """권위 있는 참조: mj_kinematics + hand_loader 합성. {link: (R(3,3), t(3))}"""
    d.qpos[:] = qpos29[7:]
    mujoco.mj_kinematics(m, d)
    h_r = np.zeros(9)
    mujoco.mju_quat2Mat(h_r, qpos29[3:7])
    h_r = h_r.reshape(3, 3)
    h_t = qpos29[:3]
    out = {}
    for link in LINK_NAMES:
        b = m.body(link)
        bid = int(b.id)
        out[link] = (h_r @ d.xmat[bid].reshape(3, 3), h_r @ d.xpos[bid] + h_t)
    return out


def run_parity(n_per_type: int, seed: int) -> bool:
    samples = sample_grasps(n_per_type, seed)
    print(f"V1 parity: {len(samples)} grasps, {len(LINK_NAMES)} links")
    fk = DexonomyShadowFK(dtype=torch.float64)
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    d = mujoco.MjData(m)

    qpos = np.stack([s["qpos"] for s in samples])
    ours = fk.link_poses(qpos)  # {link: (N,4,4)}

    max_pos, max_ang = 0.0, 0.0
    alt_max_pos = 0.0  # 대안 가설(palm offset 제거) 오차
    palm_off = np.array([0.0, 0.0, 0.034])
    for i, s in enumerate(samples):
        ref = mujoco_link_poses(m, d, s["qpos"])
        h_r = ref["rh_palm"][0] @ np.linalg.inv(d.xmat[int(m.body("rh_palm").id)].reshape(3, 3))
        for link in LINK_NAMES:
            R_ref, t_ref = ref[link]
            mat = ours[link][i].numpy()
            perr = float(np.linalg.norm(mat[:3, 3] - t_ref))
            cos = (np.trace(mat[:3, :3] @ R_ref.T) - 1) / 2
            aerr = float(np.arccos(np.clip(cos, -1, 1)))
            max_pos, max_ang = max(max_pos, perr), max(max_ang, aerr)
            # 대안: model pose에서 palm offset을 빼고 합성했다면
            t_alt = t_ref - h_r @ palm_off
            alt_max_pos = max(alt_max_pos, float(np.linalg.norm(mat[:3, 3] - t_alt)))

    print(f"  ours vs mujoco(권위 합성):  max pos err = {max_pos:.3e} m, max rot err = {max_ang:.3e} rad")
    print(f"  ours vs 대안 합성(offset 제거): max pos err = {alt_max_pos:.3e} m  (판별용, 커야 정상)")
    ok = max_pos < 1e-4 and max_ang < 1e-3
    print(f"  수용 기준 (pos<1e-4, rot<1e-3): {'PASS' if ok else 'FAIL'}")
    return ok


def resolve_object(scene_path: str) -> tuple[Path, float]:
    """scene_path -> (unit mesh 경로, bucket_scale)."""
    parts = Path(scene_path).parts  # .../<dataset>/scene_cfg/<objid>/floating/scaleNNN.npy
    dataset = parts[parts.index("scene_cfg") - 1]
    objid = parts[parts.index("scene_cfg") + 1]
    bucket = float(Path(scene_path).stem.replace("scale", "")) / 100.0
    mesh = REPO_ROOT / "data" / "dexonomy" / dataset / "processed_data" / objid / "mesh" / "simplified.obj"
    return mesh, bucket


def run_plausibility(n_grasps: int, seed: int, threshold: float) -> bool:
    from extract_contacts import compute_contacts_for_qpos, load_sdf

    rng = random.Random(seed)
    samples = sample_grasps(n_per_type=max(2, n_grasps // 31 // 3), seed=seed)
    rng.shuffle(samples)
    fk = DexonomyShadowFK()

    n_fingers_hist = np.zeros(7, dtype=np.int64)
    pen_links = 0
    total_links = 0
    scales = []
    done = 0
    sdf_cache = {}
    for s in samples:
        if done >= n_grasps:
            break
        mesh_path, bucket = resolve_object(s["scene_path"])
        if not mesh_path.exists():
            print(f"  [skip] mesh missing: {mesh_path}")
            continue
        if str(mesh_path) not in sdf_cache:
            sdf_cache.clear()  # 샘플은 물체가 계속 바뀌므로 1개만 유지
            sdf_cache[str(mesh_path)] = load_sdf(mesh_path)
        sdf = sdf_cache[str(mesh_path)]
        scale = np.array([bucket * s["scene_scale"]])
        res = compute_contacts_for_qpos(fk, sdf, s["qpos"][None], scale, threshold)
        n_c = int(res["finger_contact"][0, :5].sum())  # palm 제외 손가락 수
        n_fingers_hist[n_c] += 1
        pen_links += int((res["link_min_dist"][0] < -0.003).sum())
        total_links += res["link_min_dist"].shape[1]
        scales.append(scale[0])
        done += 1

    print(f"V2 plausibility: {done} grasps, threshold={threshold*1000:.1f}mm")
    frac2 = n_fingers_hist[2:].sum() / max(1, done)
    print(f"  접촉 손가락 개수 분포 (0..6): {n_fingers_hist.tolist()}")
    print(f"  손가락 >=2개 접촉 비율: {frac2*100:.1f}%  (기준 >95%)")
    pen = pen_links / max(1, total_links)
    print(f"  penetration(<-3mm) 링크 비율: {pen*100:.2f}%  (기준 <2%)")
    print(f"  total_scale 범위: {min(scales):.3f} ~ {max(scales):.3f} m")
    ok = frac2 > 0.95 and pen < 0.02
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def link_visual_meshes(m: mujoco.MjModel) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """body frame 기준 {link: (verts, faces)}. visual mesh geom 우선,
    mesh가 없는 body(thbase/thhub 등)는 primitive를 trimesh로 근사."""
    import trimesh

    out = {}
    for link in LINK_NAMES:
        bid = int(m.body(link).id)
        parts = []
        gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]
        mesh_gids = [g for g in gids if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH
                     and int(m.geom_contype[g]) == 0]
        use = mesh_gids if mesh_gids else gids
        for g in use:
            gtype = int(m.geom_type[g])
            size = m.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                tm = trimesh.Trimesh(vertices=m.mesh_vert[va:va + vn].copy(),
                                     faces=m.mesh_face[fa:fa + fn].copy(), process=False)
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                tm = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                tm = trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                tm = trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                tm = trimesh.creation.box(extents=2 * np.asarray(size[:3], dtype=np.float64))
            else:
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, m.geom_quat[g])
            tm.apply_transform(np.block([[rot.reshape(3, 3), m.geom_pos[g].reshape(3, 1)],
                                         [np.zeros((1, 3)), np.ones((1, 1))]]))
            parts.append(tm)
        if parts:
            merged = trimesh.util.concatenate(parts)
            out[link] = (np.asarray(merged.vertices), np.asarray(merged.faces))
    return out


def run_vis(n: int, seed: int, threshold: float, out_dir: Path):
    import plotly.graph_objects as go
    import trimesh

    from extract_contacts import compute_contacts_for_qpos, load_sdf
    from shadow_fk import FINGER_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_grasps(n_per_type=1, seed=seed)
    random.Random(seed).shuffle(samples)
    fk = DexonomyShadowFK()
    hand_meshes = link_visual_meshes(fk.model)
    made = 0
    for s in samples:
        if made >= n:
            break
        mesh_path, bucket = resolve_object(s["scene_path"])
        if not mesh_path.exists():
            continue
        scale = bucket * s["scene_scale"]
        res = compute_contacts_for_qpos(fk, load_sdf(mesh_path), s["qpos"][None],
                                        np.array([scale]), threshold)
        obj = trimesh.load(mesh_path, force="mesh")
        v = np.asarray(obj.vertices) * scale
        f = np.asarray(obj.faces)
        contact_link = res["link_contact"][0]

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="lightblue", opacity=0.45, name="object"))
        poses = fk.link_poses(s["qpos"][None])
        for li, link in enumerate(LINK_NAMES):
            if link not in hand_meshes:
                continue
            lv, lf = hand_meshes[link]
            mat = poses[link][0].numpy()
            wv = lv @ mat[:3, :3].T + mat[:3, 3]
            dist_mm = res["link_min_dist"][0, li] * 1000
            fig.add_trace(go.Mesh3d(
                x=wv[:, 0], y=wv[:, 1], z=wv[:, 2],
                i=lf[:, 0], j=lf[:, 1], k=lf[:, 2],
                color="crimson" if contact_link[li] else "lightgray", opacity=1.0,
                flatshading=True, name=f"{link} ({dist_mm:.1f}mm)", showlegend=True))
        cp = res["finger_contact_point"][0]
        cn = res["finger_contact_normal"][0]
        for fi, fname in enumerate(FINGER_NAMES):
            if res["finger_contact"][0, fi]:
                p, nv = cp[fi], cn[fi]
                fig.add_trace(go.Scatter3d(x=[p[0], p[0] + nv[0] * 0.02], y=[p[1], p[1] + nv[1] * 0.02],
                                           z=[p[2], p[2] + nv[2] * 0.02], mode="lines+markers",
                                           marker=dict(size=3), name=f"contact:{fname}"))
        name = s["file"].replace("/", "__").replace(".npy", "")
        contacts = [FINGER_NAMES[i] for i in range(6) if res["finger_contact"][0, i]]
        fig.update_layout(title=f"{s['file']} scale={scale:.3f} contacts={contacts}",
                          scene=dict(aspectmode="data"))
        fig.write_html(out_dir / f"{name}.html")
        made += 1
    print(f"V3: {made} visualizations -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--plausibility", type=int, default=0, metavar="N_GRASPS")
    ap.add_argument("--vis", type=int, default=0, metavar="N")
    ap.add_argument("--n-per-type", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "tmp" / "dexonomy_vis"))
    args = ap.parse_args()

    ok = True
    if args.parity:
        ok &= run_parity(args.n_per_type, args.seed)
    if args.plausibility:
        ok &= run_plausibility(args.plausibility, args.seed, args.threshold)
    if args.vis:
        run_vis(args.vis, args.seed, args.threshold, Path(args.out))
    sys.exit(0 if ok else 1)
