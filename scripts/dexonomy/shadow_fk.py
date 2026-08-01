"""Dexonomy Shadow hand FK layer.

Dexonomy의 grasp qpos (29,) = [trans(3), quat wxyz(4), joints(22)]를
Dexonomy 자체 MJCF(extlib/Dexonomy/assets/hand/shadow/right.xml)에 적용해
링크별 world pose와 링크 표면 샘플점을 계산한다.

Root 합성 규약 (Dexonomy dexonomy/data/hand_loader.py:201-210 에서 확정):
  - 모델에는 freejoint가 없다 (nq=22, palm이 world 원점에서 +0.034m 고정).
  - forward_kinematics(qpos[7:])는 모델 좌표계(palm offset 포함) pose를 반환.
  - world pose = T(trans=qpos[:3], rot=quat2mat(qpos[3:7])) ∘ model pose.
따라서 pk chain은 worldbody를 root로 두고(offset 유지) qpos[:7]을
Chain.forward_kinematics의 `world` 인자로 넘긴다.
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "pytorch_kinematics" / "src"))

import pytorch_kinematics as pk  # noqa: E402
import pytorch_kinematics.transforms as tf  # noqa: E402

DEFAULT_XML = REPO_ROOT / "extlib" / "Dexonomy" / "assets" / "hand" / "shadow" / "right.xml"
LINK_POINTS_CACHE = REPO_ROOT / "urdf" / "contact_dexonomy_shadow.json"

# qpos[7+i] 가 공급되는 joint 이름. mujoco jnt_qposadr 순서와 일치함을
# DexonomyShadowFK.__init__ 에서 assert 한다.
QPOS_JOINT_ORDER = [
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

# repo 관례(scripts/precompute_sample_graphs.py) 순서 + palm
FINGER_NAMES = ["index", "middle", "ring", "little", "thumb", "palm"]
FINGER_LINKS = {
    "index": ["rh_ffknuckle", "rh_ffproximal", "rh_ffmiddle", "rh_ffdistal"],
    "middle": ["rh_mfknuckle", "rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"],
    "ring": ["rh_rfknuckle", "rh_rfproximal", "rh_rfmiddle", "rh_rfdistal"],
    "little": ["rh_lfmetacarpal", "rh_lfknuckle", "rh_lfproximal", "rh_lfmiddle", "rh_lfdistal"],
    "thumb": ["rh_thbase", "rh_thproximal", "rh_thhub", "rh_thmiddle", "rh_thdistal"],
    "palm": ["rh_palm"],
}
LINK_NAMES = [ln for f in FINGER_NAMES for ln in FINGER_LINKS[f]]  # 전역 링크 순서 (L=23)
LINK_FINGER_ID = np.array(
    [fi for fi, f in enumerate(FINGER_NAMES) for _ in FINGER_LINKS[f]], dtype=np.int64
)

# 링크별 표면 샘플점 개수. 샘플 기반 최소거리는 실제 거리를 과대추정하므로
# (V2 분석: 접촉 손가락이 2~4mm로 측정됨) 밀도를 충분히 높게 잡는다.
_N_POINTS = {"distal": 64, "middle": 32, "proximal": 32, "hub": 16, "base": 16,
             "knuckle": 16, "metacarpal": 32, "palm": 128}


def n_points_for(link: str) -> int:
    for key, n in _N_POINTS.items():
        if link.endswith(key) or key in link:
            return n
    raise ValueError(f"no point-count rule for link {link}")


def _sample_geom_surface(m: mujoco.MjModel, geom_id: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """geom 표면에서 균일하게 n개 (points, normals)를 geom frame으로 샘플."""
    gtype = int(m.geom_type[geom_id])
    size = m.geom_size[geom_id]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[geom_id])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        mesh = trimesh.Trimesh(vertices=m.mesh_vert[va:va + vn], faces=m.mesh_face[fa:fa + fn], process=False)
        pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(2**31)))
        return np.asarray(pts), np.asarray(mesh.face_normals[face_idx])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return d * size[0], d
    if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hh = float(size[0]), float(size[1])
        side_area = 2 * np.pi * r * 2 * hh
        cap_area = 4 * np.pi * r**2 if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE else 2 * np.pi * r**2
        n_side = max(1, int(round(n * side_area / (side_area + cap_area))))
        n_cap = max(1, n - n_side)
        theta = rng.uniform(0, 2 * np.pi, n_side)
        z = rng.uniform(-hh, hh, n_side)
        side_p = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
        side_n = np.stack([np.cos(theta), np.sin(theta), np.zeros(n_side)], axis=1)
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            d = rng.normal(size=(n_cap, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            cap_p = d * r + np.stack([np.zeros(n_cap), np.zeros(n_cap), np.sign(d[:, 2]) * hh], axis=1)
            cap_n = d
        else:
            rad = r * np.sqrt(rng.uniform(0, 1, n_cap))
            th2 = rng.uniform(0, 2 * np.pi, n_cap)
            zs = np.where(rng.uniform(size=n_cap) < 0.5, hh, -hh)
            cap_p = np.stack([rad * np.cos(th2), rad * np.sin(th2), zs], axis=1)
            cap_n = np.stack([np.zeros(n_cap), np.zeros(n_cap), np.sign(zs)], axis=1)
        return np.concatenate([side_p, cap_p]), np.concatenate([side_n, cap_n])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.asarray(size[:3], dtype=np.float64)
        areas = np.array([half[1] * half[2], half[1] * half[2], half[0] * half[2],
                          half[0] * half[2], half[0] * half[1], half[0] * half[1]])
        face = rng.choice(6, size=n, p=areas / areas.sum())
        uv = rng.uniform(-1, 1, size=(n, 2))
        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        for i in range(n):
            axis, sign = face[i] // 2, 1.0 if face[i] % 2 == 0 else -1.0
            others = [a for a in range(3) if a != axis]
            pts[i, axis] = sign * half[axis]
            pts[i, others[0]] = uv[i, 0] * half[others[0]]
            pts[i, others[1]] = uv[i, 1] * half[others[1]]
            nrm[i, axis] = sign
        return pts, nrm
    raise NotImplementedError(f"geom type {gtype} not supported")


def _geom_surface_area(m: mujoco.MjModel, geom_id: int) -> float:
    """geom 표면적 (m^2). 샘플점당 대표 면적(접촉 면적 근사)에 사용."""
    gtype = int(m.geom_type[geom_id])
    size = m.geom_size[geom_id]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[geom_id])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        return float(trimesh.Trimesh(vertices=m.mesh_vert[va:va + vn],
                                     faces=m.mesh_face[fa:fa + fn], process=False).area)
    r, hh = float(size[0]), float(size[1])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return 4 * np.pi * r**2
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return 2 * np.pi * r * 2 * hh + 4 * np.pi * r**2
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return 2 * np.pi * r * 2 * hh + 2 * np.pi * r**2
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        a, b, c = (float(s) for s in size[:3])
        return 8 * (a * b + b * c + a * c)
    raise NotImplementedError(f"geom type {gtype}")


def sample_link_points(m: mujoco.MjModel, seed: int = 0) -> dict[str, list]:
    """collision geom(contype!=0)에서 body frame 기준 {link: [points, normals, areas]} 샘플.

    areas[k] = 샘플점 k가 대표하는 링크 표면적 (geom 면적 / geom당 점 수, m^2).
    """
    rng = np.random.default_rng(seed)
    geoms_by_body: dict[str, list[int]] = {}
    for g in range(m.ngeom):
        if int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0:
            continue
        body = m.body(int(m.geom_bodyid[g])).name
        geoms_by_body.setdefault(body, []).append(g)

    out = {}
    for link in LINK_NAMES:
        gids = geoms_by_body.get(link)
        assert gids, f"link {link} has no collision geom"
        n_total = n_points_for(link)
        n_each = max(4, n_total // len(gids))
        pts_all, nrm_all, area_all = [], [], []
        for gid in gids:
            p, nv = _sample_geom_surface(m, gid, n_each, rng)
            quat = m.geom_quat[gid]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            pts_all.append(p @ rot.T + m.geom_pos[gid])
            nrm_all.append(nv @ rot.T)
            area_all.append(np.full(len(p), _geom_surface_area(m, gid) / len(p)))
        out[link] = [np.concatenate(pts_all).tolist(), np.concatenate(nrm_all).tolist(),
                     np.concatenate(area_all).tolist()]
    return out


class DexonomyShadowFK:
    def __init__(self, xml_path: str | Path = DEFAULT_XML, device: str = "cpu",
                 dtype: torch.dtype = torch.float32, points_cache: str | Path = LINK_POINTS_CACHE):
        self.device, self.dtype = device, dtype
        xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        assert self.model.nq == 22, f"expected 22 joint dofs, got {self.model.nq}"
        mj_order = [None] * 22
        for j in range(self.model.njnt):
            assert int(self.model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE
            mj_order[int(self.model.jnt_qposadr[j])] = self.model.joint(j).name
        assert mj_order == QPOS_JOINT_ORDER, f"joint order mismatch:\n{mj_order}\nvs\n{QPOS_JOINT_ORDER}"

        self.chain = pk.build_chain_from_mjcf(xml_path)  # root = worldbody, offsets 유지 (가설 B)
        self.chain = self.chain.to(dtype=dtype, device=device)
        chain_joints = set(self.chain.get_joint_parameter_names())
        assert chain_joints == set(QPOS_JOINT_ORDER), f"chain joints mismatch: {chain_joints}"

        points_cache = Path(points_cache)
        data = None
        if points_cache.exists():
            data = json.loads(points_cache.read_text())
            if any(len(v) != 3 for v in data.values()):  # 구버전(areas 없음) 캐시
                data = None
        if data is None:
            data = sample_link_points(self.model)
            points_cache.write_text(json.dumps(data))
        assert set(data.keys()) == set(LINK_NAMES)
        # 링크별 점을 하나로 concat, link_slices[i] = (start, end) in LINK_NAMES 순서
        self.link_slices = []
        pts, nrms, areas = [], [], []
        start = 0
        for link in LINK_NAMES:
            p = np.asarray(data[link][0], dtype=np.float64)
            nv = np.asarray(data[link][1], dtype=np.float64)
            self.link_slices.append((start, start + len(p)))
            start += len(p)
            pts.append(p)
            nrms.append(nv)
            areas.append(np.asarray(data[link][2], dtype=np.float64))
        self.n_points_total = start
        self._local_points = torch.tensor(np.concatenate(pts), dtype=dtype, device=device)
        self._local_normals = torch.tensor(np.concatenate(nrms), dtype=dtype, device=device)
        self.point_areas = np.concatenate(areas)  # (P,) m^2
        self._point_link_id = np.concatenate(
            [np.full(e - s, i, dtype=np.int64) for i, (s, e) in enumerate(self.link_slices)])

    def _fk_transforms(self, qpos: np.ndarray | torch.Tensor) -> dict[str, tf.Transform3d]:
        q = torch.as_tensor(qpos, dtype=self.dtype, device=self.device)
        assert q.ndim == 2 and q.shape[1] == 29, f"qpos must be (N,29), got {tuple(q.shape)}"
        root = tf.Transform3d(rot=q[:, 3:7], pos=q[:, 0:3], dtype=self.dtype, device=self.device)
        # ensure_2d_tensor가 (N,)을 (1,N)으로 오해하므로 (N,1)로 공급
        th = {name: q[:, 7 + i:8 + i] for i, name in enumerate(QPOS_JOINT_ORDER)}
        return self.chain.forward_kinematics(th, world=root)

    def link_poses(self, qpos) -> dict[str, torch.Tensor]:
        """qpos (N,29) -> {link_name: (N,4,4)} world transforms."""
        ret = self._fk_transforms(qpos)
        return {link: ret[link].get_matrix() for link in LINK_NAMES}

    def link_points_world(self, qpos) -> tuple[torch.Tensor, torch.Tensor]:
        """qpos (N,29) -> (points (N,P,3), normals (N,P,3)) world frame.

        P = self.n_points_total; 링크 구간은 self.link_slices / self._point_link_id.
        """
        ret = self._fk_transforms(qpos)
        pts_out, nrm_out = [], []
        for link, (s, e) in zip(LINK_NAMES, self.link_slices):
            mat = ret[link].get_matrix()  # (N,4,4)
            R, t = mat[:, :3, :3], mat[:, :3, 3]
            p = self._local_points[s:e]
            nv = self._local_normals[s:e]
            pts_out.append(torch.einsum("nij,kj->nki", R, p) + t[:, None, :])
            nrm_out.append(torch.einsum("nij,kj->nki", R, nv))
        return torch.cat(pts_out, dim=1), torch.cat(nrm_out, dim=1)
