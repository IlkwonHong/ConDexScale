"""구간 합산 손-마디 접촉 추출 — 동적 v1 구현 2번 전반부 (설계 §2-②).

원천: ARCTIC 원시 npy (MANO 손 파라미터 + 물체 pose/관절각) + 물체 템플릿.
6월 산출물(JSON/그래프) 의존 없음 — 임의 프레임에서 동작.
변환 사슬은 08-07 검증 통과분(물체중심 0.00cm, 손끝→끝마디 배정 100%)의 승격.

좌표 규약: canonical 물체 좌표 (관절 반영 후 정점평균 원점). 세계 정렬은
  R(프레임별 회전)로 복원: p_world편 = R @ p_canon.
계층 문턱: 부위 존재는 5mm, 그 부위가 비면 8mm 구제 (08-07 엄지 누락 처방).
구간 합산: 표본 프레임들에서 부위 존재 = ≥40% 프레임, 접촉 정점 풀링.
"""
import json
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT.parent / "ext_proj/arctic/unpack/arctic_data"
MANO_ROOT = str(ROOT / "network/manopth/mano/models")

import sys
sys.path.insert(0, str(ROOT))
from manopth.manolayer import ManoLayer      # noqa: E402

THR_MAIN, THR_RESCUE = 0.005, 0.008
PRESENCE_FRAC = 0.4
UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]
FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]
# 손끝 구역 정점 (optimisation/robot.py human.contact_zones — 정점-마디 배정 검증용)
CONTACT_ZONES = {
    "index":  [350, 355, 329, 332, 349, 354, 343, 327, 351, 353, 348, 328, 347, 326, 337, 346],
    "middle": [462, 439, 467, 442, 461, 466, 455, 437, 459, 438, 465, 460, 436, 463, 449, 434],
    "ring":   [573, 550, 578, 577, 553, 572, 566, 570, 576, 549, 548, 547, 574, 571, 546, 569],
    "little": [690, 689, 667, 695, 670, 694, 664, 683, 688, 665, 693, 691, 666, 687, 661, 663],
    "thumb":  [743, 738, 768, 740, 763, 737, 766, 767, 764, 734, 735, 762, 745, 761, 717, 765],
}


def _vertex_link_map(layer):
    """정점 778 → (부위 0..5, 마디칸 0..2). 손끝 구역으로 손가락 정체 검증."""
    W = np.asarray(layer.th_weights)
    dom = W.argmax(1)
    triples = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)]
    finger_of = {}
    for fi, name in enumerate(FINGER_NAMES):
        j = np.bincount(dom[CONTACT_ZONES[name]], minlength=16).argmax()
        for ti, tr in enumerate(triples):
            if j in tr:
                finger_of[ti] = fi
    assert len(finger_of) == 5, f"손가락-관절 대응 실패: {finger_of}"
    unit = np.full(778, 5, np.int8)
    binv = np.full(778, 2, np.int8)
    for ti, tr in enumerate(triples):
        for k, j in enumerate(tr):
            unit[dom == j] = finger_of[ti]
            binv[dom == j] = 2 - k
    for fi in range(5):
        z = CONTACT_ZONES[FINGER_NAMES[fi]]
        assert (unit[z] == fi).all() and (binv[z] == 0).all(), "손끝 배정 검증 실패"
    return unit, binv


class SeqData:
    """시퀀스 하나의 원시 데이터 + 프레임별 변환/접촉 계산기."""

    def __init__(self, subj, seq):
        raw = ARCTIC / "data/raw_seqs" / subj
        self.mano = np.load(raw / f"{seq}.mano.npy", allow_pickle=True).item()
        self.obj = np.load(raw / f"{seq}.object.npy", allow_pickle=True)
        obj_name = seq.split("_")[0]
        tdir = ARCTIC / "data/meta/object_vtemplates" / obj_name
        import trimesh
        mesh = trimesh.load_mesh(tdir / "mesh.obj", process=False)
        self.tverts = np.array(mesh.vertices, np.float64) / 1000.0
        self.tfaces = np.array(mesh.faces)
        self.parts = np.array(json.load(open(tdir / "parts.json")), np.int32)
        self.layers = {s: ManoLayer(ncomps=45, side=s, mano_root=MANO_ROOT,
                                    use_pca=False, flat_hand_mean=False)
                       for s in ("right", "left")}
        self.vmap = {s: _vertex_link_map(self.layers[s]) for s in ("right", "left")}

    def obj_frame(self, f):
        """프레임 f 의 (R, trans, theta, canonical 관절반영 정점, centroid)."""
        theta = float(self.obj[f, 0])
        R = Rotation.from_rotvec(self.obj[f, 1:4]).as_matrix()
        trans = self.obj[f, 4:7] / 1000.0
        v = self.tverts.copy()
        top = self.parts == 0
        if abs(theta) > 1e-8:
            Ra = Rotation.from_rotvec([0, 0, -theta]).as_matrix()
            v[top] = v[top] @ Ra.T
        cen = v.mean(0)
        return R, trans, theta, v, cen

    def hand_verts_world(self, side, frames):
        m = self.mano[side]
        rot = torch.from_numpy(np.asarray(m["rot"], np.float32))[frames]
        pose = torch.from_numpy(np.asarray(m["pose"], np.float32))[frames]
        trans = torch.from_numpy(np.asarray(m["trans"], np.float32))[frames]
        shape = torch.from_numpy(np.asarray(m["shape"], np.float32)).view(1, 10)
        F = rot.shape[0]
        verts, *_ = self.layers[side](
            th_pose_coeffs=torch.cat([rot.view(F, 3), pose.view(F, 45)], 1),
            th_betas=shape.expand(F, -1), th_trans=torch.zeros(F, 3),
            root_palm=False)
        return (verts + trans.unsqueeze(1)).numpy().astype(np.float64)

    def segment_contacts(self, side, frames):
        """표본 프레임 합산 접촉 (canonical 좌표).

        반환: mask[6] bool, ratios[6,3], cpos[M,3](canonical 원점=centroid),
              cunit[M], center_c[3](풀링 정점 평균) — 접촉 없으면 None.
        """
        uu, bb = self.vmap[side]
        hv = self.hand_verts_world(side, frames)
        present_cnt = np.zeros(6)
        cnt = np.zeros((6, 3), np.float64)
        cpos, cunit = [], []
        for k, f in enumerate(frames):
            R, trans, theta, v_art, cen = self.obj_frame(f)
            h_c = (R.T @ (hv[k] - trans).T).T - cen            # canonical 원점
            o_c = v_art - cen
            d = np.linalg.norm(h_c[:, None] - o_c[None], axis=-1).min(1)
            c5, c8 = d < THR_MAIN, d < THR_RESCUE
            for u in range(6):
                cu = c5 & (uu == u)
                if not cu.any():
                    cu = c8 & (uu == u)
                if cu.any():
                    present_cnt[u] += 1
                    for b in (0, 1, 2):
                        cnt[u, b] += (cu & (bb == b)).sum()
                    cpos.append(h_c[cu])
                    cunit.append(np.full(cu.sum(), u, np.int8))
        mask = present_cnt >= max(1, PRESENCE_FRAC * len(frames))
        if not mask.any():
            return None
        keepv = np.concatenate(cpos) if cpos else np.zeros((0, 3))
        keepu = np.concatenate(cunit) if cunit else np.zeros(0, np.int8)
        sel = mask[keepu]                                       # 존재 부위 정점만
        cnt[~mask] = 0
        s = cnt.sum(1, keepdims=True)
        ratios = np.divide(cnt, s, out=np.zeros_like(cnt), where=s > 0)
        return dict(mask=mask, ratios=ratios.astype(np.float32),
                    cpos=keepv[sel].astype(np.float32), cunit=keepu[sel],
                    center=keepv[sel].mean(0) if sel.any() else None)
