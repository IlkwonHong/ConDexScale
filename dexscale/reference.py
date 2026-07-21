"""Guidance reference 생성기 (dexscale_mvp.md §2 + 2026-07-16 하이브리드 확장).

두 경로를 제공한다:

[구경로 — 데모 고정] build(θ): reference가 데모 pose에 붙은 (T,4,3) 시간표.
  M1~M3-A에서 사용. 측정 결과 실물 박스가 뒤처지면 reference가 떠나는 결함 확인.

[신경로 — 하이브리드] build_hybrid_plan(θ): 앵커를 물체 part frame(canonical)에
  정의하고, rollout이 매 스텝 실물 pose로 변환해 reference를 계산.
    x_ref = [실물 위 앵커 + (r−δ)·법선]  +  clip(데모 앵커 − 실물 앵커, d_max)
             └──── 쥐기 (물체 기준) ────┘     └──── 운반 (포화 오프셋) ────┘
  접촉 게이트(2026-07-16 그릴링): 사람 유효접촉(valid_contacts) 신호로
    - 접근 구간(첫 접촉 전) → 첫 접촉 앵커로 사전 파지
    - 재파지 갭 → 마지막 유효 앵커 홀드
    - 면 전환 히스테리시스 5프레임 + 앵커 이동 rate limit 5mm/프레임
  guidance의 hover 의미론(무접촉 구간 표면 밖 띄움)은 게이트가 대체한다.

스피어 순서 = [L1, L2, R1, R2]. part 변환 규약은 test_surface_projection.pose_apply:
canonical→world(top) = R_base · Rz(−arti) · p_c + obj_pos.
"""
import os.path as osp
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO, "scripts"))
from test_surface_projection import part_cloud, tangent_basis  # noqa: E402

from dexscale.knots import interp_knots, knot_times  # noqa: E402

ASSETS = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets")
BOX_DIR = osp.join(ASSETS, "arctic", "box")

N_SPHERES = 4
SPHERE_R = 0.015
N_CHANNELS = 3        # 구체별 채널 [du, dv, delta]
DU_MAX = 0.05         # 접선 보정 한계 [m]
DELTA_MAX = 0.015     # 프레스 한계 [m] — K=1500이면 22.5N ≈ F_MAX(20N) 근처
DMAX_CAP = 0.03       # 운반 오프셋 포화 한계 [m]
TILT_MAX = 0.3        # 기울인 프레스의 접선/수직 비율 상한 (≈ tan 17° ≤ μ_min — 마찰콘 마진)
GATE_MIN_CONTACTS = 3     # 손이 "잡고 있다" 판정할 최소 유효 접촉점 수
PART_HYSTERESIS = 5       # 면(part) 전환 확정에 필요한 연속 프레임
ANCHOR_RATE_LIMIT = 0.005  # canonical 앵커 이동 상한 [m/프레임]


# ---------------- θ 레이아웃 ----------------
def theta_dim(n_knots):
    """구경로: 4구체 × (du,dv,delta) × K."""
    return N_SPHERES * N_CHANNELS * n_knots


def theta_dim_hybrid(n_knots, tilt=False):
    """하이브리드: 구체별 (du,dv,delta[,tu,tv]) + 공유 d_max 채널 K개."""
    nch = N_CHANNELS + (2 if tilt else 0)
    return N_SPHERES * nch * n_knots + n_knots


def unpack_theta(theta, n_knots):
    a = np.asarray(theta, dtype=np.float64)[: theta_dim(n_knots)]
    a = a.reshape(N_SPHERES, N_CHANNELS, n_knots)
    return a[:, 0], a[:, 1], a[:, 2]


def unpack_theta_hybrid(theta, n_knots, tilt=False):
    """반환: du, dv, delta, dmax, tu, tv (tilt=False면 tu,tv는 0)."""
    nch = N_CHANNELS + (2 if tilt else 0)
    per = N_SPHERES * nch * n_knots
    th = np.asarray(theta, dtype=np.float64)
    a = th[:per].reshape(N_SPHERES, nch, n_knots)
    dmax = th[per:per + n_knots]
    if tilt:
        return a[:, 0], a[:, 1], a[:, 2], dmax, a[:, 3], a[:, 4]
    z = np.zeros((N_SPHERES, n_knots))
    return a[:, 0], a[:, 1], a[:, 2], dmax, z, z


def theta_bounds(n_knots, tangential=True, hybrid=False, tilt=False):
    nch = N_CHANNELS + (2 if tilt else 0)
    lo = np.zeros((N_SPHERES, nch, n_knots))
    hi = np.zeros_like(lo)
    if tangential:
        lo[:, 0:2] = -DU_MAX
        hi[:, 0:2] = DU_MAX
    hi[:, 2] = DELTA_MAX
    if tilt:
        lo[:, 3:5] = -TILT_MAX
        hi[:, 3:5] = TILT_MAX
    lo, hi = lo.ravel(), hi.ravel()
    if hybrid:
        lo = np.concatenate([lo, np.zeros(n_knots)])
        hi = np.concatenate([hi, np.full(n_knots, DMAX_CAP)])
    return lo, hi


def tangent_basis_batch(n):
    """(..., 3) 법선 → 접선기저 t1, t2 각 (..., 3). tangent_basis의 벡터화판."""
    n = np.asarray(n, dtype=np.float64)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    a = np.where(np.abs(n[..., 0:1]) < 0.9, ex, ey)
    t1 = a - n * (a * n).sum(-1, keepdims=True)
    t1 = t1 / (np.linalg.norm(t1, axis=-1, keepdims=True) + 1e-9)
    t2 = np.cross(n, t1)
    return t1, t2


def remap_theta_hybrid(theta_old, k_old, t_old, k_new, t_new,
                       tilt_old=False, tilt_new=False):
    """이전 지평선에서 최적화한 θ를 새 지평선·knot 수로 이식 (warm start).

    기존 구간에서는 knot 보간 곡선을 보존하고, 지평선이 늘어난 구간은
    마지막 값을 홀드한다. tilt 채널은 양쪽 레이아웃에 맞춰 이식/0-초기화.
    """
    du, dv, de, dm, tu, tv = unpack_theta_hybrid(theta_old, k_old, tilt=tilt_old)

    def remap(ch):
        curve = interp_knots(ch, t_old)                       # (..., t_old)
        tk = knot_times(k_new, t_new)
        idx = np.minimum(np.round(tk).astype(int), t_old - 1)
        return curve[..., idx]

    chans = [remap(du), remap(dv), remap(de)]
    if tilt_new:
        chans += [remap(tu), remap(tv)]
    per_sphere = np.stack(chans, axis=1)                      # (4, 3|5, K_new)
    return np.concatenate([per_sphere.ravel(), remap(dm[None])[0]])


def theta_sigma0(n_knots, tangential=True, hybrid=False, tilt=False):
    nch = N_CHANNELS + (2 if tilt else 0)
    s = np.zeros((N_SPHERES, nch, n_knots))
    if tangential:
        s[:, 0:2] = 0.01
    s[:, 2] = 0.004
    if tilt:
        s[:, 3:5] = 0.08   # 무차원 비율 채널 (±TILT_MAX=0.3 범위)
    s = s.ravel()
    if hybrid:
        s = np.concatenate([s, np.full(n_knots, 0.008)])
    return s


class BoxGuidanceReference:
    """guidance 로드 + 표면 스냅 precompute + θ 적용 (구경로/하이브리드)."""

    def __init__(self, hand="point4_hand", subj="s01", clip="box_use_01",
                 n_cloud=4000, frame_end=None, seed=0, slow=1):
        """slow=N: 데모를 N배 시간 보간(감속). ARCTIC 원본 30fps를 60Hz 스텝으로
        1:1 재생하면 2배속이 됨(2026-07-15 확인) — slow=2 가 실제 사람 속도."""
        path = osp.join(ASSETS, "retargeted", hand, subj, f"{clip}_vector_genesis.npy")
        d = np.load(path, allow_pickle=True).item()
        demo = d["demo_data"]
        sl = slice(0, frame_end)
        self.obj_pos = np.asarray(demo["obj_pos"], np.float64)[sl]     # (T,3) world m
        self.obj_quat = np.asarray(demo["obj_quat"], np.float64)[sl]   # (T,4) wxyz
        self.obj_arti = np.asarray(demo["obj_arti"], np.float64)[sl]   # (T,)
        kpt = np.concatenate([d["retarget_data"]["left"]["kpt_pos"],
                              d["retarget_data"]["right"]["kpt_pos"]], axis=1)
        self.kpt = np.asarray(kpt, np.float64)[sl]                     # (T,4,3)

        # 사람 유효접촉 신호 (접촉 게이트용) — 손 단위 (left→구체0,1 / right→구체2,3)
        proc = np.load(osp.join(ASSETS, "arctic", "processed", subj, f"{clip}.npy"),
                       allow_pickle=True).item()["world_coord"]
        vh = np.stack([np.asarray(proc["valid_contacts.left"]).sum(1),
                       np.asarray(proc["valid_contacts.right"]).sum(1)], axis=1)
        self.hand_valid = (vh[sl] >= GATE_MIN_CONTACTS)                # (T,2) bool

        self.slow = int(slow)
        if self.slow > 1:
            self._upsample(self.slow)
        self.T = self.obj_pos.shape[0]

        # 프레임별 변환 행렬 (pose_apply 규약)
        self._Rb = Rotation.from_quat(self.obj_quat[:, [1, 2, 3, 0]]).as_matrix()
        c, s = np.cos(self.obj_arti), np.sin(self.obj_arti)
        Ra = np.zeros((self.T, 3, 3))                                  # Rz(−arti)
        Ra[:, 0, 0], Ra[:, 0, 1] = c, s
        Ra[:, 1, 0], Ra[:, 1, 1] = -s, c
        Ra[:, 2, 2] = 1.0
        self._Ra = Ra

        np.random.seed(seed)  # part_cloud 표면 샘플링 재현성
        self._pts, self._nrm, self._tree = {}, {}, {}
        for part in ("bottom", "top"):
            mesh = trimesh.load_mesh(
                osp.join(BOX_DIR, f"{part}_watertight_tiny.obj"), process=False)
            pts, nrm = part_cloud(mesh, n=n_cloud)
            self._pts[part], self._nrm[part] = pts, nrm
            self._tree[part] = cKDTree(pts)
        self._part_names = ("bottom", "top")

        (self.anchors, self.normals, self.part_ids, self.snap_dist,
         self._can_p, self._can_n) = self._snap_world(self.kpt)
        self.t1, self.t2 = tangent_basis_batch(self.normals)

        self._build_gated_canonical()

    # ---------------- 전처리 ----------------
    def _upsample(self, n):
        """모든 데모 배열을 시간축으로 n배 선형 보간 (감속 재생용)."""
        T = self.obj_pos.shape[0]
        told = np.arange(T, dtype=np.float64)
        tnew = np.linspace(0.0, T - 1.0, (T - 1) * n + 1)

        q = self.obj_quat.copy()
        for t in range(1, T):
            if (q[t] * q[t - 1]).sum() < 0:
                q[t] = -q[t]

        def li(a):
            flat = a.reshape(T, -1)
            out = np.stack([np.interp(tnew, told, flat[:, j])
                            for j in range(flat.shape[1])], axis=1)
            return out.reshape((len(tnew),) + a.shape[1:])

        self.obj_pos = li(self.obj_pos)
        self.obj_arti = li(self.obj_arti)
        self.kpt = li(self.kpt)
        qn = li(q)
        self.obj_quat = qn / np.linalg.norm(qn, axis=1, keepdims=True)
        self.hand_valid = li(self.hand_valid.astype(np.float64)) > 0.5

    def _snap_world(self, pts):
        """world 점들 (T,4,3)을 그 프레임 part 표면(articulated)에 스냅.

        반환: anchors, normals (world), part_ids(0=bottom,1=top), dist,
              canonical 점/법선 (part frame — 하이브리드 경로용).
        """
        T = self.T
        rel = pts - self.obj_pos[:, None]
        loc_b = np.einsum("tji,tsj->tsi", self._Rb, rel)
        loc_t = np.einsum("tji,tsj->tsi", self._Ra, loc_b)
        db, jb = self._tree["bottom"].query(loc_b.reshape(-1, 3))
        dt_, jt = self._tree["top"].query(loc_t.reshape(-1, 3))
        use_t = (dt_ < db).reshape(T, N_SPHERES)
        p_c = np.where(use_t.reshape(-1)[:, None],
                       self._pts["top"][jt], self._pts["bottom"][jb]).reshape(T, N_SPHERES, 3)
        n_c = np.where(use_t.reshape(-1)[:, None],
                       self._nrm["top"][jt], self._nrm["bottom"][jb]).reshape(T, N_SPHERES, 3)
        p_rot = np.einsum("tij,tsj->tsi", self._Ra, p_c)
        n_rot = np.einsum("tij,tsj->tsi", self._Ra, n_c)
        p_eff = np.where(use_t[..., None], p_rot, p_c)
        n_eff = np.where(use_t[..., None], n_rot, n_c)
        anchors = np.einsum("tij,tsj->tsi", self._Rb, p_eff) + self.obj_pos[:, None]
        normals = np.einsum("tij,tsj->tsi", self._Rb, n_eff)
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-9)
        dist = np.where(use_t, dt_.reshape(T, N_SPHERES), db.reshape(T, N_SPHERES))
        return anchors, normals, use_t.astype(np.int8), dist, p_c, n_c

    def _build_gated_canonical(self):
        """접촉 게이트 + part 히스테리시스 + rate limit로 canonical 앵커 확정.

        산출: self.gp_c/gn_c (T,4,3), self.gpart (T,4) — 하이브리드의 골격.
        """
        T = self.T
        gp = np.empty((T, N_SPHERES, 3))
        gn = np.empty((T, N_SPHERES, 3))
        gpart = np.empty((T, N_SPHERES), dtype=np.int8)
        for s in range(N_SPHERES):
            valid = self.hand_valid[:, 0 if s < 2 else 1]
            first = int(np.argmax(valid)) if valid.any() else 0
            cur_p, cur_n, cur_part = (self._can_p[first, s].copy(),
                                      self._can_n[first, s].copy(),
                                      int(self.part_ids[first, s]))
            run = 0
            for t in range(T):
                if valid[t] and t >= first:
                    new_part = int(self.part_ids[t, s])
                    if new_part != cur_part:
                        run += 1
                        if run >= PART_HYSTERESIS:   # 면 전환 확정 → 점프 허용
                            cur_part, run = new_part, 0
                            cur_p = self._can_p[t, s].copy()
                            cur_n = self._can_n[t, s].copy()
                    else:
                        run = 0
                        step = self._can_p[t, s] - cur_p
                        d = np.linalg.norm(step)
                        if d > ANCHOR_RATE_LIMIT:    # 같은 면 위 이동 rate limit
                            step *= ANCHOR_RATE_LIMIT / d
                        cur_p = cur_p + step
                        cur_n = self._can_n[t, s]
                # invalid (접근/갭/릴리즈) → 현재 값 홀드 (사전 파지 = first 값)
                gp[t, s], gn[t, s], gpart[t, s] = cur_p, cur_n, cur_part
        self.gp_c, self.gn_c, self.gpart = gp, gn, gpart

    # ---------------- 구경로 (데모 고정) ----------------
    def build(self, theta, n_knots):
        """θ → 데모 고정 reference 시간표 (T,4,3). M1~M3 비교용으로 유지."""
        du, dv, delta = unpack_theta(theta, n_knots)
        de_t = np.clip(interp_knots(delta, self.T), 0.0, DELTA_MAX).T[:, :, None]
        if np.abs(du).max() > 1e-9 or np.abs(dv).max() > 1e-9:
            du_t = interp_knots(du, self.T).T[:, :, None]
            dv_t = interp_knots(dv, self.T).T[:, :, None]
            moved = self.anchors + du_t * self.t1 + dv_t * self.t2
            anchors, normals, _, _, _, _ = self._snap_world(moved)
        else:
            anchors, normals = self.anchors, self.normals
        return anchors + (SPHERE_R - de_t) * normals

    # ---------------- 하이브리드 (물체 기준 + 운반) ----------------
    def build_hybrid_plan(self, theta, n_knots, tilt=False):
        """θ → rollout이 매 스텝 실물 pose로 조립할 재료들 (전부 numpy).

        can_p/can_n (T,4,3) canonical 앵커·법선 (Δu,Δv 보정 + part 내 재스냅 적용)
        is_top (T,4)        part 플래그
        delta (T,4)         프레스 스케줄
        dmax (T,)           운반 오프셋 포화 스케줄
        tilt_c (T,4,3)      기울인 프레스의 접선 변위 (canonical) = δ·(u·t̂₁+v·t̂₂)
        demo_anchor (T,4,3) 데모 pose 기준 앵커 world 위치 (운반 목표)
        start_pos (4,3)     리셋용 초기 구체 위치 (t=0, 실물=데모)
        """
        du, dv, delta, dmax, tu, tv = unpack_theta_hybrid(theta, n_knots, tilt=tilt)
        du_t = interp_knots(du, self.T).T          # (T,4)
        dv_t = interp_knots(dv, self.T).T
        de_t = np.clip(interp_knots(delta, self.T), 0.0, DELTA_MAX).T
        dm_t = np.clip(interp_knots(dmax[None], self.T)[0], 0.0, DMAX_CAP)
        tu_t = np.clip(interp_knots(tu, self.T), -TILT_MAX, TILT_MAX).T
        tv_t = np.clip(interp_knots(tv, self.T), -TILT_MAX, TILT_MAX).T

        can_p, can_n = self.gp_c, self.gn_c
        if np.abs(du).max() > 1e-9 or np.abs(dv).max() > 1e-9:
            # canonical 접선 보정 + 같은 part 안에서 재스냅 (벡터화, 2026-07-17)
            can_p = can_p.copy()
            can_n = can_n.copy()
            gt1, gt2 = tangent_basis_batch(self.gn_c)          # (T,4,3)
            moved = self.gp_c + du_t[..., None] * gt1 + dv_t[..., None] * gt2
            for pid, pname in enumerate(self._part_names):
                m = self.gpart == pid                           # (T,4)
                if m.any():
                    _, j = self._tree[pname].query(moved[m])
                    can_p[m] = self._pts[pname][j]
                    can_n[m] = self._nrm[pname][j]

        is_top = self.gpart.astype(bool)
        # 데모 pose 기준 앵커 world (운반 목표)
        p_rot = np.einsum("tij,tsj->tsi", self._Ra, can_p)
        p_eff = np.where(is_top[..., None], p_rot, can_p)
        demo_anchor = np.einsum("tij,tsj->tsi", self._Rb, p_eff) + self.obj_pos[:, None]
        # t=0 실물 = 데모 → 초기 구체 위치
        n_rot0 = self._Ra[0] @ can_n[0].T
        n_eff0 = np.where(is_top[0][:, None], n_rot0.T, can_n[0])
        n_w0 = (self._Rb[0] @ n_eff0.T).T
        start_pos = demo_anchor[0] + SPHERE_R * n_w0

        # 기울인 프레스: 접선 변위를 canonical에서 미리 계산 (δ에 비례 → 마찰콘 마진 구조적 보장)
        ct1, ct2 = tangent_basis_batch(can_n)
        tilt_c = (de_t * tu_t)[..., None] * ct1 + (de_t * tv_t)[..., None] * ct2

        return dict(can_p=can_p, can_n=can_n, is_top=is_top,
                    delta=de_t, dmax=dm_t, tilt_c=tilt_c,
                    demo_anchor=demo_anchor, start_pos=start_pos)

    # ---------------- 공용 ----------------
    def start_positions(self):
        """구경로 리셋용: δ 무시, 표면 접촉 상태."""
        return self.anchors[0] + SPHERE_R * self.normals[0]

    def demo_qpos0(self):
        """물체 리셋용 qpos (8,) = [pos(3), quat wxyz(4), arti(1)]."""
        return np.concatenate([self.obj_pos[0], self.obj_quat[0], self.obj_arti[:1]])
