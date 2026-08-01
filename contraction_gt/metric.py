"""Contraction GT 품질 메트릭: 방향별 외란 내하중 배율 α의 배치 LP 계산.

설계 근거: md_claude/plans/contraction_gt_generation.md 의 D4(soft-finger 구형 팁),
D6(14방향 외란 집합), D7(초소형 LP의 GPU 배치).

정의:
    α(d) = max{ α ≥ 0 : ∃ x ≥ 0,  Σ_i W_i x_i = -α ŵ_d,  앵커별 Σ_k x_{i,k} ≤ 1 }

- 접촉 모델: 앵커당 극선(extreme ray) 6개 — 법선±접선 4개, 법선+비틀림 ±2개.
  extlib/Dexonomy/dexonomy/qp/qp_batched.py 의 E 행렬과 같은 soft-finger 구조.
- 정규화: 내부에서 위치를 (pos - com) / r_obj 로 무차원화한다. 토크 방향의 단위
  wrench (0, ê)는 물리적으로 "지렛대 r_obj 에 걸린 단위힘"과 등가이므로, 힘 방향의
  α와 토크 방향의 α가 같은 단위(앵커당 압력 상한 1 기준 배율)로 비교 가능하다.
  μ_tor 는 길이 단위(모멘트 팔, 예: 헤르츠 유도값 0.02 m)로 받아 내부에서 r_obj 로
  나눈다.
- LP는 항상 feasible (x=0, α=0) 하므로 해가 존재하며 α ≥ 0.

솔버 2종:
    alpha_reference : scipy HiGHS 로 문제당 정확히 푸는 검증 기준 (CPU, 느림)
    alpha_batched   : OSQP 형 ADMM 을 torch 로 배치 실행 (GPU, 실전용)
"""

import torch


# ---------------------------------------------------------------------------
# 기하 구성
# ---------------------------------------------------------------------------

def tangent_frames(normals):
    """법선 [..., 3] -> (n, t1, t2) 정규직교 프레임. 결정적(재현 가능) 구성."""
    n = torch.nn.functional.normalize(normals, dim=-1)
    helper = torch.zeros_like(n)
    use_z = n[..., 2].abs() < 0.9
    helper[..., 2] = torch.where(use_z, 1.0, 0.0)
    helper[..., 0] = torch.where(use_z, 0.0, 1.0)
    t1 = torch.nn.functional.normalize(torch.cross(helper, n, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return n, t1, t2


def _as_bn1(value, B, N, dtype, device):
    """스칼라 또는 [B,N] 텐서를 [B,N,1] 로 브로드캐스트."""
    t = torch.as_tensor(value, dtype=dtype, device=device)
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return t.expand(B, N).unsqueeze(-1)


def ray_wrench_matrix(pos, normal, com, r_obj, mu_t, mu_tor):
    """극선 가중치 1단위가 만드는 정규화된 물체 wrench 행렬 W: [B, N, 6, 6].

    pos, normal : [B, N, 3] (물리 단위)
    com         : [B, 3]
    r_obj       : [B] 또는 스칼라 — 물체 외접반경 (물리 단위)
    mu_t        : 접선 마찰계수, 스칼라 또는 [B, N]
    mu_tor      : 비틀림 모멘트 팔 (물리 길이 단위), 스칼라 또는 [B, N]

    반환 W[..., :, k] = k번 극선의 (힘 3, 토크 3) — com 기준, r_obj 정규화 좌표계.
    """
    B, N = pos.shape[:2]
    dtype, device = pos.dtype, pos.device
    r_obj_t = torch.as_tensor(r_obj, dtype=dtype, device=device).reshape(-1, 1, 1)
    n, t1, t2 = tangent_frames(normal)
    r = (pos - com[..., None, :]) / r_obj_t
    mu_t_b = _as_bn1(mu_t, B, N, dtype, device)
    mu_tor_b = _as_bn1(mu_tor, B, N, dtype, device) / r_obj_t

    # 극선 힘 성분 [B,N,6,3]: k=0..3 접선 피라미드, k=4,5 순수 법선(비틀림 담당)
    f = torch.stack(
        [
            n + mu_t_b * t1,
            n - mu_t_b * t1,
            n + mu_t_b * t2,
            n - mu_t_b * t2,
            n,
            n,
        ],
        dim=-2,
    )
    tau = torch.cross(r[..., None, :].expand_as(f), f, dim=-1)
    tau = tau.clone()
    tau[..., 4, :] = tau[..., 4, :] + mu_tor_b * n
    tau[..., 5, :] = tau[..., 5, :] - mu_tor_b * n
    # [B,N,6rays,6wrench] -> [B,N,6wrench,6rays]
    return torch.cat([f, tau], dim=-1).transpose(-1, -2)


def weak_axis(pos):
    """앵커 배치의 급소 회전축 [B,3]: 앵커들이 가장 일직선에 가까운 주축.

    N=2 에서는 두 앵커의 연결선과 일치 — 접촉힘의 지렛대가 0이라 비틀림
    마찰로만 저항하는 구조적 최약 모드(D6).
    """
    centered = pos - pos.mean(dim=-2, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return Vh[..., 0, :]


_CANONICAL_CACHE = {}


def _canonical_dirs(dtype, device):
    """힘 ±xyz + 토크 ±xyz 12방향 상수 텐서 (호출마다 재조립하지 않도록 캐시)."""
    key = (dtype, str(device))
    if key not in _CANONICAL_CACHE:
        c = torch.zeros(12, 6, dtype=dtype, device=device)
        for i in range(3):
            c[2 * i, i] = 1.0
            c[2 * i + 1, i] = -1.0
            c[6 + 2 * i, 3 + i] = 1.0
            c[6 + 2 * i + 1, 3 + i] = -1.0
        _CANONICAL_CACHE[key] = c
    return _CANONICAL_CACHE[key]


def direction_set(pos):
    """D6 의 14방향 외란 집합 [B, 14, 6]. pos 는 정규화 좌표계의 앵커 위치 [B,N,3].

    0..5  : 힘 ±x, ±y, ±z
    6..11 : 토크 ±x, ±y, ±z (정규화 좌표계 — 지렛대 r_obj 의 단위힘 등가)
    12,13 : 앵커 급소축 ± 토크
    """
    B = pos.shape[0]
    dtype, device = pos.dtype, pos.device
    canonical = _canonical_dirs(dtype, device).unsqueeze(0).expand(B, 12, 6)

    axis = weak_axis(pos)
    zeros3 = torch.zeros_like(axis)
    ax_p = torch.cat([zeros3, axis], dim=-1).unsqueeze(1)
    ax_m = torch.cat([zeros3, -axis], dim=-1).unsqueeze(1)
    return torch.cat([canonical, ax_p, ax_m], dim=1)


# ---------------------------------------------------------------------------
# LP 조립
# ---------------------------------------------------------------------------

def _flatten_W(W):
    """[B,N,6,6] -> [B, 6, 6N] (wrench 행, 전 앵커 극선 열)."""
    B, N = W.shape[:2]
    return W.permute(0, 2, 1, 3).reshape(B, 6, 6 * N)


def _pressure_rows(N, dtype, device):
    """앵커별 압력 합 행렬 [N, 6N+1] (α 열은 0)."""
    A = torch.zeros(N, 6 * N + 1, dtype=dtype, device=device)
    for j in range(N):
        A[j, 6 * j : 6 * j + 6] = 1.0
    return A


# ---------------------------------------------------------------------------
# 참조 솔버 (scipy HiGHS) — 검증 기준
# ---------------------------------------------------------------------------

def alpha_reference(W, dirs):
    """정확한 LP 해. W:[B,N,6,6], dirs:[B,D,6] -> alpha [B,D] (torch, CPU)."""
    import numpy as np
    from scipy.optimize import linprog

    B, N = W.shape[:2]
    D = dirs.shape[1]
    n = 6 * N + 1
    Wf = _flatten_W(W).cpu().double().numpy()
    dirs_np = dirs.cpu().double().numpy()
    A_ub = _pressure_rows(N, torch.float64, "cpu").numpy()
    b_ub = np.ones(N)
    c = np.zeros(n)
    c[-1] = -1.0
    bounds = [(0.0, None)] * n

    out = np.zeros((B, D))
    for b in range(B):
        for d in range(D):
            A_eq = np.concatenate([Wf[b], dirs_np[b, d][:, None]], axis=1)
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(6),
                bounds=bounds, method="highs",
            )
            out[b, d] = -res.fun if res.status == 0 else 0.0
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# 배치 솔버 (OSQP 형 ADMM, torch) — 실전용
# ---------------------------------------------------------------------------

def alpha_batched(W, dirs, iters=40, tau=0.99):
    """GPU 배치 내점법(Mehrotra predictor-corrector)으로 LP 를 정확히 푼다.

    W:[B,N,6,6], dirs:[B,D,6] -> alpha [B,D].

    표준형:  min c'v  s.t.  A v = b,  v ≥ 0
      v = [x (6N), α, s (N)]  (s 는 압력 슬랙),  c = [0,…,0,-1,0,…,0]
      A = [ G  ŵ  0 ]   b = [0 (6)]
          [ Pr 0  I ]       [1 (N)]

    설계 이력(테스트로 확인): 직접 LP-ADMM 은 α*=0 퇴화 방향에서 정체하고,
    이분탐색+잔차 QP 는 wrench 집합 경계를 스치는(grazing) 방향에서 res_tol
    이 α 를 구조적으로 부풀린다. 내점법은 허용치 파라미터 없이 ~30회 고정
    반복으로 기계 정밀도 해를 주며, 반복당 비용이 (6+N)² 촐레스키라 더 싸다.
    이 LP 는 항상 feasible(원점)이고 유계(압력 상한)라 예외 처리가 없다.
    """
    B, N = W.shape[:2]
    D = dirs.shape[1]
    dtype, device = W.dtype, W.device
    n6 = 6 * N
    nv = n6 + 1 + N
    m = 6 + N
    BD = B * D

    A = torch.zeros(B, D, m, nv, dtype=dtype, device=device)
    A[:, :, :6, :n6] = _flatten_W(W).unsqueeze(1)
    A[:, :, :6, n6] = dirs
    A[:, :, 6:, :n6] = _pressure_rows(N, dtype, device)[:, :n6]
    A[:, :, 6:, n6 + 1 :] = torch.eye(N, dtype=dtype, device=device)
    A = A.reshape(BD, m, nv)

    # 행 등화: 비틀림 토크 행(~μ_tor)과 힘 행(~1)의 스케일 격차 해소
    row_scale = A.norm(dim=-1).clamp(min=1e-12)                # [BD, m]
    A = A / row_scale.unsqueeze(-1)
    b = torch.zeros(m, dtype=dtype, device=device)
    b[6:] = 1.0
    b = b.unsqueeze(0) / row_scale                             # [BD, m]
    c = torch.zeros(nv, dtype=dtype, device=device)
    c[n6] = -1.0

    v = torch.ones(BD, nv, dtype=dtype, device=device)
    z = torch.ones(BD, nv, dtype=dtype, device=device)
    y = torch.zeros(BD, m, dtype=dtype, device=device)
    eye_m = torch.eye(m, dtype=dtype, device=device)

    def solve_dy(LU, pivots, rp, rd, rc):
        """소거된 정규방정식 A D A' Δy = rp - A((rc - v·rd)/z) 의 해."""
        rhs = rp - torch.einsum("bmn,bn->bm", A, (rc - v * rd) / z)
        dy = torch.linalg.lu_solve(LU, pivots, rhs.unsqueeze(-1)).squeeze(-1)
        dz = rd - torch.einsum("bmn,bm->bn", A, dy)
        dv = (rc - v * dz) / z
        return dv, dy, dz

    def max_step(w, dw):
        """w + t·dw ≥ 0 를 유지하는 최대 t ≤ 1 (fraction-to-boundary)."""
        ratio = torch.where(dw < 0, -w / dw.clamp(max=-1e-30), torch.full_like(w, float("inf")))
        return (tau * ratio.min(dim=-1).values).clamp(max=1.0)

    # 수렴한 문제는 동결(latch): IPM 은 수렴 후 계속 반복하면 퇴화 스케일링이
    # 해를 파괴한다(테스트로 확인). 안전망으로 최소 merit 반복의 α 도 추적.
    done = torch.zeros(BD, dtype=torch.bool, device=device)
    best_alpha = torch.zeros(BD, dtype=dtype, device=device)
    best_merit = torch.full((BD,), float("inf"), dtype=dtype, device=device)

    for _ in range(iters):
        rp = b - torch.einsum("bmn,bn->bm", A, v)
        rd = c - torch.einsum("bmn,bm->bn", A, y) - z
        mu = (v * z).mean(dim=-1, keepdim=True)

        pn, dn, gap = rp.norm(dim=-1), rd.norm(dim=-1), mu.squeeze(-1)
        merit = pn + dn + gap
        improved = merit < best_merit
        best_merit = torch.where(improved, merit, best_merit)
        best_alpha = torch.where(improved, v[:, n6], best_alpha)
        done = done | ((pn < 1e-7) & (dn < 1e-7) & (gap < 1e-9))
        if bool(done.all()):
            break

        Dvz = (v / z).clamp(1e-14, 1e14)
        M = torch.einsum("bmn,bn,bkn->bmk", A, Dvz, A)
        reg = 1e-11 * (1.0 + M.diagonal(dim1=-2, dim2=-1).mean(-1))
        M = M + reg[:, None, None] * eye_m
        LU, pivots = torch.linalg.lu_factor(M)

        # predictor (affine scaling)
        rc_aff = -v * z
        dv_a, dy_a, dz_a = solve_dy(LU, pivots, rp, rd, rc_aff)
        tp = max_step(v, dv_a).unsqueeze(-1)
        td = max_step(z, dz_a).unsqueeze(-1)
        mu_aff = ((v + tp * dv_a) * (z + td * dz_a)).mean(dim=-1, keepdim=True)
        sig = (mu_aff / mu.clamp(min=1e-300)).clamp(0.0, 1.0) ** 3

        # corrector
        rc = sig * mu - v * z - dv_a * dz_a
        dv, dy, dz = solve_dy(LU, pivots, rp, rd, rc)
        tp = max_step(v, dv).unsqueeze(-1)
        td = max_step(z, dz).unsqueeze(-1)
        frozen = done.unsqueeze(-1)
        v = torch.where(frozen, v, v + tp * dv)
        y = torch.where(frozen, y, y + td * dy)
        z = torch.where(frozen, z, z + td * dz)

    return best_alpha.clamp(min=0.0).reshape(B, D)


# ---------------------------------------------------------------------------
# 상위 API
# ---------------------------------------------------------------------------

def compute_alpha(pos, normal, com=None, r_obj=1.0, mu_t=0.6, mu_tor=0.02,
                  dirs=None, backend="batched", **solver_kwargs):
    """앵커 배치 -> (alpha [B,D], dirs [B,D,6]).

    pos, normal : [B, N, 3] 물리 단위. com 기본값은 원점, r_obj 기본값 1.0
    (이미 정규화된 입력이라는 뜻). dirs 를 주지 않으면 D6 의 14방향을 생성.
    """
    if com is None:
        com = torch.zeros(pos.shape[0], 3, dtype=pos.dtype, device=pos.device)
    W = ray_wrench_matrix(pos, normal, com, r_obj, mu_t, mu_tor)
    if dirs is None:
        r_obj_t = torch.as_tensor(r_obj, dtype=pos.dtype, device=pos.device).reshape(-1, 1, 1)
        pos_norm = (pos - com[..., None, :]) / r_obj_t
        dirs = direction_set(pos_norm)
    if backend == "batched":
        alpha = alpha_batched(W, dirs, **solver_kwargs)
    elif backend == "reference":
        alpha = alpha_reference(W, dirs)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return alpha, dirs
