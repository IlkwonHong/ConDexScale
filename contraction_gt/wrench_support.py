"""렌치 공간의 그림자 길이(지지 함수)와 사람 손 대비 닮음 점수.

계획서: md_claude/plans/wrench_space_similarity.md (2026-10-06 그릴링 확정).
논문 CHORD(식 1~2)의 접촉 렌치 공간 비교를 우리 데이터 규약(metric.py 와 같은
정규화)으로 옮긴 것. 선형계획 없이 곱셈·최댓값만 쓰므로 수십만 기록을 한 번에 돈다.

규약
- 접촉 i: 위치 p_i(물리 단위), 안쪽 법선 n_i(접촉힘이 물체를 미는 방향).
  정규화 위치 r_i = (p_i − com)/r_obj  → 토크 1 = "지렛대 r_obj 에 힘 1".
- 각뿔 모서리 힘 f_ik = n_i + μ(cosθ_k t1_i + sinθ_k t2_i), θ_k = 2πk/K.
  법선 성분이 1 → "접촉당 누르는 힘 상한 1" (metric.py 의 앵커별 압력 ≤ 1 과 같은 단위).
  (논문 코드는 합력 크기를 1 로 맞추고 순수 법선 모서리를 하나 더 넣는다. 기울인 모서리만 보면
   크기가 전부 √(1+μ²) 라 균일 배율 차이지만, 순수 법선 모서리 때문에 전체로는
   우리 값/√(1+μ²) ≤ 논문 값 ≤ 우리 값 의 사이 관계다. test_wrench_support 에서 둘 다 대조.)
- 모서리 렌치 w_ik = [f_ik, r_i × f_ik] ∈ R⁶.
- 그림자(논문 규칙)   σ(b) = max(0, max_{i,k} b·w_ik)          — 접촉 개수에 무관
- 그림자(더하는 규칙) h(b) = Σ_i max(0, max_k b·w_ik)          — α·ρ 렌치 집합의 지지 함수
  (앵커별 압력 ≤ 1 집합의 민코프스키 합 → 지지 함수는 합). h(b) ≥ α(b), 최약 방향에서 등호.
"""

import math

import torch

from .metric import tangent_frames


def sample_directions(n=512, seed=0, dtype=torch.float64, device="cpu"):
    """6차원 단위 방향 n 개 (고정 시드 — 모든 기록·양쪽(사람/로봇)에 같은 집합을 쓸 것)."""
    g = torch.Generator().manual_seed(int(seed))
    b = torch.randn(n, 6, generator=g, dtype=dtype)
    return torch.nn.functional.normalize(b, dim=-1).to(device)


def cone_edge_wrenches(pos, normal, com, r_obj, mu=0.6, n_edges=8, active=None):
    """접촉들의 각뿔 모서리 렌치 W: [B, C, K, 6].

    pos, normal : [B, C, 3] 물리 단위 (normal 은 안쪽 방향; 크기는 내부에서 정규화)
    com         : [B, 3]
    r_obj       : [B] 또는 스칼라
    mu          : 스칼라 또는 [B, C]
    active      : [B, C] bool — False 인 접촉(또는 법선이 0 인 접촉)은 렌치 0 (기여 없음)
    """
    B, C = pos.shape[:2]
    dtype, device = pos.dtype, pos.device
    r_obj_t = torch.as_tensor(r_obj, dtype=dtype, device=device).reshape(-1, 1, 1)
    mu_t = torch.as_tensor(mu, dtype=dtype, device=device)
    while mu_t.dim() < 2:
        mu_t = mu_t.unsqueeze(0)
    mu_t = mu_t.expand(B, C)[..., None, None]                       # [B,C,1,1]

    n, t1, t2 = tangent_frames(normal)                               # 법선 0 → 전부 0 (NaN 없음)
    theta = torch.arange(n_edges, dtype=dtype, device=device) * (2.0 * math.pi / n_edges)
    cs, sn = torch.cos(theta)[:, None], torch.sin(theta)[:, None]  # [K,1]
    f = n[..., None, :] + mu_t * (cs * t1[..., None, :] + sn * t2[..., None, :])   # [B,C,K,3]
    r = (pos - com[:, None, :]) / r_obj_t                            # [B,C,3]
    tau = torch.cross(r[..., None, :].expand_as(f), f, dim=-1)
    W = torch.cat([f, tau], dim=-1)
    if active is not None:
        W = W * active.to(dtype)[..., None, None]
    return W


def from_ray_matrix(W_rays):
    """metric.ray_wrench_matrix 출력 [B, N, 6, K] → 이 모듈 규약 [B, N, K, 6]."""
    return W_rays.transpose(-1, -2)


def _projections(W, dirs):
    """b·w 전부: W [B,C,K,6], dirs [D,6] → [B, D, C, K]."""
    return torch.einsum("bcke,de->bdck", W, dirs)


def support_max(W, dirs):
    """논문 규칙 그림자 σ(b): 방향마다 가장 잘 미는 모서리 힘 하나. → [B, D] (≥ 0)."""
    d = _projections(W, dirs)
    return d.flatten(2).amax(-1).clamp(min=0.0)


def support_sum(W, dirs):
    """더하는 규칙 그림자 h(b): 접촉마다 최선 모서리를 고른 뒤 합. → [B, D] (≥ 0)."""
    d = _projections(W, dirs)
    return d.amax(-1).clamp(min=0.0).sum(-1)


def similarity(sig_h, sig_r, beta=0.1, eps=1e-3):
    """사람 그림자 sig_h 와 로봇 그림자 sig_r ([B, D]) 의 닮음 요약.

    - active   : 사람 값 > eps 인 방향 (사람이 밀 수 있는 방향만 비교; 넘침은 세지 않음)
    - ratio    : 로봇 ÷ 사람 (비활성 방향 NaN)
    - lost     : active ∧ ratio < 1 − beta
    - lost_frac: 잃은 방향 수 ÷ 활성 방향 수          (요약 1; 활성 0 이면 NaN)
    - mean_short: 잃은 방향에서 (1 − ratio) 평균     (요약 2; 잃은 방향 없으면 0)
    - paper_score: Σ_active max(0, (1−β)σ_h − σ_r)² ÷ 활성 방향 수 (논문 식 3 첫 항을 방향 수로 나눈 것)
    """
    active = sig_h > eps
    nan = torch.full_like(sig_h, float("nan"))
    ratio = torch.where(active, sig_r / sig_h.clamp(min=eps), nan)
    lost = active & (ratio < (1.0 - beta))
    n_active = active.sum(-1)
    n_lost = lost.sum(-1)
    dt = sig_h.dtype
    lost_frac = torch.where(n_active > 0, n_lost.to(dt) / n_active.clamp(min=1).to(dt), nan[:, 0])
    short = torch.where(lost, 1.0 - ratio, torch.zeros_like(sig_h))
    mean_short = torch.where(n_lost > 0, short.sum(-1) / n_lost.clamp(min=1).to(dt),
                             torch.zeros_like(lost_frac))
    deficit = torch.clamp((1.0 - beta) * sig_h - sig_r, min=0.0) ** 2
    paper_score = torch.where(n_active > 0, (deficit * active).sum(-1) / n_active.clamp(min=1).to(dt),
                              nan[:, 0])
    return dict(active=active, ratio=ratio, lost=lost, n_active=n_active, n_lost=n_lost,
                lost_frac=lost_frac, mean_short=mean_short, paper_score=paper_score)
