"""wrench_support 단위 테스트.

실행: ./venv/bin/python -m contraction_gt.test_wrench_support

CLAUDE.md §4 규칙에 따라:
  1) 손으로 유도한 해석 정답 (두 손가락 예시, 6방향 장난감 예시)
  2) 수학적 관계 (단일 접촉에서 두 규칙 일치, 그림자 ≥ α(선형계획))
  3) 권위 있는 출처(논문 공개 코드, extlib/chord_ref) 와의 대조
"""

import sys
from pathlib import Path

import numpy as np
import torch

from contraction_gt.metric import ray_wrench_matrix, alpha_reference
from contraction_gt.wrench_support import (
    sample_directions, cone_edge_wrenches, from_ray_matrix,
    support_max, support_sum, similarity,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extlib"))
from chord_ref import wrench_support_ref as ref  # noqa: E402

torch.manual_seed(0)
DT = torch.float64
PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    PASS.append(bool(cond))


def test_two_finger_toy():
    """박스 왼쪽 면을 위(+1)·아래(−1) 손가락이 오른쪽(+x)으로 밈, 마찰 0."""
    pos = torch.tensor([[[-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0]]], dtype=DT)
    nrm = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=DT)
    W = cone_edge_wrenches(pos, nrm, torch.zeros(1, 3, dtype=DT), 1.0, mu=0.0, n_edges=1)
    s2 = 1 / np.sqrt(2)
    dirs = torch.tensor([[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, -1, 0], [s2, 0, 0, 0, s2, 0], [s2, 0, 0, 0, -s2, 0]], dtype=DT)
    smax = support_max(W, dirs)[0]
    ssum = support_sum(W, dirs)[0]
    # 손 유도: A = (1,0,0, 0,+1,0), B = (1,0,0, 0,−1,0).
    #   +x: A,B 각 1 → 논문 1 / 더함 2.   −x: 둘 다 음수 → 0.
    #   +τy: A 1, B −1 → 1 / 1.          −τy: A −1, B 1 → 1 / 1  (B 가 반대 회전을 냄)
    #   (+x,+τy)/√2: A √2, B 0 → √2 / √2.  (+x,−τy)/√2: A 0, B √2 → √2 / √2.
    exp_max = torch.tensor([1, 0, 1, 1, np.sqrt(2), np.sqrt(2)], dtype=DT)
    exp_sum = torch.tensor([2, 0, 1, 1, np.sqrt(2), np.sqrt(2)], dtype=DT)
    check("두 손가락: 논문 규칙 그림자", torch.allclose(smax, exp_max, atol=1e-12), smax.tolist())
    check("두 손가락: 더하는 규칙 그림자", torch.allclose(ssum, exp_sum, atol=1e-12), ssum.tolist())


def test_single_contact_rules_agree():
    pos = torch.randn(5, 1, 3, dtype=DT)
    nrm = torch.nn.functional.normalize(torch.randn(5, 1, 3, dtype=DT), dim=-1)
    W = cone_edge_wrenches(pos, nrm, torch.zeros(5, 3, dtype=DT), 1.0, mu=0.6, n_edges=8)
    dirs = sample_directions(256)
    a, b = support_max(W, dirs), support_sum(W, dirs)
    check("단일 접촉: 두 규칙 일치", torch.allclose(a, b, atol=1e-12), f"최대 차 {float((a-b).abs().max()):.2e}")


def test_inactive_contact_zero():
    pos = torch.randn(2, 3, 3, dtype=DT)
    nrm = torch.nn.functional.normalize(torch.randn(2, 3, 3, dtype=DT), dim=-1)
    act = torch.tensor([[True, False, True], [True, True, True]])
    W = cone_edge_wrenches(pos, nrm, torch.zeros(2, 3, dtype=DT), 1.0, active=act)
    W2 = cone_edge_wrenches(pos[:, [0, 2]], nrm[:, [0, 2]], torch.zeros(2, 3, dtype=DT), 1.0)
    dirs = sample_directions(128)
    check("비활성 접촉 = 기여 0", torch.allclose(support_sum(W, dirs)[0], support_sum(W2, dirs)[0])
          and bool(torch.isfinite(W).all()))


def test_shadow_geq_alpha():
    """같은 렌치 집합(metric 의 6광선 soft-finger)에서 더하는 규칙 그림자 ≥ α(LP)."""
    B, N, D = 3, 3, 48
    pos = torch.nn.functional.normalize(torch.randn(B, N, 3, dtype=DT), dim=-1)
    nrm = -pos.clone()
    W6 = ray_wrench_matrix(pos, nrm, torch.zeros(B, 3, dtype=DT), 1.0, 0.6, 0.02)   # [B,N,6,6rays]
    dirs = sample_directions(D, seed=1)
    walk = alpha_reference(W6, (-dirs).unsqueeze(0).expand(B, D, 6))               # ΣWx = +α·b
    shadow = support_sum(from_ray_matrix(W6), dirs)
    viol = int((shadow < walk - 1e-9).sum())
    ratio = (shadow / walk.clamp(min=1e-9))[walk > 1e-6]
    check("그림자 ≥ α (위반 0)", viol == 0,
          f"위반 {viol}/{B*D}, 비율 중앙값 {float(ratio.median()):.3f}, 최소 {float(ratio.min()):.3f}")


def test_against_paper_code():
    """논문 공개 코드와 대조. (a) 지지 함수 단계: 같은 렌치 행렬 → 정확 일치.
    (b) 모서리 구성(순수 법선 모서리 없이): K 를 크게 하면 접선 기저 회전 차이가 사라져
        균일 배율 1/√(1+μ²) 로 일치.
    (c) 논문 기본 구성(순수 법선 모서리 포함): 우리 값/√(1+μ²) ≤ 논문 값 ≤ 우리 값."""
    B, C, mu = 4, 3, 0.6
    pts = torch.randn(B, C, 3, dtype=DT) * 0.05
    nrm = torch.nn.functional.normalize(torch.randn(B, C, 3, dtype=DT), dim=-1)
    basis = sample_directions(512)
    # (a)
    K = 8
    th = torch.linspace(0, 2 * np.pi, K + 1, dtype=DT)[:-1]
    ws = ref.compute_wrench_space(pts, nrm, torch.cos(th).view(1, -1, 1), torch.sin(th).view(1, -1, 1),
                                  rc=1.0, friction_coefficients=mu)                  # [B,6,C*(K+1)]
    theirs = ref.compute_wrench_space_support_function(ws, basis)
    W_from_theirs = ws.transpose(1, 2).reshape(B, C, K + 1, 6)
    ours_same_W = support_max(W_from_theirs, basis)
    check("논문 코드 대조 (a) 지지 함수 단계 정확 일치",
          torch.allclose(ours_same_W, theirs, atol=1e-12), f"최대 차 {float((ours_same_W-theirs).abs().max()):.1e}")
    # (b)
    K = 512
    th = torch.linspace(0, 2 * np.pi, K + 1, dtype=DT)[:-1]
    cs, sn = torch.cos(th).view(1, -1, 1), torch.sin(th).view(1, -1, 1)
    edges_wo = ref.compute_friction_cone_edges(nrm, cs, sn, friction_coefficients=mu, append_normal=False)
    tq = torch.cross(pts.unsqueeze(2).expand_as(edges_wo), edges_wo, dim=-1)
    ws_wo = torch.cat([edges_wo, tq], dim=-1).view(B, -1, 6).transpose(1, 2)
    theirs_wo = ref.compute_wrench_space_support_function(ws_wo, basis)
    W_ours = cone_edge_wrenches(pts, nrm, torch.zeros(B, 3, dtype=DT), 1.0, mu=mu, n_edges=K)
    ours = support_max(W_ours, basis)
    rel = float((ours / np.sqrt(1 + mu ** 2) - theirs_wo).abs().max() / theirs_wo.abs().max())
    check("논문 코드 대조 (b) 모서리 구성(법선 모서리 없이) 수렴 일치 (상대 오차 < 1e-3)",
          rel < 1e-3, f"상대 오차 {rel:.1e}")
    ws = ref.compute_wrench_space(pts, nrm, cs, sn, rc=1.0, friction_coefficients=mu)   # 기본: 법선 모서리 포함
    theirs = ref.compute_wrench_space_support_function(ws, basis)
    lo, hi = ours / np.sqrt(1 + mu ** 2), ours
    tol = 1e-3 * float(ours.abs().max())
    check("논문 코드 대조 (c) 기본 구성은 우리 값/√(1+μ²) 와 우리 값 사이",
          bool(((theirs >= lo - tol) & (theirs <= hi + tol)).all()),
          f"논문/우리 비율 범위 {float((theirs/ours.clamp(min=1e-9))[ours>1e-6].min()):.3f}~"
          f"{float((theirs/ours.clamp(min=1e-9))[ours>1e-6].max()):.3f} (이론 {1/np.sqrt(1+mu**2):.3f}~1)")


def test_similarity_toy():
    sig_h = torch.tensor([[1.0, 1.0, 0.5, 0.8, 0.0, 1.0]], dtype=DT)
    sig_r = torch.tensor([[1.0, 0.3, 0.5, 0.2, 0.4, 1.2]], dtype=DT)
    s = similarity(sig_h, sig_r, beta=0.2)
    check("장난감 예시: 활성 5, 잃음 2", int(s["n_active"]) == 5 and int(s["n_lost"]) == 2)
    check("장난감 예시: 요약 1 = 0.4", abs(float(s["lost_frac"]) - 0.4) < 1e-12, float(s["lost_frac"]))
    check("장난감 예시: 요약 2 = 0.725", abs(float(s["mean_short"]) - 0.725) < 1e-12, float(s["mean_short"]))
    check("장난감 예시: 넘침(방향 6)은 안 셈", not bool(s["lost"][0, 5]) and bool(torch.isnan(s["ratio"][0, 4])))


def test_gpu_float32():
    if not torch.cuda.is_available():
        check("GPU float32 (건너뜀: CUDA 없음)", True)
        return
    pos = torch.randn(64, 4, 3, dtype=DT) * 0.05
    nrm = torch.nn.functional.normalize(torch.randn(64, 4, 3, dtype=DT), dim=-1)
    dirs = sample_directions(512)
    W = cone_edge_wrenches(pos, nrm, torch.zeros(64, 3, dtype=DT), 0.08)
    cpu = support_max(W, dirs)
    gpu = support_max(W.cuda().float(), dirs.cuda().float()).double().cpu()
    check("GPU float32 vs CPU float64", torch.allclose(cpu, gpu, atol=1e-4), f"최대 차 {float((cpu-gpu).abs().max()):.1e}")


if __name__ == "__main__":
    test_two_finger_toy()
    test_single_contact_rules_agree()
    test_inactive_contact_zero()
    test_shadow_geq_alpha()
    test_against_paper_code()
    test_similarity_toy()
    test_gpu_float32()
    print(f"\n{sum(PASS)}/{len(PASS)} 통과")
    sys.exit(0 if all(PASS) else 1)
