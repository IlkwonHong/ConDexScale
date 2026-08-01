"""메트릭 모듈 단위 테스트 — 해석적 정답 케이스 + 참조/배치 솔버 교차 검증.

실행: ./venv/bin/python -m contraction_gt.test_metric

계획서(D 항목)와 CLAUDE.md 검증 규칙에 따라, 답을 손으로 유도할 수 있는
결정적 케이스를 먼저 통과시킨다. 케이스별 기대값 유도는 각 함수 주석에 있다.
"""

import torch

from contraction_gt.metric import (
    ray_wrench_matrix, direction_set, weak_axis,
    alpha_reference, alpha_batched, compute_alpha,
)

torch.manual_seed(0)
DTYPE = torch.float64
MU_T, MU_TOR = 0.6, 0.02

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    PASS.append(bool(cond))


def antipodal_sphere():
    """단위구 위 대향 2앵커: p = (±1,0,0), 법선은 안쪽(∓x)."""
    pos = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]], dtype=DTYPE)
    normal = -pos.clone()
    return pos, normal


def same_face_pair():
    """같은 평면 위 2앵커: p = (±0.5, 0, 1), 법선 둘 다 (0,0,-1)."""
    pos = torch.tensor([[[0.5, 0, 1.0], [-0.5, 0, 1.0]]], dtype=DTYPE)
    normal = torch.tensor([[[0, 0, -1.0], [0, 0, -1.0]]], dtype=DTYPE)
    return pos, normal


def test_antipodal_exact():
    """대향 2앵커의 해석적 정답 3개.

    1) 힘 ±x: 한쪽 앵커가 법선력만으로 상쇄, 압력 상한 1 → α = 1 정확.
    2) 급소축(=x축) 토크: 접촉힘 지렛대 0, 비틀림만 저항. |n·û|=1 양쪽 합
       → α = 2·μ_tor 정확 (비틀림 극선은 ± 둘 다 있어 피라미드 손실 없음).
    3) 힘 ±z (전단): 접선 피라미드의 방향 의존 → 2μ_t/√2 ≤ α ≤ 2μ_t.
    """
    pos, normal = antipodal_sphere()
    a = alpha_reference(
        ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR),
        direction_set(pos),
    )[0]
    check("대향 2앵커: 힘 +x α = 1", abs(a[0] - 1.0) < 1e-8, f"α={a[0]:.6f}")
    check("대향 2앵커: 힘 -x α = 1", abs(a[1] - 1.0) < 1e-8, f"α={a[1]:.6f}")
    expected_tor = 2 * MU_TOR
    check("대향 2앵커: 급소축 토크 α = 2μ_tor", abs(a[12] - expected_tor) < 1e-8,
          f"α={a[12]:.6f} (기대 {expected_tor})")
    lo, hi = 2 * MU_T / (2 ** 0.5) - 1e-8, 2 * MU_T + 1e-8
    check("대향 2앵커: 전단 α ∈ [2μ/√2, 2μ]", lo <= a[4] <= hi, f"α={a[4]:.6f}")


def test_torsion_off():
    """μ_tor = 0 이면 N=2 급소축 토크 α 가 정확히 0 (D4 의 존재 이유)."""
    pos, normal = antipodal_sphere()
    a = alpha_reference(
        ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, 0.0),
        direction_set(pos),
    )[0]
    check("비틀림 0: 급소축 α = 0", abs(a[12]) < 1e-9, f"α={a[12]:.2e}")
    check("비틀림 0: 힘 방향은 영향 없음", abs(a[0] - 1.0) < 1e-8, f"α={a[0]:.6f}")


def test_same_face():
    """같은 면 2앵커의 해석적 정답 3개.

    1) 미는 방향(+z 외란 → 접촉이 -z 로 상쇄): 압력 2 로 α = 2 정확.
    2) 당기는 방향(-z 외란): 법선이 -z 뿐이라 +z 힘 생성 불가 → α = 0 정확.
    3) 급소축(x축) 토크: n·û = 0 이라 비틀림이 투영되지 않음 → μ_tor > 0
       이어도 α = 0 정확 (법선 정렬 물리의 결정적 검증).
    """
    pos, normal = same_face_pair()
    a = alpha_reference(
        ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR),
        direction_set(pos),
    )[0]
    check("같은 면: 미는 방향 α = 2", abs(a[4] - 2.0) < 1e-8, f"α={a[4]:.6f}")
    check("같은 면: 당기는 방향 α = 0", abs(a[5]) < 1e-9, f"α={a[5]:.2e}")
    check("같은 면: 급소축 토크 α = 0 (n⊥û)", abs(a[12]) < 1e-9, f"α={a[12]:.2e}")


def test_weak_axis_n2():
    """N=2 급소축 = 연결선 방향."""
    pos, _ = antipodal_sphere()
    ax = weak_axis(pos)[0]
    check("급소축(N=2) = 연결선", abs(abs(ax[0]) - 1.0) < 1e-9, f"axis={ax.tolist()}")


def test_homogeneity():
    """LP 선형성: 방향 wrench 를 2배 하면 α 는 정확히 절반."""
    pos = torch.rand(1, 3, 3, dtype=DTYPE) * 2 - 1
    normal = torch.nn.functional.normalize(-pos + 0.2 * torch.randn_like(pos), dim=-1)
    W = ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    dirs = direction_set(pos)
    a1 = alpha_reference(W, dirs)
    a2 = alpha_reference(W, 2.0 * dirs)
    ratio = (a1 / a2.clamp(min=1e-12))[a1 > 1e-6]
    check("동차성 α(2ŵ) = α(ŵ)/2", bool(((ratio - 2.0).abs() < 1e-6).all()),
          f"max편차={float((ratio - 2.0).abs().max()):.2e}")


def test_three_anchor_closure():
    """정삼각형 3앵커(단위구 적도, 법선 안쪽): 14방향 전부 α > 0."""
    ang = torch.tensor([0.0, 2.0943951, 4.1887902], dtype=DTYPE)
    pos = torch.stack([ang.cos(), ang.sin(), torch.zeros(3, dtype=DTYPE)], dim=-1).unsqueeze(0)
    normal = -pos.clone()
    a = alpha_reference(
        ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR),
        direction_set(pos),
    )[0]
    check("정삼각형 3앵커: 전 방향 force closure", bool((a > 1e-4).all()),
          f"min α={float(a.min()):.4f}")


def test_batched_vs_reference():
    """배치 ADMM vs scipy 정확해 — N ∈ {2,3,4}, 무작위 구성 교차 검증."""
    for N in (2, 3, 4):
        B = 24
        pos = torch.nn.functional.normalize(torch.randn(B, N, 3, dtype=DTYPE), dim=-1)
        normal = torch.nn.functional.normalize(-pos + 0.3 * torch.randn_like(pos), dim=-1)
        mu_t = 0.3 + 0.7 * torch.rand(B, N, dtype=DTYPE)
        mu_tor = 0.01 + 0.04 * torch.rand(B, N, dtype=DTYPE)
        W = ray_wrench_matrix(pos, normal, torch.zeros(B, 3, dtype=DTYPE), 1.0, mu_t, mu_tor)
        dirs = direction_set(pos)
        ref = alpha_reference(W, dirs)
        adm = alpha_batched(W, dirs)
        err = (adm - ref).abs() / (ref.abs() + 0.05)
        check(f"배치 vs 참조 (N={N})", float(err.max()) < 0.02,
              f"max상대오차={float(err.max()):.4f}")


def test_gpu_float64():
    """GPU float64 실전 설정이 CPU 참조와 일치하는지 (있을 때만).

    내점법은 수렴 말기에 정규방정식 조건수가 1/gap 까지 커져 float32 로는
    원리적으로 불가 — GPU 도 float64 로 쓴다 (10×10 행렬이라 대역폭 병목,
    fp64 페널티는 벤치마크로 확인).
    """
    if not torch.cuda.is_available():
        print("[SKIP] GPU 없음")
        return
    B, N = 32, 3
    pos = torch.nn.functional.normalize(torch.randn(B, N, 3, dtype=DTYPE), dim=-1)
    normal = torch.nn.functional.normalize(-pos + 0.3 * torch.randn_like(pos), dim=-1)
    W = ray_wrench_matrix(pos, normal, torch.zeros(B, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    dirs = direction_set(pos)
    ref = alpha_reference(W, dirs)
    adm = alpha_batched(W.cuda(), dirs.cuda()).cpu()
    err = (adm - ref).abs() / (ref.abs() + 0.05)
    check("GPU float64 vs 참조", float(err.max()) < 1e-4,
          f"max상대오차={float(err.max()):.2e}")


def test_compute_alpha_api():
    """상위 API 가 두 백엔드에서 같은 답."""
    pos, normal = antipodal_sphere()
    a_ref, _ = compute_alpha(pos, normal, mu_t=MU_T, mu_tor=MU_TOR, backend="reference")
    a_adm, _ = compute_alpha(pos, normal, mu_t=MU_T, mu_tor=MU_TOR, backend="batched")
    err = (a_adm - a_ref).abs().max()
    check("compute_alpha API 백엔드 일치", float(err) < 5e-3, f"max오차={float(err):.2e}")


if __name__ == "__main__":
    test_antipodal_exact()
    test_torsion_off()
    test_same_face()
    test_weak_axis_n2()
    test_homogeneity()
    test_three_anchor_closure()
    test_batched_vs_reference()
    test_gpu_float64()
    test_compute_alpha_api()
    total, passed = len(PASS), sum(PASS)
    print(f"\n{passed}/{total} 통과")
    raise SystemExit(0 if passed == total else 1)
