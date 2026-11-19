"""ℓ∞ 강건성 여유 ρ (compute_rho) 단위 테스트.

실행: ./venv/bin/python -m contraction_gt.test_rho

CLAUDE.md 검증 규칙에 따라 3층으로 검증한다:
  1) 손으로 유도한 해석적 정답 (대향 2앵커 / 같은 면 2앵커의 정확값)
  2) 정의 자체의 브루트포스 검증 — ρ 에서 모든 정점의 집합 포함(w̃+ρDv ∈ W_c)을
     독립 scipy 가능성 LP 로 직접 확인 (우리 솔버끼리 비교가 아님)
  3) 배치 IPM vs scipy 참조 교차 검증 (w_req ≠ 0 오프셋 경로)
"""

import numpy as np
import torch

from contraction_gt.metric import (
    ray_wrench_matrix, direction_set, vertex_direction_set,
    alpha_reference, compute_alpha, compute_rho, _flatten_W, _pressure_rows,
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
    pos = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]], dtype=DTYPE)
    return pos, -pos.clone()


def same_face_pair():
    pos = torch.tensor([[[0.5, 0, 1.0], [-0.5, 0, 1.0]]], dtype=DTYPE)
    normal = torch.tensor([[[0, 0, -1.0], [0, 0, -1.0]]], dtype=DTYPE)
    return pos, normal


def wrench_feasible(W, target):
    """독립 가능성 LP: target ∈ W_c ? (W:[1,N,6,6], target:[6] numpy)."""
    from scipy.optimize import linprog
    N = W.shape[1]
    Wf = _flatten_W(W).cpu().double().numpy()[0]
    A_ub = _pressure_rows(N, torch.float64, "cpu").numpy()[:, :6 * N]
    res = linprog(np.zeros(6 * N), A_ub=A_ub, b_ub=np.ones(N),
                  A_eq=Wf, b_eq=np.asarray(target, dtype=np.float64),
                  bounds=[(0.0, None)] * (6 * N), method="highs")
    return res.status == 0


def random_case(B, N):
    pos = torch.nn.functional.normalize(torch.randn(B, N, 3, dtype=DTYPE), dim=-1)
    normal = torch.nn.functional.normalize(-pos + 0.3 * torch.randn_like(pos), dim=-1)
    return pos, normal


def feasible_wreq(pos, normal, scale=0.5):
    """무작위 방향 용량의 scale 배 명목 렌치 — 구성상 w̃ ∈ W_c 보장."""
    B = pos.shape[0]
    W = ray_wrench_matrix(pos, normal, torch.zeros(B, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    u = torch.nn.functional.normalize(torch.randn(B, 6, dtype=DTYPE), dim=-1)
    cap = alpha_reference(W, u.unsqueeze(1))[:, 0]     # 생성 −cap·u ∈ W_c
    return -scale * cap.unsqueeze(-1) * u


def test_vertex_set():
    v = vertex_direction_set(DTYPE, "cpu")
    ok = v.shape == (64, 6) and bool((v.abs() == 1).all())
    uniq = len({tuple(r.tolist()) for r in v}) == 64
    check("정점 집합 64개, 전부 ±1, 중복 없음", ok and uniq)
    v8 = vertex_direction_set(DTYPE, "cpu", force_only=True)
    ok8 = v8.shape == (8, 6) and bool((v8[:, 3:] == 0).all()) \
        and bool((v8[:, :3].abs() == 1).all())
    check("force_only 정점 8개, 토크 성분 0", ok8)


def test_wreq_offset_exact():
    """대향 2앵커 + w̃ = 0.5x̂ 의 해석적 정답.

    x 방향 순힘 범위 = [−1, +1] (각 앵커 법선 용량 1, 토크는 짝 극선으로 상쇄).
    w̃ = 0.5x̂ 에서: +x̂ 로 추가 생성 여유 = 0.5, −x̂ 로 = 1.5 (정확).
    솔버 규약: dirs = d ⇒ Σ W x = w̃ − t·d (생성 방향은 −d).
    """
    pos, normal = antipodal_sphere()
    W = ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    wreq = torch.tensor([[0.5, 0, 0, 0, 0, 0]], dtype=DTYPE)
    dirs = torch.tensor([[[-1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]]], dtype=DTYPE)
    t = alpha_reference(W, dirs, w_req=wreq)[0]
    check("오프셋 LP: +x̂ 잔여 여유 = 0.5", abs(t[0] - 0.5) < 1e-8, f"t={t[0]:.6f}")
    check("오프셋 LP: -x̂ 잔여 여유 = 1.5", abs(t[1] - 1.5) < 1e-8, f"t={t[1]:.6f}")


def test_same_face_coupling():
    """같은 면 2앵커 (r_z=1): τ_x = −F_y 강제 결합 → 박스 ρ = 0 정확.

    정점은 F_y 와 τ_x 를 동시에 ±1 로 요구 → v_fy = v_τx = +1 정점에서 t=0.
    반면 w̃ = −0.5ẑ(누름) 상태의 단일 축 +x̂ 여유는 0.25 (fz1 ≤ 0 한계로 정확)
    — 축별 α 는 놓치고 ℓ∞ 정점은 잡는 결합의 결정적 예."""
    pos, normal = same_face_pair()
    wreq = torch.tensor([[0, 0, -0.5, 0, 0, 0]], dtype=DTYPE)
    out = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                      backend="reference")
    check("같은 면: w̃=-0.5ẑ 실현 가능", bool(out["feasible"][0]),
          f"α_nom={float(out['alpha_nominal'][0]):.4f}")
    check("같은 면: 명목 용량 α_nom = 2", abs(float(out["alpha_nominal"][0]) - 2.0) < 1e-8)
    check("같은 면: 박스 ρ = 0 (τx=−Fy 결합)", abs(float(out["rho"][0])) < 1e-9,
          f"ρ={float(out['rho'][0]):.2e}")
    out8 = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                       force_only=True, backend="reference")
    check("같은 면: force_only ρ = 0 (같은 결합)", abs(float(out8["rho"][0])) < 1e-9)
    W = ray_wrench_matrix(pos, normal, torch.zeros(1, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    tx = alpha_reference(W, torch.tensor([[[-1.0, 0, 0, 0, 0, 0]]], dtype=DTYPE),
                         w_req=wreq)[0, 0]
    check("같은 면: 단일 축 +x̂ 여유 = 0.25 (대조)", abs(tx - 0.25) < 1e-8,
          f"t={tx:.6f}")


def test_infeasible_signed():
    """같은 면 2앵커는 +z(당김) 렌치 생성 불가 → feasible=False, ρ = −0.5 정확."""
    pos, normal = same_face_pair()
    wreq = torch.tensor([[0, 0, 0.5, 0, 0, 0]], dtype=DTYPE)
    out = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                      backend="reference")
    check("비가능 명목: feasible=False", not bool(out["feasible"][0]))
    check("비가능 명목: ρ = −‖w̃‖ (부족분)", abs(float(out["rho"][0]) + 0.5) < 1e-8,
          f"ρ={float(out['rho'][0]):.6f}")
    check("비가능 명목: t_vertex 전부 NaN", bool(out["t_vertex"].isnan().all()))


def test_containment_bruteforce():
    """정의 검증 — ρ* 에서 (1−ε) 수준은 64 정점 전부 포함, (1+ε) 수준은
    최소 한 정점 이탈함을 독립 scipy 가능성 LP 로 직접 확인."""
    B, N = 6, 3
    pos, normal = random_case(B, N)
    wreq = feasible_wreq(pos, normal)
    out = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                      backend="reference")
    verts = vertex_direction_set(DTYPE, "cpu").numpy()
    W = ray_wrench_matrix(pos, normal, torch.zeros(B, 3, dtype=DTYPE), 1.0, MU_T, MU_TOR)
    ok_in, ok_out = True, True
    for b in range(B):
        rho = float(out["rho"][b])
        w = wreq[b].numpy()
        if rho > 1e-6:
            lo = rho * (1 - 1e-6) - 1e-9
            ok_in &= all(wrench_feasible(W[b:b + 1], w + lo * v) for v in verts)
        hi = rho * (1 + 1e-3) + 1e-6
        ok_out &= any(not wrench_feasible(W[b:b + 1], w + hi * v) for v in verts)
    check("포함 검증: ρ(1−ε) 정점 64개 전부 ∈ W_c", ok_in)
    check("포함 검증: ρ(1+ε) 에서 이탈 정점 존재 (타이트)", ok_out)


def test_batched_vs_reference_wreq():
    """w_req 오프셋 경로의 배치 IPM vs scipy 참조 교차 검증 (N ∈ {2,3,4})."""
    for N in (2, 3, 4):
        B = 12
        pos, normal = random_case(B, N)
        wreq = feasible_wreq(pos, normal)
        ref = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                          backend="reference")
        adm = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                          backend="batched")
        err = (adm["t_vertex"] - ref["t_vertex"]).abs() / (ref["t_vertex"].abs() + 0.05)
        rho_err = (adm["rho"] - ref["rho"]).abs() / (ref["rho"].abs() + 0.05)
        check(f"배치 vs 참조 (N={N}): 정점 여유", float(err.max()) < 0.02,
              f"max상대오차={float(err.max()):.4f}")
        check(f"배치 vs 참조 (N={N}): ρ", float(rho_err.max()) < 0.02,
              f"max상대오차={float(rho_err.max()):.4f}")


def test_rho_leq_axis_alpha():
    """ρ 상한 검증: B∞(ρ) ∋ ρ·e_i 이므로 ρ ≤ (축별 α 12개의 min) (w̃=0, D=I)."""
    B, N = 12, 3
    pos, normal = random_case(B, N)
    a, _ = compute_alpha(pos, normal, mu_t=MU_T, mu_tor=MU_TOR, backend="reference")
    out = compute_rho(pos, normal, torch.zeros(B, 6, dtype=DTYPE),
                      mu_t=MU_T, mu_tor=MU_TOR, backend="reference")
    axis_min = a[:, :12].min(dim=-1).values
    check("ρ ≤ min 축별 α", bool((out["rho"] <= axis_min + 1e-8).all()),
          f"max위반={float((out['rho'] - axis_min).max()):.2e}")


def test_gpu_batched():
    if not torch.cuda.is_available():
        print("[SKIP] GPU 없음")
        return
    B, N = 8, 3
    pos, normal = random_case(B, N)
    wreq = feasible_wreq(pos, normal)
    ref = compute_rho(pos, normal, wreq, mu_t=MU_T, mu_tor=MU_TOR,
                      backend="reference")
    gpu = compute_rho(pos.cuda(), normal.cuda(), wreq.cuda(),
                      mu_t=MU_T, mu_tor=MU_TOR, backend="batched")
    err = (gpu["rho"].cpu() - ref["rho"]).abs() / (ref["rho"].abs() + 0.05)
    check("GPU 배치 vs CPU 참조: ρ", float(err.max()) < 1e-3,
          f"max상대오차={float(err.max()):.2e}")


if __name__ == "__main__":
    test_vertex_set()
    test_wreq_offset_exact()
    test_same_face_coupling()
    test_infeasible_signed()
    test_containment_bruteforce()
    test_batched_vs_reference_wreq()
    test_rho_leq_axis_alpha()
    test_gpu_batched()
    total, passed = len(PASS), sum(PASS)
    print(f"\n{passed}/{total} 통과")
    raise SystemExit(0 if passed == total else 1)
