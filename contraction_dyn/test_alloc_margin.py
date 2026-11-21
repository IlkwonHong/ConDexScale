"""alloc_margin 단위시험 — 해석 정답, scipy 참조 대조, 여유 닫힌식, 시간.

실행: ./venv/bin/python -m contraction_dyn.test_alloc_margin
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.alloc_margin import (MarginAllocator, reference_stage1,   # noqa: E402
                                          reference_rmax, pyramid)

MU = 0.5


def test_two_tip_x():
    """마주보는 2팁, 요구 +2N(x), x 외란만(r=1) → 오른팁 0.5N, 왼팁 2.5N."""
    al = MarginAllocator(mu=MU, f_cap=30.0, r_req=1.0)
    al.setup(P=[[0.1, 0, 0], [-0.1, 0, 0]], N_in=[[-1, 0, 0], [1, 0, 0]],
             com=[0, 0, 0], r_obj=0.1, d_diag=[1, 0, 0, 0, 0, 0])
    out = al.solve(np.array([2.0, 0, 0, 0, 0, 0]))
    f = out["f"]
    a = f[0] @ np.array([-1, 0, 0])          # 오른팁 누르는 힘
    b = f[1] @ np.array([1, 0, 0])           # 왼팁 누르는 힘
    print(f"[2팁 x외란] stage={out['stage']} status={out['status']} 오른팁 {a:.4f}N 왼팁 {b:.4f}N "
          f"(기대 0.5 / 2.5)  여유 r={al.margin_of(f):.4f}")
    assert out["stage"] == 1 and abs(a - 0.5) < 1e-3 and abs(b - 2.5) < 1e-3
    assert np.allclose(al.G @ f.reshape(-1), [2, 0, 0, 0, 0, 0], atol=1e-6)


def test_two_tip_y():
    """마주보는 2팁, y 외란만(r=1): 각 팁 접선 변화 ±0.5 → 누르는 힘 ≥ 0.5/μ_r (모서리가 y 방향)."""
    al = MarginAllocator(mu=MU, f_cap=30.0, r_req=1.0)
    al.setup(P=[[0.1, 0, 0], [-0.1, 0, 0]], N_in=[[-1, 0, 0], [1, 0, 0]],
             com=[0, 0, 0], r_obj=0.1, d_diag=[0, 1, 0, 0, 0, 0])
    out = al.solve(np.array([2.0, 0, 0, 0, 0, 0]))
    f = out["f"]
    a = f[0] @ np.array([-1, 0, 0])
    b = f[1] @ np.array([1, 0, 0])
    # 기대값: 면 k 의 조건 a·(h·n) ≥ 0.5·|h·ŷ| → a ≥ 0.5·max_k |h_y|/(h·n)
    _E, H = pyramid([-1, 0, 0], MU)
    n = np.array([-1.0, 0, 0])
    a_exp = 0.5 * max(abs(h[1]) / (h @ n) for h in H)
    mu_r = MU * np.cos(np.pi / 8)
    print(f"[2팁 y외란] 오른팁 {a:.4f}N 왼팁 {b:.4f}N (기대 {a_exp:.4f} / {a_exp+2:.4f}; "
          f"0.5/μ_r={0.5/mu_r:.4f})  여유 r={al.margin_of(f):.4f}")
    assert abs(a - a_exp) < 1e-3 and abs(b - a_exp - 2) < 1e-3


def seg_placement(seq="box_grab_01", seg=1):
    d = json.load(open(ROOT / "data/contraction_dyn/static_capacity" / f"s01__{seq}.json"))
    row = d["rows"][seg]
    plc = row["tuned"]
    arms = sorted(plc["arms"])
    P = np.array([plc["arms"][a]["pos"] for a in arms])
    N = np.array([plc["arms"][a]["nrm_in"] for a in arms])
    return row, P, N, np.array(row["com_ref"])


def test_seg_vs_reference():
    """실제 6팁 배치: OSQP 1단계 vs SLSQP 참조, 2단계 r_max vs HiGHS 참조."""
    row, P, N, com = seg_placement()
    mg = row["mass"] * 9.81
    rng = np.random.default_rng(0)
    al = MarginAllocator(mu=MU, f_cap=30.0, r_req=0.15)
    al.setup(P, N, com, row["r_obj"])
    for trial in range(5):
        w = np.concatenate([[0, 0, mg], np.zeros(3)]) + rng.normal(0, [2, 2, 2, 0.3, 0.3, 0.3])
        out = al.solve(w)
        ref = reference_stage1(al, w)
        assert out["stage"] == 1, out
        assert ref is not None
        lam, f_ref = ref
        obj_o, obj_r = out["lam"] @ out["lam"], lam @ lam
        rm = al.margin_of(out["f"])
        print(f"[seg1 #{trial}] OSQP ‖λ‖²={obj_o:.5f} SLSQP={obj_r:.5f} 힘차 max={np.abs(out['f']-f_ref).max():.2e}N "
              f"여유 r={rm:.4f} 팁 누르는힘={np.round((out['f']*N).sum(1),2)}")
        assert abs(obj_o - obj_r) < 1e-3 * max(1.0, obj_r)
        assert rm >= 0.15 - 1e-5
        assert np.allclose(al.G @ out["f"].reshape(-1), w, atol=1e-6)
    # 2단계: 도달 불가능한 r_req → 최대 r
    w = np.concatenate([[0, 0, mg], np.zeros(3)])
    out = al.solve(w, r_req=10.0)
    r_ref = reference_rmax(al, w, r_req=10.0)
    print(f"[seg1 r_max] stage={out['stage']} OSQP r_max={out['r_max']:.5f} HiGHS={r_ref:.5f} "
          f"사용 r={out['r']:.5f} 여유 r(f)={al.margin_of(out['f']):.5f}")
    assert out["stage"] == 2 and abs(out["r_max"] - r_ref) < 1e-3
    # ρ (재배분 허용) 와 비교 — r_max ≤ ρ 여야 함 (같은 μ 로)
    try:
        import torch
        from contraction_gt.metric import compute_rho
        td = dict(dtype=torch.float64)
        o = compute_rho(torch.tensor(P[None], **td), torch.tensor(N[None], **td),
                        torch.tensor(w[None], **td), com=torch.tensor(com[None], **td),
                        r_obj=row["r_obj"], f_cap=30.0, mu_t=MU, mu_tor=row["mu_tor"],
                        backend="batched")
        rho = float(o["rho"][0])
        print(f"[seg1 r_max vs ρ] r_alloc,max={out['r_max']:.4f} ≤ ρ(μ=0.5)={rho:.4f} : "
              f"{'OK' if out['r_max'] <= rho + 1e-3 else 'VIOLATION'}")
    except Exception as e:  # noqa: BLE001
        print("ρ 비교 생략:", e)


def test_stage3():
    """불가능한 요구(거대 토크) → 3단계 잔차 최소화, 힘은 각뿔 안."""
    row, P, N, com = seg_placement()
    al = MarginAllocator(mu=MU, f_cap=30.0, r_req=0.15)
    al.setup(P, N, com, row["r_obj"])
    w = np.array([0, 0, 10.7, 0, 0, 50.0])
    out = al.solve(w)
    print(f"[3단계] stage={out['stage']} status={out['status']} 잔차={out['resid']:.2f} "
          f"최소 면여유={al.cone_slack(out['f']).min():.3e} 누르는힘 max={(out['f']*N).sum(1).max():.2f}")
    assert out["stage"] == 3 and al.cone_slack(out["f"]).min() > -1e-5


def test_timing():
    row, P, N, com = seg_placement()
    mg = row["mass"] * 9.81
    al = MarginAllocator(mu=MU, f_cap=30.0, r_req=0.15)
    rng = np.random.default_rng(1)
    t_setup, t_solve = [], []
    for _ in range(100):
        R = np.eye(3)
        t0 = time.perf_counter()
        al.setup(P @ R.T, N @ R.T, R @ com, row["r_obj"])
        t1 = time.perf_counter()
        al.solve(np.concatenate([[0, 0, mg], np.zeros(3)]) + rng.normal(0, [1, 1, 1, .2, .2, .2]))
        t2 = time.perf_counter()
        t_setup.append(t1 - t0)
        t_solve.append(t2 - t1)
    print(f"[시간] setup 평균 {1e3*np.mean(t_setup):.2f}ms  solve 평균 {1e3*np.mean(t_solve):.2f}ms "
          f"(최대 {1e3*np.max(t_solve):.2f}ms)  — 예산 16ms")


if __name__ == "__main__":
    test_two_tip_x()
    test_two_tip_y()
    test_seg_vs_reference()
    test_stage3()
    test_timing()
    print("전부 통과")
