#!/usr/bin/env python3
"""Robustness envelope 평가 — 학습(TO)보다 넓은 물성 그리드에서 θ*를 재생.

질량 스케일 × 마찰 비율 그리드의 각 칸에서 novoc 연속 rollout을 돌리고,
낙하 여부·평균/최종 추적 오차를 표로 출력한다 (dexscale_mvp.md §6 M4).
TO의 DR 범위(질량 ×0.7~1.5, 마찰 ×0.5~1.5)보다 넓게 잡아 경계를 관측한다.

실행:
  python scripts/eval_dexscale_envelope.py --theta data/dexscale/theta_hybrid_full_K5.npz --frames 220
"""
import argparse
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from dexscale import reference as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--masses", default="0.5,1.0,2.0,3.0", help="질량 스케일 목록")
    ap.add_argument("--frics", default="0.3,0.6,1.0,2.0", help="마찰 비율 목록")
    ap.add_argument("--hinge", type=float, default=0.005)
    args = ap.parse_args()

    z = np.load(args.theta)
    theta, K = z["best_theta"], int(z["n_knots"])
    tilt = bool(z["tilt"]) if "tilt" in z else False
    assert bool(z["hybrid"]) if "hybrid" in z else False, "hybrid θ 전용 스크립트"
    slow = int(z["slow"]) if "slow" in z else 1

    masses = [float(x) for x in args.masses.split(",")]
    frics = [float(x) for x in args.frics.split(",")]
    grid = [(m, f) for m in masses for f in frics]

    ref = R.BoxGuidanceReference(frame_end=args.frames, slow=slow)
    from dexscale.rollout import DexScaleEvaluator
    ev = DexScaleEvaluator(ref, n_dr=len(grid), chunk=1)
    pt = ev.plan_to_tensors([ref.build_hybrid_plan(theta, K, tilt=tilt)])
    ev.set_physics(np.array([g[0] for g in grid]),
                   np.array([g[1] for g in grid]),
                   np.full(len(grid), args.hinge))
    ev.reset(start_positions=pt["start_pos"])
    costs, diag = ev.rollout_hybrid(pt)

    print(f"θ={osp.basename(args.theta)} | T={ref.T} (slow={slow}) | "
          f"TO 학습범위: 질량 ×0.7~1.5, 마찰 ×0.5~1.5")
    print("질량×  마찰×  | 낙하  평균pos[cm]  최종pos[cm]  평균arti[deg]  비용")
    n_ok = 0
    for i, (m, f) in enumerate(grid):
        ok = not diag["dropped"][i]
        n_ok += ok
        mark = " " if ok else "×"
        print(f" {m:4.1f}  {f:4.1f}  |  {mark}    "
              f"{diag['mean_pos_err'][i] * 100:8.1f}  {diag['final_pos_err'][i] * 100:9.1f}  "
              f"{np.degrees(diag['mean_arti_err'][i]):11.1f}  {costs[i]:6.3f}")
    print(f"\n낙하 없음: {n_ok}/{len(grid)} 칸")
    out = osp.join("md_claude", "dexscale",
                   f"envelope_{osp.splitext(osp.basename(args.theta))[0]}.npz")
    np.savez(out, grid=np.array(grid), costs=costs, **diag)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
