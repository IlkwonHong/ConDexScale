#!/usr/bin/env python3
"""Onset 해부 프로브 — θ=0 재생에서 "힘 포화 vs 접촉 상실"의 인과 순서를 측정.

θ=0(또는 지정 θ)을 단일 nominal 물성에서 재생하며 매 스텝 구체 위치·제어힘·물체
상태를 기록하고, 실물 박스 표면까지의 거리(접촉 여부)와 힘 크기의 사건 순서를 출력.

실행:
  python scripts/probe_dexscale_onset.py --slow 1 --frames 260
  python scripts/probe_dexscale_onset.py --slow 2 --frames 260
"""
import argparse
import os.path as osp
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from dexscale import reference as R  # noqa: E402

F_MAX = 20.0


def surface_separation(ref, sphere_pos, obj_qpos):
    """(T,4,3) 구체 중심 → 실물(로그된) 박스 표면과의 간극 [m] (0≈접촉)."""
    T = sphere_pos.shape[0]
    pos, quat, arti = obj_qpos[:, :3], obj_qpos[:, 3:7], obj_qpos[:, 7]
    Rb = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    sep = np.empty((T, R.N_SPHERES))
    for t in range(T):
        loc_b = Rb[t].inv().apply(sphere_pos[t] - pos[t])
        loc_t = Rotation.from_rotvec([0, 0, +arti[t]]).apply(loc_b)
        db, _ = ref._tree["bottom"].query(loc_b)
        dt_, _ = ref._tree["top"].query(loc_t)
        sep[t] = np.minimum(db, dt_) - R.SPHERE_R
    return sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow", type=int, default=1)
    ap.add_argument("--frames", type=int, default=260)
    ap.add_argument("--theta", default=None)
    ap.add_argument("--hybrid", action="store_true", help="물체 기준 앵커 + 운반 경로")
    ap.add_argument("--grip-delta", type=float, default=0.0,
                    help="[hybrid] 전 구간 프레스 깊이 [mm] (E1 쥐기 검증용 프리셋)")
    ap.add_argument("--dmax", type=float, default=0.0,
                    help="[hybrid] 운반 오프셋 포화 [mm] (0=쥐기만)")
    args = ap.parse_args()

    K = 5
    tilt = False
    if args.theta:
        z = np.load(args.theta)
        theta, K = z["best_theta"], int(z["n_knots"])
        tilt = bool(z["tilt"]) if "tilt" in z else False
    elif args.hybrid:
        theta = np.zeros(R.theta_dim_hybrid(K))
        th = theta[: R.theta_dim(K)].reshape(R.N_SPHERES, R.N_CHANNELS, K)
        th[:, 2, :] = args.grip_delta / 1000.0
        theta[R.theta_dim(K):] = args.dmax / 1000.0
    else:
        theta = np.zeros(R.theta_dim(K))

    ref = R.BoxGuidanceReference(frame_end=args.frames, slow=args.slow)
    from dexscale.rollout import DexScaleEvaluator
    import torch
    ev = DexScaleEvaluator(ref, n_dr=1, chunk=1)
    ev.set_physics(np.array([1.0]), np.array([1.0]), np.array([0.005]))
    if args.hybrid:
        pt = ev.plan_to_tensors([ref.build_hybrid_plan(theta, K, tilt=tilt)])
        ev.reset(start_positions=pt["start_pos"])
        costs, diag = ev.rollout_hybrid(pt, log_states=True)
    else:
        refs = torch.as_tensor(ref.build(theta, K)[None], dtype=torch.float32,
                               device=ev.DEV)
        ev.reset()
        costs, diag = ev.rollout(refs, log_states=True)
    log = diag["log"]

    sep = surface_separation(ref, log["sphere_pos"], log["obj_qpos"])   # (T,4)
    fmag = np.linalg.norm(log["force"], axis=-1)                        # (T,4)
    perr = np.linalg.norm(log["obj_qpos"][:, :3] - ref.obj_pos, axis=1)

    dt = 1.0 / 60.0
    disp = np.linalg.norm(ref.obj_pos - ref.obj_pos[0], axis=1)
    onset = int(np.argmax(disp > 0.005))
    acc = np.gradient(np.gradient(ref.obj_pos, dt, axis=0), dt, axis=0)
    a_peak = np.abs(acc[onset:onset + 60]).max() if onset else 0.0
    m_box = 1.094
    print(f"=== slow={args.slow} | T={ref.T} | cost={costs[0]:.3f} | dropped={bool(diag['dropped'][0])}")
    print(f"demo onset frame {onset} | onset+60 내 |a|peak {a_peak:.2f} m/s² "
          f"(요구력 개산 m(a+g) ≈ {m_box * (a_peak + 9.81):.1f} N, 4팔 예산 {4 * F_MAX:.0f} N)")

    names = ["L1", "L2", "R1", "R2"]
    for s in range(4):
        c = sep[:, s] < 0.005
        sat = fmag[:, s] > 0.975 * F_MAX
        # onset 이후 3프레임 연속 이탈/포화 첫 시점
        def first_run(mask, start, k=3):
            for t in range(start, len(mask) - k):
                if mask[t:t + k].all():
                    return t
            return -1
        t_sep = first_run(~c, max(onset - 30, 0))
        t_sat = first_run(sat, max(onset - 30, 0))
        print(f"{names[s]}: 접촉이탈 1st frame {t_sep:4d} | 힘포화 1st frame {t_sat:4d} "
              f"| onset시 sep {sep[onset, s] * 1000:5.1f} mm, |F| {fmag[onset, s]:5.1f} N")

    print("\nframe :  box_err[cm]  min_sep[mm]  n_contact  max|F|[N]")
    for t in range(max(onset - 10, 0), min(onset + 50, ref.T), 5):
        print(f"{t:5d} :  {perr[t] * 100:9.1f}  {sep[t].min() * 1000:9.1f}  "
              f"{int((sep[t] < 0.005).sum()):9d}  {fmag[t].max():8.1f}")


if __name__ == "__main__":
    main()
