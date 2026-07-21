#!/usr/bin/env python3
"""DexScale 롤아웃 시각화 — M1(θ=0) 눈검증 및 임의 θ 재생 (dexscale_mvp.md §6).

단일 물성(기본 nominal)에서 θ를 임피던스로 재생하고 mp4 + 스냅샷 PNG + 오차
trace를 저장한다. 판정은 novoc 연속 rollout — 이 스크립트가 곧 Stage 3 배포 코드.

실행 (conda dexmachina):
  # M1: θ=0 (guidance를 임피던스로 재생)
  python scripts/visualize_dexscale_rollout.py
  # 최적화 결과 재생
  python scripts/visualize_dexscale_rollout.py --theta data/dexscale/theta_delta_K5.npz
  # unseen 물성
  python scripts/visualize_dexscale_rollout.py --mass-scale 2.0 --fric 0.3
"""
import argparse
import os
import os.path as osp
import sys

import imageio
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from dexscale import reference as R  # noqa: E402

SNAP_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default=None, help="TO 결과 npz (없으면 θ=0 = M1)")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--slow", type=int, default=None,
                    help="데모 감속 배수 (기본: theta npz에 저장된 값, 없으면 1)")
    ap.add_argument("--mass-scale", type=float, default=1.0)
    ap.add_argument("--fric", type=float, default=1.0)
    ap.add_argument("--hinge-damp", type=float, default=0.005)
    ap.add_argument("--record-every", type=int, default=2)
    ap.add_argument("--tag", default=None, help="출력 파일명 태그")
    args = ap.parse_args()

    slow = args.slow
    hybrid = False
    tilt = False
    if args.theta:
        z = np.load(args.theta)
        theta, K = z["best_theta"], int(z["n_knots"])
        hybrid = bool(z["hybrid"]) if "hybrid" in z else False
        tilt = bool(z["tilt"]) if "tilt" in z else False
        if slow is None:
            slow = int(z["slow"]) if "slow" in z else 1
        tag = args.tag or osp.splitext(osp.basename(args.theta))[0]
    else:
        K = 5
        theta = np.zeros(R.theta_dim(K))
        tag = args.tag or "m1_theta0"
    slow = slow or 1
    tag += f"_m{args.mass_scale}_f{args.fric}"

    ref = R.BoxGuidanceReference(frame_end=args.frames, slow=slow)
    from dexscale.rollout import DexScaleEvaluator  # gs.init 지연
    ev = DexScaleEvaluator(ref, n_dr=1, chunk=1, with_camera=True)

    import torch
    ev.set_physics(np.array([args.mass_scale]), np.array([args.fric]),
                   np.array([args.hinge_damp]))
    if hybrid:
        pt = ev.plan_to_tensors([ref.build_hybrid_plan(theta, K, tilt=tilt)])
        ev.reset(start_positions=pt["start_pos"])
        costs, diag = ev.rollout_hybrid(pt, record_every=args.record_every)
    else:
        refs = torch.as_tensor(ref.build(theta, K)[None], dtype=torch.float32,
                               device=ev.DEV)
        ev.reset()
        costs, diag = ev.rollout(refs, record_every=args.record_every)

    out_dir = osp.join("md_claude", "dexscale")
    os.makedirs(out_dir, exist_ok=True)
    frames = diag["frames"]
    mp4 = osp.join(out_dir, f"{tag}.mp4")
    imageio.mimsave(mp4, frames, fps=max(1, round(60 / args.record_every)))
    for frac in SNAP_FRACS:
        i = min(int(frac * (len(frames) - 1)), len(frames) - 1)
        imageio.imwrite(osp.join(out_dir, f"{tag}_p{int(frac * 100):03d}.png"), frames[i])

    pt, at = diag["pos_trace"], diag["arti_trace"]
    print(f"[cost] 스텝평균 = {costs[0]:.4f} | dropped = {bool(diag['dropped'][0])}")
    print(f"[final] pos err = {diag['final_pos_err'][0] * 100:.1f} cm, "
          f"arti err = {np.degrees(diag['final_arti_err'][0]):.1f} deg")
    print("[trace] (프레임 : pos err cm / arti err deg)")
    for frac in SNAP_FRACS:
        i = min(int(frac * (len(pt) - 1)), len(pt) - 1)
        print(f"  t={i * args.record_every:4d} ({int(frac * 100):3d}%) : "
              f"{pt[i] * 100:6.1f} cm / {np.degrees(at[i]):6.1f} deg")
    np.savez(osp.join(out_dir, f"{tag}_trace.npz"), pos_trace=pt, arti_trace=at,
             record_every=args.record_every, cost=costs[0])
    print(f"[saved] {mp4} + 스냅샷 {len(SNAP_FRACS)}장")


if __name__ == "__main__":
    main()
