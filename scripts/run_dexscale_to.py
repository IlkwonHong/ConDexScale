#!/usr/bin/env python3
"""DexScale Stage 2 — 오프라인 TO 엔트리포인트 (dexscale_mvp.md §2, §6).

CEM이 θ(Δu,Δv,δ knot)를 세대 단위로 진화. 매 세대 물성 K개 공유 샘플(공통 난수)
에서 폐루프 임피던스 롤아웃 총점으로 채점. 산출: θ* 1개 + 히스토리 npz.

실행 (conda dexmachina):
  # 배선 검증 (M1: θ=0 재생 + 2세대)
  /home/robin/miniforge3/envs/dexmachina/bin/python scripts/run_dexscale_to.py --smoke
  # 베이스라인 ② (δ만): M2
  ... run_dexscale_to.py --mode delta --gens 30
  # full A안 (Δu,Δv,δ): M3
  ... run_dexscale_to.py --mode full --gens 50
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from dexscale.cem import CEM  # noqa: E402
from dexscale import reference as R  # noqa: E402

# TO용 좁은 DR (설계문서 §1 부속 — 평가 그리드는 이보다 넓게, evaluate 스크립트에서)
DR_MASS = (0.7, 1.5)
DR_HINGE = (0.001, 0.02)   # 절대 damping [N·m·s]


def sample_dr(rng, n_dr, fric_lo=0.5):
    return {
        "mass_scale": rng.uniform(*DR_MASS, n_dr),
        "fric_ratio": rng.uniform(fric_lo, 1.5, n_dr),
        "hinge_damp": rng.uniform(*DR_HINGE, n_dr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "delta"], default="delta",
                    help="delta=베이스라인②(δ만) / full=A안(Δu,Δv,δ)")
    ap.add_argument("--hybrid", action="store_true",
                    help="물체 기준 앵커 + 운반 오프셋 d_max 채널 (2026-07-16 설계)")
    ap.add_argument("--warm-start", default=None,
                    help="이전 TO 결과 npz — 곡선을 새 지평선/knot로 이식해 초기 μ로 사용")
    ap.add_argument("--tilt", action="store_true",
                    help="기울인 프레스 채널 (u,v): 능동 접선력 = kp·δ·|(u,v)|, |(u,v)|≤0.3")
    ap.add_argument("--fric-lo", type=float, default=0.5,
                    help="학습 DR 마찰 하한 (tilt 사용 시 0.3 권장 — 마찰 의존 전략 규율)")
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--n-dr", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32,
                    help="패스당 동시 후보 수. 32(=128env)가 실측 최속 (2026-07-17 벤치)")
    ap.add_argument("--frames", type=int, default=None,
                    help="클립 앞부분만 사용 (원본 데모 프레임 기준, slow 적용 전)")
    ap.add_argument("--slow", type=int, default=1,
                    help="데모 감속 배수. 2=실제 사람 속도 (원본 30fps ↔ 60Hz 스텝)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="배선 검증: 소형 설정 + θ=0 평가(M1) + 2세대")
    args = ap.parse_args()

    if args.smoke:
        args.pop, args.elite, args.gens = 4, 2, 2
        args.n_dr, args.chunk = 2, 2
        args.frames = args.frames or 240

    K = args.knots
    tang = args.mode == "full"
    ref = R.BoxGuidanceReference(frame_end=args.frames, slow=args.slow)
    print(f"[ref] T={ref.T} frames (slow={args.slow}), "
          f"snap p95={np.percentile(ref.snap_dist, 95) * 100:.1f}cm", flush=True)

    from dexscale.rollout import DexScaleEvaluator  # gs.init 지연
    ev = DexScaleEvaluator(ref, n_dr=args.n_dr, chunk=args.chunk)
    print(f"[env] n_envs={ev.n_envs} (chunk {args.chunk} × dr {args.n_dr})")

    rng = np.random.default_rng(args.seed)
    dim = R.theta_dim_hybrid(K, tilt=args.tilt) if args.hybrid else R.theta_dim(K)
    lo, hi = R.theta_bounds(K, tangential=tang, hybrid=args.hybrid, tilt=args.tilt)
    mu0 = np.zeros(dim)
    if args.warm_start:
        assert args.hybrid, "warm start는 hybrid 경로 전용"
        w = np.load(args.warm_start)
        tilt_old = bool(w["tilt"]) if "tilt" in w else False
        mu0 = np.clip(R.remap_theta_hybrid(w["best_theta"], int(w["n_knots"]),
                                           int(w["frames"]), K, ref.T,
                                           tilt_old=tilt_old, tilt_new=args.tilt),
                      lo, hi)
        print(f"[warm] {args.warm_start} (K{int(w['n_knots'])},T{int(w['frames'])},"
              f"tilt={tilt_old}) → (K{K},T{ref.T},tilt={args.tilt}) 이식", flush=True)
    cem = CEM(mu0=mu0,
              sigma0=R.theta_sigma0(K, tangential=tang, hybrid=args.hybrid,
                                    tilt=args.tilt),
              pop=args.pop, n_elite=args.elite, lo=lo, hi=hi)

    def score(thetas, dr):
        if args.hybrid:
            return ev.evaluate_hybrid(ref, thetas, dr, K, tilt=args.tilt)
        return ev.evaluate(thetas, dr, K)

    # 기준점: θ=0 재생 비용
    dr0 = sample_dr(np.random.default_rng(args.seed + 9999), args.n_dr,
                    fric_lo=args.fric_lo)
    t0 = time.time()
    c0 = score(np.zeros((1, dim)), dr0)
    print(f"[M1] θ=0 비용 = {c0[0]:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    history = []
    for g in range(args.gens):
        dr = sample_dr(np.random.default_rng((args.seed, g).__hash__() & 0xFFFFFFFF),
                       args.n_dr, fric_lo=args.fric_lo)   # 세대 내 공통 난수
        thetas = cem.ask(rng)
        t0 = time.time()
        costs = score(thetas, dr)
        info = cem.tell(thetas, costs)
        info["sec"] = round(time.time() - t0, 1)
        history.append(info)
        print(f"[gen {info['gen']:3d}] best={info['best']:.4f} "
              f"elite={info['elite_mean']:.4f} pop={info['pop_mean']:.4f} "
              f"sigma={info['sigma_mean']:.4f} ({info['sec']}s)")

    tag = ("hybrid_" if args.hybrid else "") + ("tilt_" if args.tilt else "") + args.mode
    out = args.out or osp.join("data", "dexscale",
                               f"theta_{tag}_K{K}{'_smoke' if args.smoke else ''}.npz")
    os.makedirs(osp.dirname(out), exist_ok=True)
    np.savez(out, best_theta=cem.best_theta, best_cost=cem.best_cost,
             mu=cem.mu, sigma=cem.sigma, n_knots=K, mode=args.mode,
             hybrid=args.hybrid, tilt=args.tilt,
             frames=ref.T, slow=args.slow, theta0_cost=c0[0],
             history=np.array([(h["gen"], h["best"], h["elite_mean"], h["pop_mean"])
                               for h in history]))
    print(f"[saved] {out}  (best={cem.best_cost:.4f}, θ=0 기준 {c0[0]:.4f})")


if __name__ == "__main__":
    main()
