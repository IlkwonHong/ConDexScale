#!/usr/bin/env python3
"""dexscale 시뮬-무관 코어 단위테스트: knots / CEM toy 수렴 / reference θ=0 빌드.

실행: /home/robin/miniforge3/envs/dexmachina/bin/python scripts/test_dexscale_core.py
"""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from dexscale.knots import interp_knots, refine_knots  # noqa: E402
from dexscale.cem import CEM  # noqa: E402
from dexscale import reference as R  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------- knots ----------
kv = np.array([[0.0, 8.0, 12.0, 6.0, 0.0]])
curve = interp_knots(kv, 889)
check("knots endpoints", curve[0, 0] == 0.0 and curve[0, -1] == 0.0)
check("knots mid interp", abs(curve[0, 333] - np.interp(333, np.linspace(0, 888, 5), kv[0])) < 1e-9)
kv2 = refine_knots(kv, 9, 889)
curve2 = interp_knots(kv2, 889)
check("refine preserves curve", np.abs(curve - curve2).max() < 1.0,
      f"(max diff {np.abs(curve - curve2).max():.3f})")

# ---------- CEM toy: noisy quadratic, D=60 ----------
rng = np.random.default_rng(0)
target = rng.uniform(-1, 1, 60)
opt = CEM(mu0=np.zeros(60), sigma0=0.4 * np.ones(60), pop=64, n_elite=8)
for g in range(80):
    thetas = opt.ask(rng)
    costs = ((thetas - target[None]) ** 2).sum(1) + 0.01 * rng.standard_normal(64)
    info = opt.tell(thetas, costs)
err = np.abs(opt.mu - target).max()
check("CEM converges (D=60)", err < 0.15, f"(max |mu-target| = {err:.3f}, best={opt.best_cost:.4f})")

# ---------- reference: θ=0 / random θ ----------
ref = R.BoxGuidanceReference(frame_end=120, n_cloud=3000)
K = 5
refs0 = ref.build(np.zeros(R.theta_dim(K)), K)
check("refs shape/finite", refs0.shape == (120, 4, 3) and np.isfinite(refs0).all())
snap_p95 = np.percentile(ref.snap_dist, 95)
check("surface snap sane", snap_p95 < 0.05, f"(kpt→표면 스냅거리 p95 = {snap_p95 * 100:.1f} cm)")
step_jump = np.linalg.norm(np.diff(refs0, axis=0), axis=-1)
check("refs continuity", np.percentile(step_jump, 99) < 0.05,
      f"(프레임간 이동 p99 = {np.percentile(step_jump, 99) * 100:.2f} cm, max = {step_jump.max() * 100:.2f} cm)")

rng2 = np.random.default_rng(1)
lo, hi = R.theta_bounds(K)
theta_r = np.clip(rng2.normal(0, 0.01, R.theta_dim(K)), lo, hi)
refs_r = ref.build(theta_r, K)
dev = np.linalg.norm(refs_r - refs0, axis=-1).max()
check("random θ bounded", dev < R.DU_MAX * 2 + R.DELTA_MAX, f"(max 편차 {dev * 100:.2f} cm)")

top_frac = ref.part_ids.mean()
print(f"info: top part 접촉 비율 {top_frac:.2f} (오른손 뚜껑 우세 예상)")
print(f"\n=== {'ALL PASS' if fails == 0 else f'{fails} FAIL'} ===")
sys.exit(1 if fails else 0)
