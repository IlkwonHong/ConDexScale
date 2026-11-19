"""검증: runner_batched._rho_many 의 ρ0 == metric.compute_rho(w_req=0) (표본 200 조합)."""
import sys
sys.path.insert(0, ".")
from types import SimpleNamespace

import numpy as np
import torch

from contraction_gt.runner_batched import _rho_many
from contraction_gt.metric import compute_rho

rng = np.random.default_rng(0)
B, N = 200, 3
pts = rng.normal(size=(B * N, 3))
nrm = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
samples = SimpleNamespace(points=pts, normals=nrm, com=np.zeros(3), r_obj=1.3)
combos = np.arange(B * N).reshape(B, N)
cfg = SimpleNamespace(device="cuda", mu_t=0.6)
mt = 0.02
r = _rho_many([(samples, combos, mt)], cfg).min(axis=1)
o = compute_rho(torch.as_tensor(pts[combos]), torch.as_tensor(nrm[combos]),
                torch.zeros(B, 6, dtype=torch.float64),
                r_obj=1.3, mu_t=0.6, mu_tor=mt, backend="reference")
ref = o["rho"].numpy()
err = np.abs(r - ref)
print(f"ρ0 배치 vs 참조(HiGHS): 최대 오차 {err.max():.2e}, 중앙값 {np.median(ref):.3f}, "
      f"0 아닌 비율 {np.mean(ref > 1e-6):.2f}")
sys.exit(0 if err.max() < 1e-4 else 1)
