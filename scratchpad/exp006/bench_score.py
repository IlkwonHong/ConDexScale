"""α(14방향) 대 ρ0(정점 64) 채점 속도 비교 — 같은 조합 수."""
import sys
import time
sys.path.insert(0, ".")
from types import SimpleNamespace

import numpy as np
import torch

from contraction_gt.runner_batched import _alpha_many, _rho_many

rng = np.random.default_rng(0)
B, N = 6000, 3
pts = rng.normal(size=(B * N, 3))
nrm = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
samples = SimpleNamespace(points=pts, normals=nrm, com=np.zeros(3), r_obj=1.0)
combos = np.arange(B * N).reshape(B, N)
cfg = SimpleNamespace(device="cuda", mu_t=0.6)
for name, fn in (("alpha14", _alpha_many), ("rho64", _rho_many)):
    fn([(samples, combos[:100], 0.02)], cfg)
    torch.cuda.synchronize()
    t = time.time()
    fn([(samples, combos, 0.02)], cfg)
    torch.cuda.synchronize()
    print(f"{name}: {B} 조합 {time.time() - t:.2f}s")
