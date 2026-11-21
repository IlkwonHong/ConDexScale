"""EXP-008 파지 목록: 학습 6.5% 부분집합(EXP-005 와 같은 물체 해시 규칙, frac 0.065) + 검증 파지 전부."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import hashlib  # noqa: E402

from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402

tr = Stage1Dataset(split="train", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
u = lambda gid: int(hashlib.md5(f"frac:{tr.obj_of[gid][1]}".encode()).hexdigest()[:8], 16) / 16**8
g_tr = sorted(g for g, _e in tr.samples if u(g) < 0.065)
g_va = sorted(g for g, _e in va.samples)
Path("scratchpad/exp008/gids_train065.txt").write_text("\n".join(g_tr))
Path("scratchpad/exp008/gids_val.txt").write_text("\n".join(g_va))
Path("scratchpad/exp008/gids_all.txt").write_text("\n".join(g_tr + g_va))
print(f"학습 6.5% {len(g_tr)} / 검증 {len(g_va)}")
