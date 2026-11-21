"""Net1 손가락 노드가 접촉 위치 정보를 갖고 있는가 (선형 탐침).

학습된 Net1(v1.1 설정, 접촉 위치 입력 0)의 트렁크 출력에서 손가락 노드 6개 표현을 꺼내,
그 손가락의 실제 접촉 위치(표면 투영점 / r_obj, 물체 좌표)를 능선 회귀로 복원 → 검증 R².
비교: 같은 탐침을 '손가락 번호 + 마디 비율'(입력에 직접 있는 것)만으로 했을 때, 트렁크 입력 직후(첫 블록 전) 표현.
함께: unit_feat[9] (닿았는지) 통계.
"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset, N_UNIT  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

ds_geo = Stage1Dataset(split="val", augment=False, contact_geom=True, sigma_max=0.08, n_patch=128)   # 정답 위치 꺼내기용
ds = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
idx = list(range(0, len(ds), 1))
dev = "cuda"
model = Stage1Net(len(ds.vocab), n_patch=128).to(dev)
model.load_state_dict(torch.load("checkpoints/exp005_f1.0_s0/best.pt", map_location=dev, weights_only=False)["model"])
model.eval()
H_out, H_in, Y, M, F_in = [], [], [], [], []
with torch.no_grad():
    for b, bg in zip(DataLoader(Subset(ds, idx), batch_size=256, num_workers=4),
                     DataLoader(Subset(ds_geo, idx), batch_size=256, num_workers=4)):
        b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        x, _h = model.encode(b)
        H_out.append(x[:, 128:128 + N_UNIT].float().cpu())
        uid = model.unit_id(torch.arange(N_UNIT, device=dev)).expand(x.shape[0], -1, -1)
        H_in.append(model.unit_enc(torch.cat([b["unit_feat"], uid], -1)).float().cpu())
        F_in.append(b["unit_feat"][..., 6:10].float().cpu())
        Y.append(bg["unit_feat"][..., :3].float())
        M.append(bg["unit_feat"][..., 9].float())
H_out, H_in, Y, M, F_in = (torch.cat(t).numpy() for t in (H_out, H_in, Y, M, F_in))
mask = M > 0.5
print(f"닿았는지(unit_feat[9]) = 1 인 비율: {mask.mean():.3f} (손가락별 {np.round(mask.mean(0), 3).tolist()})")


def probe(Xall, name):
    r2s = []
    for u in range(N_UNIT):
        sel = mask[:, u]
        X, y = Xall[sel, u], Y[sel, u]
        n = len(X)
        tr, te = slice(0, int(n * 0.7)), slice(int(n * 0.7), n)
        Xa = np.c_[X, np.ones(n)]
        lam = 1.0
        W = np.linalg.solve(Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1]), Xa[tr].T @ y[tr])
        pred = Xa[te] @ W
        r2s.append(1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[tr].mean(0)) ** 2).sum())
    print(f"{name:34s}: 접촉 위치 복원 R² 손가락별 {np.round(r2s, 2).tolist()} (평균 {np.mean(r2s):.2f})")


probe(F_in, "입력 마디 비율 + 닿음 (직접 입력)")
probe(H_in, "첫 블록 전 손가락 노드 표현")
probe(H_out, "트렁크 출력 손가락 노드 표현")
