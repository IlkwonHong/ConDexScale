"""Net1 top-K 재채점: 출력 상위 K개 중 하나라도 정답 점수 ≥ k × 1등 이면 맞음.

K ∈ {1, 3, 5}, k ∈ {1.0, 0.95, 0.9, 0.8}. 비교: 같은 K 개를 무작위로 고른 기댓값(우연 수준),
후보 파티션 수가 K 이하라 자동으로 맞는 비율. 체크포인트 = EXP-005 100% 두 시드 평균.
"""
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

KS, TOPS = (1.0, 0.95, 0.9, 0.8), (1, 3, 5)
dev = "cuda"
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
dl = DataLoader(va, batch_size=256, num_workers=4)
acc = defaultdict(list)
for ck in ("checkpoints/exp005_f1.0_s0/best.pt", "checkpoints/exp005_f1.0_s1/best.pt"):
    model = Stage1Net(len(va.vocab), n_patch=128).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    model.eval()
    hit, rnd, cnt, small = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    nvalid = defaultdict(list)
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits, _r, _p = model(b)
            for n in (3, 4):
                t, v = b[f"target_n{n}"], b[f"valid_n{n}"]
                rows = t.sum(-1) > 0
                if not rows.any():
                    continue
                lg = logits[rows].masked_fill(~v[rows], float("-inf"))
                tt, vv = t[rows], v[rows]
                best = tt.max(-1).values
                nv = vv.sum(-1)
                nvalid[n] += nv.tolist()
                order = lg.argsort(-1, descending=True)
                for K in TOPS:
                    topk = order[:, :K]
                    okk = torch.arange(K, device=dev)[None, :] < nv[:, None]   # 유효 후보 수보다 많이 고르지 않음
                    got = torch.where(okk, tt.gather(1, topk), torch.full_like(topk, -1, dtype=tt.dtype)).max(-1).values
                    small[(n, K)] += (nv <= K).float().sum().item()
                    for k in KS:
                        good = (tt >= k * best[:, None] - 1e-12) & vv               # 정답 묶음
                        hit[(n, K, k)] += (got >= k * best - 1e-12).float().sum().item()
                        # 무작위 K개(비복원)로 묶음을 하나라도 뽑을 확률 = 1 - C(nv-g, K)/C(nv, K)
                        g = good.sum(-1).double()
                        nvd = nv.double()
                        p_miss = torch.ones_like(nvd)
                        for j in range(K):
                            p_miss = p_miss * torch.clamp((nvd - g - j) / torch.clamp(nvd - j, min=1), min=0)
                        rnd[(n, K, k)] += (1 - p_miss).sum().item()
                cnt[n] += rows.sum().item()
    for n in (3, 4):
        for K in TOPS:
            for k in KS:
                acc[(n, K, k)].append(hit[(n, K, k)] / cnt[n])
                acc[("rnd", n, K, k)].append(rnd[(n, K, k)] / cnt[n])
            acc[("small", n, K)].append(small[(n, K)] / cnt[n])
    acc[("nv", 3)] = [np.median(nvalid[3])]
    acc[("nv", 4)] = [np.median(nvalid[4])]

for n in (3, 4):
    print(f"\n팔 {n}대 (유효 파티션 수 중앙값 {acc[('nv', n)][0]:.0f})")
    for K in TOPS:
        cells = ", ".join(f"≥{k:.2f}배 {np.mean(acc[(n, K, k)]) * 100:.1f}% (무작위 {np.mean(acc[('rnd', n, K, k)]) * 100:.1f}%)" for k in KS)
        print(f"  top-{K}: {cells} | 후보 수 ≤ {K} 인 파지 {np.mean(acc[('small', n, K)]) * 100:.0f}%")
