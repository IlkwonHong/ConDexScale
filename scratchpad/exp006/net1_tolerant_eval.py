"""기존 Net1 을 '동률 허용' 기준으로 다시 채점 (학습 없음).

맞음 기준: 고른 파티션의 정답 점수(학습 목표 분포 = 합격 레코드의 min-α 비례) ≥ k × 1등 점수, k ∈ {1(=top-1), 0.95, 0.9, 0.8}.
함께: 정답 쪽 동률 근접 비율(2등 ≥ 0.95 × 1등). 대상: 검증 4,648 파지, 체크포인트 = EXP-005 100% 두 시드 + stage1_v11.
"""
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

KS = (1.0, 0.95, 0.9, 0.8)
dev = "cuda"
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
dl = DataLoader(va, batch_size=256, num_workers=4)
for ck in ("checkpoints/exp005_f1.0_s0/best.pt", "checkpoints/exp005_f1.0_s1/best.pt", "checkpoints/stage1_v11/best.pt"):
    model = Stage1Net(len(va.vocab), n_patch=128).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    model.eval()
    hit, cnt, tie = defaultdict(float), defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits, _reg, _pl = model(b)
            for n in (3, 4):
                t, v = b[f"target_n{n}"], b[f"valid_n{n}"]
                rows = t.sum(-1) > 0
                if not rows.any():
                    continue
                lg = logits[rows].masked_fill(~v[rows], float("-inf"))
                tt = t[rows]
                pred = lg.argmax(-1)
                best = tt.max(-1).values
                got = tt.gather(1, pred[:, None])[:, 0]
                for k in KS:
                    hit[(n, k)] += (got >= k * best - 1e-12).float().sum().item()
                second = tt.topk(2, dim=-1).values[:, 1]
                tie[n] += (second >= 0.95 * best).float().sum().item()
                cnt[n] += rows.sum().item()
    line = " | ".join(f"N{n}: " + ", ".join(f"≥{k:.2f}배 {hit[(n, k)] / cnt[n]:.3f}" for k in KS)
                      + f" (정답 동률 근접 {tie[n] / cnt[n]:.2f}, {int(cnt[n])}개)" for n in (3, 4))
    print(f"{ck}: {line}", flush=True)
