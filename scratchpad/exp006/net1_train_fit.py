"""Net1 이 학습 파지 자체를 맞히는가 (외우기 확인). 학습 파지 4,000개 표본에서 Net1 / 빈도 규칙 / 무작위, 0.95배 기준."""
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

tr = Stage1Dataset(split="train", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
top_cnt = {3: Counter(), 4: Counter()}
for _g, e in tr.samples:
    for n in (3, 4):
        if len(e[f"ids{n}"]) and e[f"p{n}"].sum() > 0:
            top_cnt[n][int(e[f"ids{n}"][e[f"p{n}"].argmax()])] += 1
idx = list(range(0, len(tr), len(tr) // 4000))[:4000]
dev = "cuda"
for ck in ("checkpoints/exp005_f1.0_s0/best.pt", "checkpoints/exp005_f1.0_s0/last.pt"):
    model = Stage1Net(len(tr.vocab), n_patch=128).to(dev)
    sd = torch.load(ck, map_location=dev, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    picks = {3: [], 4: []}
    with torch.no_grad():
        for b in DataLoader(Subset(tr, idx), batch_size=256, num_workers=4):
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits, _r, _p = model(b)
            for n in (3, 4):
                picks[n] += logits.masked_fill(~b[f"valid_n{n}"], float("-inf")).argmax(-1).tolist()
    for n in (3, 4):
        h_net, h_pri, h_rnd, m = [], [], [], 0
        for j, i in enumerate(idx):
            e = tr.samples[i][1]
            ids, p = e[f"ids{n}"], e[f"p{n}"]
            if len(ids) < 2 or p.sum() <= 0:      # 후보 1개는 자동 정답이라 뺌
                continue
            sc = dict(zip(ids.tolist(), p.tolist()))
            b_ = max(sc.values())
            h_net.append(sc.get(picks[n][j], -1) >= 0.95 * b_)
            h_pri.append(sc[max(ids.tolist(), key=lambda c: top_cnt[n][c])] >= 0.95 * b_)
            h_rnd.append(np.mean([s >= 0.95 * b_ for s in sc.values()]))
        print(f"{ck.split('/')[-2]}/{ck.split('/')[-1]} (에폭 {sd['epoch']}) 학습 파지 팔 {n}대, 후보 2개 이상 {len(h_net)}건: "
              f"Net1 {np.mean(h_net) * 100:.1f} | 빈도 {np.mean(h_pri) * 100:.1f} | 무작위 {np.mean(h_rnd) * 100:.1f}")
