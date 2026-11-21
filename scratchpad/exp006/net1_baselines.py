"""Net1 대 단순 기준선 (검증 파지, 정답 = 지금 GT).

- 무작위: 유효 후보 중 하나를 고르게 (기댓값)
- 빈도(사전) 기준선: 입력을 보지 않고, 학습 파지에서 1등이었던 횟수가 가장 많은 파티션(유효 후보 중)을 고름
- 빈도(점수합) 기준선: 학습 파지에서 목표 확률 p 합이 가장 큰 파티션을 고름
- 후보 수별로 나눠 Net1 과 비교 (후보 1개 = 자동 정답)
검산: 무작위 기댓값을 직접 표본 추출로도 계산.
"""
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

KS = (1.0, 0.95, 0.9)
tr = Stage1Dataset(split="train", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)

top_cnt = {3: Counter(), 4: Counter()}
psum = {3: defaultdict(float), 4: defaultdict(float)}
for _gid, e in tr.samples:
    for n in (3, 4):
        ids, p = e[f"ids{n}"], e[f"p{n}"]
        if len(ids) and p.sum() > 0:
            top_cnt[n][int(ids[p.argmax()])] += 1
            for c, w in zip(ids, p):
                psum[n][int(c)] += float(w)

# 검증: 파지별 (후보 id, 점수) 와 Net1 선택
dev = "cuda"
model = Stage1Net(len(va.vocab), n_patch=128).to(dev)
model.load_state_dict(torch.load("checkpoints/exp005_f1.0_s0/best.pt", map_location=dev, weights_only=False)["model"])
model.eval()
net_pick = {3: [], 4: []}
with torch.no_grad():
    for b in DataLoader(va, batch_size=256, num_workers=4):
        b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        logits, _r, _p = model(b)
        for n in (3, 4):
            lg = logits.masked_fill(~b[f"valid_n{n}"], float("-inf"))
            net_pick[n] += lg.argmax(-1).tolist()

rng = np.random.default_rng(0)
for n in (3, 4):
    rows = []
    for i, (_gid, e) in enumerate(va.samples):
        ids, p = e[f"ids{n}"], e[f"p{n}"]
        if len(ids) == 0 or p.sum() <= 0:
            continue
        score = dict(zip(ids.tolist(), p.tolist()))
        best = max(score.values())
        prior = max(ids.tolist(), key=lambda c: top_cnt[n][c])
        prior2 = max(ids.tolist(), key=lambda c: psum[n][c])
        rand_draw = int(rng.choice(ids))
        rows.append(dict(nv=len(ids), best=best, score=score, net=net_pick[n][i], prior=prior, prior2=prior2,
                         rand=rand_draw))
    print(f"\n팔 {n}대: 채점 {len(rows)}건")
    for k in KS:
        def acc(key, sel=lambda r: True):
            rs = [r for r in rows if sel(r)]
            return np.mean([r["score"].get(r[key], -1) >= k * r["best"] - 1e-12 for r in rs]) * 100
        exp_rand = np.mean([np.mean([s >= k * r["best"] - 1e-12 for s in r["score"].values()]) for r in rows]) * 100
        print(f"  ≥{k:.2f}배: Net1 {acc('net'):.1f} | 빈도(1등 횟수) {acc('prior'):.1f} | 빈도(점수합) {acc('prior2'):.1f} | "
              f"무작위 기댓값 {exp_rand:.1f} (표본 추출 검산 {acc('rand'):.1f})")
    for lo, hi, name in ((1, 1, "후보 1개"), (2, 3, "후보 2~3개"), (4, 999, "후보 4개 이상")):
        sel = lambda r, lo=lo, hi=hi: lo <= r["nv"] <= hi
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        net = np.mean([r["score"].get(r["net"], -1) >= 0.95 * r["best"] - 1e-12 for r in rs]) * 100
        pri = np.mean([r["score"].get(r["prior"], -1) >= 0.95 * r["best"] - 1e-12 for r in rs]) * 100
        rnd = np.mean([np.mean([s >= 0.95 * r["best"] - 1e-12 for s in r["score"].values()]) for r in rs]) * 100
        print(f"  [{name}] {len(rs)}건 ({len(rs) / len(rows) * 100:.0f}%), 0.95배: Net1 {net:.1f} | 빈도 {pri:.1f} | 무작위 {rnd:.1f}")
