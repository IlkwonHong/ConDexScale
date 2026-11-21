"""EXP-007 평가: 조건별 Net1(best/last) 의 0.95배 정확도 — 학습 파지 4,000 표본 / 검증 파지, 후보 2개 이상만.
비교: 빈도 규칙(학습 파지 전체 기준 1등 빈도), 무작위 기댓값.
주의: 조건 (가)~(다) 모두 학습 25% 부분집합으로 학습했으므로 '학습 파지 표본'은 그 부분집합에서 뽑는다.
"""
import hashlib
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

tr = Stage1Dataset(split="train", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
u = lambda gid: int(hashlib.md5(f"frac:{tr.obj_of[gid][1]}".encode()).hexdigest()[:8], 16) / 16**8
sub = [i for i, (g, _e) in enumerate(tr.samples) if u(g) < 0.25]
top_cnt = {3: Counter(), 4: Counter()}
for i in sub:
    e = tr.samples[i][1]
    for n in (3, 4):
        if len(e[f"ids{n}"]) and e[f"p{n}"].sum() > 0:
            top_cnt[n][int(e[f"ids{n}"][e[f"p{n}"].argmax()])] += 1
tr_idx = sub[:: max(1, len(sub) // 4000)][:4000]
dev = "cuda"


def picks_of(model, ds, idx):
    out = {3: [], 4: []}
    with torch.no_grad():
        for b in DataLoader(Subset(ds, idx), batch_size=256, num_workers=4):
            b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg, _r, _p = model(b)
            for n in (3, 4):
                out[n] += lg.masked_fill(~b[f"valid_n{n}"], float("-inf")).argmax(-1).tolist()
    return out


def score(ds, idx, picks):
    res = {}
    for n in (3, 4):
        hn, hp, hr = [], [], []
        for j, i in enumerate(idx):
            e = ds.samples[i][1]
            ids, p = e[f"ids{n}"], e[f"p{n}"]
            if len(ids) < 2 or p.sum() <= 0:
                continue
            sc = dict(zip(ids.tolist(), p.tolist()))
            b_ = max(sc.values())
            hn.append(sc.get(picks[n][j], -1) >= 0.95 * b_ - 1e-12)
            hp.append(sc[max(ids.tolist(), key=lambda c: top_cnt[n][c])] >= 0.95 * b_ - 1e-12)
            hr.append(np.mean([s >= 0.95 * b_ - 1e-12 for s in sc.values()]))
        res[n] = (np.mean(hn) * 100, np.mean(hp) * 100, np.mean(hr) * 100, len(hn))
    return res


va_idx = list(range(len(va)))
first = True
for m in ("prop", "set95", "top1"):
    for which in ("best", "last"):
        ck = torch.load(f"checkpoints/exp007_{m}_f0.25_s0/{which}.pt", map_location=dev, weights_only=False)
        model = Stage1Net(len(tr.vocab), n_patch=128).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        rt = score(tr, tr_idx, picks_of(model, tr, tr_idx))
        rv = score(va, va_idx, picks_of(model, va, va_idx))
        if first:
            print(f"기준선 — 학습 파지: 빈도 N3 {rt[3][1]:.1f} N4 {rt[4][1]:.1f}, 무작위 N3 {rt[3][2]:.1f} N4 {rt[4][2]:.1f} "
                  f"({rt[3][3]}/{rt[4][3]}건) | 검증: 빈도 N3 {rv[3][1]:.1f} N4 {rv[4][1]:.1f}, "
                  f"무작위 N3 {rv[3][2]:.1f} N4 {rv[4][2]:.1f} ({rv[3][3]}/{rv[4][3]}건)")
            first = False
        print(f"{m:5s} {which} (에폭 {ck['epoch']:2d}): 학습 파지 N3 {rt[3][0]:.1f} N4 {rt[4][0]:.1f} | "
              f"검증 N3 {rv[3][0]:.1f} N4 {rv[4][0]:.1f}", flush=True)
