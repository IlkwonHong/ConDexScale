"""입력 정보 확인: '사람 접촉 중심 앵커의 α' 규칙이 1등 파티션을 빈도 규칙보다 잘 맞히는가 (검증 파지, 지금 정답).

규칙: 파티션마다 그룹별 사람 접촉 중심(면적 가중)에 가장 가까운 표면 점을 앵커로 두고(탐색 없음),
그 앵커 조합의 14방향 α 최솟값이 가장 큰 파티션을 고른다. 입력(접촉 패치 + 물체 표면)만 쓴다.
채점: 0.95배 기준, 후보 2개 이상, Net1 / 빈도 규칙 / 무작위와 비교.
"""
import json
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from contraction_gt.contacts_io import load_grasp_contacts  # noqa: E402
from contraction_gt.object_cache import load_samples  # noqa: E402
from contraction_gt.runner_batched import CONTACT_ROOT, CACHE_ROOT, _alpha_many, _coarse_jobs  # noqa: E402
from contraction_gt.search import SearchConfig, mu_tor_from_pad  # noqa: E402
from contraction_gt.units import enumerate_partitions  # noqa: E402
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

cfg = SearchConfig()
tr = Stage1Dataset(split="train", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
top_cnt = {3: Counter(), 4: Counter()}
for _g, e in tr.samples:
    for n in (3, 4):
        if len(e[f"ids{n}"]) and e[f"p{n}"].sum() > 0:
            top_cnt[n][int(e[f"ids{n}"][e[f"p{n}"].argmax()])] += 1

dev = "cuda"
model = Stage1Net(len(va.vocab), n_patch=128).to(dev)
model.load_state_dict(torch.load("checkpoints/exp005_f1.0_s0/best.pt", map_location=dev, weights_only=False)["model"])
model.eval()
net_pick = {3: [], 4: []}
with torch.no_grad():
    for b in DataLoader(va, batch_size=256, num_workers=4):
        b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        lg, _r, _p = model(b)
        for n in (3, 4):
            net_pick[n] += lg.masked_fill(~b[f"valid_n{n}"], float("-inf")).argmax(-1).tolist()

rel_of = {}
for line in open("data/contraction_gt_300k/pilot_manifest.tsv").read().splitlines()[1:]:
    t, ds, oid, rel, ridx, _s, _n = line.split("\t")
    rel_of[f"{rel}#{ridx}"] = (ds, oid, rel, int(ridx))

res = {3: [], 4: []}
for i, (gid, e) in enumerate(va.samples):
    need = [n for n in (3, 4) if len(e[f"ids{n}"]) >= 2 and e[f"p{n}"].sum() > 0]
    if not need:
        continue
    ds, oid, rel, ridx = rel_of[gid]
    patches, ts = load_grasp_contacts(CONTACT_ROOT, rel, ridx)
    samples = load_samples(CACHE_ROOT, ds, oid, scale=ts)
    for n in need:
        parts = enumerate_partitions(patches, n)
        jobs = _coarse_jobs(samples, patches, parts, cfg)
        cid = {}
        for j in jobs:
            c = va.vocab.id_of(j["part"])
            if c not in cid:
                cid[c] = j
        cand = [c for c in e[f"ids{n}"].tolist() if c in cid]
        if len(cand) < 2:
            continue
        idx = np.array([[int(np.argmin(np.linalg.norm(samples.points - ctr, axis=1))) for ctr in cid[c]["centers"]]
                        for c in cand])
        mt = mu_tor_from_pad(cid[cand[0]]["pad_rel"] * samples.r_obj, cfg.mu_t, cfg.hertz_frac)
        a = _alpha_many([(samples, idx, mt)], cfg).min(axis=1)
        sc = dict(zip(e[f"ids{n}"].tolist(), e[f"p{n}"].tolist()))
        best = max(sc.values())
        ok = lambda c: sc.get(c, -1) >= 0.95 * best - 1e-12
        rule = cand[int(np.argmax(a))]
        prior = max(e[f"ids{n}"].tolist(), key=lambda c: top_cnt[n][c])
        res[n].append((ok(rule), ok(net_pick[n][i]), ok(prior), np.mean([ok(c) for c in sc]),
                       float(np.corrcoef(a, [sc.get(c, 0) for c in cand])[0, 1]) if len(cand) > 2 else np.nan))
    if i % 500 == 0:
        print(f"{i}/{len(va.samples)}", flush=True)

for n in (3, 4):
    r = np.array(res[n], dtype=float)
    print(f"팔 {n}대 ({len(r)}건, 후보 2개 이상): 접촉 중심 α 규칙 {r[:, 0].mean() * 100:.1f} | Net1 {r[:, 1].mean() * 100:.1f} | "
          f"빈도 {r[:, 2].mean() * 100:.1f} | 무작위 {r[:, 3].mean() * 100:.1f} | 규칙 점수와 정답 점수 상관(후보 3개 이상) 중앙값 "
          f"{np.nanmedian(r[:, 4]):.2f}")
