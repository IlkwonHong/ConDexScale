"""탐색 운(시드 노이즈)만으로 생기는 Net1 정확도 상한.

'예측기' = 다른 시드의 정답 1등 파티션, '채점 정답' = 시드 0 의 파티션 점수 (ρ0, 200 파지).
동률 허용 기준 k (고른 파티션 점수 ≥ k × 1등) 별로 맞힌 비율. 예측기가 완벽해도 이 이상은 못 나온다.
비교: 3시드 최댓값 점수로 채점할 때 (노이즈가 줄어든 정답) 다른 시드 1등의 정확도.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
gids = set(Path("scratchpad/exp006/pilot_gids_200.txt").read_text().split())
KS = (1.0, 0.95, 0.9, 0.8)


def scores(sub):
    out = defaultdict(dict)
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        g, p, n, a = z["grasp_id"], z["partition"], z["n_anchors"], z["alpha"]
        for i in range(len(g)):
            if str(g[i]) in gids and int(n[i]) >= 3:
                k = (str(g[i]), int(n[i]))
                out[k][str(p[i])] = max(out[k].get(str(p[i]), -1), float(a[i].min()))
    return out


S = [scores(f"search_records_exp006_rho64_s{s}") for s in (0, 1, 2)]
keys = [k for k in sorted(set(S[0]) & set(S[1]) & set(S[2])) if len(S[0][k]) >= 2 and max(S[0][k].values()) > 1e-6]
top = lambda d: max(d, key=d.get)
for N in (3, 4):
    ks = [k for k in keys if k[1] == N]
    res = {}
    for label_name, label in (("시드0 정답", lambda k: S[0][k]),
                              ("3시드 최댓값 정답", lambda k: {p: max(S[s][k].get(p, -1) for s in range(3)) for p in S[0][k]})):
        for k_ in KS:
            hits = []
            for k in ks:
                lab = label(k)
                b = max(lab.values())
                for s in (1, 2):              # 예측기 = 시드 1, 2 의 1등
                    hits.append(lab.get(top(S[s][k]), -1) >= k_ * b - 1e-12)
            res[(label_name, k_)] = np.mean(hits)
    for label_name in ("시드0 정답", "3시드 최댓값 정답"):
        print(f"N={N} ({len(ks)} 파지) 채점={label_name}: " + ", ".join(f"≥{k_:.2f}배 {res[(label_name, k_)]:.2f}" for k_ in KS))
