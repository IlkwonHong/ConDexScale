"""정답 시드 수(1/2/3)별 안정성 — 편향 없는 측정 (시드 4개, 200 파지, rho64).

정답 = k개 시드의 파티션별 최댓값, 예측 = 정답에 쓰지 않은 시드 하나의 1등 파티션. 모든 조합 평균.
사용: python seed_count_effect.py <subdir 접두사>   (기본 search_records_exp006_rho64_s → s0..s3)
"""
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
gids = set(Path("scratchpad/exp006/pilot_gids_200.txt").read_text().split())
prefix = sys.argv[1] if len(sys.argv) > 1 else "search_records_exp006_rho64_s"
seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else "0,1,2,3".split(","))]
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


S = {s: scores(f"{prefix}{s}") for s in seeds}
keys = [k for k in sorted(set.intersection(*[set(S[s]) for s in seeds]))
        if len(S[seeds[0]][k]) >= 2 and max(S[seeds[0]][k].values()) > 1e-6]
top = lambda d: max(d, key=d.get)
for N in (3, 4):
    ks = [k for k in keys if k[1] == N]
    parts = []
    for m in range(1, len(seeds)):
        res = {k_: [] for k_ in KS}
        for lab_seeds in itertools.combinations(seeds, m):
            for pred_seed in (s for s in seeds if s not in lab_seeds):
                for k in ks:
                    lab = {p: max(S[s][k].get(p, -1) for s in lab_seeds) for p in S[seeds[0]][k]}
                    b = max(lab.values())
                    got = lab.get(top(S[pred_seed][k]), -1)
                    for k_ in KS:
                        res[k_].append(got >= k_ * b - 1e-12)
        parts.append(f"{m}시드: " + ", ".join(f"≥{k_:.2f} {np.mean(res[k_]):.3f}" for k_ in KS))
    print(f"N={N} ({len(ks)} 파지) | " + " | ".join(parts))
