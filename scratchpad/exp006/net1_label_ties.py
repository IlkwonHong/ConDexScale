"""Net1 정답(파티션 1등)이 시드에 따라 바뀌는가, 동률에 가까운 파티션이 얼마나 많은가 (200 파지 × 3시드, rho64).

파지 × N(3,4) 마다 파티션별 점수 = 그 파티션 레코드들(패드 반경별) 중 최대 ρ0.
- 시드 간 1등 파티션 일치율
- 1등과 2등 점수 차가 5% / 10% 이내인 비율 (동률 근접)
- 3시드 최댓값으로 점수를 매겼을 때 1등이 시드 0 과 같은 비율
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
gids = set(Path("scratchpad/exp006/pilot_gids_200.txt").read_text().split())


def scores(sub):
    out = defaultdict(dict)       # (gid, N) -> {partition: best ρ0}
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        g, p, n, a = z["grasp_id"], z["partition"], z["n_anchors"], z["alpha"]
        for i in range(len(g)):
            if str(g[i]) not in gids or int(n[i]) < 3:
                continue
            k = (str(g[i]), int(n[i]))
            v = float(a[i].min())
            out[k][str(p[i])] = max(out[k].get(str(p[i]), -1), v)
    return out


S = [scores(f"search_records_exp006_rho64_s{s}") for s in (0, 1, 2)]
keys = sorted(set(S[0]) & set(S[1]) & set(S[2]))
for N in (3, 4):
    ks = [k for k in keys if k[1] == N and len(S[0][k]) >= 2 and max(S[0][k].values()) > 1e-6]
    top = lambda d: max(d, key=d.get)
    agree = np.mean([top(S[0][k]) == top(S[1][k]) for k in ks] + [top(S[0][k]) == top(S[2][k]) for k in ks])
    gaps = []
    for k in ks:
        v = sorted(S[0][k].values(), reverse=True)
        gaps.append((v[0] - v[1]) / v[0])
    gaps = np.array(gaps)
    best3 = [{p: max(S[s][k].get(p, -1) for s in range(3)) for p in S[0][k]} for k in ks]
    agree3 = np.mean([top(b) == top(S[0][k]) for b, k in zip(best3, ks)])
    npart = np.median([len(S[0][k]) for k in ks])
    print(f"N={N}: 파지 {len(ks)}, 파티션 수 중앙값 {npart:.0f} | 시드 간 1등 파티션 일치 {agree:.2f} | "
          f"1·2등 차 5% 이내 {np.mean(gaps < 0.05):.2f}, 10% 이내 {np.mean(gaps < 0.10):.2f} | "
          f"3시드 최댓값 1등 = 시드0 1등 {agree3:.2f}")
