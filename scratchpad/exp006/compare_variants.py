"""B0/B1/B2 비교: 시간, ρ0 품질, 시드 간 정답 위치 일치, 파티션 1등 안정성(1~3시드, 편향 없음)."""
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
RUNS = Path("harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-006-gt-v2-pilot/runs")
gids = set(Path("scratchpad/exp006/pilot_gids_200.txt").read_text().split())
SEEDS = (0, 1, 2, 3)


def load(sub):
    out = {}
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        c = {k: z[k] for k in ("grasp_id", "partition", "pad_rel", "alpha", "n_anchors", "anchor_pos", "r_obj")}
        for i in range(len(c["grasp_id"])):
            if str(c["grasp_id"][i]) in gids:
                out[(str(c["grasp_id"][i]), str(c["partition"][i]), round(float(c["pad_rel"][i]), 4))] = \
                    {k: v[i] for k, v in c.items()}
    return out


def part_scores(recs):
    out = defaultdict(dict)
    for (g, p, _pad), r in recs.items():
        if int(r["n_anchors"]) >= 3:
            k = (g, int(r["n_anchors"]))
            out[k][p] = max(out[k].get(p, -1), float(r["alpha"].min()))
    return out


def same_place(a, b):
    m = int(a["n_anchors"])
    return np.linalg.norm(a["anchor_pos"][:m] - b["anchor_pos"][:m], axis=1).max() <= 0.078 * float(a["r_obj"])


top = lambda d: max(d, key=d.get)
data = {v: [load(f"search_records_exp006_{v}_s{s}") for s in SEEDS] for v in ("B0", "B1", "B2")}
base_keys = set.intersection(*[set(d) for v in data for d in data[v]])
print(f"공통 레코드 {len(base_keys)}")
ref = np.array([np.mean([data["B0"][s][k]["alpha"].min() for s in range(4)]) for k in sorted(base_keys)])
for v in ("B0", "B1", "B2"):
    D = data[v]
    secs = []
    for s in SEEDS:
        log = (RUNS / f"direct-search_records_exp006_{v}_s{s}" / "output.log").read_text()
        t = re.findall(r"times=\{(.*)\}", log)[-1]
        secs.append(sum(float(x) for x in re.findall(r": ([\d.]+)", t)))
    keys = sorted(base_keys)
    rho = np.array([[D[s][k]["alpha"].min() for s in range(4)] for k in keys])
    ok = ref > 1e-6
    place = {N: np.mean([same_place(D[a][k], D[b][k]) for k in keys if int(D[0][k]["n_anchors"]) == N
                         for a, b in itertools.combinations(range(4), 2)]) for N in (2, 3, 4)}
    P = [part_scores(D[s]) for s in range(4)]
    pk = [k for k in sorted(set.intersection(*[set(p) for p in P])) if len(P[0][k]) >= 2 and max(P[0][k].values()) > 1e-6]
    stab = {}
    for N in (3, 4):
        ks = [k for k in pk if k[1] == N]
        for m in (1, 2, 3):
            h = []
            for lab in itertools.combinations(range(4), m):
                for pr in (s for s in range(4) if s not in lab):
                    for k in ks:
                        L = {p: max(P[s][k].get(p, -1) for s in lab) for p in P[0][k]}
                        h.append(L.get(top(P[pr][k]), -1) >= 0.95 * max(L.values()) - 1e-12)
            stab[(N, m)] = np.mean(h)
    print(f"{v}: 파지당 {np.mean(secs) / 200:.2f}s | ρ0 (시드 평균) / B0 중앙값 {np.median(rho.mean(1)[ok] / ref[ok]):.3f} | "
          f"시드 간 같은 자리 N2 {place[2]:.2f} N3 {place[3]:.2f} N4 {place[4]:.2f} | "
          f"파티션 안정(0.95) N3 1/2/3시드 {stab[(3, 1)]:.3f}/{stab[(3, 2)]:.3f}/{stab[(3, 3)]:.3f} "
          f"N4 {stab[(4, 1)]:.3f}/{stab[(4, 2)]:.3f}/{stab[(4, 3)]:.3f}")
