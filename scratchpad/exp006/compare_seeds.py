"""EXP-006 3시드 분석 (200 파지, rho64): 시드 간 정답 일치와 3시드 최댓값의 ρ0 상승.

일치 기준 = EXP-001 과 같음: 모든 앵커가 물체 반경의 0.078배 이내 (앵커 순서는 그룹 순서로 대응).
"""
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
gids = set(Path("scratchpad/exp006/pilot_gids_200.txt").read_text().split())


def load(sub):
    out = {}
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        c = {k: z[k] for k in ("grasp_id", "partition", "pad_rel", "alpha", "n_anchors", "anchor_pos", "r_obj")}
        for i in range(len(c["grasp_id"])):
            g = str(c["grasp_id"][i])
            if g in gids:
                out[(g, str(c["partition"][i]), round(float(c["pad_rel"][i]), 4))] = {k: v[i] for k, v in c.items()}
    return out


S = [load(f"search_records_exp006_rho64_s{s}") for s in (0, 1, 2)]
keys = sorted(set(S[0]) & set(S[1]) & set(S[2]))
r = np.array([[S[s][k]["alpha"].min() for s in range(3)] for k in keys])
n = np.array([int(S[0][k]["n_anchors"]) for k in keys])


def same(a, b):
    m = int(a["n_anchors"])
    return np.linalg.norm(a["anchor_pos"][:m] - b["anchor_pos"][:m], axis=1).max() <= 0.078 * float(a["r_obj"])


agree01 = np.array([same(S[0][k], S[1][k]) for k in keys])
agree02 = np.array([same(S[0][k], S[2][k]) for k in keys])
best = r.max(axis=1)
ok = r[:, 0] > 1e-6
print(f"공통 레코드 {len(keys)} (200 파지)")
for name, sel in (("전체", np.ones(len(keys), bool)), ("N=3", n == 3), ("N=4", n == 4), ("N=2", n == 2)):
    s_ok = sel & ok
    print(f"{name:4s} n={sel.sum():4d} | 시드0 대비 시드1·2 일치(0.078r) {agree01[sel].mean():.3f} / {agree02[sel].mean():.3f} | "
          f"3시드 최댓값 / 시드0 ρ0 중앙값 {np.median(best[s_ok] / r[s_ok, 0]):.3f}, 평균 {np.mean(best[s_ok] / r[s_ok, 0]):.3f}, "
          f"5% 이상 오른 비율 {np.mean(best[s_ok] >= 1.05 * r[s_ok, 0]):.3f} | 시드 간 ρ0 변동(최대/최소) 중앙값 "
          f"{np.median(r[s_ok].max(1) / np.maximum(r[s_ok].min(1), 1e-9)):.3f}")
