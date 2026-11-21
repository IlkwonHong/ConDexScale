"""EXP-008 시드 합치기: (파지, 파티션, 패드) 레코드마다 두 시드 중 ρ0 가 큰 쪽을 남겨 한 폴더로 저장.

입력: data/contraction_gt_300k/search_records_exp008_s{0,1}/  출력: search_records_exp008_merged/shard_0_part0.npz
배치 묶음 필드(mode_*)는 남긴 쪽 레코드의 것을 그대로 쓴다. src_seed 필드에 어느 시드에서 왔는지 기록.
"""
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")


def load(sub):
    cols = None
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        keys = [k for k in z.files if k not in ("times", "n_grasp", "n_fail", "config")]
        d = {k: z[k] for k in keys}
        cols = d if cols is None else {k: np.concatenate([cols[k], d[k]]) for k in keys}
    return cols


A, B = load("search_records_exp008_s0"), load("search_records_exp008_s1")
key = lambda C, i: (str(C["grasp_id"][i]), str(C["partition"][i]), round(float(C["pad_rel"][i]), 4))
ib = {key(B, i): i for i in range(len(B["grasp_id"]))}
rows, src = [], []
for i in range(len(A["grasp_id"])):
    k = key(A, i)
    j = ib.get(k)
    if j is not None and float(B["alpha"][j].min()) > float(A["alpha"][i].min()):
        rows.append(("B", j)); src.append(1)
    else:
        rows.append(("A", i)); src.append(0)
seen = {key(A, i) for i in range(len(A["grasp_id"]))}
for k, j in ib.items():                     # 시드 1 에만 있는 레코드
    if k not in seen:
        rows.append(("B", j)); src.append(1)
out = {k: np.stack([(A if s == "A" else B)[k][i] for s, i in rows]) for k in A}
out["src_seed"] = np.array(src, np.int8)
d = ROOT / "search_records_exp008_merged"
d.mkdir(exist_ok=True)
np.savez_compressed(d / "shard_0_part0.npz", **out)
print(f"레코드 시드0 {len(A['grasp_id'])} / 시드1 {len(B['grasp_id'])} → 합침 {len(rows)} (시드1 에서 온 것 {sum(src)})")
