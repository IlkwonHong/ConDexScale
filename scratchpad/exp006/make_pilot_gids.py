"""EXP-006 시범 생성용 파지 목록: 기존 GT 가 있는 파지 중 학습 쪽(물체 해시, Stage1Dataset 규칙 seed 0·5%)에서
고정 난수로 1,000개, 그중 200개를 3시드용으로."""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
have = set()
for f in sorted((ROOT / "search_records").glob("shard_*.npz")):
    have.update(str(g) for g in np.load(f, allow_pickle=False)["grasp_id"])


def is_val(oid):
    h = int(hashlib.md5(f"0:{oid}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < 0.05 * 10_000


train = sorted({f"{r[3]}#{r[4]}" for r in rows if f"{r[3]}#{r[4]}" in have and not is_val(r[2])})
rng = np.random.default_rng(20261120)
g1000 = sorted(rng.choice(train, 1000, replace=False).tolist())
g200 = sorted(rng.choice(g1000, 200, replace=False).tolist())
Path("scratchpad/exp006/pilot_gids_1000.txt").write_text("\n".join(g1000))
Path("scratchpad/exp006/pilot_gids_200.txt").write_text("\n".join(g200))
print(f"학습 쪽 GT 파지 {len(train):,} → 1000 / 200")
