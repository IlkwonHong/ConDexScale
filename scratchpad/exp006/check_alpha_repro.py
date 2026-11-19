"""검증: runner_multi --score alpha --seed 0 이 기존 search_records_multi 레코드를 재현하는가 (50 파지).

사용: python scratchpad/exp006/check_alpha_repro.py prepare   → gid 목록 작성
      (runner_multi 실행)
      python scratchpad/exp006/check_alpha_repro.py compare
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_300k")
OLD = ROOT / "search_records_multi"
NEW = ROOT / "search_records_exp006_check"
GL = Path("scratchpad/exp006/check_gids.txt")


def load(d, pattern="shard_*.npz"):
    out = {}
    for f in sorted(d.glob(pattern)):
        z = np.load(f, allow_pickle=False)
        for i in range(len(z["grasp_id"])):
            k = (str(z["grasp_id"][i]), str(z["partition"][i]), round(float(z["pad_rel"][i]), 4))
            out[k] = (z["anchor_idx"][i], z["alpha"][i])
    return out


if sys.argv[1] == "prepare":
    z = np.load(sorted(OLD.glob("shard_0_part0.npz"))[0], allow_pickle=False)
    gids = list(dict.fromkeys(str(g) for g in z["grasp_id"]))[:50]
    GL.write_text("\n".join(gids))
    print(f"{len(gids)} gids")
else:
    gids = set(GL.read_text().split())
    old = {k: v for k, v in load(OLD, "shard_0_part0.npz").items() if k[0] in gids}
    new = load(NEW)
    same_idx = sum(np.array_equal(old[k][0], new[k][0]) for k in old if k in new)
    missing = sum(k not in new for k in old)
    err = max(float(np.nanmax(np.abs(old[k][1] - new[k][1]))) for k in old if k in new)
    print(f"레코드 기존 {len(old)} / 새 {len(new)} / 새에 없음 {missing} | 같은 앵커 {same_idx}/{len(old) - missing} | α 최대 차이 {err:.2e}")
    sys.exit(0 if missing == 0 and same_idx == len(old) and err < 1e-6 else 1)
