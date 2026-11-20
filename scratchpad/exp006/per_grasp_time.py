"""파지별 생성 시간: 주어진 gid 목록을 하나씩 alpha / rho64 로 생성해 시간과 조합 수를 잰다."""
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from contraction_gt.runner_multi import run_shard  # noqa: E402

gids = Path(sys.argv[1]).read_text().split()
modes = sys.argv[2].split(",")
tmp = Path("scratchpad/exp006/_one_gid.txt")
for g in gids:
    tmp.write_text(g)
    row = [g]
    for mode in modes:
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            run_shard(0, 1, None, "data/contraction_gt_300k", 48, 0, score=mode, seed=0,
                      out_subdir=f"search_records_exp006_pergrasp_{mode}", gid_list={g})
        row.append(f"{mode} {time.time() - t:.1f}s")
    print(" | ".join(row), flush=True)
