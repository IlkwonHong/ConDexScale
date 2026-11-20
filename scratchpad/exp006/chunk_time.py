"""묶음 크기별 ρ 생성 시간 (같은 48 파지). 인자: 묶음 크기 목록, 시간 상한(초)."""
import contextlib
import faulthandler
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from contraction_gt.runner_multi import run_shard  # noqa: E402

gids = set(Path("scratchpad/exp006/main48_gids.txt").read_text().split())
faulthandler.dump_traceback_later(int(sys.argv[2]), exit=True,
                                  file=open("scratchpad/exp006/chunk_time_stack.log", "w"))
for chunk in [int(c) for c in sys.argv[1].split(",")]:
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        run_shard(0, 1, None, "data/contraction_gt_300k", chunk, 0, score="rho64", seed=0,
                  out_subdir=f"search_records_exp006_chunk{chunk}", gid_list=gids)
    print(f"chunk {chunk}: {time.time() - t:.0f}s", flush=True)
