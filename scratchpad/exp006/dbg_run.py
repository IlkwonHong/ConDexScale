"""ρ 생성 지연 원인 추적: 시범 목록 앞 48 파지를 rho64 로 돌리며 2분마다 파이썬 스택을 기록."""
import faulthandler
import runpy
import sys
from pathlib import Path

sys.path.insert(0, ".")
gids = Path("scratchpad/exp006/pilot_gids_1000.txt").read_text().split()[:48]
Path("scratchpad/exp006/dbg48_gids.txt").write_text("\n".join(gids))
log = open("scratchpad/exp006/dbg48_stack.log", "w")
faulthandler.dump_traceback_later(120, repeat=True, file=log)
sys.argv = ["runner_multi", "0", "1", "0", "data/contraction_gt_300k", "48", "0", "--score", "rho64",
            "--seed", "0", "--gid-list", "scratchpad/exp006/dbg48_gids.txt",
            "--out-subdir", "search_records_exp006_dbg48"]
runpy.run_module("contraction_gt.runner_multi", run_name="__main__")
