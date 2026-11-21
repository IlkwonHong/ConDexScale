"""runner_multi 를 그대로 실행하되 5분마다 파이썬 스택을 <out-subdir>.stack 에 덧붙인다 (지연 원인 추적용)."""
import faulthandler
import runpy
import sys

sys.path.insert(0, ".")
sub = sys.argv[sys.argv.index("--out-subdir") + 1]
faulthandler.dump_traceback_later(300, repeat=True, file=open(f"scratchpad/exp006/{sub}.stack", "a"))
sys.argv = ["runner_multi"] + sys.argv[1:]
runpy.run_module("contraction_gt.runner_multi", run_name="__main__")
