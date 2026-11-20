#!/bin/bash
# EXP-006 후속: 탐색 무작위성 줄이기 비교 (200 파지 × 시드 4, two_stage + 조건표 생략).
# B0 = 기준(후보 32/12, 다듬기 1), B1 = 후보 64/24, B2 = 후보 64/24 + 다듬기 최대 4회.
cd "$(dirname "$0")/../.."
M=data/contraction_gt_300k
G=scratchpad/exp006/pilot_gids_200.txt
RUNS=harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-006-gt-v2-pilot/runs
run() {  # run <name> <seed> <extra args...>
  local name=$1 seed=$2; shift 2
  local sub=search_records_exp006_${name}_s${seed}
  grep -q 완료 $RUNS/direct-$sub/output.log 2>/dev/null && { echo "skip $sub"; return; }
  rm -rf $M/$sub; mkdir -p $RUNS/direct-$sub
  { echo "command: runner_multi two_stage seed $seed $* gid=$G"; echo "git_commit: $(git rev-parse HEAD)";
    echo "started: $(date -u +%FT%TZ)"; } > $RUNS/direct-$sub/run.txt
  ./venv/bin/python -m contraction_gt.runner_multi 0 1 0 $M 48 0 --score two_stage --seed $seed --skip-cond \
    --gid-list $G --out-subdir $sub "$@" > $RUNS/direct-$sub/output.log 2>&1
  { echo "finished: $(date -u +%FT%TZ)"; echo "exit_code: $?"; } >> $RUNS/direct-$sub/run.txt
  echo "[$(date +%T)] 끝 $sub $(grep 완료 $RUNS/direct-$sub/output.log | cut -c1-200)"
}
for s in 0 1 2 3; do
  run B0 $s
  run B1 $s --k-cand 64 --k-cand-n4 24
  run B2 $s --k-cand 64 --k-cand-n4 24 --refine-iters 4
done
echo VARIANTS_DONE
