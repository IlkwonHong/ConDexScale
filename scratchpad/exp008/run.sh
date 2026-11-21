#!/bin/bash
# EXP-008: 노이즈를 줄인 정답(B1: 후보 64/24, two_stage, 2시드, 조건표 생략)으로 학습 6.5% + 검증 전부 생성 → 시드 합치기 → 힘 제어 게이트.
cd "$(dirname "$0")/../.."
M=data/contraction_gt_300k
RUNS=harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-008-net1-clean-labels/runs
mkdir -p $RUNS
prov() { local name=$1; shift; local d=$RUNS/direct-$name; mkdir -p $d
  { echo "command: $*"; echo "git_commit: $(git rev-parse HEAD)"; echo "started: $(date -u +%FT%TZ)"; } > $d/run.txt
  "$@" > $d/output.log 2>&1; local rc=$?
  { echo "finished: $(date -u +%FT%TZ)"; echo "exit_code: $rc"; } >> $d/run.txt; echo "[$(date +%T)] 끝 $name (exit $rc)"; }
for s in 0 1; do
  sub=search_records_exp008_s$s
  if grep -q 완료 $RUNS/direct-$sub/output.log 2>/dev/null; then echo "skip $sub"; continue; fi
  rm -rf $M/$sub
  prov $sub ./venv/bin/python -m contraction_gt.runner_multi 0 1 0 $M 48 0 --score two_stage --seed $s \
    --k-cand 64 --k-cand-n4 24 --skip-cond --gid-list scratchpad/exp008/gids_all.txt --out-subdir $sub
done
prov merge ./venv/bin/python scratchpad/exp008/merge_seeds.py
prov gate ./venv/bin/python -m contraction_gt.gate_runner $M 8 0.000001 --records-subdir search_records_exp008_merged \
  --score rho64 --normal-mode force --out-name gate_exp008_merged_force.npz --all-records
echo EXP008_DONE
