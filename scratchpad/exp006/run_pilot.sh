#!/bin/bash
# EXP-006 시범 생성: 끝난 단계는 건너뛰고 차례로 실행.
# 2026-11-20: harness/scripts/run_experiment.py 로 감싸면 ρ 생성만 10배 이상 느려지는 현상(원인 미상)이 있어
# 직접 실행하고, 기록기와 같은 정보(명령·커밋·dirty·시작/종료·종료 코드)를 runs/ 아래에 직접 남긴다.
cd "$(dirname "$0")/../.."
M=data/contraction_gt_300k
G1=scratchpad/exp006/pilot_gids_1000.txt
G2=scratchpad/exp006/pilot_gids_200.txt
RUNS=harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-006-gt-v2-pilot/runs

prov() {  # prov <name> <command...> : 실행하고 runs/direct-<name>/ 에 기록
  local name=$1; shift
  local d=$RUNS/direct-$name; mkdir -p $d
  { echo "command: $*"; echo "git_commit: $(git rev-parse HEAD)"; echo "git_dirty: $(git status --porcelain | grep -v '^??' | wc -l) tracked changes";
    echo "python: $(./venv/bin/python --version 2>&1)"; echo "started: $(date -u +%FT%TZ)"; } > $d/run.txt
  "$@" > $d/output.log 2>&1
  local rc=$?
  { echo "finished: $(date -u +%FT%TZ)"; echo "exit_code: $rc"; } >> $d/run.txt
  return $rc
}

gen() {  # gen <score> <seed> <gidlist> <subdir>
  if grep -q "완료" scratchpad/exp006/$4.done 2>/dev/null; then echo "skip $4"; return; fi
  rm -rf $M/$4
  echo "[$(date +%T)] 시작 $4"
  prov $4 ./venv/bin/python scratchpad/exp006/traced_runner.py 0 1 0 $M 48 0 \
    --score $1 --seed $2 --gid-list $3 --out-subdir $4
  grep "완료" $RUNS/direct-$4/output.log | tail -1 > scratchpad/exp006/$4.done
  echo "[$(date +%T)] 끝 $4: $(cut -c1-160 scratchpad/exp006/$4.done)"
}
gen alpha     0 $G1 search_records_exp006_alpha_s0
gen rho64     0 $G1 search_records_exp006_rho64_s0
gen two_stage 0 $G1 search_records_exp006_two_s0
gen rho64     1 $G2 search_records_exp006_rho64_s1
gen rho64     2 $G2 search_records_exp006_rho64_s2

gate() {  # gate <subdir> <score> <mode> <out>
  [ -f $M/$4 ] && { echo "skip $4"; return; }
  echo "[$(date +%T)] 시작 $4"
  prov ${4%.npz} ./venv/bin/python -m contraction_gt.gate_runner $M 8 --records-subdir $1 --score $2 \
    --normal-mode $3 --out-name $4
  echo "[$(date +%T)] 끝 $4: $(grep -E '판정 분포' $RUNS/direct-${4%.npz}/output.log)"
}
gate search_records_exp006_rho64_s0 rho64 force gate_exp006_rho64_force.npz
gate search_records_exp006_rho64_s0 rho64 servo gate_exp006_rho64_servo.npz
gate search_records_exp006_alpha_s0 alpha servo gate_exp006_alpha_servo.npz
echo EXP006_DONE
