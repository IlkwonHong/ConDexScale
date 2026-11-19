#!/bin/bash
# EXP-006 시범 생성: EXP-005 가 끝난 뒤(GPU 경합 방지) 차례로 실행. 끝난 단계는 건너뜀.
cd "$(dirname "$0")/../.."
M=data/contraction_gt_300k
G1=scratchpad/exp006/pilot_gids_1000.txt
G2=scratchpad/exp006/pilot_gids_200.txt
RUN="python3 harness/scripts/run_experiment.py --exp EXP-006 -- ./venv/bin/python -m"

until grep -q EXP005_DONE scratchpad/exp005_all.log 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] EXP-005 끝남, 시범 생성 시작"

gen() {  # gen <score> <seed> <gidlist> <subdir>
  if ls $M/$4/shard_0_part*.npz >/dev/null 2>&1 && grep -q "완료" scratchpad/exp006/$4.done 2>/dev/null; then
    echo "skip $4"; return; fi
  rm -rf $M/$4
  $RUN contraction_gt.runner_multi 0 1 0 $M 48 0 --score $1 --seed $2 --gid-list $3 --out-subdir $4 \
    2>&1 | tee scratchpad/exp006/$4.log | tail -1 > scratchpad/exp006/$4.done
}
gen alpha     0 $G1 search_records_exp006_alpha_s0
gen rho64     0 $G1 search_records_exp006_rho64_s0
gen two_stage 0 $G1 search_records_exp006_two_s0
gen rho64     1 $G2 search_records_exp006_rho64_s1
gen rho64     2 $G2 search_records_exp006_rho64_s2

gate() {  # gate <subdir> <score> <mode> <out>
  [ -f $M/$4 ] && { echo "skip $4"; return; }
  $RUN contraction_gt.gate_runner $M 8 --records-subdir $1 --score $2 --normal-mode $3 --out-name $4 \
    2>&1 | tee scratchpad/exp006/$4.log | tail -3
}
gate search_records_exp006_rho64_s0 rho64 force gate_exp006_rho64_force.npz
gate search_records_exp006_rho64_s0 rho64 servo gate_exp006_rho64_servo.npz
gate search_records_exp006_alpha_s0 alpha servo gate_exp006_alpha_servo.npz
echo EXP006_DONE
