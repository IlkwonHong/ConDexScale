#!/bin/bash
# EXP-007: Net1 목표 모양 비교 (prop / set95 / top1), 데이터 25%, seed 0, 30 에폭.
cd "$(dirname "$0")/../.."
RUNS=harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-007-net1-target-shape/runs
for m in prop set95 top1; do
  out=checkpoints/exp007_${m}_f0.25_s0
  d=$RUNS/direct-$m; mkdir -p $d
  { echo "command: train_stage1 --target-mode $m --train-frac 0.25 --seed 0 --epochs 30 (v1.1 인자)";
    echo "git_commit: $(git rev-parse HEAD)"; echo "started: $(date -u +%FT%TZ)"; } > $d/run.txt
  rm -f $out/log.jsonl
  ./venv/bin/python -m contraction_net.train_stage1 --epochs 30 --batch 256 --lr 3e-4 --workers 4 --sigma-max 0.08 \
    --no-contact-geom --train-frac 0.25 --seed 0 --target-mode $m --out $out > $d/output.log 2>&1
  { echo "finished: $(date -u +%FT%TZ)"; echo "exit_code: $?"; } >> $d/run.txt
  echo "[$(date +%T)] 끝 $m"
done
./venv/bin/python scratchpad/exp007/eval.py > scratchpad/exp007/eval.log 2>&1
echo EXP007_DONE
