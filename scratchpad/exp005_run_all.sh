#!/bin/bash
# EXP-005: Net1 데이터 규모 시험 6회 차례로 실행 (30에폭 다 끝난 것은 건너뜀)
cd "$(dirname "$0")/.."
for f in 0.25 0.5 1.0; do for s in 0 1; do
  out=checkpoints/exp005_f${f}_s${s}
  if [ -f $out/log.jsonl ] && [ "$(grep -c '"epoch"' $out/log.jsonl)" -ge 30 ]; then
    echo "skip $out (완료)"; continue
  fi
  rm -f $out/log.jsonl      # 중단된 실행의 기록은 지우고 처음부터
  python3 harness/scripts/run_experiment.py --exp EXP-005 -- ./venv/bin/python -m contraction_net.train_stage1 \
    --epochs 30 --batch 256 --lr 3e-4 --workers 4 --sigma-max 0.08 --no-contact-geom \
    --train-frac $f --seed $s --out $out
done; done
echo EXP005_DONE
