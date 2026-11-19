#!/bin/bash
# EXP-005: Net1 데이터 규모 시험 6회 차례로 실행
cd "$(dirname "$0")/.."
for f in 0.25 0.5 1.0; do for s in 0 1; do
  python3 harness/scripts/run_experiment.py --exp EXP-005 -- ./venv/bin/python -m contraction_net.train_stage1 \
    --epochs 30 --batch 256 --lr 3e-4 --workers 4 --sigma-max 0.08 --no-contact-geom \
    --train-frac $f --seed $s --out checkpoints/exp005_f${f}_s${s}
done; done
echo EXP005_DONE
