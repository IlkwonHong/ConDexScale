#!/usr/bin/env bash
# EXP-004 의 남은 단계를 차례로 실행한다 (연구자가 정한 순서, 2026-11-14):
#   모델 A 학습 → 모델 B 학습을 시작하면서 동시에 모델 A 평가 → A·B 비교 평가.
# 단계마다 실행 기록 도구를 거친다. 학습 하나가 2.5시간을 넘으면 끊는다 (전체 10시간 상한을 지키기 위함).
set -u
cd /home/robin/a_workspace/condex
PY=./venv/bin/python
RUN="$PY harness/scripts/run_experiment.py --exp EXP-004 --"
TRAIN="timeout 9000 $PY -m contraction_net.train_stage2_cond --cand-from-records --only-n 3 --epochs 150 --seed 0"
echo "[$(date '+%F %T')] 모델 A 학습 시작"
$RUN $TRAIN --records-dir data/contraction_gt_300k/search_records_c128a --out checkpoints/stage2_cond_c128a \
  || { echo "[$(date '+%F %T')] 모델 A 학습이 정상 종료하지 않음 — 멈춥니다"; exit 1; }
echo "[$(date '+%F %T')] 모델 A 학습 끝. 모델 A 평가와 모델 B 학습을 동시에 시작"
$RUN $PY scratchpad/eval_c128_learnability.py --models a &
EV=$!
sleep 5
$RUN $TRAIN --records-dir data/contraction_gt_300k/search_records_c128b --out checkpoints/stage2_cond_c128b \
  || { echo "[$(date '+%F %T')] 모델 B 학습이 정상 종료하지 않음 — 멈춥니다"; wait $EV; exit 1; }
wait $EV
echo "[$(date '+%F %T')] 모델 B 학습 끝. A·B 비교 평가 시작"
$RUN $PY scratchpad/eval_c128_learnability.py
echo "[$(date '+%F %T')] 모두 끝"
