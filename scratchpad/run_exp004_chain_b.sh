#!/usr/bin/env bash
# EXP-004 의 남은 단계 (2026-11-14 다시 시작): 모델 B 학습 → A·B 비교 평가.
# 앞선 묶음 실행은 모델 B 학습 4에폭째에 세션이 끝나면서 함께 죽었다 (run-005). 이번에는 세션과 떼어 실행한다.
set -u
cd /home/robin/a_workspace/condex
PY=./venv/bin/python
RUN="$PY harness/scripts/run_experiment.py --exp EXP-004 --"
echo "[$(date '+%F %T')] 모델 B 학습 시작"
$RUN timeout 9000 $PY -m contraction_net.train_stage2_cond --cand-from-records --only-n 3 --epochs 150 --seed 0 \
  --records-dir data/contraction_gt_300k/search_records_c128b --out checkpoints/stage2_cond_c128b \
  || { echo "[$(date '+%F %T')] 모델 B 학습이 정상 종료하지 않음 — 멈춥니다"; exit 1; }
echo "[$(date '+%F %T')] 모델 B 학습 끝. A·B 비교 평가 시작"
$RUN $PY scratchpad/eval_c128_learnability.py
echo "[$(date '+%F %T')] 모두 끝"
