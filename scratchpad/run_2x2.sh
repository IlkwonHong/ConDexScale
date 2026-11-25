#!/bin/bash
# 2×2 비교 (2026-10-24): 시험 생성 데이터(파지 2만) — 독립/조건부 × 정답 한 벌/여러 벌. 차례로 학습 + 평가.
cd /home/robin/a_workspace/condex
COMMON="--records-dir data/contraction_gt_300k/search_records_multi --records-format multi --epochs 60"
run() {  # 이름, 추가 인자
  name=$1; shift
  echo "== $name 시작 $(date)"
  ./venv/bin/python -m contraction_net.train_stage2_cond $COMMON "$@" --out checkpoints/x2_$name > checkpoints/x2_$name.log 2>&1
  ./venv/bin/python scratchpad/net2_cond_eval.py checkpoints/x2_$name/best.pt scratchpad/x2_eval_$name.pkl > scratchpad/x2_eval_$name.log 2>&1
  echo "== $name 끝 $(date)"
}
run indep_k1 --no-cond --max-k 1
run indep_k3 --no-cond --max-k 3
run cond_k1  --max-k 1
run cond_k3  --max-k 3
echo "== 전부 끝 $(date)"
