# 자주 쓰는 명령어

> 모두 프로젝트 루트에서 실행. venv = `./venv/bin/python`
> ```bash
> cd /home/robin/a_workspace/condex
> ```

---

## 🖥️ 데이터셋 뷰어 (켜기 / 끄기)

인터랙티브 뷰어 — 오브젝트 드롭다운 + 슬라이더로 전체 데이터셋을 `[사진 | 3D: 오브젝트 메쉬 + 양손 메쉬 + 접촉점 + GT/PRED]` 양식으로 브라우징.

**켜기 (train 데이터, 기본):**
```bash
./venv/bin/python scripts/dataset_viewer.py
```
→ http://localhost:8050

**켜기 (s05 held-out, 포트 8051):**
```bash
./venv/bin/python scripts/dataset_viewer.py \
  --results-dir data/results_arctic_val \
  --graph-dir data/arctic_cache/sample_graphs_val \
  --gate-max-dist 0.03 --port 8051
```
→ http://localhost:8051

**백그라운드로 켜기 (터미널 닫아도 유지):**
```bash
nohup ./venv/bin/python scripts/dataset_viewer.py > /tmp/viewer.log 2>&1 &
```

**끄기:**
```bash
pkill -f dataset_viewer.py
```

**상태 확인:**
```bash
pgrep -af dataset_viewer.py        # 실행 중인 뷰어 PID/명령
```

옵션: `--results-dir --graph-dir --ckpt --gate-max-dist --num-anchors --port`

---

## 📦 데이터 생성 / 재contraction

**recontract (라벨 human_anchor만 재계산, 빠름 ~2.5분, force closure 없음):**
```bash
# train (1.5cm 게이팅)
./venv/bin/python scripts/recontract_arctic_genhand.py \
  --split train --results-dir data/results_arctic --contact-threshold 0.015
# s05 (p1 val)
./venv/bin/python scripts/recontract_arctic_genhand.py \
  --split val --results-dir data/results_arctic_val --contact-threshold 0.015
```

**fresh 생성 (run_arctic, force closure 제거돼 빠름):**
```bash
# train 전체
./venv/bin/python scripts/run_arctic.py --robot Robotiq --subjects all \
  --split train --frame-stride 30 --workers 4 --device cuda:0
# s05(p1 val) → 별도 경로
./venv/bin/python scripts/run_arctic.py --robot Robotiq --subjects all \
  --split val --frame-stride 30 --workers 4 --out-base data/results_arctic_val
```
옵션: `--anchors-per-hand N`(손당 anchor 수), `--contact-threshold`, `--out-base`

**sample_graphs (모델 입력 그래프):**
```bash
# train
./venv/bin/python scripts/precompute_arctic_sample_graphs.py --overwrite
# s05
./venv/bin/python scripts/precompute_arctic_sample_graphs.py \
  --results-dir data/results_arctic_val --graph-dir data/arctic_cache/sample_graphs_val --overwrite
```

---

## 🎓 학습 / 평가

**학습 (~36분):**
```bash
./venv/bin/python -m training.train --dataset arctic --robot Robotiq \
  --epochs 500 --gate-max-dist 0.03
```

**평가 — 내부 val (frame-split):**
```bash
./venv/bin/python scripts/visualize_eval_predictions.py \
  --ckpt checkpoints/best.pt --gate-max-dist 0.03
```

**평가 — s05 cross-subject held-out:**
```bash
./venv/bin/python scripts/visualize_eval_predictions.py --ckpt checkpoints/best.pt \
  --all --results-dir data/results_arctic_val --graph-dir data/arctic_cache/sample_graphs_val \
  --gate-max-dist 0.03 --save md_claude/arctic/eval_s05.html
```

**rich 시각화 (사진 + 손/오브젝트 메쉬, HTML 파일 생성):**
```bash
./venv/bin/python scripts/visualize_eval_predictions_rich.py --ckpt checkpoints/best.pt \
  --all --results-dir data/results_arctic_val --graph-dir data/arctic_cache/sample_graphs_val \
  --n 12 --save md_claude/arctic/eval_s05_rich.html
```
옵션: `--no-pred`(GT만), `--require-image`(사진 있는 샘플만), `--cam`, `--n`

---

## 🗂️ 데이터 경로

| 경로 | 내용 |
|---|---|
| `data/results_arctic/` + `data/arctic_cache/sample_graphs/` | train (8 subject) |
| `data/results_arctic_val/` + `data/arctic_cache/sample_graphs_val/` | held-out s05 |
| `data/results_arctic.bak_*` | 구버전 스냅샷 |
| `checkpoints/best.pt` | 최신 모델 |

---

## 🤖 DexMachina 점접촉 Multi-Arm RL (extlib/dexmachina, 2026-07-05)

**환경**: conda env `dexmachina` (`/home/robin/miniforge3/envs/dexmachina/bin/python`)
Genesis fork `0122` 브랜치(0.3.13) + torch 2.7.0+cu128. 패치 내역: `md_claude/plans/dexmachina_integration.md` §3.5

**Guidance 데이터 생성 (contraction → retargeted/contact_retarget npy):**
```bash
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/export_dexmachina_guidance.py \
  --obj box --subject s01 --clip 01 --spheres-per-side 2   # point4용 (1이면 point2용)
```

**Anchor 표면 재투영 검증 (CLAUDE.md §4, median <10mm 통과 기준):**
```bash
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/verify_dexmachina_guidance.py --obj box --hand point4_hand
```

**G1 학습 (contraction guidance, point4 = 4암) — ⚠️ 커리큘럼 플래그 전체 필수 (누락 시 VOC 감쇠 고장, 07-06 교훈):**
```bash
cd extlib/dexmachina && WANDB_MODE=disabled /home/robin/miniforge3/envs/dexmachina/bin/python \
  dexmachina/rl/train_rl_games.py --hand point4_hand -B 1024 -obf -obt --max_epochs 3000 \
  --actuate_object --horizon 16 -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
  --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
  --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 \
  --fixed_mode uniform --uniform_mode slow --action_penalty 0.01 --dialback_ep_len 80 \
  --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
  --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01 \
  --clip box-30-230 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp g1full
```

**G0 baseline (task-only):** 같은 명령에서 `-imi 0 -bc 0 -con 0` + `--use_retarget_contact`·`--contact_beta` 제거 (retargeted npy는 base action용으로 여전히 필요)

**평가 (VOC 끄고 순수 정책 + 비디오 mp4):**
```bash
cd extlib/dexmachina && WANDB_MODE=disabled /home/robin/miniforge3/envs/dexmachina/bin/python \
  dexmachina/rl/eval_rl_games.py --hand point4_hand -B 4 -am hybrid --hybrid_scales 0.1 1.0 \
  --task_rew_betas 10 1 5 --clip box-30-230 -imi 0.3 -bc 0.3 -con 3 --use_retarget_contact \
  --record_video --checkpoint logs/rl_games/point4_hand/<런이름>/nn/point4_hand.pth \
  --eval_episodes 1 --output_render --render_dir <출력경로>
```
`--actuate_object`를 **빼면** "novoc"(가상 컨트롤러 없는 본실력) 평가. 넣으면 VOC가 물체를 움직여줘 과대평가됨.
평가 지표는 do-nothing 기준선·engagement(물체 이동량)와 함께 볼 것 (07-06 "무행동 함정" 교훈).

⚠️ **eval 주의사항 (07-13 발견)**:
- **B>1이면 마지막 env는 정책 미제어**(is_eval 시 시각화 예약, robot init 0) — env별 통계에서 제외할 것.
- eval은 기본으로 마찰 랜덤화를 끔(μ=1.0 고정). 추가 플래그: `--stochastic`(확률적 액션 = 학습시 행동),
  `--keep_rand`(마찰 DR 유지 + env별 마찰값 npy 기록), `--train_like`(조기종료+전체 env 제어 = 학습 재현,
  `--n_steps 800` 등과 함께 에피소드 길이 실측용).
- 학습 wandb의 pos_dist/ep_len은 조기종료 리셋 사이클 탓에 낙관 편향 — novoc 연속 rollout으로만 판정.

**학습 재현 기록:** 런별 `extlib/dexmachina/logs/rl_games/point4_hand/<런이름>/params/`
(`agent.yaml`=PPO·네트워크 설정, `env.yaml`=환경·보상·커리큘럼 설정) + stdout 로그 `md_claude/dexmachina/train_logs/`

**URDF 재생성 (스피어 수/반경 변경 시):**
```bash
python3 scripts/gen_point_urdf.py --radius 0.015 --spheres 1 2
```

## 🦾 DexScale — TO+임피던스 multi-arm (dexscale/, 2026-07-15)

**환경**: conda `dexmachina`. 설계: `md_claude/plans/dexscale_mvp.md`

```bash
# Genesis 임피던스/DR 게이트 (6/6 PASS 확인용)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/test_genesis_impedance_smoke.py
# 시뮬-무관 코어 테스트 (knots/CEM/reference)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/test_dexscale_core.py
# TO 배선 스모크 (M1 θ=0 + 2세대)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/run_dexscale_to.py --smoke
# M2 베이스라인② (δ만, 전체 클립): ~15분
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/run_dexscale_to.py --mode delta --gens 30
# M3 full A안 (Δu,Δv,δ)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/run_dexscale_to.py --mode full --gens 50
```

**하이브리드 (물체 기준 앵커 + 운반, 2026-07-16):**
```bash
# TO (E2 재현): θ=65차원 (Δu,Δv,δ + d_max 채널)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/run_dexscale_to.py --hybrid --mode full --slow 2 --frames 220 --gens 40
# 접촉 유지 프로브 / 시각화 (npz의 hybrid/slow 자동 인식)
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/probe_dexscale_onset.py --hybrid --theta data/dexscale/theta_hybrid_full_K5.npz --slow 2 --frames 220
/home/robin/miniforge3/envs/dexmachina/bin/python scripts/visualize_dexscale_rollout.py --theta data/dexscale/theta_hybrid_full_K5.npz --frames 220
```
