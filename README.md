# ConDexScale

**Scaling Up Dexterous Manipulation through Contact Graph Contraction**

사람 손의 파지 접촉을 소수의 접촉 지점(앵커)으로 줄이는 작업(수축, contraction)을 합니다. 이렇게 줄인 앵커로 사람 손보다 큰 물체를 로봇 팔 2~4대가 함께 다루게 하는 연구 코드입니다.

- **Net1 (Stage 1)**: 사람 손가락 접촉을 팔 몇 대에, 어떻게 나눠 맡길지(파티션)를 고릅니다.
- **Net2 (Stage 2)**: 고른 파티션에서 각 팔이 닿을 물체 표면 위치(앵커)를 정합니다.

환경 설치는 [`INSTALL.md`](INSTALL.md), 자주 쓰는 명령어는 [`commands.md`](commands.md), 지난 진행 기록은 [`Summary.md`](Summary.md)와 [`PROGRESS.md`](PROGRESS.md)를 보세요.
모든 명령은 프로젝트 루트에서 `./venv/bin/python`으로 실행합니다.

---

## 1. 정답 데이터(GT) 만들기

### 1-1. 순서

| 단계 | 하는 일 | 명령 | 결과 위치 |
|---|---|---|---|
| ① | Dexonomy 파지 데이터에서 사람 손 접촉 추출 | `scripts/dexonomy/extract_contacts.py` (인자는 스크립트 머리말 참고) | `data/dexonomy/contact_cache/` |
| ② | 파지 목록 뽑기 | `python -m contraction_gt.pilot_select <성공 파지 폴더> <출력 폴더> [총 개수]` | `data/contraction_gt_300k/pilot_manifest.tsv` |
| ③ | 물체 표면 점 2,048개 캐시 | `python -m contraction_gt.object_cache <에셋 폴더> <출력 폴더> <물체 ID 목록.txt>` | `data/contraction_gt_pilot/object_cache/` |
| ④ | 정답 탐색 (파지 × 파티션마다 최선의 앵커와 점수) | `python -m contraction_gt.runner_multi <샤드 번호> <샤드 수> <개수, 0=전부> <매니페스트 폴더> [묶음 크기] [시작 행] --score <alpha\|rho64> --seed <씨앗> --out-subdir <이름>` | `data/contraction_gt_300k/<이름>/shard_*_part*.npz` |
| ⑤ | 시뮬레이션 합격 판정(게이트) | `python -m contraction_gt.gate_runner_all <매니페스트 폴더> <작업자 수> [α 하한] [개수 상한]` | `data/contraction_gt_300k/gate_records_all.npz` |
| ⑥ | Net1 입력용 패치 캐시 | `python -m contraction_net.build_patch_cache [작업자 수]` → `extract_patch_contacts [작업자 수]` → `build_patch_interp` | `data/contraction_net/` |

- 매니페스트(`pilot_manifest.tsv`)는 정답을 만들 파지 목록 파일입니다. 한 줄이 파지 하나입니다.
- ④의 실제 사용 예 (EXP-006 기록):
  ```bash
  ./venv/bin/python -m contraction_gt.runner_multi 0 1 0 data/contraction_gt_300k 48 0 \
      --score rho64 --seed 0 --gid-list scratchpad/exp006/pilot_gids_1000.txt \
      --out-subdir search_records_exp006_rho64_s0
  ```
- ④는 파지 1개에 약 1.8초가 걸립니다. 10만 파지를 시드 2개로 만들면 약 108시간(4.5일)으로 추정합니다.

### 1-2. 지금 저장된 데이터

| 위치 | 내용 |
|---|---|
| `data/dexonomy/` | Dexonomy 원본 파지, 물체 에셋(22GB), 사람 접촉 캐시(23GB) |
| `data/contraction_gt_300k/search_records/` | **지금 Net1이 학습하는 정답 (v1)**. α 점수, 탐색 1회, 위치 서보 게이트 |
| `data/contraction_gt_300k/gate_records_all.npz` | v1 정답의 게이트 판정 결과 |
| `data/contraction_gt_300k/search_records_multi/` | Net2 학습용 정답 (배치 여러 벌 저장) |
| `data/contraction_gt_300k/search_records_exp008_merged/` | **새 정답 일부 (v2 방식)**. 10,931 파지(학습 6.5% + 검증 전부), 후보 64/24, 시드 2개 최댓값, ρ 점수 |
| `data/contraction_gt_300k/gate_exp008_merged_force.npz` | 새 정답의 게이트 판정 결과 (법선 방향 힘 제어) |
| `data/contraction_gt_300k/search_records_exp00*` | 그 밖의 실험(EXP-001~009)용 정답 |
| `data/contraction_net/` | Net1 입력 캐시 (패치 128개, 256개판) |

---

## 2. 학습 실행

### Net1 (지금 쓰는 v1.1 설정)
```bash
./venv/bin/python -m contraction_net.train_stage1 \
    --epochs 30 --batch 256 --lr 3e-4 --workers 4 \
    --sigma-max 0.08 --no-contact-geom --seed 0 \
    --out checkpoints/stage1_새이름
```
| 인자 | 뜻 |
|---|---|
| `--smoke` | 파지 2,000개, 2에폭으로 코드 연결만 확인 |
| `--train-frac 0.25` | 학습 데이터 중 일부만 사용 (물체 단위로 자름) |
| `--target-mode prop\|set95\|top1` | 학습 목표 모양: 점수 비례 / 1등의 0.95배 이상 묶음 / 1등만 |
| `--sigma-max` | 손가락 접촉 위치를 흔드는 증강의 세기 (물체 반경 대비) |
| `--no-contact-geom` | 손가락 접촉 위치·법선 입력을 끔 (v1.1 기본) |
| `--records-dir`, `--verdict-file` | 다른 정답·게이트 파일로 학습 (새 정답 학습 시. `worktree-gt-v2-pilot` 브랜치에만 있음) |

평가: `./venv/bin/python -m contraction_net.eval_stage1 <체크포인트.pt> [--no-contact-geom]`
기존 체크포인트: `checkpoints/stage1_v11/` (v1.1), 실험별 `checkpoints/exp0NN_*/`

### Net2 (조건부)
```bash
./venv/bin/python -m contraction_net.train_stage2_cond \
    --records-dir data/contraction_gt_300k/search_records_multi --records-format multi \
    --epochs 60 --out checkpoints/stage2_cond_새이름
```
`--no-cond`는 앞 팔의 선택을 조건으로 쓰지 않는 비교 모델, `--max-k`는 정답 배치를 몇 벌까지 쓸지입니다.

---

## 3. Net1의 현재 문제 (2026-11-25 기준)

자세한 수치는 [`upgrade_net1.md`](upgrade_net1.md)와 `harness/research/IDEA-001-imitate-verified-gt/`의 실험별 `analysis.md`에 있습니다.

**문제: Net1이 입력을 보고 파티션을 고르지 못합니다.**
검증 파지에서 1등 점수의 0.95배 이상인 파티션을 고른 비율이 팔 3대 47.0%, 팔 4대 44.4%입니다. 입력을 보지 않고 "학습 데이터에서 가장 자주 1등이었던 파티션"을 고르는 규칙(빈도 규칙)과 0.6~1.7%p밖에 차이 나지 않습니다. 학습에 쓴 파지에서도 빈도 규칙보다 4~7%p만 높습니다.

**원인을 하나씩 확인한 결과**

| 의심한 원인 | 결과 | 근거 |
|---|---|---|
| 학습 데이터가 적다 | 아님. 데이터 4배에 +1.2~1.4%p | EXP-005 |
| 학습 목표가 너무 평평하다 | 아님. 날카롭게 하면 학습 파지만 오르고 검증은 그대로 | EXP-007 |
| 새 물체로 일반화를 못 한다 | 아님. 본 물체의 다른 파지도 똑같이 못 맞힘 | 같은 물체 시험 |
| 증강이 정답과 어긋나는 것을 가르친다 | 아님. 증강 세기 0/0.02/0.08 모두 같은 성능 | EXP-010 |
| 정답이 탐색 운으로 흔들린다 | **일부 맞음.** 시드만 바꿔도 1등이 절반 바뀜. 새 정답으로 학습하면 팔 4대만 빈도 규칙보다 +3.4%p | EXP-001, 002, 008 |
| 정답이 접촉의 아주 작은 차이에 민감하다 | **맞음 (팔 4대).** 접촉을 물체 반경의 5%만 옮겨도 정답 일치율이 17.6%p 떨어짐 | EXP-009 |
| Net1 입력이 너무 거칠다 | **다시 유력.** Net1은 접촉을 패치(간격 약 물체 반경의 15%) 단위로만 보는데, 정답은 반경의 5% 차이에도 바뀜 | EXP-010 |

**지금 진행 중**
- EXP-011: 손가락 접촉 위치·법선을 연속값으로 다시 넣어(`--no-contact-geom` 끔) 새 정답으로 학습. 검증이 +3%p 이상 오르면 입력 해상도가 원인입니다. (2026-11-25 승인, 결과 분석 전)

**당장 쓸 수 있는 방법 (안전장치)**
Net1 상위 3개 파티션을 탐색하고 시뮬레이션으로 확인하면 팔 3대 73.2%, 팔 4대 82.0%에서 쓸 수 있는 배치를 얻습니다. 무작위 3개와는 1%p 차이라 Net1의 기여는 작습니다. 다만 모든 파티션을 탐색하는 것(76.5% / 84.8%)보다 탐색량이 1/2~1/3로 줄어듭니다.

**정해야 할 것**
- 새 정답(v2) 전량 생성(약 4.5일)을 할지. EXP-008 결과로는 효과가 작을 것으로 추정합니다.
- 동률 기준을 0.95배와 0.9배 중 무엇으로 할지.
- 정답이 작은 차이에 민감한 것이 확인되면, 정답 정의 자체를 바꿀지 (예: 접촉을 흔들어도 유지되는 품질로 파티션 평가).

---

## 기반 코드

이 저장소는 [GenHand](https://www.nature.com/articles/s44182-026-00076-1) (npj Robotics 2026) 공식 코드에서 출발했습니다. 데이터셋은 [DexYCB](https://dex-ycb.github.io/), [ARCTIC](https://arctic.is.tue.mpg.de/), [Dexonomy](https://pku-epic.github.io/Dexonomy/)를 사용합니다.
