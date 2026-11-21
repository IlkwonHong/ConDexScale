# EXP-005-net1-data-scaling: Experiment plan
- **Status:** APPROVED (2026-11-20, 연구자가 계획 승인 화면에서 승인).
- **Associated idea (folder):** IDEA-001-imitate-verified-gt
- **Question:** GT 를 다시 만들 때 파지 수를 늘릴 가치가 있는가 (학습 데이터가 늘면 Net1 이 좋아지는가).
- **Hypothesis:** 학습 데이터를 50% → 100% 로 늘리면 검증 top-1(N=3) 이 시드 간 차이보다 크게 오른다.
- **Setup:** `contraction_net/train_stage1.py` 에 `--train-frac`(물체 해시 기준 고정·중첩 부분집합, 시드 무관), `--seed` 추가.
  나머지 인자는 `checkpoints/stage1_v11/best.pt` 에 저장된 v1.1 인자와 동일 (epochs 30, batch 256, lr 3e-4, workers 4, sigma_max 0.08, --no-contact-geom).
  검증 세트(물체 해시 5%, 4,648 파지)는 바꾸지 않음. 학습 파지 100% = 95,066, 50% = 47,791.
- **Runs:** frac {0.25, 0.5, 1.0} × seed {0, 1} = 6회, 차례로.
- **Primary metric and fixed criterion (실행 전 고정):** 최고 에폭(best.pt 기준 = top-1 N3·N4 평균 최대)의 top-1 N=3, N=4, 두 시드 평균.
  - 50%→100% 상승 ≥ 2%p 이고 두 시드 차이보다 크면 "데이터 부족" (생성 수를 늘릴 가치 있음).
  - 상승 < 1%p 면 "평평" (지금 규모 또는 더 적게).
  - 그 사이는 연구자가 판단.
- **Reference:** 기존 stage1_v11 top-1 N=3 46.5%, N=4 41.4% (시드 미고정, 단일 실행).
- **Budget / stop:** 예상 72분 (전체 30에폭 ≈ 21분 실측 기준). 2시간 넘으면 멈추고 보고. GPU RTX 5080 1장.
- **Outputs:** `checkpoints/exp005_f{frac}_s{seed}/` (저장소 밖), 실행 기록 runs/.
- **Caveats:** 시드 2개뿐이라 1~2%p 차이는 우연과 구분이 약함. 지금 α 기반 GT 로 학습하므로 ρ 기반 새 GT 에서의 데이터 효율은 따로 확인해야 함.
- **Approval record:** (연구자 확인 필요) 2026-11-20 연구자가 "데이터 규모에 대해서 한 시간 시험을 하고 1000파지 시범 생성으로 시간체크를 하자"고 지시, 계획 승인.
