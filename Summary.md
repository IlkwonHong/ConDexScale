# Project Summary — 진행 단계별 요약 및 코드 위치

**프로젝트 목표:** 사람의 5손가락 파지(micro-manipulation)를 소수의 접촉 앵커로 "수축(contraction)"하여, 스케일업된 물체를 다루는 2~4대 다중 로봇 암 조작(macro-manipulation)으로 변환한다.

상세 세션 로그는 `PROGRESS.md`, 일일 기록은 `md_claude/daily_progress/` 참고.

---

## 전체 타임라인 한눈에 보기

| 기간 | 단계 | 결과 요약 |
|---|---|---|
| 2026-04-22 ~ 05-22 | 1. GenHand 환경 구축 + DexYCB 데이터 생성 | 데모 재현 성공, 배치 파이프라인 구축 (배치화 한계 확인) |
| 2026-05-25 ~ 06-06 | 2. 수축 네트워크 첫 학습 (DexYCB) | GraspContractionNet 구현, canonical frame 리팩토링, 1,272샘플로는 underfitting |
| 2026-06-10 ~ 07-01 | 3. ARCTIC 양손 데이터셋 확장 | critical 버그 다수 수정, 수축 알고리즘 재설계(joint k-means), 8명 6,536샘플, held-out 오차 3.07cm |
| 2026-07-05 ~ 07-14 | 4. 동적 확장 1차 — DexMachina 점접촉 강화학습 | guidance 필수성 실증, 그러나 점접촉 수축의 지지면 소실이 병목 → 강화학습 트랙 동결 |
| 2026-07-14 ~ 07-17 | 5. DexScale — 강화학습 없는 궤적 최적화 | 최초 무낙하 운반 성공, 질량 ×2·×3 zero-shot 일반화. 현 병목은 뚜껑 자유도 |
| 2026-07-27 ~ 현재 | 6. 대규모 Contraction GT + Stage 1/2 네트워크 | Dexonomy 기반 GT 85.7만 판정 구축, Stage 1 완료·동결(top-3 정찰병 구조), Stage 2 설계 중 |

---

## 1단계 — GenHand 환경 구축 + DexYCB 데이터 생성 (2026-04-22 ~ 05-22)

### 요약

npj Robotics 2026에 출판된 GenHand(사람 손 + 물체 이미지를 입력받아 로봇별 파지 구성을 출력하는 human-to-robot grasp retargeting 프레임워크) 코드를 Python 3.11 / CUDA 12.4 환경에서 되살렸다. 데모 3단계 전체 실행에 성공한 뒤, DexYCB(한 손 파지 데이터셋, 물체 21종)에서 GT(ground truth, 정답 데이터) 파지를 배치 생성하는 파이프라인을 만들었다.

속도 개선을 위해 텐서 배치화와 subprocess 병렬화를 모두 시도했으나, force closure 최적화가 샘플마다 다른 SDF(signed distance function, 부호 있는 거리 함수)를 쓰기 때문에 배치화 이득이 1.5~1.8배에 그친다는 한계를 확인했다. 또한 8개 카메라 뷰가 동일 GT 포즈의 좌표 변환본(중복)임을 파악하고 카메라 1개만 쓰도록 정리해 데이터 87%를 삭제했다.

### 코드 위치

| 파일 | 역할 |
|---|---|
| `demo/run_demo.py` | GenHand 원본 데모 실행 진입점 (3단계: 네트워크 추론 → 최적화 → 시뮬레이션 검증) |
| `optimisation/Optimize.py` | GenHand 핵심 최적화 (Stage 1 초기화 + Stage 2 refine + force closure) |
| `optimisation/dataset_opt.py` | 데이터셋 로더 (DexYCB / 이후 ARCTIC의 `ArcticFeed` 클래스도 여기 추가됨) |
| `optimisation/robot.py` | 로봇 손 모델의 forward kinematics와 표면 점 계산 (배치 shape 버그 2건 수정됨) |
| `optimisation/optimize_batch.py` | 텐서 배치화 시도 (`BatchedOptimization` 클래스, 1.84배 한계 확인) |
| `scripts/run_batch.py` | subprocess 병렬 배치 러너 (persistent worker 방식) |
| `scripts/run_partial.py` | subject/robot/camera 인자로 부분 배치 실행하는 범용 러너 |
| `scripts/clean_camera_data.py` | 8뷰 중복 데이터 정리 (카메라 1개만 유지) |
| `network/`, `simulation/` | GenHand 원본의 추론 네트워크와 PyBullet 시뮬레이션 검증 코드 |

**데이터:** `optimisation/results_train/<subject>/<robot>/mu<mu>/` (DexYCB 결과 JSON)

---

## 2단계 — 수축 네트워크(GraspContractionNet) 첫 학습 (2026-05-25 ~ 06-06)

### 요약

사람 손 접촉을 K개 앵커로 수축하는 네트워크를 처음 구현하고 학습했다. 구조는 물체 패치 토큰 + 접촉 토큰 → 상호작용 트랜스포머 → 앵커 쿼리 디코더 → 패치 히트맵 헤드(4.3M 파라미터). 학습 라벨은 1단계에서 생성한 result JSON의 `human_anchor`(pseudo-label)를 쓴다.

이 과정에서 좌표계 불일치 버그(패치·손 관절은 hand-local frame인데 접촉·앵커는 object-centric frame이라 네트워크가 구조적으로 GT에 도달 불가)를 발견해 수정하고, 이어서 물체 canonical frame으로 전면 리팩토링했다(같은 물체면 패치 좌표가 샘플 무관하게 동일). DexYCB 1,272샘플(subject-01)로 학습한 결과 epoch 19 이후 정체 — 데이터가 부족하다는 결론으로 3단계 확장의 동기가 됐다.

### 코드 위치

| 파일 | 역할 |
|---|---|
| `training/model.py` | `GraspContractionNet` 본체 (ObjectTokenEncoder → ContactTokenEncoder → InteractionTransformer → AnchorQueryDecoder → PatchHeatmapHead + AssignmentHead) |
| `training/dataset.py` | 그래프 캐시 + result JSON을 로드해 고정 shape 텐서 배치를 반환 (DexYCB용 + 이후 ARCTIC용 `ArcticContractionDataset` 추가) |
| `training/loss.py` | Hungarian 매칭 L2 / 표면 / 벌어짐(spread) 손실 |
| `training/train.py` | 학습 진입점 (`--dataset arctic` 등 인자는 3단계에서 추가) |
| `scripts/precompute_object_patches.py` | YCB 메시 → FPS(farthest point sampling) → kNN → 물체 패치 캐시 |
| `scripts/precompute_sample_graphs.py` | 패치 캐시 + result JSON → 샘플별 그래프 캐시 (canonical frame 변환 포함) |
| `scripts/visualize_grasp.py` | Plotly 2패널 시각화 (파지 vs 수축 결과 GT/예측 비교) |

---

## 3단계 — ARCTIC 양손 데이터셋 확장 (2026-06-10 ~ 07-01)

### 요약

프로젝트 핵심이 양손 → 다중 암 변환이므로, 한 손뿐인 DexYCB를 넘어 ARCTIC(양손 + 1자유도 관절 물체 조작 데이터셋, 물체 11종)으로 전환했다. 이 단계에서 critical 버그를 연달아 발견·수정했다:

- **articulation 반대 적용 버그** — 가동부가 아닌 고정 본체를 회전시키고 있었음. 사진 재투영(GT 카메라에 3D를 투영해 사진과 겹치는지 확인)으로 검증하는 규칙이 여기서 확립됨 (`CLAUDE.md` §4).
- **genhand HDBSCAN(밀도 기반 군집화)이 한 번도 실행된 적 없던 버그** — NumPy ragged array 크래시를 try/except가 삼키고 fingertip fallback으로 위장하고 있었음.
- **접촉 게이팅 버그** — 5cm 임계값이 느슨해 안 닿은 손가락까지 수축에 포함 → 1.5cm로 수정.

수축 알고리즘 자체도 재설계했다: combined HDBSCAN(한 손으로 collapse하는 문제) → per-hand genhand → 최종적으로 **joint k-means**(게이팅 후 [정규화 위치, 가중 물체 법선]에 k-means, 불균형 핀치에서도 antipodal 복구). 캐시는 design B(모델 입력과 GT 라벨을 분리 저장 → 수축만 재생성해도 입력 캐시 재활용)로 재구성했다.

최종: 8명(subject) 6,536샘플로 학습해 내부 검증 오차 2.57cm, 처음 보는 사람(s05, 물리적으로 분리된 held-out) 3.07cm — cross-subject 일반화를 확인했다.

### 코드 위치

| 파일 | 역할 |
|---|---|
| `scripts/run_arctic.py` | ARCTIC 배치 러너 (subprocess 병렬, `--subjects` / `--anchors-per-hand` / `--out-base` 인자) |
| `optimisation/dataset_opt.py` (`ArcticFeed` 클래스) | MANO(사람 손 파라메트릭 모델)/물체 npy 로드, articulation 변환, 양손 데이터 반환 |
| `optimisation/Optimize.py` (`run_idx_json_bimanual`) | 양손 파이프라인 진입점 (손별 수축 → 라벨 생성) |
| `optimisation/objects.py` (`contract_kmeans`, `hand_contact_cluster_hdbscan`) | 수축 알고리즘 본체 — 최종 채택은 joint k-means, ragged array 버그도 여기서 수정 |
| `scripts/precompute_arctic_object_patches.py` | ARCTIC 물체 11종 패치 캐시 (rest pose, 가동부 표시 포함) |
| `scripts/precompute_arctic_sample_graphs.py` | result JSON + 패치 → 양손 그래프 캐시 (articulation 적용의 단일 지점) |
| `scripts/recontract_arctic.py` | 오프라인 재수축 (result JSON의 앵커 라벨만 갱신, design B 활용) |
| `training/dataset.py` (`ArcticContractionDataset`) | ARCTIC 그래프 캐시 로드 + 앵커를 result JSON에서 `Rg.T` 변환, `--gate-max-dist` 게이팅 |
| `scripts/visualize_arctic_grasp.py` | ARCTIC 파지/수축 Plotly 시각화 |
| `scripts/dataset_viewer.py` | Dash 기반 인터랙티브 뷰어 (http://localhost:8050, 사진 + 3D + GT/예측 토글) |
| `scripts/visualize_eval_predictions.py`, `scripts/visualize_eval_predictions_rich.py` | 평가 결과 시각화 (12패널 rich 뷰 등) |

**데이터:** `data/results_arctic/` (학습용 result JSON), `data/results_arctic_val/` (s05 held-out), `data/arctic_cache/` (object_patches + sample_graphs + anchors)

**문서:** `md_claude/arctic/data_generation_v1.md`, `md_claude/contraction_methods/genhand_vs_kmeans.md`

---

## 4단계 — 동적 확장 1차: DexMachina 점접촉 강화학습 (2026-07-05 ~ 07-14)

### 요약

정적 파지를 넘어 동적 조작으로 확장하기 위해 DexMachina(functional retargeting 강화학습 프레임워크, Genesis 시뮬레이터 기반)를 채택했다. 추상화는 **"로봇 암 1대 = 접촉점 1개 = 접촉력 1개"**(그리퍼가 아닌 점접촉체, 팔 수 K는 파라미터). 수축 결과를 guidance로 넣고 VOC(virtual object controller — 물체를 가상 힘으로 잡아주다가 점차 줄여가는 커리큘럼) 감쇠로 학습했다.

**성과:** guidance 없이 task 보상만으로는 접촉조차 학습하지 못함을 실증 (물체 이동 0.6cm) — "수축 guidance가 필수"라는 핵심 주장의 1차 지지 증거.

**한계 (트랙 동결 사유):** 정책 없이 기준 궤적을 재생(kinematic replay)만 해도 박스가 넘어짐 — 손바닥 면접촉의 지지 다각형이 점접촉 수축에서 소실되기 때문. 즉 병목은 학습량이 아니라 guidance 품질. 표면 구속 액션(슬라이드 Δu,Δv + 프레스 d) ablation을 2차례 수행했고, 학습 지표가 조기종료 리셋 사이클 때문에 낙관적으로 왜곡되는 "reset-farming 착시"도 규명했다(판정은 연속 rollout으로만 할 것).

### 코드 위치

| 파일 | 역할 |
|---|---|
| `extlib/dexmachina/` | DexMachina fork (point-contact 브랜치 — 환경/보상/액션 수정 본체) |
| `extlib/Genesis/` | Genesis 시뮬레이터 fork (RTX 5080 sm_120 호환 패치 적용) |
| `extlib/rl_games/` | 강화학습 라이브러리 |
| `scripts/gen_point_urdf.py` | 점접촉 로봇 URDF 생성 |
| `scripts/export_dexmachina_guidance.py` | 수축 결과 → DexMachina guidance npy 변환 |
| `scripts/verify_dexmachina_guidance.py` | guidance 재투영 검증 (오차 1.9mm 확인) |
| `scripts/visualize_dexmachina_guidance.py` | guidance 시각화 |
| `scripts/test_surface_action_torch.py`, `scripts/test_surface_insim.py`, `scripts/test_surface_projection.py` | 표면 구속 액션 검증 테스트 |

**산출물:** `md_claude/dexmachina/` (rollout 비디오), `md_claude/plans/{offline_keyframe_planning,dexmachina_integration,surface_action_ablation}.md`

---

## 5단계 — DexScale: 강화학습 없는 궤적 최적화 + 임피던스 제어 (2026-07-14 ~ 07-17)

### 요약

강화학습 트랙을 동결하고, 수축 결과를 prior로 쓰는 **폐루프 CEM(Cross-Entropy Method) 궤적 최적화 + 임피던스 제어** 방식으로 전환했다. 목표 task는 ARCTIC 실궤적(box_use_01) 추종, 시뮬레이터는 Genesis, 팔당 최대 힘 20N.

핵심 설계는 **하이브리드 reference**: `x_ref = 실물 위 앵커 + (r−δ)·법선 + clip(데모−실물, d_max)` — 쥐는 힘은 물체 기준으로, 운반은 포화 오프셋으로 만든다. 데모 고정 reference가 실패 근본 원인(접촉 이탈이 힘 포화 없이 발생)임을 진단한 결과다.

**성과:** 파이프라인 **최초 무낙하 운반 성공** (raw 220프레임). 학습 범위 밖 질량 ×2·×3에서도 zero-shot 일반화 (최종 오차 1~2cm, robustness envelope 16칸 중 11칸 통과) — 스케일업 서사와 정합. **현 병목:** 뚜껑(관절 자유도) 제어 — 운반 힘이 본체 대신 뚜껑을 움직임. 현재 메인 방향.

### 코드 위치

| 파일 | 역할 |
|---|---|
| `dexscale/cem.py` | CEM 궤적 최적화 본체 |
| `dexscale/reference.py` | 하이브리드 reference 생성 (실물 앵커 + 포화 오프셋 + 접촉 게이트) |
| `dexscale/rollout.py` | Genesis rollout 실행 (임피던스 제어 포함) |
| `dexscale/knots.py` | 궤적 knot 파라미터화 |
| `scripts/run_dexscale_to.py` | 궤적 최적화 실행 진입점 |
| `scripts/eval_dexscale_envelope.py` | robustness envelope 평가 (질량·마찰 그리드) |
| `scripts/probe_dexscale_onset.py` | 파지 onset 구간 진단 |
| `scripts/visualize_dexscale_rollout.py`, `scripts/visualize_rollout_overlay.py` | rollout 시각화 |
| `scripts/test_dexscale_core.py`, `scripts/test_genesis_impedance_smoke.py` | 핵심 로직·임피던스 게이트 테스트 |

**문서:** `md_claude/plans/dexscale_mvp.md` (설계), `md_claude/dexscale/optimization_formulation.html` (수식)

---

## 6단계 — 대규모 Contraction GT + Stage 1/2 네트워크 (2026-07-27 ~ 현재)

### 요약

수축을 제대로 된 학습 문제로 풀기 위해, ARCTIC(물체 11종)보다 훨씬 큰 **Dexonomy 기반**(Shadow 손 파지 대규모 데이터셋, 물체 10,690종) GT를 구축했다: 10만 파지 → 탐색 레코드 1,219,442건 + 시뮬 합격 판정 856,656건 전량 (합격률 68.9%). DexYCB 병합 학습은 제외 결정(1:360 불균형, 라벨 의미 차이).

**Stage 1 (완료·동결, v1.1):** "어떤 접촉 파티션(토폴로지 — 어느 접촉 부위 묶음에 팔을 배치할지)을 고를지" 분류하는 그래프 트랜스포머. 전역 파티션 클래스 사전 + 유효성 마스크, 관계 유형 7종의 헤드별 바이어스. 단발 top-1은 39%대에 정체(개선 시도 4연속 기각)했지만, 구제 가능한 파지에서 **상위 3개 후보 안에 시뮬 합격 후보가 95.7~96.4% 포함** → "정찰병(top-3 제안) + 검산(LP 재계산 + 짧은 시뮬)" 구조로 확정. 체크포인트: `checkpoints/stage1_v11/best.pt`.

**Stage 2 (설계·구현 진행 중):** 각 파티션의 실제 앵커 위치를 생성하는 단계. 실측으로 확정된 사양 — 검산은 LP(linear program) + 짧은 시뮬 병행(LP 단독은 불합격 30% 선택), 생성 목표는 다듬기 반경(물체 반경의 5~10%) 진입, 다듬기는 GT refine 재사용. 현재 방안 논의(`Memo.md`): diffusion 원안(B) 대신 **표면 히트맵 + 조합 LP 선택(A)으로 정적 v1을 먼저 완성**하고 diffusion은 동적 확장 시점에 재평가하는 C안이 추천된 상태.

### 코드 위치

| 파일 | 역할 |
|---|---|
| `contraction_gt/search.py`, `contraction_gt/runner_batched.py` | GT 탐색 생성 (10만 파지 → 탐색 레코드) |
| `contraction_gt/sim_eval.py`, `contraction_gt/gate_runner.py`, `contraction_gt/gate_runner_all.py` | 시뮬 합격 판정 (전 파티션 85.7만 건) |
| `contraction_gt/metric.py`, `contraction_gt/benchmark_metric.py` | 품질 지표 (LP α 등) |
| `contraction_gt/pilot_*.py` | 파일럿 실험 (probe/runner/sim/analysis/select) |
| `contraction_gt/package_parquet.py`, `contraction_gt/contacts_io.py`, `contraction_gt/object_cache.py`, `contraction_gt/surface.py`, `contraction_gt/units.py` | 데이터 패키징·입출력·물체 캐시·표면 연산·단위 처리 |
| `scripts/dexonomy/setup_assets.py`, `scripts/dexonomy/extract_contacts.py` | Dexonomy 원본 자산 준비 + 접촉 추출 (`data/dexonomy/contact_cache`) |
| `scripts/dexonomy/shadow_fk.py`, `scripts/dexonomy/validate_fk.py` | Shadow 손 forward kinematics 재현·검증 |
| `scripts/dexonomy/INTERFACE.md` | Dexonomy 데이터 인터페이스 문서 |
| `contraction_net/dataset.py` (`ContractionGTDataset`) | GT 3원 join 로더 (탐색 레코드 + 판정 + 패치) |
| `contraction_net/build_patch_cache.py`, `contraction_net/extract_patch_contacts.py` | 패치 캐시(10,690물체 × 128패치) + 실측 패치 접촉 |
| `contraction_net/partition_vocab.py` | 전역 파티션 클래스 사전 |
| `contraction_net/stage1_model.py`, `contraction_net/stage1_dataset.py`, `contraction_net/train_stage1.py`, `contraction_net/eval_stage1.py` | Stage 1 그래프 트랜스포머 (모델·데이터셋·학습·평가기) |
| `contraction_net/stage2_model.py`, `contraction_net/stage2_dataset.py`, `contraction_net/train_stage2.py` | Stage 2 (진행 중) |
| `extlib/Dexonomy/` | Dexonomy 원본 저장소 |

**데이터:** `data/contraction_gt_300k/search_records/`, `data/dexonomy/contact_cache/`

**문서:** `md_claude/plans/stage1_net_diagram.html` (Stage 1 구조도 + 코드 대응 + 실험 대장 — 단일 기준점), `md_claude/daily_progress/07_27.md`

---

## 현재 상태 (2026-08-01 기준)

**활성 트랙 2개:**

1. **정적 수축 네트워크 (6단계):** Stage 1 동결, Stage 2 설계 확정 직전 (표면 히트맵 + 조합 LP 방안 유력).
2. **동적 조작 — DexScale (5단계):** 뚜껑 자유도 제어 설계가 다음 과제. 강화학습 트랙(4단계)은 비교 베이스라인으로 보존.

**주요 체크포인트:**

| 경로 | 내용 |
|---|---|
| `checkpoints/stage1_v11/best.pt` | Stage 1 v1.1 확정본 (top-1 39.5%/37.3%, top-3 적중 74%/68%) |
| `checkpoints/best.pt` | ARCTIC 수축 네트워크 (joint k-means 라벨 + 1.5cm 게이팅, val_loss 0.0779) |
