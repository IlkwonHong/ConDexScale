# Project Progress

## Project Overview

**GenHand** — Generalised Human Grasp Kinematic Retargeting  
Published in [npj Robotics (2026)](https://www.nature.com/articles/s44182-026-00076-1)

A learning-based framework for human-to-robot grasp retargeting. Takes human hand + object images as input and produces robot-specific grasp configurations that preserve contact relationships. Supports Shadow, Allegro, Barrett, and Robotiq robots.

---

## Session Log

### 2026-04-22 — Environment Setup

**Goal:** Get the development environment running in a Python 3.11 / CUDA 12.4 Docker container.

**Completed:**

- [x] Diagnosed and fixed SSH submodule failure in `dex-ycb-toolkit` install (git SSH→HTTPS URL rewrite)
- [x] Fixed `chumpy` build failure (wheel + `--no-build-isolation`)
- [x] Resolved `torchvision==0.14.1` unavailability (yanked from PyPI)
- [x] Resolved `numpy==1.21.6` Python 3.11 incompatibility
- [x] Installed `torch 2.6.0+cu124` / `torchvision 0.21.0+cu124` from PyTorch index (CUDA 12.4 aligned)
- [x] Built and installed `pytorch3d 0.7.9` from source with GPU support (CUDA kernel compilation ~15 min)
- [x] Installed all remaining dependencies: `tensorboard`, `loguru`, `pybullet`, `mujoco`, `fire`, `lmdb`, `kornia`, `hdbscan`, `urdf-parser-py`
- [x] Verified all core imports: `torch` (CUDA=True), `torchvision`, `pytorch3d`, `numpy`, `scipy`, `sklearn`, `cv2`, `open3d`, `pandas`, `mujoco`, `pybullet`, `kornia`, `loguru`, `lmdb`

**Note:** Packages were installed into system Python (`/usr/local/lib/python3.11/dist-packages/`), not the venv. See 2026-04-23 session for venv fix.

---

### 2026-04-23 — venv Fix + Demo Running

**Goal:** Fix broken venv, install all packages properly, run the sample demo end-to-end.

#### Root cause: venv was broken by folder rename

The project folder was renamed from `/workspace/Generalised_Human_Grasp_Kinematic_Retargeting/` to `/workspace/genhand/`. The `venv/bin/activate` script had the old absolute path hardcoded.

**Fix:**
```bash
grep -rl "Generalised_Human_Grasp_Kinematic_Retargeting" venv/bin/ | \
  xargs sed -i 's|/workspace/Generalised_Human_Grasp_Kinematic_Retargeting/venv|/workspace/genhand/venv|g'
```

#### Compatibility patches applied

**chumpy** — `inspect.getargspec` → `inspect.getfullargspec`, numpy alias imports patched  
**dex-ycb-toolkit** — missing `__init__.py` added before pip install  
**MANO model files** — must be placed at `network/manopth/mano/models/MANO_{RIGHT,LEFT}.pkl`

#### Code patches for sample-data compatibility

- `optimisation/dataset_opt.py` — DexYCBDataset wrapped in try/except; fallback to local images
- `simulation/opt_results_loader.py` — `_DexYCBMock` class added; reads from JSON annotation
- `simulation/xyz.urdf` — created (phantom anchor body for PyBullet floating-arm trick)

#### Demo result

```bash
python demo/run_demo.py --sdf-source gt_pv --method genhand \
    --cluster-method hdbscan --contact-threshold 0.05 \
    --sample-idx 70 --robot Shadow --device cuda:0 --sim-render DIRECT
```

All 3 stages completed (~22 sec Stage 2). Output: `optimisation/results/Shadow/mu0.9/70_Shadow_0_mu_0.9.json`

---

### 2026-04-25 — 배치 파이프라인 + 텐서 배치화 시도

**Goal:** 5928개 전체 샘플에 대해 gt_pv 모드로 GT 데이터셋 빠르게 생성 (PLAN.md Week 4).

#### 방향 결정

- **gt_pv 채택**: Stage 1(네트워크 추론) 없이 DexYCB GT 메시 직접 사용 → 깨끗한 contact 데이터, Module B 학습용으로 적합
- **recon_pv**: 나중에 end-to-end 시스템 검증 때 사용
- **Isaac Lab 기각 (단기)**: Stage 2는 gradient descent — 물리 시뮬레이터가 아님. Stage 3 교체에는 유효하나 Stage 2 병목 해결 불가. 장기 과제로 분류.

#### 생성 파일

| 파일 | 내용 |
|---|---|
| `scripts/run_batch.py` | Persistent worker subprocess 배치 러너 |
| `scripts/__init__.py` | 패키지 init |
| `optimisation/optimize_batch.py` | BatchedOptimization 클래스 (텐서 배치화) |

#### 버그 수정

| 파일 | 내용 |
|---|---|
| `simulation/robot_base.py` | VS Code 서버 오류 메시지 삽입 → SyntaxError 복원 |
| `optimisation/robot.py` L150 | `get_surface_points_updated`: `.squeeze()` → `.squeeze(0)` (B=1 호환 유지하면서 B>1 지원) |
| `optimisation/robot.py` L276 | `get_keypoints`: `global_translation.unsqueeze(1)` 제거 → `(B,1,3)`이 이미 올바른 shape, unsqueeze(1) 시 (B,1,1,3)이 되어 (B,B,K,3) broadcasting 오류 발생 |

**버그 근본 원인:** 원래 코드는 batch=1 전용으로 설계되었고, `.squeeze()` (인자 없음 = 모든 size-1 차원 제거)가 배치 차원을 자동 제거. batch=N 지원 시 이 숨겨진 shape 버그들이 드러남.

#### 속도 벤치마크 (RTX 4090, Shadow, gt_pv)

**프로파일링 결과:**
```
FC optimization:      9.36s  (2000 iter, force closure)
FK (get_surface):    32.20ms/call  (batch=1)
FK batch=8:          34.41ms/call  → 처리량 7.5× (GPU 병렬)
SDF pv_sdf:           2.53ms/call
penetration_check:  100.63ms/call  ← 주요 병목
```

**run_batch.py (subprocess 병렬화):**

| 설정 | 4샘플 처리 | 시간당 처리량 |
|---|---|---|
| N=1 순차 | 6.2분 | ~39 샘플/h |
| N=4 병렬 | 4.1분 | ~59 샘플/h |
| **실제 향상** | | **1.5×** (예상 4-6× 빗나감) |

이유: GPU 컨텍스트 경합으로 샘플당 시간이 62초 → 109초로 역증가.

**BatchedOptimization (텐서 배치화):**

| 설정 | 2샘플 처리 | 샘플당 |
|---|---|---|
| N=1 | ~101s | ~101s |
| N=2 | ~110s | ~55s |
| **실제 향상** | | **1.84×** |

이유: q_optimization은 FK 배치화로 향상되나, fc_optimization이 샘플마다 다른 object SDF 때문에 병렬화 한계. CUDA 스트림 시도했으나 메모리 대역폭 병목으로 추가 개선 없음.

**5928샘플 전체 추정:**
```
BatchedOpt N=2:    ~90시간 (3.8일)
run_batch.py N=4:  ~100시간 (4.2일)
```

#### DexYCB 데이터셋

- `/workspace/dataset/`에 `dex-ycb-20210415.tar.gz` (119GB) 압축 해제 중
- 2026-04-25 기준 10개 subject 중 일부 완료
- `calibration/` 폴더 미완료 → DexYCBDataset mock fallback 동작 중
- Stage 3: mock에서 물체 메시 경로 미제공 → `createCollisionShape failed` (압축 완료 후 해결 예정)

#### 결론 및 다음 단계

fc_optimization 배치화 한계 확인.

#### 재검토: Isaac Lab의 실제 활용 범위

초기 결론은 "Isaac Lab으로 전환"이었으나, 다시 따져보니 단순 이식으로는 병목이 해결되지 않음.

**핵심 오해 정정:**

병목은 "물리 시뮬이 느려서"가 아니라 **"샘플마다 SDF가 달라서 gradient descent 배치화 불가"**. Isaac Lab은 gradient descent를 빠르게 해주는 도구가 아니라 **물리 시뮬을 병렬화**해주는 도구. 따라서 `fc_optimization` 코드를 Isaac Lab 환경에 그대로 옮겨도 이득 없음.

**Isaac Lab으로 진짜 이득을 보려면 알고리즘 재설계 필요:**

| 항목 | 현재 fc_optimization | Isaac Lab 활용 방식 |
|---|---|---|
| 패러다임 | **생성적** (gradient로 최적 grasp 탐색) | **평가적** (후보 sampling → 물리 검증) |
| 답하는 질문 | wrench-space analysis상 force closure인가? | 중력/외란 하에 실제로 잡히는가? |
| 병렬화 | 샘플간 SDF 달라서 어려움 | 환경별 독립 → 자연스럽게 GPU 벡터화 |

이는 논문 알고리즘 변경 → 학습 데이터 분포 변화 → npj 결과 재현성 영향. **연구급 변경**이라 단기 데이터 확보 목적과 분리해야 함.

#### 수정된 우선순위

1. **단기 (데이터 확보):** BatchedOpt N=2로 5928샘플 직진. 검증된 알고리즘으로 깨끗한 GT 우선 확보 (~4일).
2. **중기 (Isaac Lab 도입 범위 한정):** Stage 3만 PyBullet → PhysX 5 교체. fc_optimization은 그대로 유지. grasp이 실제 물리적으로 잡히는지 정밀 검증 용도.
3. **장기 (별도 연구 과제):** sample-and-validate 패러다임 탐색 — fc_optimization을 Isaac Lab 벡터화 물리 검증으로 대체하는 알고리즘. 논문 분량 작업.

#### venv 충돌 처리

Isaac Lab은 Python 3.10 + Isaac Sim 번들 환경 필요 (genhand venv는 3.11). **분리 설치**로 해결:
- genhand venv: `/workspace/genhand/venv/` (Python 3.11) — Stage 1+2 그대로
- Isaac Lab env: `/workspace/isaaclab_env/` (Python 3.10) — Stage 3 검증
- 인터페이스: JSON 파일 I/O (Stage 2 결과 → Isaac Lab 검증 결과)

이 구조에서는 fc_optimization 병목이 그대로 남지만, 단기 우선순위가 데이터 확보이므로 OK. 검증 정밀도만 PhysX로 향상.

---

### 2026-05-22 — 데이터 파이프라인 완성 + 결과 포맷 확장

**완료:**
- [x] DexYCB `calibration.tar.gz` / `models.tar.gz` 압축 해제 완료 (21개 YCB 모델)
- [x] `dexycb_trainfeed` 클래스 구현 — train split, on-the-fly 메시 생성
- [x] `scripts/run_partial.py` 구현 — `--subject` / `--robot` 인자로 범용 배치 실행
- [x] **Module A 개념 정정:** GenHand의 `build_contact_target()` + HDBSCAN이 이미 Contraction 수행
  - `--robot Robotiq` → 2개 anchor (2-arm), `--robot Barrett` → 3개 anchor (3-arm)
  - 별도 Module A 구현 불필요
- [x] Stage 2를 `refine` 버전으로 교체 (실제 2차 최적화 수행)
- [x] 결과 JSON에 학습용 필드 추가: `file_name`, `ycb_id`, `obj_center`, `obj_transform`, `hand_joints`, `human_finger_contacts` (손가락별), `human_anchor`

**결과 저장 경로:** `optimisation/results_train/<subject>/<robot>/mu<mu>/`

**좌표계:** 전체 hand-local frame (손목=원점). `obj_transform` (4×4)으로 object canonical frame 역변환 가능.

> ⚠️ **좌표계 정정 (2026-05-29):** 실제로는 result JSON의 `human_finger_contacts`, `human_anchor`가 **object-centric frame** (obj_centre 기준)으로 저장됨. 위 설명은 오류. 자세한 내용은 05_29 세션 참고.

---

## Environment State

```
Python:           3.11 (venv at ./venv)
CUDA:             12.4 (nvcc) / driver supports 13.0
torch:            2.6.0+cu124  (CUDA available: True)
pytorch3d:        0.7.9
venv python:      /workspace/genhand/venv/bin/python
GPU:              RTX 4090 (24GB VRAM)
DexYCB:           /workspace/dataset/ (calibration + models 완료)
```

---

---

### 2026-05-25 — Grasp Contraction Network 학습 파이프라인 구축

**완료:**
- [x] result JSON 필드 분석 → `human_anchor` (pseudo-label), `human_finger_contacts`, `hand_joints` 확인
- [x] 오프라인 캐시 vs 데이터로더 분리 결정 (Object patch graph + Finger-Object edge → 오프라인)
- [x] `scripts/precompute_object_patches.py` — YCB 메시 → FPS → kNN → `offline_cache/ycb_{id}_patches.json`
- [x] `scripts/precompute_sample_graphs.py` — 패치 캐시 + result JSON → hand-local graph → `graph_cache/{file_name}_graph.json`
- [x] `training/dataset.py` — 두 JSON 로드 → 고정 shape 텐서 배치 반환
- [x] `training/model.py` — `GraspContractionNet` MVP (4.3M 파라미터)
  - `ObjectTokenEncoder` → `ContactTokenEncoder` → `InteractionTransformer` → `AnchorQueryDecoder` → `PatchHeatmapHead` + `AssignmentHead`
- [x] `training/loss.py` — Hungarian L2, surface, spread 손실 구현
- [x] forward → loss → backward 전체 정상 동작 확인

**핵심 결정:**
- 데이터로더가 두 JSON 로드: contracted grasp JSON (pseudo-label) + graph JSON (구조)
- `PatchHeatmapHead`: dot-product 방식 → P 크기 무관 (resolution-agnostic)
- `hand_trans`는 `dexycb_train_s0.json`에서 `file_name` 기준 lookup으로 복원

---

### 2026-05-29 — 환경 세팅 + 좌표계 버그 수정 + 시각화 구현

**완료:**
- [x] 새 컴퓨터(Robin-X870) venv 세팅 — Python 3.11, torch 2.5.1+cu121, pytorch3d 0.7.9
  - RTX 5080 (sm_120) ↔ torch 2.5.1+cu121 미호환 → CPU fallback 중
- [x] `scripts/visualize_grasp.py` 구현 — Plotly 2-panel (Panel A: dexterous grasp, Panel B: contracted grasp GT vs Pred)
- [x] **좌표계 버그 발견 및 수정 (Critical)**
  - `precompute_sample_graphs.py`에서 `patch_positions`와 `hand_joints`가 hand-local frame으로 저장됐으나, `contact_positions`·`human_anchor`는 object-centric frame → 불일치
  - `PatchHeatmapHead` 구조상 예측값은 항상 patch 범위에 갇혀 GT(object-centric) 도달 불가
  - 수정: `patch_pos -= obj_centre`, `hand_joints -= obj_centre` 적용
  - graph_cache 1272개 전체 재생성 완료
- [x] 기존 `checkpoints/` (500 epoch) 폐기 확정 — 잘못된 좌표로 학습된 무효 체크포인트

**버그 증거:**
```
l_anchor: epoch 0 = 0.167  →  epoch 499 = 0.150  (500 epoch에 0.017만 감소, 수렴 실패)
이론적 하한: ~0.22m (obj_centre 오프셋) — 아키텍처상 도달 불가
```

---

### 2026-05-31 — Canonical Frame 리팩토링 + torch 2.7.0 + 1000 epoch 훈련

**완료:**
- [x] torch 2.7.0+cu128 + pytorch3d 재빌드 → RTX 5080 (sm_120) GPU 정상 동작
- [x] **Canonical frame 리팩토링 (핵심)**
  - `precompute_sample_graphs.py`: patch 변환 제거, `R.T` 적용으로 contacts/hand_joints → canonical, `obj_rotation` [3×3] 저장
  - `training/dataset.py`: patch 데이터를 `offline_cache`에서 직접 로드, `human_anchor`에 `R.T` 적용
  - 검증: 같은 YCB 오브젝트 두 샘플의 patch_positions 차이 = **0.00000000** ✓
  - graph_cache 1272개 재생성 완료
- [x] 1000 epoch 훈련 완료, best: **epoch 19 / val_loss=0.0480**
- [x] GT anchor 품질 검증: gf_0 mean=0.0009, **1272개 100% force closure 통과**
- [x] TensorBoard 이전 run 정리 (`checkpoints/tb_old/`로 이동)

**현재 한계:**
- val_loss epoch 19 이후 plateau (~0.054) — 1272 샘플(subject-01)로 underfitting
- 데이터 확장 없이는 성능 개선 불가

---

## Environment State

```
Python:           3.11 (venv at ./venv)
CUDA:             nvcc 12.8 / driver 13.2 (RTX 5080 sm_120)
torch:            2.7.0+cu128  (CUDA available: True)
pytorch3d:        0.7.9  (torch 2.7.0 기준 재빌드)
GPU:              RTX 5080 (24GB VRAM)
DexYCB:           dataset/data/ (annotations + partial meshes)
graph_cache:      1272개 (object canonical frame, 2026-05-31 재생성)
checkpoints:      best.pt = epoch 19 / val_loss=0.0480 (canonical frame)
```

---

---

### 2026-06-06 — DexYCB 8뷰 중복 제거 + 데이터 정리

**완료:**
- [x] DexYCB 구조 파악: 8개 카메라 entry는 동일 GT 포즈의 좌표 변환본 → 중복
- [x] `run_partial.py` 수정: `--camera` 인자 추가, 카메라 1개만 처리 (기본: `836212060125`)
- [x] `scripts/clean_camera_data.py` 작성 + 실행: 기존 데이터 87% 삭제 (419MB → 57MB)
  - results_train: 28,476개 → 3,566개
  - graph_cache: 26,891개 → 3,378개
- [x] RGB 미사용 확인 (`color_img=None` 고정) → 1카메라 사용 문제 없음 확인

**핵심 결정:**
- DexYCB GT는 멀티뷰 fusion으로 이미 최적화된 값 → 카메라 1개로도 동일 품질
- 데이터 다양성을 위해 여러 카메라를 쓸 이유 없음 (canonical frame에서 거의 동일)

---

### 2026-06-14 — ARCTIC articulation 버그 수정 (Critical)

- **버그:** 물체 articulation이 **반대 part**에 적용됨. ARCTIC 공식은 `parts.json==0`(가동부) 회전, 우리는 `parts==1`(고정 본체) 회전 → 큰 본체가 엉뚱하게 휘둘림
- **수정:** `dataset_opt.py:397`, `visualize_arctic_grasp.py:204` → `top = parts == 0`
- **영향:** s01 621개 중 **arti>0.05rad 299개(48%) 무효** → 재생성 필요 (arti≈0인 322개는 영향 없음)
- 검증: 사진 재투영으로 box·capsulemachine 6개 샘플 전부 GT와 일치 확인 (오른손 contact 2~5mm)
- ⚠️ **arti>0 result JSON + 그로부터 만든 graph_cache 전부 무효** → 재생성·재훈련 필요
- (교훈) 내부 일관성(centroid/bbox)이 아니라 **공식 소스와 대조**해야 공유 버그를 잡음

---

### 2026-06-10~11 — ARCTIC Bimanual 데이터셋 확장 (방향 전환)

**Goal:** DexYCB(single-hand) 한계를 넘어, ARCTIC bimanual 데이터셋으로 양손(10손가락) → K-anchor contraction 파이프라인 확장. Robin-X870(RTX 5080)에서 진행.

#### 방향 전환 배경

- DexYCB는 single-hand. 본 프로젝트 핵심은 micro(사람 손) → macro(다중 로봇 암) 변환 → **양손 manipulation 데이터가 더 적합.**
- ARCTIC: 양손 + articulated object(1-DOF) bimanual manipulation 데이터셋. 사람 양손 contact를 HDBSCAN으로 K개 `human_anchor`로 수축 → K대 robot arm pseudo-label.

#### 생성/수정 파일

| 파일 | 내용 |
|---|---|
| `scripts/run_arctic.py` (신규) | ARCTIC 배치 러너. subprocess worker 병렬화, `run_idx_json_bimanual` 호출 |
| `scripts/visualize_arctic_grasp.py` (신규) | Plotly 2/3-panel 시각화 (Panel A: 양손 grasp, Panel B: contracted anchor) |
| `optimisation/dataset_opt.py` L299 (+303줄) | `ArcticFeed` 클래스 — MANO/object npy 로드, articulation 변환, bimanual mano_pack 반환 |
| `optimisation/Optimize.py` L938 (+187줄) | `run_idx_json_bimanual()` — left HumanModel 추가, bimanual combined HDBSCAN, Stage1+`second_stage_opt_wo_gq_refine` |
| `optimisation/objects.py` L492 (+15줄) | `hand_contact_cluster_hdbscan(extra_contact_pack)` — 양손 contact concat (80→160 points) |
| `md_claude/arctic/data_generation_v1.md` | 파이프라인 전체 문서 (좌표계/articulation/MANO forward/JSON 필드) |

#### 좌표계 (ARCTIC)

| 항목 | 단위 | 좌표계 |
|---|---|---|
| MANO trans | m | World (Vicon) |
| Object trans | mm | World (Vicon) |
| Canonical mesh | mm | Object canonical |
| 최적화 내부 | m | Object-centric (obj_centre 빼기) |

#### 결과 JSON 추가 필드 (bimanual)

`hand_joints_left`, `human_finger_contacts_left`, `obj_arti_angle`, `obj_name` 추가. 기존 `human_anchor`/`hand_joints`/`human_finger_contacts`/`obj_center`/`gf_0` 등 유지.

**결과 저장 경로:** `data/results_arctic/{subject}/{seq_name}/{robot}/mu{mu}/{idx}_{robot}_0_mu_{mu}.json`

#### 실행 결과

```bash
python scripts/run_arctic.py --robot Robotiq --subjects s01 --frame-stride 30 --workers 4 --device cuda:0
```

- **s01 / Robotiq / stride=30 → JSON 812개 생성 완료** (`data/results_arctic/`)
- 시각화 HTML 24개 생성 (`md_claude/arctic/`), 예: `vis_s01_capsulemachine_use_01_113_Robotiq_0_mu_0.9.html`
  - 생성 명령: `python scripts/visualize_arctic_grasp.py --result data/results_arctic/s01/capsulemachine_use_01/Robotiq/mu0.9/113_Robotiq_0_mu_0.9.json`

#### 미완료 / 다음 (v2 예정)

- [ ] `precompute_object_patches.py` ARCTIC 오브젝트(11종) 패치 캐시 생성
- [ ] `precompute_sample_graphs.py` ARCTIC bimanual 필드 처리 (joint_left 등)
- [ ] `training/dataset.py` ARCTIC graph cache 로드 지원
- [ ] left hand contact_zones 정확도 검증 (MANO left topology index 호환성)
- [ ] s02~s10 확장 (현재 s01만)

---

### 2026-06-14 (후속) — s01 재생성 + ARCTIC 전처리 캐시 완료

버그 수정 직후, 영향분 재생성과 학습용 전처리 캐시 생성까지 완료. (우선순위 0 + 우선순위 1 전처리 단계 소진)

**완료:**
- [x] ARCTIC object patch 캐시 **11종** 생성 (`data/offline_cache/arctic_*.json`, 06-14 02:15) — canonical rest pose, theta-independent
- [x] s01 articulation 영향분 재생성: arti>0 **490개** 재생성 (06-14 03:23 완료, 36.9분). arti≈0인 322개는 06-10/11 결과 유지 → **result JSON 합계 812개**
- [x] `graph_cache_arctic` **812개 전체 재생성** (canonical frame, bimanual 필드 포함)
- [x] `precompute_arctic_sample_graphs.py`에 경험적 검증된 좌표계 주석 반영 ("Verified frames 2026-06-14": obj_center=world articulated verts 평균, joints=world, contacts/anchor=object-centric world, obj_transform=identity → Rg는 raw npy에서)
- [x] 재투영 검증 산출물 `md_claude/arctic/regen10/` (13개, _photo_verify6 / _montage 포함)

**생성 파일 (06-10~11 세션 표에 누락분):**

| 파일 | 내용 |
|---|---|
| `scripts/precompute_arctic_object_patches.py` | ARCTIC 11종 → FPS/kNN/curvature → `arctic_{obj}_patches.json` (parts==0 movable patch 표시) |
| `scripts/precompute_arctic_sample_graphs.py` | result JSON + patch → canonical frame bimanual graph (`Rg.T` 정규화, articulation은 dataloader에 위임) |
| `scripts/module_a_contract.py` | (DexYCB 계열, 보류) GenHand 결과 → M-anchor 압축 standalone. 05-22 "별도 Module A 불필요" 결론 이후의 잔재 실험 |

**데이터 인벤토리 (현재):**

| 산출물 | 개수 | 경로 |
|---|---|---|
| result JSON | 812 | `data/results_arctic/s01/<seq>/Robotiq/mu0.9/` |
| graph_cache | 812 | `data/graph_cache_arctic/` |
| object patch | 11종 | `data/offline_cache/arctic_*.json` |

graph_cache 키: `file_name, obj_name, arti_angle, canonical_centroid, finger_ids, finger_side, finger_names, contact_positions, hand_joints, hand_joints_left, human_anchor, num_anchors`

**미완료 (학습 측 — 이게 현재 진짜 다음 작업):**
- `training/dataset.py` — ARCTIC graph_cache 로드 미지원 (DexYCB 전용, `arctic`/`hand_joints_left`/`finger_side` 참조 없음)
- `training/train.py` — ARCTIC 참조 없음 (DexYCB 학습 그대로)
- `checkpoints/best.pt` = **2026-06-03 DexYCB canonical 모델** → ARCTIC 재훈련 안 됨

---

### 2026-06-18 — ARCTIC 전처리 캐시 재구성 (design B) + sample_graphs 재생성

**핵심 결정 — design B (입력/GT 분리):**
- sample_graph는 **모델 입력만** 저장, **`human_anchor`(GT)는 미저장** → dataloader가 result JSON에서 직접 읽음 (DexYCB 방식과 동일)
- 효과: **FO cache가 contraction과 독립** → contraction 재생성해도 sample_graphs 재활용 가능
- 의존: `object_patches`(독립, 재활용 OK) / `sample_graphs = f(result JSON, object_patches)`

**캐시 일원화 → `data/arctic_cache/` (49MB):**
- `object_patches/` 11종 (rest pose, θ 무관, contraction 무관)
- `sample_graphs/` 812개 (θ articulated patch + 양손 contact + contact_normals + FO-edge + gating 거리)
- 구 위치(`offline_cache/arctic_*`, `graph_cache_arctic/`) 정리 완료

**구현 포인트:**
- patch articulation을 `precompute_arctic_sample_graphs.py` **단일 지점**에서 적용 (`parts==0` canonical −Z θ 회전, `_apply_articulation` 포팅 — CLAUDE.md §4)
- centering `art_pos − canonical_centroid`로 patch·라벨 동일 frame
- contact_normals/FO-edge는 DexYCB `build_finger_object_edges` 재사용 (단일 출처)
- FO-edge·obj-edge·hand_joints는 모델 forward 미사용 → 저장만 (V2 GNN 대비)

**검증 (단일 샘플, CLAUDE.md §4):** phone_grab_01_612 (arti 166°) — movable 9.6cm 회전 / **static 0.0000mm 고정**(06-14 버그 미재발), patch·contact frame 정합, 오른손 min_dist 2.7mm

**gating 신호 (캐시에 거리 저장):** 양손 <30mm 532개(66%) / 한 손 >50mm 236개(29%) → 학습 단계 필터 예정 (캐시 재생성 불필요)

---

### 2026-06-19 — ARCTIC 학습 파이프라인(1·2·3) 완성 + 생성 데이터 검증

**완료 — 학습 측 (우선순위 1의 진짜 다음 작업):**
- [x] `training/dataset.py` — `ArcticContractionDataset` (design B): 패치는 graph 내장, **anchor는 result JSON에서 `Rg.T` 변환**, bimanual 10-contact, `max_anchors=4` padding+mask
- [x] precompute에 `obj_rotation`(Rg) 추가 (contraction 독립 → design B 유지) → 812개 재생성
- [x] `training/train.py` — `--dataset arctic` 경로, `num_anchors=4`, `--gate-max-dist`
- [x] Contact gating 검증: `gate=0.03` → 812→**532개** (노트와 정확히 일치)
- [x] 스모크 3 epoch: val_loss 0.491→0.453, 루프 정상

**생성 데이터 검증 (4번 판정용):**
- [x] per-finger contact 양손 **정확** (fingertip 6.5~10mm), **왼손 topology 유효**(L≈R) → 4번 "왼손 검증" 항목 해소
- [x] **contraction(anchor) 의심** 🔴: 2-anchor 샘플에서 anchor가 양손 contact 평균이라 표면서 **최대 246mm(허공)**, 2-arm pseudo-label 부적절
- [x] 시각화 HTML 5개 생성(`md_claude/arctic/`) — 사용자 눈 검증 대기

**잠정 결론:** contact 입력 정확 → **4번 = contraction 알고리즘 수정**(전체 재생성 아님). design B로 sample_graphs 재활용, result JSON anchor만 갱신.

**학습 실행:** `python -m training.train --dataset arctic --robot Robotiq --epochs 500 --gate-max-dist 0.03`

---

### 2026-06-21 — Contraction 심층 조사 + per-hand 결정 (06-19 "허공 246mm" 진단 정정)

**핵심 정정 (실측):**
- 06-19의 "anchor 허공 246mm"는 **오진** — anchor는 표면 위(4~16mm), 246mm는 무시된 반대손 contact 거리. 진짜 문제는 **mid-air 아니라 한 손 collapse + count 불일치(2/4)**.
- 실측: **human_anchor 94% 양손+표면**(대체로 멀쩡) / **actual_contact 93% 한 손 collapse**(로봇 단일 그리퍼) → B 탈락.

**collapse 원인 (4-agent 워크플로 + 결정적 후속 체크):**
- force closure 탓 아님(FC loss에 kinematics 無 + 접점 벌리는 항), gripper span 한계도 아님(max 138mm, 물체크기와 +0.51).
- **seed x가 이미 95% 한 손** → collapse는 로봇 kinematics가 아니라 **contraction(combined HDBSCAN=contact_target) 그 자체**. 로봇은 따라갈 뿐.

**결정:**
- anchor 생성 = **per-hand**(손별 1개) — combined가 한 손 고르는 문제를 구조적 제거. label=human_anchor 계열.
- 표면 보장 위해 **per-hand centroid → 표면 projection** 예정.

**구현물:** `scripts/recontract_arctic.py`(오프라인 재contraction, design B), `training/dataset.py` `anchor_dir` hook, `scripts/visualize_dataloader_sample.py`, anchor 캐시 812개(`data/arctic_cache/anchors/`). 문서 2종(`md_claude/arctic/contraction_*.html`).

---

### 2026-06-23 — 분포형/생성 Contraction 방향 조사 (continuous_grasp.md)

**방향 전환 검토:** per-hand(결정론 단일)를 넘어 **분포형 contraction**(여러 가능한 grasp) + **초기 grasp 조건화** + **continuous 확장** 탐색. → `PLAN.md §4.4`·`RESEARCH.md §3`(T(R,O) diffusion)에 이미 있던 비전과 일치 확인.

**핵심 결정:**
- 멀티모달 출처 = **멀티-후보 라벨**(Module A가 샘플당 유효 grasp K개). 단일 라벨+full contact 조건이면 분포 안 생김.
- 범위 = **점진적**(per-hand 베이스라인 → 멀티-후보 라벨 → 생성넷 → continuous).
- **human을 prior로**: flow matching / diffusion bridge는 p₀를 genhand prior로 둘 수 있어 few-step·빠름 → genhand를 **생성 경로 출발점**으로 재활용. 초기-grasp 조건화 = 동일 메커니즘.
- 속도: anchor 6~12D + 조건 1회 인코딩 → diffusion/flow 둘 다 ms대("느리다"는 고차원 얘기, 비적용).

**산출물:** `md_claude/plans/continuous_grasp.md` — 4계열(Flow Matching / CVAE / Diffusion Bridge / 1-step 증류) 서베이 + arxiv 링크 + 비교표. 양손 직접 청사진 = **DAGDiff**.

**생성 계열은 미정** (flow matching from prior 유력). 다음: 특정 계열 딥다이브.

---

### 2026-06-24 — 누더기 제거 + per-hand genhand contraction 재구축 + genhand HDBSCAN 버그 수정 (Critical)

**Goal:** ARCTIC bimanual 라벨(`human_anchor`)을 누더기 경로에서 떼어내고, **각 손에 genhand contraction을 따로 돌린 결과**를 라벨로. anchor 개수 파라미터화. 학습까지.

**누더기 진단 (왜 이랬나):** 단일손 DexYCB → 양손 확장 과정에서 쌓인 패치.
- `human_anchor`(라벨)는 genhand를 **안 쓰고** 손가락 5개 zone **기하평균**에 별도 HDBSCAN(`_cluster_hand_anchors`, Optimize.py:999-1029) → **SDF 게이팅 없음**, 항상 4개 강제.
- combined HDBSCAN(`extra_contact_pack`, Optimize.py:989)은 collapse 유발, 로봇 seed에만 영향.
- genhand `hand_contact_cluster_hdbscan`은 이미 `SDF≤thr AND antipodal` 게이팅을 하는데, 라벨이 그 경로를 우회.

**🔴 숨은 Critical 버그 발견·수정 (`objects.py` p_collection):**
- `p_collection[labels] = np.array(p_collector)` — `p_collector`가 길이 다른 배열들의 ragged 리스트 → **modern NumPy(≥1.24)에서 ValueError**. `build_contact_target`의 try/except가 이를 삼키고 **fingertip fallback**으로 우회.
- 즉 **genhand HDBSCAN(`hand_contact_cluster_hdbscan`)이 이 환경에서 내내 크래시**하고 있었음. 모든 과거 결과의 로봇 seed = fingertip fallback, 라벨 = 기하평균(`_cluster_hand_anchors`)이었음 (진짜 genhand contraction이 한 번도 안 돌았음).
- 수정: 1D object array 명시 생성. → genhand HDBSCAN이 실제로 동작.

**수정 파일:**

| 파일 | 내용 |
|---|---|
| `optimisation/objects.py` | `hand_contact_cluster_hdbscan(contact_pack, n_target)` 파라미터화(per-hand + 개수). 빈-게이팅/HDBSCAN실패 시 external pack은 `contact_target=None`(잘못된 손 fallback 방지). `n_target<=1` 분기, 단일클러스터 KeyError 가드. **p_collection ragged 버그 수정.** `build_contact_target`도 동일 파라미터 + external fallback 게이트 |
| `optimisation/Optimize.py` | `_contract_single_hand`(hand_model swap 없이 pack 전달, contact_target 복원)·`_fallback_anchor_from_fingertips` 헬퍼. `__init__ anchors_per_hand`(기본=robot.contacts_num). `run_idx_json_bimanual`: 누더기 2개 + L1031-1036 seed 재할당 삭제(init seed 유지), 라벨=per-hand concat |
| `scripts/run_arctic.py` | `--anchors-per-hand` 배선(default None→robot.contacts_num) |

**좌표계:** per-hand contraction도 object-centric world frame(`initialization`이 obj_centre 센터링) → 다운스트림 `Rg.T` 그대로. frame 변화 없음.

**검증 (CLAUDE.md §4, 스모크 4샘플):** K=4(손당2), **양손 기여(collapse 없음)**, 표면 정합 **|sdf|<2cm**, bbox 내부, nearest_hand `R,R,L,L`(왼손 mirror 없음 → 법선 flip+센터링 정합). 버그 수정 전엔 둘 다 None→fingertip fallback(K=2 오른손)으로 위장돼 있었음.

**재생성·학습 (전부 완료):**
- result JSON **812 재생성** (35.3분, workers=4, 에러 0). K분포: **K=2 567개(70%, 단일손) / K=4 245개(30%, 양손)** — 프레임별로 닿은 손만 anchor(가변, padding+mask 흡수). 11 오브젝트.
- sample_graphs **812 재생성** (입력 일관성, 에러 0).
- 학습: `--dataset arctic --robot Robotiq --epochs 500 --gate-max-dist 0.03` → gate 후 **535 samples**(train 482/val 53), **best val_loss=0.1549** (l_anchor val 0.123→0.050 수렴).

**한계 (s01-only):** 482 train으로 overfitting(train 0.052 / val 0.16). → 같은 날 s02~s10 확장으로 해소.

**s02~s10 확장 (당일 후속):** result JSON **6539개 전체 생성**(4.01h, workers=4, 에러 3개=s08 box/notebook use 엣지). s03·s05는 p1 train split에 시퀀스 없음(0 프레임) → **8개 subject 6536 valid**. K=2 74%/K=4 26%. sample_graphs 6536 재생성(에러 3). **전체 재학습**: gate 후 **4454 samples**(train 4009/val 445 = s01-only 535의 8.3×) → **best val_loss=0.0848** (s01-only 0.1549 대비 **−45%**). train/val 격차 축소(l_anchor train 0.021 / val 0.032) → **overfitting 완화 확인**. 구 결과 백업: `data/results_arctic.bak_oldcruft`.

---

### 2026-06-25 — contraction 재설계: genhand HDBSCAN → robust joint k-means

- **결함 발견(시각화):** 양손 프레임 다수가 K=2로 collapse. 진단 = **genhand normal-HDBSCAN은 "균형 양면 핀치"에서만 군집 형성**, 단일면·불균형·wrapping은 0군집(또는 antipodal 소실). 실측: antipodal 실재 손의 **54% 소실**(절반은 HDBSCAN "성공"인데도).
- **교체:** `contract_kmeans` = 게이팅 후 **[정규화 위치, w·물체법선] k-means(k=n_target)** + centroid·snap. noise 안 버림 → 불균형 핀치 antipodal 복구(notebook dot **+1.0→−0.69**). 받침은 위치로 분할.
- **일반화:** `contract(P,N,k)` primitive + 정책(per_hand/combined, k) 분리 → 미래 구성(손당1/양손결합)은 설정만으로 대응.
- recontract(2.5분) → 새 라벨 K분포 `{2:1820, 4:4699}`. GT-only 시각화 2종.
- **force closure:** generation에서 Stage1/2 호출 제거(라벨과 무관, self.x=오른손2점 단일Robotiq 잔재였음). FC 머신러리·DexYCB HDBSCAN은 보존.
- 상세 비교: `md_claude/contraction_methods/genhand_vs_kmeans.md`. 일일로그: `md_claude/daily_progress/06_25.md`.
- **미완:** 새 라벨로 재학습(~36분) → GT-vs-PRED 비교 (대기).

---

### 2026-06-29 — joint-kmeans 재학습 + cross-subject held-out 평가 구축

- **재학습(joint-kmeans 라벨):** best val_loss **0.0761** (이전 0.0848/0.0866 대비 개선).
- **제대로 된 held-out 구축:** 우리 task엔 외부 GT 없음 → GT=우리 contraction. p1 split 실태: train=6539(8 subj), **val=878 전부 s05(train에 없는 사람)**, test=0(ARCTIC 비공개).
- **s05 별도 경로 생성:** `run_arctic.py --out-base` 추가 → `data/results_arctic_val/`(878, 1.1분) + `sample_graphs_val/`. train과 **물리적 분리(누수 없음)**.
- **평가 결과:** 내부 val(frame-split) **2.82cm** / **s05 cross-subject held-out 3.37cm** (+0.55cm) → **처음 보는 사람에 양호한 일반화**.
- eval/rich 스크립트에 `--results-dir/--graph-dir/--all` 추가. s05 rich 12패널(`eval_s05_rich.html`, 물체+양손 메쉬+접촉+GT/pred).
- 데이터 경로: train=`results_arctic`, held-out=`results_arctic_val`, 구버전=`.bak_*`. 상세: `md_claude/daily_progress/06_29.md`.

---

### 2026-07-01 — 게이팅 1.5cm 수정 + 인터랙티브 뷰어

- **게이팅 버그(사용자 지적):** `contact_threshold=5cm`이 느슨해 **안 닿은 손가락(엄지 3~4cm)까지** contraction에 포함. 실측: 닿은 손가락 SDF≈0, 안 닿은 2.7~4.3cm.
- **1.5cm로 수정 + 재생성(recontract) + 재학습:** best val_loss 0.0779. **평가 개선: 내부 val 2.82→2.57cm, s05 held-out 3.37→3.07cm** (닿은 손가락만 → task 깨끗해짐).
- **인터랙티브 뷰어:** `scripts/dataset_viewer.py` (dash) — object 필터 + 슬라이더 + prev/next + PRED 토글, `[사진|3D: 오브젝트+양손 메쉬+접촉점+GT/pred]`, **http://localhost:8050**. HTML 매번 생성 불필요.
- **유연화:** `run_arctic --out-base`, eval/rich `--results-dir/--graph-dir/--all/--no-pred`, recontract `--split/--contact-threshold`. **`commands.md`** 추가.
- 상세: `md_claude/daily_progress/07_01.md`.

---

### 2026-07-05~07 — 정적→동적 확장: DexMachina 통합 (점접촉 Multi-Arm RL)

**방향 전환 (브레인스토밍):**
- 정적 contraction → **동적 조작**으로 확장. **DexMachina**(functional retargeting RL, Genesis) 채택.
- 추상화 확정: **"로봇 암 1대 = 접촉점 1개 = 접촉력 1개"** (그리퍼 아님, 점접촉체). K=암 수 파라미터, 기본 4.
- 차별점: contraction이 **kinematic retargeting 대체**(비의인 multi-arm은 손가락 대응 부재로 그들 방법 시작 불가). 학습 기여 = schedule-guided RL 실행. real-time·(i)생성모델은 N+1로.
- 설계 문서: `md_claude/plans/{offline_keyframe_planning,dexmachina_integration}.md`

**구현 (P0~P3 완주, extlib/dexmachina point-contact 브랜치):**
- 환경: RTX 5080(sm_120) → Genesis fork 0122(0.3.13)+torch 2.7.0+cu128 (논문 0.2.1 스택 불가). 호환 패치 다수.
- `scripts/gen_point_urdf.py`(점접촉 로봇), `export_dexmachina_guidance.py`(contraction→guidance npy), `verify_dexmachina_guidance.py`(재투영 1.9mm), `visualize_dexmachina_guidance.py`.
- 학습 G1(guidance) vs G0(task-only): VOC 감쇠 G1 7단계(kp→38.3)/G0 4단계(→52.5).

**핵심 결과 — 병목 = guidance 품질(학습량 아님):**
- novoc 평가+비디오: **guidance 없이는 접촉조차 학습 못함**(G0 물체 이동 0.6cm) = 핵심주장 1차 지지.
- **kinematic replay(사용자 관찰 발단)**: 정책없이 기준궤적 재생 → **박스 전도**(rot max146°). 손바닥 면접촉의 지지다각형이 점접촉 수축에서 소실. replay=G1정책 지표 동일 → G1은 guidance 재생기로 수렴.
- 비디오: `md_claude/dexmachina/{rollout_g0full,rollout_g1full,replay_kinematic}_novoc.mp4`
- 상세: `md_claude/daily_progress/07_07.md`

---

### 2026-07-08~14 — 표면 구속 액션 ablation (surfc1·surfc2) + 학습지표 착시 규명

- **표면 구속 액션 도입**: 스피어당 (Δu,Δv 슬라이드, d 프레스) — 잔차 스케일 축소 대신 표면 매니폴드 제약 (사용자 결정)
- **surfc1 판정**: 무행동 함정 — d 대칭 매핑이 이탈 방향 선호, novoc 물체 이동 1.6cm
- **surfc2** (비대칭 d 매핑 split=−0.5 + con 3 + 마찰 DR 0.3~2.0): **무행동 탈출 성공**(프레스율 65%, 뚜껑 112° 개방) — 그러나 **밀쳐냄 실패**: 전 env가 프레임 25~40에 박스를 받침대에서 이탈시킴 (슬라이드 92% 포화)
- 🔴 **reset-farming 착시 확정**: 학습 wandb "잘 추적"(pos 4.4cm, ep_len 147)은 조기종료 리셋 사이클의 낙관 편향. train_like 재현 실측 ep_len median 46. → **판정은 novoc 연속 rollout으로만, 학습 지표 금지**
- **eval 인프라 결함 수정**: ① is_eval+B>1 마지막 env 정책 미제어(과거 B=4 평가 median 1/4 오염) ② eval 기본 마찰 DR off(μ=1.0). `--stochastic/--keep_rand/--train_like/--n_steps` 플래그 추가
- **다음 갈림길(사용자 결정 대기)**: ① 상태 기반 즉시 조기종료 ② 슬라이드 레이트 제한 ③ 1+2 후 (c)-4점 재학습. 접촉유지 보상·aux contact는 실패 원인(보상 exploit)과 처방 불일치로 보류
- 상세: `md_claude/plans/surface_action_ablation.md`, `md_claude/daily_progress/07_14.md`

---

### 2026-07-14~17 — DexScale 신설: RL 없는 TO+임피던스로 첫 무낙하 운반 (방향 전환)

- **방향 전환**: DexMachina RL 트랙(07-14 갈림길) 동결 → **DexScale** = contraction prior + 폐루프 CEM 궤적최적화 + 임피던스 제어 (No-RL). `/grill-me`로 설계 확정: goal task = ARCTIC 실궤적(box_use_01), 시뮬 = Genesis(임피던스/DR 게이트 6/6 통과), F_max=20N/팔
- **측정 진단**: ① 2× 재생 속도 발견(30fps↔60Hz, `--slow 2`=실속도) ② 실패 근본 원인 = 데모 고정 reference(접촉이탈이 힘 포화 없이 발생 — 힘 예산 기각) ③ M3에서 비용 결함(제곱 오차+상대 낙하판정)이 "포기 exploit" 유발 → 수정
- **하이브리드 reference** (사용자 제안): `x_ref = 실물 위 앵커 + (r−δ)·법선 + clip(데모−실물, d_max)` — 쥐기는 물체 기준, 운반은 포화 오프셋. 접촉 게이트(사전 파지 + 재파지 갭 홀드)
- 🎉 **E2: 파이프라인 최초 무낙하 운반** (raw 220프레임, DR 비용 −86%, onset 4접촉 유지)
- 🎉 **Robustness envelope 11/16칸**: 학습 밖 질량 ×2·×3 **zero-shot 일반화**(최종오차 1~2cm). 실패 경계는 가벼운 쪽(×0.5, 힘 대비 관성 부족) — 스케일업 서사와 정합
- **점진 지평선**: E4(전체 클립, warm start) 앞 구간 보존 + 자가 회복 실증, 실패 지점 = 첫 뚜껑 열기. E5(raw 360) 비용 1.36 미수렴 — **현 병목 = 뚜껑 자유도** (운반 힘이 본체 대신 뚜껑을 움직임)
- Genesis fork 지뢰 문서화: `batch_dofs_info`/`batch_links_info` 필수, `set_mass_shift` 오동작(→`set_links_inertial_mass`)
- 산출물: `dexscale/` 패키지 + 스크립트 6종, 설계 `md_claude/plans/dexscale_mvp.md`, 수식 `md_claude/dexscale/optimization_formulation.html`, 영상 8종
- 다음: E5 연장 → 뚜껑 제어 설계 → raw 520/889 확장 → ×0.5 커버 실험 → slow=1 킬샷 비교표

---

## Next Steps

**우선순위 A′ — DexScale (현재 메인 방향, 2026-07-17):**
- [ ] E5 연장 or 뚜껑 제어 설계(W_ARTI↑ / 뚜껑 δ 정밀화 / part 재배정)
- [ ] 지평선 사다리: raw 520 → 889(릴리즈 δ→0)
- [ ] 학습 DR ×0.4~3.0 확대 실험(가벼운 물체 커버 가능성 판정)
- [ ] slow=1 + 킬샷 비교표(replay/구TO/하이브리드), LP 사전판정, 세대 오버헤드 프로파일링

**우선순위 A — 동적 확장 (DexMachina RL, 동결 — DexScale의 비교 베이스라인으로 보존):**
- [ ] **조기종료 재설계(상태 기반 즉시 컷) ± 슬라이드 레이트 제한 → (c)-4점 재학습** (07-14 갈림길, 사용자 옵션 선택 대기)
- [ ] ablation grid 후속: #2 aux contact(5/6점) → #3 (b) 접촉유지만 → #4 (a) 순수 task
- [ ] articulation 저항 randomization (스케일업 단계에서, TODO)
- [ ] **support-aware contraction** (병목 확정, 07-07): anchor 선택에 힘평형/지지LP 점수(기존 gf_0·지지LP 재사용), 지지손 anchor 벌려 지지다각형 확보
- [ ] 법선-인지 클러스터링 (06-25 정식 `contract()` [pos, w·normal] 이식) + 스피어 반경 확대(점→패치)
- [ ] G2 schedule guidance, 물체·클립 확장, fleet(N) 분석

**우선순위 0 — articulation 버그 후처리 (✅ 완료, 2026-06-14):**
- [x] s01 전체 재생성 (arti>0 영향 490개) — result JSON 812개
- [x] graph_cache 재생성 (812개) — 재훈련은 우선순위 1으로 이월

**우선순위 1 — ARCTIC bimanual 학습 파이프라인 완성 (현재 메인 방향):**
- [x] `precompute_arctic_object_patches.py` — 11종 패치 캐시 → `data/arctic_cache/object_patches/` ✅
- [x] `precompute_arctic_sample_graphs.py` — design B 재작성, 812개 → `data/arctic_cache/sample_graphs/` ✅
- [x] `training/dataset.py` — ARCTIC 지원 (design B): sample_graphs 입력 + anchor는 result JSON에서 `Rg.T` 변환 ✅
- [x] `training/train.py` — `--dataset arctic`, `max_anchors=4` padding+mask ✅
- [x] Contact gating: `--gate-max-dist` (812→532 검증) ✅
- [x] left hand contact_zones 정확도 검증 — **유효 확인**(L≈R, fingertip 6.5~10mm) ✅
- [x] contraction 조사·진단 완료 → **per-hand 방식 결정** (06-21, 위 참고). `recontract_arctic.py` + anchor 캐시 812개 생성
- [x] **per-hand genhand contraction (in-pipeline)** — `recontract_arctic.py --rep surface_proj`(오프라인) 대신 `run_idx_json_bimanual`에서 손별 `build_contact_target` 직접 호출로 채택. genhand HDBSCAN 게이팅이 라벨에 그대로 반영됨 (06-24 세션). offline `recontract_arctic.py`는 보류(불필요).
- [x] 좌표 검증(스모크) + 812 재생성 + sample_graphs + 학습(best val_loss=0.1549) ✅
- [x] **s02~s10 확장** (`run_arctic.py --subjects all`, 6536 valid / 8 subject) + sample_graphs + 전체 재학습 → **val_loss 0.1549→0.0848 (−45%)**, overfitting 완화 ✅
- [ ] (선택) `anchors_per_hand`를 2 외 값으로 실험 시 `train.py num_anchors` 동기화

**(참고) DexYCB single-hand 잔여 — 보류:**
- [ ] subject-10 + 누락 subject 데이터 생성 (`run_partial.py --camera 836212060125`)
- [ ] `precompute_sample_graphs.py` 신규 결과 포함 재실행 / 재훈련

**우선순위 2 — Loss 고도화 (V2):**
- [ ] Assignment supervision (anchor → finger 매핑 loss)
- [ ] Intent preservation loss (OT 기반)
- [ ] Force closure loss (differentiable QP)

**우선순위 3 — 장기 (분포형/생성 contraction → continuous, 설계: `md_claude/plans/continuous_grasp.md`):**
- [ ] 생성 계열 딥다이브 → 확정 (flow matching from genhand prior 유력 / DAGDiff식 diffusion bridge)
- [ ] 멀티-후보 라벨 생성(`recontract_arctic.py`: 손별 서브클러스터 → 조합 FC proxy → top-K NMS) → 분포 검증
- [ ] 생성 네트워크(human prior 출발) → T(R,O) 스타일 + continuous `p(A_{0:h}|c)` 확장
- [ ] Object pose estimator 연동 (FoundationPose / ICG)
