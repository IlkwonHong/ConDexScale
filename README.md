# ConDexScale

**Scaling Up Dexterous Manipulation through Contact Graph Contraction**

사람의 5손가락 파지(micro-manipulation)를 소수의 접촉 앵커로 **수축(contraction)** 하여, 스케일업된 물체를 다루는 2~4대 다중 로봇 암 조작(macro-manipulation)으로 변환하는 연구 코드베이스입니다.

## 핵심 아이디어 — Micro-to-Macro

사람 손의 파지는 손가락과 손바닥의 풍부한 접촉으로 이루어지지만, 물체가 사람 손 크기를 넘어서면 같은 파지 의도를 소수의 로봇 암 접촉으로 다시 표현해야 합니다. 이 프로젝트는 그 변환을 두 단계로 풉니다.

1. **수축 (Contraction):** 사람 손(양손 포함)의 접촉 그래프를 K개의 접촉 앵커로 압축합니다. 어느 접촉 부위 묶음에 팔을 배치할지(파티션 선택)와 각 앵커의 실제 위치를 네트워크로 예측합니다.
2. **실행 (Execution):** 수축된 앵커를 로봇 암 K대의 목표 접촉으로 삼아, 궤적 최적화와 임피던스 제어로 스케일업된 물체를 실제로 조작합니다.

## 연구 트랙

현재 활성 트랙 2개와 동결 트랙 1개로 진행 중입니다.

| 트랙 | 내용 | 상태 |
|---|---|---|
| 정적 수축 네트워크 | Dexonomy 기반 대규모 GT(시뮬 합격 판정 85.7만 건) 위에서 Stage 1(파티션 분류, top-3 후보 제안) + Stage 2(앵커 위치 생성) 계층 학습 | Stage 1 완료·동결, Stage 2 설계·구현 중 |
| 동적 조작 (DexScale) | 수축 앵커를 prior로 쓰는 CEM(Cross-Entropy Method) 궤적 최적화 + 임피던스 제어. ARCTIC 실궤적 추종으로 무낙하 운반 성공, 질량 ×2·×3 zero-shot 일반화 확인 | 진행 중 (현 병목: 관절 물체의 뚜껑 자유도 제어) |
| 점접촉 강화학습 (DexMachina) | "로봇 암 1대 = 접촉점 1개" 추상화의 강화학습. 수축 guidance의 필수성을 실증했으나 점접촉의 지지면 소실이 병목 | 동결 (비교 베이스라인으로 보존) |

## 진행 타임라인

| 기간 | 단계 | 결과 요약 |
|---|---|---|
| 2026-04 ~ 05 | GenHand 환경 구축 + DexYCB 데이터 생성 | 데모 재현, GT 파지 배치 생성 파이프라인 구축 |
| 2026-05 ~ 06 | 수축 네트워크 첫 학습 (DexYCB) | GraspContractionNet 구현, canonical frame 정립 |
| 2026-06 ~ 07 | ARCTIC 양손 데이터셋 확장 | 수축 알고리즘 재설계(joint k-means), held-out subject 오차 3.07cm |
| 2026-07 | 동적 확장 1차 — 점접촉 강화학습 | guidance 필수성 실증, 지지면 소실 병목 확인 후 동결 |
| 2026-07 | DexScale — 궤적 최적화 전환 | 최초 무낙하 운반, 질량 스케일 zero-shot 일반화 |
| 2026-07 ~ 현재 | 대규모 Contraction GT + Stage 1/2 | Dexonomy 10,690물체 GT 구축, Stage 1 동결(top-3 적중 95%+), Stage 2 진행 |

단계별 상세 요약과 코드 위치는 [`Summary.md`](Summary.md)에 정리되어 있습니다. 처음 코드베이스에 진입한다면 여기서 시작하세요.

## 저장소 구조

```text
.
├── contraction_gt/    # Dexonomy 기반 대규모 수축 GT 생성 (탐색 + 시뮬 합격 판정)
├── contraction_net/   # Stage 1/2 네트워크 (파티션 분류 + 앵커 생성)
├── training/          # DexYCB/ARCTIC 수축 네트워크 (GraspContractionNet)
├── dexscale/          # CEM 궤적 최적화 + 임피던스 제어 (동적 조작)
├── optimisation/      # 파지 최적화 + 데이터 생성 (DexYCB/ARCTIC 로더, GenHand 기반)
├── scripts/           # 배치 러너·전처리·시각화 스크립트
├── extlib/            # 외부 fork (DexMachina, Genesis, rl_games, Dexonomy)
├── network/           # 손·물체 포즈 추정 및 메시 복원 (GenHand 원본)
├── simulation/        # PyBullet 파지 검증 (GenHand 원본)
├── demo/              # GenHand 데모 스크립트
├── data/              # 데이터 캐시 (result JSON, 그래프 캐시, GT 레코드)
├── checkpoints/       # 학습 체크포인트
├── md_claude/         # 연구 노트·계획·일일 기록
└── thirdparty/        # pytorch_kinematics, pytorch_volumetric (수정본)
```

## 사용 데이터셋

- **[DexYCB](https://dex-ycb.github.io/)** — 한 손 파지, 물체 21종. 초기 수축 네트워크 학습에 사용.
- **[ARCTIC](https://arctic.is.tue.mpg.de/)** — 양손 + 1자유도 관절 물체 조작, 물체 11종. 양손 수축과 동적 조작 궤적에 사용.
- **[Dexonomy](https://pku-epic.github.io/Dexonomy/)** — Shadow 손 파지 대규모 데이터셋, 물체 10,690종. Stage 1/2 학습용 대규모 GT의 원천.

## 문서 안내

| 문서 | 내용 |
|---|---|
| [`Summary.md`](Summary.md) | 단계별 상세 요약 + 코드 위치 (신규 진입 시작점) |
| [`PROGRESS.md`](PROGRESS.md) | 세션 단위 진행 로그 |
| [`INSTALL.md`](INSTALL.md) | 환경 설치 안내 |
| `md_claude/` | 계획·분석·실험 문서 (Stage 1 구조도, DexScale 수식 등) |

## 기반 코드

이 저장소는 [GenHand: generalised human grasp kinematic retargeting](https://www.nature.com/articles/s44182-026-00076-1) (npj Robotics 2026)의 공식 코드베이스에서 출발했습니다. `network/`, `simulation/`, `demo/`, `optimisation/`의 파지 최적화 골격이 GenHand에서 왔으며, 이후 배치화·양손 확장·수축 알고리즘 등이 대폭 수정·확장되었습니다.

```text
@article{qi2026genhand,
  title={GenHand: generalised human grasp kinematic retargeting},
  author={Qi, Liyuan and Popoola, Olaoluwa and Imran, Muhammad Ali and Ahmad, Wasim},
  journal={npj Robotics},
  volume={4},
  number={1},
  pages={19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

GenHand이 기반한 [AlignSDF](https://github.com/zerchen/AlignSDF), [DFC](https://github.com/tengyu-liu/diverse-and-stable-grasp), [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp), [manopth](https://github.com/hassony2/manopth), [Pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics), [Pytorch_volumetric](https://github.com/UM-ARM-Lab/pytorch_volumetric)에도 감사를 표합니다.
