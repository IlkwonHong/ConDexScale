**Scaling Up Dexterous Manipulation through Contact Graph Contraction & Diffusion**

## 1. Core Contributions (논문 핵심 주장 - The Pitch)
본 연구는 인간의 미시적 조작(Micro-manipulation)을 다중 로봇 암의 거시적 조작(Macro-manipulation)으로 변환하는 새로운 프레임워크를 제안한다.

1. **Micro-to-Macro Paradigm:** 사람의 5손가락 파지(Grasp) 의도를 분석하여, 스케일업 된 물체를 다루는 2~3대의 로봇 암(Multi-arm) 매니퓰레이션으로 변환.
2. **Contact Graph Contraction:** `GenHand`의 클러스터링 기법을 발전시켜, N개의 인간 접촉점을 M개의 로봇 암 접촉점으로 압축(Contraction)하는 알고리즘 제안.
3. **Real-time Multi-arm Grasp Synthesis:** `T(R,O) Grasp`의 Graph Diffusion 구조를 다중 로봇 암 환경으로 확장하여, 기존 최적화 기반 방법론의 느린 연산 속도를 극복하고 1초 이내에 물리적으로 안정적인(Force-closure) 파지 자세 생성.

---

## 2. System Architecture (파이프라인 설계)
전체 시스템은 크게 3단계 모듈로 구성된다.

### Module A: Contact Contraction (Offline/Prior)
* **Input:** 사람의 파지 데이터 (e.g., DexYCB 데이터셋) + 원본 물체 Mesh
* **Process:** HDBSCAN 기반 클러스터링을 통해 5개의 접촉점을 핵심 힘 성분(Force Components)으로 압축하여 M개(로봇 암 개수)의 Contact Anchor 생성.
* **Comment**: 파인튜닝용 데이터셋 필요 + hand tracking 

### Module B: Graph Diffusion Network (Core AI)
* **Input:** 스케일업(x5~x10) 된 물체의 Point Cloud (VQ-VAE 특징 추출) + Module A에서 얻은 Contact Anchor (Condition으로 사용)
* **Process:** T(R,O) Graph 구조를 차용하되, Robot Node를 '각 로봇 암의 End-effector'로 정의. 노이즈 상태에서 M개 로봇 암의 6D Pose를 Denoising.
* **Comment**: 손이랑 좀 다르게 가야하지 않을까 + 리파인을 RL 섞는것도 

### Module C: Differentiable Refinement (Physics Guarantee)
* **Process:** Diffusion이 예측한 초기 Pose를 바탕으로, 미분 가능한 Force-closure 최적화(e.g., GraspQP, PyroKi)를 수행하여 최종적으로 물리적 충돌(Penetration)이 없고 안정적인 파지 자세 도출.

---

## 3. Tech Stack & Environment
* **Simulator:** `NVIDIA Isaac Sim` (또는 Isaac Gym) - 대규모 병렬 데이터 생성 및 Multi-arm 제어 최적화.
* **Base Code:** 
  * `T(R,O) Grasp` 오픈소스 (Graph Diffusion 뼈대 활용)
  * `GenHand` 오픈소스 (HDBSCAN 클러스터링 및 Differentiable Force-closure 로직 활용)
* **Hardware (Demo):** UR5 또는 Franka Emika Panda 2~3대 + 평행 그리퍼.

---

## 4. 12-Week Fast-Track Timeline (주차별 실행 계획)

### Phase 1: Setup & Data Generation (Week 1-4)
* **Week 1:** 환경 세팅 (Isaac Sim, PyTorch) 및 `GenHand`, `T(R,O)` 베이스라인 코드 분석.
* **Week 2:** **Toy Problem 구현.** 2D 평면 또는 단순 3D 도형에서 5-point grasp을 3-point grasp으로 줄이는 Contraction 알고리즘 검증.
* **Week 3:** Isaac Sim 내에 Multi-arm (e.g., 3-Arm) 환경 구축 및 스케일업 된 물체 로드.
* **Week 4:** **대규모 데이터 생성.** DexYCB 데이터를 기반으로 Contraction -> Isaac Sim 내 최적화(GraspQP)를 거쳐 정답(GT) Multi-arm Grasp 데이터셋 10만 개 병렬 생성.

### Phase 2: Network Training & Integration (Week 5-8)
* **Week 5:** `T(R,O)`의 Graph Diffusion 모델 수정. Robot Node를 Multi-arm End-effector로 재정의.
* **Week 6:** 생성된 GT 데이터셋을 이용하여 Diffusion 모델 학습 시작. (Condition: Contracted Contact Anchors).
* **Week 7:** 학습된 모델의 Inference 테스트 및 하이퍼파라미터 튜닝.
* **Week 8:** Module C (Differentiable Refinement) 통합. Diffusion 출력값을 초기값으로 하여 물리적 안정성(Force-closure) 보장 모듈 연결.

### Phase 3: Evaluation & Real-world Demo (Week 9-12)
* **Week 9:** **시뮬레이션 평가 (Ablation Study).** 
  * 비교군: 순수 최적화 기반 방법(Baseline) vs 제안하는 Diffusion+Opt 방법.
  * 지표: Success Rate, Inference Time, Net Wrench Residual.
* **Week 10:** 실제 로봇 하드웨어(Multi-arm) 세팅 및 Sim-to-Real 테스트.
* **Week 11:** 데모 영상 촬영 (작은 물체를 다루는 사람 영상 -> 거대한 물체를 다루는 다중 로봇 암 영상).
* **Week 12:** 논문 작성 (Intro, Method, Exp) 및 Figure 제작. (CoRL, RSS, ICRA 타겟 제출).

---

## 5. Evaluation Strategy (리뷰어 설득 전략)
논문의 우수성을 증명하기 위해 다음 세 가지를 중점적으로 어필한다.

1. **Speed (효율성):** 기존 최적화 기반(GenHand 스타일)으로 Multi-arm Grasp을 계산하면 수십 초가 걸리지만, 우리의 Diffusion 모델은 1초 이내에 생성함을 증명.
2. **Stability (안정성):** 단순 Network Regression이 아니라, Refinement를 거치기 때문에 Net Wrench Residual이 0에 수렴함을 증명.
3. **Scalability (확장성):** 물체의 크기가 변하거나, 로봇 암의 개수가 2개에서 3개로 변해도 Graph 구조 덕분에 유연하게 대처 가능함을 증명.

---

## 6. Risk Management (플랜 B)
* **Risk:** Diffusion 모델이 Multi-arm의 6D Pose를 제대로 수렴시키지 못할 경우.
* **Plan B:** Diffusion 모델의 역할을 '6D Pose 전체 예측'에서 **'물체 표면 상의 Contact Region(Heatmap) 예측'**으로 축소. 이후 해당 Region을 기반으로 Inverse Kinematics(IK)와 QP Solver를 강하게 적용하여 해결. (학습 난이도를 대폭 낮추는 전략).


---
# Progress
## 4월 
### 3주차
- **PLAN**
	- 1. Contraction
		- genhand 기반 데이터 생성
		- pybullet(genhand) 에서 검증 후 isaacsim으로 진행
	- 2. Training(Graph diffusion?)
		- input : pcd, dexterous grasp info(taxonomy?, contact surface?,,,)
		- output : contracted grasp
	- 3. Refinement optimization
		- force closure 보장하는 fine-tuning?
- **DONE**
	- genhand baseline으로 dataset 생성 demo
		- pybullet
		- image &rarr; 3D recon or stl + MANO &rarr; Grasp contraction &rarr; Simulation 검증 40초 정도 소요
- **TODO**
	- Isaacsim으로 이동
		- Grasp contraction 멀티프로세싱
			- isaac sim에서 진행 가능한지 확인
		- DexYCB dataset 설치
		- GT 생성


Date: 2026-04-16
time: 20:59


