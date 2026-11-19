# Cross-morphology grasp/manipulation transfer and physical verification of human-derived grasps — prior-art notes (as of 2026-11-08)

작성 목적: "contact graph contraction"(사람 5손가락 파지·조작 → 2~3대 다중 로봇 암의 접촉 앵커, 렌치 공간 품질(힘 폐쇄 스케일 α, ℓ∞ 강건성 여유 ρ)로 채점, MuJoCo 검증) 연구의 참신성 주장을 정확히 하기 위해, (A) 형태가 크게 다른 몸체 사이의 파지·조작 전이와 (B) 사람 유래 파지의 물리적 성공 검증·예측에 관한 선행 연구를 정리한다. 2024–2026년 연구를 중심으로 하되 고전 파지 해석 문헌은 배경으로 인용한다.

표기 규칙(모든 항목에 적용):
- **[원문 확인]** — arXiv HTML/PDF 본문이나 공식 페이지를 직접 열어 확인한 내용.
- **[검색 요약]** — 검색 결과 요약문(초록·2차 요약 사이트 포함)에서만 확인한 내용. 수치는 원문 재확인 권장.
- **[추론]** — 위 자료에서 내가 끌어낸 해석. 출처가 직접 말한 것이 아님.
- **[기억, 식별자 미확인]** — 널리 알려진 문헌이지만 이번 세션에서 식별자(arXiv 번호 등)를 검색으로 재확인하지 못한 것.

약어는 처음 나올 때 풀어 쓴다. 예: 역기구학(inverse kinematics, IK), 강화학습(reinforcement learning, RL), 모션 캡처(motion capture, MoCap), 시각-언어 모델(vision-language model, VLM), 힘 폐쇄(force closure), 파지 렌치 공간(grasp wrench space, GWS), Ferrari–Canny의 ε 품질(ε = 렌치 볼록 껍질에 내접하는 최대 구의 반지름).

---

## A1. 사람 한 손의 파지/조작을 여러 대의 로봇 암(로봇 팀)으로 옮긴 연구가 있는가?

### Takeaway
"사람 한 손의 손가락을 각각 별개의 로봇 암(또는 로봇)으로 사상하고, 접촉·렌치 공간 추론으로 물리적 성공을 보장하는" 연구는 2026-11-08 기준으로 찾지 못했다. 가장 가까운 선행 연구는 (i) 사람 손의 손끝 운동으로 쿼드로터 편대를 "날아다니는 손"처럼 원격조작한 The Flying Hand (ICRA 2014) 계열의 가상 물체(virtual object) 매개 원격조작과, (ii) 사람 시연 없이 시각-언어 모델로 대형 물체의 다중 로봇 파지점을 고르는 CollaBot (2025)이다. 둘 다 "사람 파지 데이터로부터 다중 암 접촉 앵커를 학습·예측하고 렌치 여유로 최적화·물리 검증"하는 구성은 아니다.

### Cited Findings

**직접 선행 연구 부재(검색 결과)**
- 다음 검색어로 찾았으나 "사람 한 손 → 여러 로봇 암" 직접 선행 연구는 나오지 않았다: "human hand demonstration to multiple robot arms grasp mapping fingers to arms"(확장 검색), "multiple robots as fingers of a hand cooperative grasping large object multi-arm grasp synthesis", "dual-arm cooperative grasp derived from single human hand demonstration thumb and fingers mapped to two arms", "learning multi-robot collaborative transport from human demonstration video", "grasping large object with multiple robot arms inspired by human multi-finger grasp contact points distributed across arms". 각 검색은 사람 손→단일 다지 손 리타게팅(예: [Multi-Finger Grasping Like Humans](https://arxiv.org/pdf/2211.07304), [CEDex](https://arxiv.org/pdf/2509.24661)), 단일 팔 데이터를 두 팔로 확장하는 [One-to-Two Acting (ExS2D)](https://arxiv.org/pdf/2606.19897), 한 번의 양팔 시연으로 배우는 [One-Shot Dual-Arm Imitation Learning (ODIL)](https://arxiv.org/pdf/2503.06831), 두 사람의 톱질 영상에서 리더–팔로워 역할과 임피던스를 추출해 두 대의 Panda에 재현한 Iodice et al.([Polimi 저장소](https://re.public.polimi.it/retrieve/b5e6f0bf-a283-4bdd-a251-7c660c9f07f1/11311-1228206_Jodice.pdf)) 정도만 돌려주었다. **[검색 요약]**
- 두 사람 시연→두 로봇 암은 특허 수준에서 존재한다: [US12179350B2 "Dual arm robot teaching from dual hand human demonstration"](https://patents.google.com/patent/US12179350B2). 이는 "양손 → 양팔"이지 "한 손의 손가락들 → 여러 팔"이 아니다. **[검색 요약]**

**가장 가까운 선행 연구 1: 가상 물체 매개 원격조작(Prattichizzo 그룹)**
- *The Flying Hand: a Formation of UAVs for Cooperative Aerial Tele-Manipulation* — G. Gioioso, A. Franchi, G. Salvietti, S. Scheggi, D. Prattichizzo; ICRA 2014, pp. 4335–4341, DOI 10.1109/ICRA.2014.6907490. 초록: "The flying hand is a robotic hand consisting of a swarm of UAVs able to grasp an object where each UAV contributes to the grasping task with a single contact point at the tooltip. The swarm of robots is teleoperated by a human hand whose fingertip motions are tracked, e.g., using an RGB-D camera." 구조는 (1) 손끝 운동→원하는 물체 운동으로 해석하는 hand interpreter, (2) 물체 운동→계획된 접촉점에서 벗어나는 가상 접촉점으로 바꾸는 mapping algorithm, (3) 쿼드로터를 간접 3D 힘 작용기로 쓰는 순응 힘 제어의 3층. 시각 피드백으로 내부력(internal force)을 암시. 검증은 "human-in-the-loop simulations including the full physical model of the object, contact points and UAVs"로 **시뮬레이션만**. [HAL](https://hal.archives-ouvertes.fr/hal-01990572), [UTwente](https://research.utwente.nl/en/publications/the-flying-hand-a-formation-of-uavs-for-cooperative-aerial-tele-m/), [dblp](https://dblp.dagstuhl.de/rec/conf/icra/GioiosoFSSP14.html). 선행 워크숍판 "Hand Driven UAV Formation for Cooperative Grasping and Transportation: the Flying Hand"(2013) [LAAS 페이지](https://homepages.laas.fr/afranchi/robotics/?q=node/167). **[검색 요약: 초록 문장 수준]**
  - 프로젝트와의 차이 **[추론]**: 접촉점은 사전에 계획된 것이고 사람 손은 물체 운동 명령 소스로만 쓰인다. 사람의 파지 데이터에서 접촉 앵커를 예측하거나 렌치 여유 ρ로 접촉 위치를 최적화하지 않고, 물리 시뮬레이터로 성공률을 통계적으로 검증하지도 않는다. 하지만 "손가락 하나 = 로봇 하나 = 접촉점 하나"라는 틀은 명확히 선점되어 있으므로, 참신성 주장은 "개념" 자체가 아니라 "접촉 앵커 예측 + 렌치 여유 최적화 + 물리 검증 + 동적 궤적"에 두어야 한다.
- *Object-based bilateral telemanipulation between dissimilar kinematic structures* — Salvietti, Meli, Gioioso, Malvezzi, Prattichizzo, IROS 2013 ([PDF](https://sirslab.dii.unisi.it/papers/2013/Salvietti.IROS.2013.Med.Fin.pdf), [Siena 저장소](https://usiena-air.unisi.it/handle/11365/46429)): 엄지·검지 두 햅틱 장치로 DLR-HIT Hand II를 조작, 마스터·슬레이브 가상 물체에 같은 렌치가 작용하도록 힘 피드백 정의. 2017년 IEEE/ASME Transactions on Mechatronics 다중 접촉판은 "결과를 여러 로봇이 같은 물체를 원격조작하는 협동 파지 시나리오로 확장할 수 있다"고 언급 ([Siena 저장소](https://usiena-air.unisi.it/handle/11365/1009820)). **[검색 요약]**
- *An Object-Based Approach to Map Human Hand Synergies onto Robotic Hands with Dissimilar Kinematics* — Gioioso et al., RSS 2012 ([PDF](https://www.roboticsproceedings.org/rss08/p13.pdf)). 제목만 검색 결과로 확인. 사람 손 시너지를 가상 물체(구) 변형을 통해 운동학이 다른 로봇 손으로 사상하는 "물체 중심" 매핑의 원조로, 손가락 수에 의존하지 않는 매핑의 고전. **[검색 결과 제목만 확인; 내용은 기억]**

**가장 가까운 선행 연구 2: 대형 물체의 다중 (이동)매니퓰레이터 파지 — 사람 시연 없음**
- *CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation* — Kun Song 외 7인, arXiv [2508.03526](https://arxiv.org/abs/2508.03526v1) (2025-08-05). **[원문 확인]**
  - 몸체: 전방향 이동 베이스 + JAKA Zu7 팔 + Robotiq 2F-85 2지 그리퍼, 로봇 수 임의("arbitrary number of robots distributed randomly").
  - 표현/방법: SEEM 분할 → 물체 점군 → 국소 파지 생성(LoGNet, PointNet 기반, 2048점 국소 점군에서 평행·접근 벡터·면내 회전 회귀, 학습 데이터 ShapeNet/DA2의 실내 대형 물체 752종, 88,976 국소 점군) → 전역 협조: 충돌 없는 표면점을 K-Means로 K개 후보로 묶고 번호를 그려 이미지로 만든 뒤 **시각-언어 모델이 파지점을 고른다**. 명시된 선택 기준: "the object's center of mass is as close as possible to the interior of the convex hull" of the grasp points; "grasp poses are as dispersed as possible to prevent potential robot collisions during motion"; "the line connecting its fingers is parallel to the direction of external force"; 국소 안정성(형상·마찰).
  - 렌치 추론: 파지 행렬 기반 완화 힘 폐쇄 지표, 최소 특이값, 중력 하 마찰원뿔 제약 접촉력 최적화를 **평가에만** 사용. 선택 루프 자체는 렌치 계산을 하지 않는다.
  - 사람 시연: 없음. "teleoperation data for multi-robot systems is hard to collect"가 동기이며 사람 파지 행동은 2단계 설계의 "영감"에 그친다.
  - 수치: 전체 52% (13/25). 과제별 TABLE 3/5, CHAIR 3/5, Flip TABLE 1/5, Align 4/5, 3로봇 TABLE-3 2/5. 실패 원인 약 50%가 협동 파지 생성(국소 파지 충돌, VLM이 고른 파지의 로봇 간 충돌), 약 33.3%가 계획, 약 16.7%가 SEEM 검출 실패.
  - 한계(저자 진술): 물체 궤적 계획이 로봇 충돌을 무시; 최소 로봇 수를 구하지 않음; 단일 이미지에서 파지점 선택; "local grasp quality metrics alone are judged insufficient for collaborative grasping"; 과제당 5회의 소규모 시험.
- *Collaborative Multi-Robot Non-Prehensile Manipulation via Flow-Matching Co-Generation* — Yorai Shaoul, Zhe Chen, M. N. Gul Mohamed, F. Pecora, M. Likhachev, J. Li, arXiv [2511.10874](https://arxiv.org/abs/2511.10874v1) (v1 2025-11-14, v2 2026-02-17). 시각 관측으로부터 "contact formations and manipulation trajectories"를 공동 생성하는 비파지(밀기) 다중 로봇 조작. 사람 시연 언급 없음, 로봇 수·수치는 초록에 없음. **[원문 초록 확인]**
- *ConPoSe* arXiv [2510.08705](https://arxiv.org/abs/2510.08705?context=cs) (2025-10): LLM + 국소 탐색으로 협동 밀기의 로봇–물체 접촉점 선택(직육면체·원기둥·T형). 파지가 아니라 밀기. **[검색 요약]**
- *Cooperative Optimal Multi-Robot Formation Arrangement for Object Pushing* ([ICR 2026 게재 목록](https://ia.spcras.ru/index.php/icr2026/article/view/17354)): 입자 군집 최적화로 물체 경계의 접촉점을 고르되 기준을 파지 행렬·Gram 행렬 해석(넓은 렌치 범위, 고른 밀기 방향)에서 가져옴. 밀기 과제. **[검색 요약]**
- *Distributed Motion Control of Multiple Mobile Manipulators for Reducing Interaction Wrench in Object Manipulation* — Wenhang Liu, Meng Ren, Kun Song, Gaoming Chen, Michael Yu Wang, Zhenhua Xiong, arXiv [2406.05613](https://arxiv.org/abs/2406.05613) (v4 2025-04-07). 관절 속도 제어만으로 통신 지연을 고려해 불필요한 상호작용 렌치를 줄이는 분산 운동 제어(Lyapunov 안정성 증명). 접촉점 계획 여부는 초록에 없음. **[원문 초록 확인]**
- *Decentralized Adaptive Control for Collaborative Manipulation of Rigid Bodies* — P. Culbertson, J.-J. Slotine, M. Schwager, IEEE T-RO 37(6):1906–1920, 2021, arXiv [2005.03153](https://arxiv.org/pdf/2005.03153). 물체 질량·마찰·**자기 부착점(attachment point)을 모르는 상태**에서 각 로봇이 적응 제어로 SE(3) 궤적 추종; 3D 시뮬레이션과 평면 하드웨어. 접촉점은 주어진 것이며 계획 대상이 아니다. **[검색 요약]**
- 고전 케이징: *Decentralized Algorithms for Multirobot Manipulation via Caging* — Pereira, Kumar, Campos (2003) ([UPenn 저장소](https://repository.upenn.edu/meam_papers/36/)): 형상 폐쇄/힘 폐쇄보다 약한 "object closure"(물체가 로봇들에 갇힘) 조건을 분산 제어로 유지하며 운반. Wang & Kumar 2002(평면), Wang·Hirata·Kosuge 2005(변형 가능한 케이징 편대) ([CityU](https://scholars.cityu.edu.hk/en/publications/deformable-caging-formation-control-for-cooperative-object-transp/)). **[검색 요약]**
- 다중 손 협동 파지 계획: UPC의 MUMA 프로젝트(Raúl Suárez)는 "각 손이 단독으로는 파지를 이룰 수 없는 접촉을 제공하는 협동 파지(collaborative grasps)"를 다룬다 ([프로젝트 페이지](https://iocnet.upc.edu/usuaris/raulsuarez/proyectos/muma/objetivo-MUMA-eng.html)). 사람 시연 기반은 아님. **[검색 요약]**

### Inferences
- **[추론]** "손가락 = 로봇" 틀은 The Flying Hand(2014)가 선점했으나, 그 연구는 (a) 접촉점을 사전 계획, (b) 사람 손을 운동 명령원으로만 사용, (c) 시뮬레이션 원격조작만 검증했다. 따라서 프로젝트의 차별점은 "사람 파지 데이터셋(Dexonomy)으로 학습한 앵커 히트맵 → 렌치 여유 ρ 최적화 → 물리 시뮬레이션·도메인 무작위화 검증 → ARCTIC 동적 궤적으로 확장"이라는 폐루프 전체이며, 이를 명시적으로 대비시켜 써야 한다.
- **[추론]** CollaBot은 "대형 물체 다중 로봇 파지점 선택"이라는 문제 설정은 같지만 선택기가 시각-언어 모델이고 렌치 지표는 사후 평가용이다. 실패의 절반이 파지점 선택에서 나온다는 CollaBot의 자체 분석은, 렌치 여유를 선택 단계에 넣는 프로젝트 접근의 동기로 인용할 수 있다.
- **[추론]** 다중 로봇 협동 조작 제어 문헌(Culbertson 2021, Liu 2024)은 "접촉점이 이미 주어졌을 때의 힘/렌치 분배"를 다루므로, 프로젝트의 QP 힘 할당기와 역할이 겹친다. 프로젝트의 기여는 접촉점 자체를 사람 데이터에서 유도한다는 점에 두는 것이 안전하다.

### Gaps
- "finger-to-arm mapping", "hand-to-arms", "macro manipulation from micro"라는 어구 자체로는 어떤 논문도 검색되지 않았다. 해당 용어는 프로젝트가 새로 정의하는 것으로 보아도 될 듯하나, 비영어권 학회(예: 일본·중국 국내 학회)나 특허까지는 확인하지 못했다.
- The Flying Hand의 후속 하드웨어 실험 여부(2014 이후), Salvietti 2017 TMECH의 다중 로봇 확장 실험 유무는 원문을 열지 못해 확인 못 함.
- CollaBot 외에 2024–2026년 "다중 매니퓰레이터 대형 물체 파지점 최적화 + 렌치 분배"를 한 번에 다루는 논문은 검색에서 나오지 않았다(검색어: "multi-robot cooperative manipulation large object contact point optimization wrench distribution 2025").

---

## A1b–A2. 접촉 기반 표현으로 사람 손 → 다른 손/그리퍼(접촉 수·형태 변화 포함)로 옮긴 연구

### Takeaway
접촉 맵·접촉점을 물체 표면에 두는 "물체 중심" 표현으로 손 형태 차이를 넘는 계보는 ContactDB/ContactGrasp(2019) → GenDexGrasp(2023) → RobotFingerPrint(2024) → CEDex(2025)로 이어지며, 손가락 수 변화는 (i) 접촉 맵을 공유하고 각 손이 자기 운동학으로 맞추기(GenDexGrasp, ContactGrasp), (ii) 사람 손 부위를 규칙으로 병합(CEDex: 약지+새끼손가락), (iii) 그리퍼당 N개 접촉점 직접 선택(UniGrasp)의 세 방식으로 처리된다. 다만 **2지 그리퍼로의 전이는 대부분 제외되거나(CEDex: "two-finger grasps are inherently unable to achieve stable multi-directional force application") 별도 검출기로 처리되고, "5접촉 → 2~3접촉으로 줄일 때 어떤 접촉을 남길지"를 힘 폐쇄/렌치 여유 기준으로 고르는 연구는 찾지 못했다.**

### Cited Findings

**접촉 표현의 원천 데이터**
- *ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging* — S. Brahmbhatt, C. Ham, C. C. Kemp, J. Hays; CVPR 2019 Oral; arXiv [1904.06830](https://arxiv.org/pdf/1904.06830). 50개 가정용 물체, 접촉 맵이 입혀진 3,750개 3D 메시, 375K 프레임 RGB-D+열화상. 사람 접촉 맵의 표준 출처. **[검색 요약]** ContactPose(ECCV 2020)는 ContactDB 페이지에 "접촉+손 자세+RGB-D"를 담은 후속 데이터셋으로 언급됨 ([ContactDB 페이지](https://contactdb.cc.gatech.edu/)). **[검색 요약; arXiv 식별자 미확인]**

**접촉 맵 → 다른 손 합성(접촉 수를 손마다 다르게 소화)**
- *ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact* — Brahmbhatt, Handa, Hays, Fox; IROS 2019; arXiv [1904.03754](https://arxiv.org/pdf/1904.03754); [프로젝트](https://contactdb.cc.gatech.edu/contactgrasp.html). 물체 표면에 N개 점을 균일 샘플링하고 접촉 값 +1(끌림)/−1(밀어냄)을 두는 접촉 맵. 사람 접촉 시연으로 **사람 손 모델, Allegro(4지), Barrett(3지)** 세 손에 'use'/'handoff' 의도의 기능적 파지를 합성. GraspIt! 샘플링 + DART 최적화의 sample-and-rank. 저자들은 사람 손 파지를 Allegro로 옮기는 운동학 리타게팅을 구현해 비교했고, "deterministic mapping of grasps between hand models does not reliably reproduce contact"라는 결론을 냈다(접촉 맵 불일치 지표가 ContactGrasp에서 유의하게 낮음). 한계: 같은 접촉 패턴을 내는 손 구성이 여럿이라 순위화된 집합을 출력. **[검색 요약]**
- *Contact Transfer: A Direct, User-Driven Method for Human to Robot Transfer of Grasps and Manipulations* — A. Lakshmipathy, D. Bauer, C. Bauer, N. S. Pollard; arXiv [2110.15532](https://arxiv.org/abs/2110.15532) (v2 2022-06-02). 접촉 **면적(shape)**을 완전히 보존하며 사람 손→다른 손으로 옮김; 파지 군(family)·학습·샘플링 불필요, 조작기 형태·운동학 가정 없음; 수 초 안에 운동학적으로 실현 가능한 전손 자세 합성; 맞춤 의수에서 시연. 물리 검증·정량 지표는 초록에 없음. **[원문 초록 확인]**
- *Multi-Finger Grasping Like Humans* — Yuming Du, Philippe Weinzaepfel, Vincent Lepetit, Romain Brégier (LIGM/École des Ponts, NAVER LABS Europe); arXiv [2211.07304](https://arxiv.org/pdf/2211.07304) (2022-11-14; 검색 결과는 IROS 2022로 표기). 입력은 물체 메시 + MANO 손 자세. 목적함수는 "similar contact surfaces and global orientation for the human and the robotic grasps, while penalizing interpenetrations"의 4개 손실, 3단계 최적화(전역 자세 초기화 → 손끝을 접촉 영역으로 닫기 → 전체 목적). **Allegro(4지, n=16)와 BarrettHand(3지, n=7)**에 "gripper-specific tuning 없이" 적용. 저자들은 기존 리타게팅이 "handcrafted correspondences between the human hand and the robotic gripper ... and they do not consider the object"라고 비판. 평가: YCB-affordance 사람 파지, ε 품질·관통 부피·접촉 히트맵 차이, 사용자 연구에서 Allegro 51%/Barrett 73%가 본 방법 선호(무작위 기대 25%/33%); 비교 대상 DexPilot·GraspIt!. Panda+Allegro 실로봇 시연. **[원문 PDF 텍스트 확인]**
  - 접촉 수 변화 처리 **[추론]**: 명시적 손가락 대응 없이 "접촉 면적 유사성"으로 처리하므로 손가락 수가 줄면 자동으로 일부 사람 접촉이 버려진다. 어떤 접촉이 버려지는지에 대한 힘 폐쇄 기준은 없다.
- *GenDexGrasp: Generalizable Dexterous Grasping* — Puhao Li, Tengyu Liu, Yuyang Li, Yiran Geng, Yixin Zhu, Yaodong Yang, Siyuan Huang; ICRA 2023; arXiv [2210.00722](https://arxiv.org/pdf/2210.00722); [코드](https://www.github.com/tengyu-liu/GenDexGrasp). 손에 무관한(hand-agnostic) 접촉 맵을 CVAE로 생성한 뒤 각 손이 자기 운동학으로 맞춤. **EZGripper(2지), Barrett(3지), ShadowHand(5지)** 등 여러 손. 성공 판정: "applying an external acceleration to the object and measuring the movement of the object" (Isaac Gym). 정렬 거리 접촉 맵 ablation: EZGripper 38.59% vs 유클리드 29.53%. 2차 집계 사이트의 미검증 수치: Allegro 51%, Barrett 67%, ShadowHand 54.2%. **[검색 요약]**
- *UniGrasp: Learning a Unified Model to Grasp with Multifingered Robotic Hands* — Lin Shao 외; IEEE RA-L 5(2):2286–2293, 2020; arXiv [1910.10900](https://arxiv.org/pdf/1910.10900); [프로젝트](https://sites.google.com/view/unigrasp). "If the gripper has N fingers, our model UniGrasp selects a set of N contact points from the object point cloud" — Point Set Selection Network(PSSN)가 **힘 폐쇄이면서 손이 닿을 수 있는** 접촉점 집합을 출력. 학습 데이터는 2지·3지 손만. 실로봇: 알려진 2지·3지 그리퍼 >90%, 미학습 2지 그리퍼 93%, 미학습 다지 의인화 손 83%·90%. **[검색 요약]** 사람 시연은 쓰지 않는다 **[추론: 초록상 언급 없음]**.
- *RobotFingerPrint: Unified Gripper Coordinate Space for Multi-Gripper Grasp Synthesis and Transfer* — N. Khargonkar, L. F. Casas, B. Prabhakaran, Y. Xiang; arXiv [2409.14519](https://arxiv.org/pdf/2409.14519); [프로젝트](https://irvlutd.github.io/RobotFingerPrint). 모든 그리퍼 손바닥 표면의 점에 구면 경도·위도 (λ, φ)를 부여하는 통일 좌표계("finger print"). CVAE가 물체 표면점마다 이 좌표를 예측 → 그리퍼–물체 대응 → 자세·관절 최적화. 전이는 원·목표 그리퍼의 좌표 맵으로 표면점 대응을 초기화. **사람 손 → Fetch 2지 그리퍼**: YCB 물체에서 성공 11/14 (0.785), "most failures came from noisy human hand pose inference". 미학습 그리퍼에도 적용 가능. 한계: 충돌 인지·부분 관측·실시간 폐루프는 향후 과제. **[검색 요약]**
- *CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations* — arXiv [2509.24661](https://arxiv.org/html/2509.24661v1) (2025-09). **[원문 확인]**
  - 표현: 사람 파지 데이터로 사전학습한 CVAE가 접촉 맵 + 부위 맵(MANO 16부위) 생성(ContactOpt 방식, 방향 맵은 미사용).
  - 손가락 수 변화 처리: "predefined kinematic human-robot mapping"으로 사람 부위를 **쌍으로 병합** — 두 부위의 접촉을 두 중심점 사이의 한 점으로 투영한 뒤 합집합, 3개 이상은 반복. 본문 예시: "We adaptively consolidate multiple human hand parts, such as the ring and pinky fingers, into fewer robotic components." 그리퍼별 상세 매핑은 Fig. 3에만 있어 본문으로는 특정 불가.
  - 몸체: Barrett Hand, Robotiq-3F, Allegro, Shadow (3~5지). **2지 그리퍼 제외**, 이유: "two-finger grasps are inherently unable to achieve stable multi-directional force application."
  - 규모: 실물 58종(ContactDB·YCB) + Objaverse 503,409종; 최종 약 20M 파지. 실물 부분 후보 237,568(손당 59,392), Objaverse 부분 후보 32M.
  - 검증: Isaac Gym, 단순 파지 제어기, 6개 직교 방향 힘을 각 1초, "A grasp is considered successful if the object's displacement remains below 2 cm." 수치 관통 임계는 미기재. 통과율 미기재(32M→20M이면 약 62.5%라는 계산은 **[추론]**).
  - 수치(미학습 물체): Barrett 93.1%, Allegro 88.1%, Shadow 85.0%, 평균 88.7%; Robotiq-3F 91.9%. Ablation: 운동학 정렬 없이 27.7%(있으면 89.3%), 물리 제약 없이 30.9%.
  - 비판 문장: 물리 기반 방법은 "focus solely on static equilibrium and neglect human grasping kinematics ... leads to low success rates in practical scenarios"; 시연 기반 방법은 관절 보정·재매핑이 필요해 의인화 손에 국한.
- *D(R,O) Grasp* arXiv [2410.01702](https://arxiv.org/html/2410.01702v4) (2024-10, 2025-03 개정): 로봇–물체 상호작용의 통일 표현(거리 행렬)을 점군에서 예측, LEAP hand 실험. **[검색 요약]**
- *MachaGrasp* arXiv [2510.06068](https://arxiv.org/pdf/2510.06068v3) (2025-10): URDF에서 eigengrasp와 형태 임베딩을 뽑아 새 손에 few-shot 적응. **[검색 요약]**
- 어포던스 기반 arXiv [2601.08246](https://arxiv.org/pdf/2601.08246v1) (2026-01): 시각-언어 사전지식으로 손가락별 어포던스 장을 만들어 하드웨어별 파지 데이터 없이 손 간 전이. **[검색 요약]**
- *Cross-Embodiment Dexterous Grasping with Reinforcement Learning (CrossDex)* — Haoqi Yuan, Bohan Zhou, Yuhui Fu, Zongqing Lu; ICLR 2025; arXiv [2410.02479](https://arxiv.org/pdf/2410.02479); [코드](https://github.com/PKU-RL/CrossDex). 사람 손 기반 **eigengrasp 공간**을 공통 행동 공간으로 쓰고 손마다 리타게팅; 관측은 손끝·손바닥 위치로 통일. 4개 손 YCB 80%, 미학습 2개 손에 zero-shot. 한계: 손 4개, 파지만. 접촉 표현은 아니고 행동 공간 공유 방식. **[검색 요약]**
- *AnyDexGrasp* — Hao-Shu Fang 외; arXiv [2502.16420](https://arxiv.org/abs/2502.16420v1) (2025-02). 장면 기하 → 손에 무관한 **접촉 중심 파지 표현** → 손마다 실세계 시행착오로 학습한 "grasp decision model"(손당 약 2,000~8,000회, 축소 설정 400~1,000회). 실세계 75–95%(또는 150개 미학습 물체에서 80–98%; 두 수치의 설정 대응은 원문 확인 필요). **[검색 요약]**
- *GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training* — NVIDIA; arXiv [2507.13097](https://arxiv.org/pdf/2507.13097) (2025-07); [코드](https://github.com/NVlabs/GraspGen). 생성기가 만든 파지로 판별기를 학습("on-generator")하여 생성기의 전형적 오류를 낮게 채점. 3종 그리퍼(Franka-Panda, Robotiq-2F-140, 흡착), 53M+(README 57M+) 파지. FetchBench 최고 성능, 실로봇. 다른 그리퍼는 GraspGen-X로 안내. 사람 데이터 미사용 **[추론]**. **[검색 요약]**
- *DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness* — Zhong, Jiang, Yu, Ma; CVPR 2025 (pp. 22584–22594); arXiv [2503.08257](https://arxiv.org/abs/2503.08257v2). 확산 모델의 학습·샘플링 양쪽에 물리 제약(관통·접촉 부족) 주입, 3.4M 파지/15k 물체. 단일 손 계열, 교차 형태 아님. **[검색 요약]**
- *Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy* — Jiayi Chen, Yubin Ke, Lin Peng, He Wang; RSS 2025; arXiv [2504.18829](https://arxiv.org/html/2504.18829v2); [프로젝트](https://pku-epic.github.io/Dexonomy). **[원문 확인]** 31개 GRASP 분류 유형, 10.7k 물체, 9.5M 파지; "requires only one template per grasp type"(Shadow는 유형당 수작업 템플릿 1개, Allegro 비교엔 2개). **의인화 손 전용**(Shadow 데이터셋, Allegro 비교; Inspire·Unitree G1 통합은 감사의 글에만). 비의인화 그리퍼·가변 손가락 수는 다루지 않음. 저자 한계: 정적 자세만("focuses on synthesizing static grasp poses rather than generating sequential trajectories"), 단일 물체, 모션 플래닝 성공률 <30%, 실제 Shadow의 저작동 차이.
- *DexGraspNet 3.0 / DexVLG* — Galbot·BAAI 외; ICCV 2025; arXiv [2507.02747](https://arxiv.org/abs/2507.02747v1) (2025-07-03). Objaverse 174k 물체에 170M 파지, 각 파지 물리 시뮬레이션 검사, 부위 캡션; VLM+flow-matching 자세 헤드, 시뮬레이션 zero-shot 76%+. 단일 손, 교차 형태 아님. **[검색 요약]**
- 배경(단일 ShadowHand, 교차 형태 아님): *UniDexGrasp++* arXiv [2304.00464](https://arxiv.org/pdf/2304.00464) (ICCV 2023; 학습 85.4%/시험 78.2%), *UniGraspTransformer* arXiv [2412.02699](https://arxiv.org/pdf/2412.02699) (CVPR 2025, Microsoft Research Asia; 물체별 RL 정책을 Transformer로 증류). **[검색 요약]**
- *FunGrasp: Functional Grasping for Diverse Dexterous Hands* — Linyi Huang, Hui Zhang, Zijian Wu, Sammy Christen, Jie Song; arXiv [2411.16755](https://arxiv.org/pdf/2411.16755) (2024-11). 사람 기능적 파지 RGBD 한 장 → 손 자세 추정 → 물체 좌표계에서 링크 방향 정렬로 초기화 → "optimized to preserve precise contact positions and human-like postures" → RL(시뮬레이션) + privileged learning/시스템 식별/도메인 무작위화/중력 보상으로 실로봇. 여러 손에 배치. 2차 요약의 "75%+ 시뮬레이션 성공"은 미확인. **[검색 요약]**
- 사람 영상 → 평행 그리퍼: *Hand-centric Human-to-Robot Trajectory Transfer from Video Demonstrations via Open-World Contact Localization (HOWTransfer)* arXiv [2606.10743](https://arxiv.org/pdf/2606.10743) (2026-06-09). 접촉 시작 시점을 찾아 "retarget human grasp intent into multi-modal parallel-jaw grasp hypotheses", 손목 궤적을 따라 전파, 86% 성공, 원격조작 궤적보다 선호. 접촉 선택 기준은 초록에 없음. **[검색 요약]** *MAPLE* arXiv [2504.06084](https://arxiv.org/html/2504.06084v2): 한 장의 이미지에서 접촉 순간의 손–물체 접촉점과 3D 손 자세를 예측(라벨은 자기중심 영상 VISOR-HOS). **[검색 요약]**
- 신경과학 배경 — 가상 손가락(virtual finger)과 대립 공간(opposition space): Iberall, Bingham, Arbib (1986, Experimental Brain Research), Iberall & MacKenzie "Opposition Space and Human Prehension"(1990). "Most grasps can be defined by just two VFs exerting opposing forces"; 대립 공간은 "the area within the coordinates of the hand where opposing forces can be exerted between virtual finger surfaces in effecting a stable grasp." 삼각 파지(tripod)는 엄지와, 두 손가락 중간에 놓인 가상 손가락 사이에 대립 공간이 형성됨 ([J. Neurophysiol. 2001](https://www.sciencegate.app/document/10.1152/jn.2001.86.2.604)). ([CMU 노트](https://www.cs.cmu.edu/afs/cs/usr/sbk/www/thesis/virtual-finger.html), [Springer](https://www.springerprofessional.de/opposition-space-and-human-prehension/13787620)) **[검색 요약]**

### Inferences
- **[추론]** 접촉 수를 줄이는 기존 방식은 세 가지로 요약된다: (1) 접촉 맵 공유 후 목표 손이 "닿을 수 있는 만큼만" 닿기(ContactGrasp, GenDexGrasp, Du et al.) — 어떤 접촉이 살아남는지는 최적화 부산물; (2) 사람 부위를 사전 규칙으로 병합(CEDex의 약지+새끼손가락) — 힘 폐쇄와 무관한 운동학 규칙; (3) N지 그리퍼에 N개 접촉점을 힘 폐쇄+도달성 조건으로 직접 선택(UniGrasp) — 그러나 사람 시연을 쓰지 않음. "사람 5접촉 중 렌치 여유를 최대화하는 2~3접촉 부분집합(또는 클러스터)을 고른다"는 명시적 정식화는 이번 검색에서 발견되지 않았고, 이것이 프로젝트의 'contraction'이 차지할 수 있는 빈자리다. 가상 손가락 이론(두 개의 대립하는 가상 손가락)은 그 생물학적 근거로 인용 가능하다.
- **[추론]** CEDex가 2지 그리퍼를 "다방향 안정 힘 작용 불가"로 배제한 사실은, 프로젝트가 암 2대(2접촉)일 때 왜 ℓ∞ 여유 ρ가 작아지는지(또는 3대가 필요해지는지)를 설명하는 외부 근거가 된다. 역으로 RobotFingerPrint·HOWTransfer가 사람 손→2지 전이를 성공시켰다는 점은 "2접촉 전이가 불가능하다"는 주장을 막으므로, 프로젝트는 "접촉 수에 따른 ρ의 변화"를 정량적으로 보여주는 쪽이 설득력 있다.

### Gaps
- GenDexGrasp·AnyDexGrasp·FunGrasp의 손별 성공률과 세부 설정은 2차 요약에 의존했다. 논문에 수치를 쓰려면 원문 표 재확인 필요.
- CEDex의 그리퍼별 병합 규칙(Fig. 3)은 본문 텍스트로 특정되지 않아, "어느 사람 손가락을 어느 로봇 손가락에 합쳤는지"는 그림을 직접 봐야 한다.
- ContactPose의 arXiv 식별자와 ECCV 2020 서지는 이번 세션에서 재확인하지 못했다.

---

## A2b. 텔레오퍼레이션 리타게팅(DexPilot, AnyTeleop/dex-retargeting, ACE, GeoRT)과 접촉 보존 실패

### Takeaway
리타게팅 라이브러리 계열은 속도·운동학 일관성을 최적화하며 접촉은 목표가 아니다. 2026년의 접촉 고정 리타게팅 논문(DexGEM Lab)은 같은 100개 MANO 궤적에서 DexPilot의 **접촉 유지율(손끝이 물체에 5 mm 이내로 닿은 접촉 프레임 비율)이 2.0%**, 관절 한계 위반 21.4%, dex-retargeting Vector 방식은 유지율 32.0%이지만 위반 29.9%임을 보고했다. 즉 운동학 리타게팅은 접촉을 체계적으로 잃는다.

### Cited Findings
- *DexPilot* (Handa et al., ICRA 2020; arXiv 1910.03135 **[기억, 식별자 미확인]**), *AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System* (Yuzhe Qin 외, UCSD/NVIDIA, RSS 2023; arXiv [2307.04577](https://arxiv.org/abs/2307.04577v2)), *ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation* (Shiqi Yang 외, arXiv [2408.11805](https://arxiv.org/html/2408.11805v1), 2024-08; AnyTeleop와 달리 말단 운동 전달에 집중). dex-retargeting 라이브러리는 AnyTeleop에서 파생되었고 DexPilot형 최적화기를 포함 ([PyPI](https://pypi.org/project/dex-retargeting)). **[검색 요약]**
- *Geometric Retargeting (GeoRT)* — arXiv [2503.07541](https://arxiv.org/pdf/2503.07541) (2025-03-10; Dexterity Gen 시스템의 일부). 주장: "the robot and human hands keypoint spaces are not directly related through a linear mapping as suggested by previous works"; 비지도 1 kHz 신경 리타게팅. 접촉 보존에 대한 정량 주장은 검색 결과에 없음. **[검색 요약]**
- *DexFlow: A Unified Approach for Dexterous Hand Pose Retargeting and Interaction* — arXiv [2505.01083](https://arxiv.org/html/2505.01083v1) (2025-05). **[원문 확인]** 비판 문장: 전통적 리타게팅은 "suffer from penetration artifacts and unstable contact patterns"; "AnyTeleop and DexPilot often prioritize speed over spatial precision, leading to misalignments in delicate tasks." 방법: 이중 임계 접촉 검출 + 관통·자기관통·법선 정렬·관절 정규화 가중 에너지, 엄지부터 손가락별 순차 최적화. 수치(Table II, ShadowHand): 관통 지표 PD — DexFlow 8.5, **DexRetarget 84.4**, DexGraspNet 13.5, SpringGrasp 16.2, FRoGGeR 2.17, BODex 0.37; 시뮬레이션 성공률 SSR — DexFlow 40.32%, **DexRetarget 5.35%**, FRoGGeR 41.97%, BODex 89.55%, SpringGrasp 37.24%, DexGraspNet 31.37%. 한계(저자): 재구성 메타데이터 품질에 민감, "grasp success is not guaranteed in all scenarios", 접촉 정보는 영상에서 직접 뽑는 편이 더 신뢰할 수 있을 것.
- *Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning* — Zihao Yang, Chengyuan Liu, Yu Zhou, Runze Lv, Tianyu Cui, Sheng Yi, Haohua Zhu, Irvine Lu, JieQ Sun (DexGEM Lab, 상하이교통대, 동제대, DexRobot); arXiv [2609.24093](https://arxiv.org/pdf/2609.24093) v2 2026-11-03; [코드](https://github.com/DexGEM-Lab/real2sim2real). **[원문 PDF 텍스트 확인]**
  - 핵심 주장: "what survives the change from a human hand to a robot hand is the contact structure of a demonstration—which finger regions touch which object locations, and in what order—rather than its joint motion." 그리고 "the contact forces that determine whether a grasp succeeds are absent from every scalable source of human demonstrations."
  - 운동학 리타게팅 비판: "mapping joint angles or fingertip positions, treats hands with different kinematics as scaled copies of one another and fails precisely when morphology differs"; "Video alone, however, recovers neither precise hand–object pose nor contact force"; 모션 캡처도 "the contact forces that distinguish stable grasps from infeasible poses (Brahmbhatt et al., 2019) are unrecorded".
  - 3단계: (1) 시뮬레이션 MANO 손으로 물리 정제(단일 잔차 RL 정책이 모든 시연을 가로질러 운동학 기록을 "repair"), (2) 접촉 고정 리타게팅(지정된 손가락 표면 영역이 기록된 물체 접촉 위치에 닿도록, 목적함수는 손 운동학이 아니라 표면 기하에만 의존), (3) 로봇 잔차 정책.
  - 수치: 단일 손 25,454 궤적 성공 **7.3% → 59.3%**, 양손 25과제 **16.0% → 62.4%**; 한 사람 데이터셋을 4개 형태의 손으로 전이(+62.4 pp); 교차 형태 재구성(손당 200 에피소드) 재생 34.7% → 잔차 46.5%; 실로봇 양손 4과제 50% 이상.
  - Table 5(100 MANO 궤적 → DexHand 021): DexPilot — 충실도 15.8 mm, 유지율(≤5 mm) **2.0%**, (≤10 mm) 35.1%, 관절 한계 위반 **21.4%**, 지터 2.94 rad/s; dex-retargeting Position — 15.7 mm, 0.3%, 26.9%, 10.9%, 1.63; dex-retargeting Vector — **10.6 mm**, 32.0%, 53.5%, **29.9%**, 3.46; Spider(기본) — 49.9 mm, 0.2%, 1.4%, 3.27%, 0.73; 제안 방법 — 14.7 mm, 13.9%, 44.6%, 3.2%, 0.53 (단, 시간 8513 s로 가장 느림).
- CEDex **[원문 확인]**: 시연 기반 방법은 "joint calibration and remapping"이 필요해 의인화 손에 국한된다고 서술.
- Du et al. 2022 **[원문 확인]**: 기존 리타게팅은 "handcrafted correspondences ... in terms of joints, surfaces, or key vectors, and they do not consider the object".
- *RealDexUMI* arXiv [2606.06033](https://arxiv.org/html/2606.06033): 수집과 실행에 같은 말단 모듈을 써서 "retargeting을 거치며 배치 가능한 손재주를 잃는 것"을 피하겠다고 명시. **[검색 요약]**
- 2차 자료(비공식 서베이 페이지): dex-retarget의 한계를 "naive mapping loses physical feasibility"로 요약 ([survey-robot-hand-tactile-sensor](https://survey-robot-hand-tactile-sensor.pages.dev/en/ch15)). **[검색 요약, 출처 신뢰도 낮음]**

### Inferences
- **[추론]** 프로젝트가 주장하려는 "주류 리타게팅(손 추적→IK)은 물리적 성공 보장이 없다"는 명제는 이미 DexFlow(2025), ManipTrans(2025), Human2Sim2Robot(2025), 접촉 고정 리타게팅(2026)이 수치로 뒷받침했다. 따라서 이 명제 자체는 새롭지 않다. 새롭게 말할 수 있는 것은 "리타게팅 후에도 남는 밀리미터·수 도 수준의 실현 오차가 성공을 가르며, 렌치 공간 유사도는 그렇지 않다"는 프로젝트의 실험적 발견이다(아래 B2 참조).

### Gaps
- DexPilot 원문(접촉 관련 자체 진술)은 열지 않았다. AnyTeleop·ACE·GeoRT도 초록 수준만 확인.
- Spider(Pan et al., 2025) 리타게팅의 서지는 미확인.

---

## A3. 일반 교차 형태 로봇 학습(배경, 포인터만)

### Takeaway
대규모 교차 형태 정책(Open X-Embodiment, Octo, CrossFormer, HPT)과 시각 편집 기반 전이(Mirage, Shadow, Phantom, Masquerade), 손 착용형 수집(UMI, DexUMI), 자기중심 영상 학습(EgoMimic, EgoDex)은 모두 **말단 자세/관절 행동 공간을 공유하거나 영상을 편집**해 몸체 차이를 넘으며, 접촉·렌치 수준의 물리적 성공 조건은 명시적으로 다루지 않는다.

### Cited Findings
- *Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting* — L. Y. Chen, K. Hari, K. Dharmarajan, C. Xu, Q. Vuong, K. Goldberg (UC Berkeley·Google DeepMind); RSS 2024; arXiv [2402.19249](https://arxiv.org/abs/2402.19249). 목표 로봇을 가리고 원 로봇을 그려 넣음 + 순방향 동역학 모델로 행동 변환; 9과제·6 로봇/그리퍼 설정. URDF·좌표계·카메라 파라미터 필요. **[검색 요약]**
- *Shadow: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer* — M. Lepert, R. Doshi, J. Bohg; CoRL 2024 (PMLR 270); arXiv [2503.00774](https://arxiv.org/pdf/2503.00774). 원·목표 로봇의 합성 분할 마스크로 대체, "keeps the gripper-object interaction and the surrounding scene intact", 실로봇에서 최강 기준선(Mirage 재구현) 대비 2배 이상. **[검색 요약]**
- *Phantom: Training Robots Without Robots Using Only Human Videos* — Stanford; CoRL 2025; arXiv [2503.00779](https://arxiv.org/pdf/2503.00779). 손 자세 추정 → 로봇 행동 변환 → 손 인페인팅 후 가상 로봇 합성; 같은 사람 영상으로 Franka·UR5·Kinova 학습. *Masquerade* arXiv [2508.09976](https://arxiv.org/html/2508.09976v1): 야외 자기중심 영상(EPIC-KITCHENS 등)으로 확장, 양팔 로봇 렌더링, 성능이 편집 영상량에 로그 비례. *H2R-Grounder* arXiv [2512.09406](https://arxiv.org/html/2512.09406v1): 사람 상호작용 영상→"물리적으로 접지된" 로봇 영상 변환(짝 데이터 불필요). 검색 요약은 RoviAug·Masquerade가 여전히 로봇 데이터 공동학습에 의존한다고 서술. **[검색 요약]**
- *DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation* — Mengda Xu 외(Shuran Song 그룹); CoRL 2025 (PMLR v305); arXiv [2505.21864](https://arxiv.org/html/2505.21864v3). 목표 로봇 손의 관절–손끝 사상에 맞게 링크 길이를 최적화한 착용형 외골격 + 사람 손·외골격을 지우고 로봇 손을 인페인팅. Inspire(저작동)·XHand, 평균 86%. **[검색 요약]** UMI(2024) **[기억, 식별자 미확인: arXiv 2402.10329]**.
- *EgoMimic: Scaling Imitation Learning via Egocentric Video* — Kareer et al.; arXiv [2410.24221](https://arxiv.org/pdf/2410.24221) (2024-10). Project Aria 안경, 3D 손 추적, 사람·로봇 데이터 공동학습, 로봇 데이터만 대비 34–228% 향상. *EgoDex* — Hoque et al. (Apple); arXiv [2505.11709](https://arxiv.org/abs/2505.11709) (2025-05). Vision Pro, 829시간·90M 프레임·338k 시연·194과제, 손 궤적 예측 벤치마크. **[검색 요약]**
- 대규모 정책: Open X-Embodiment(2023, arXiv 2310.08864), Octo(RSS 2024, arXiv 2405.12213), CrossFormer(CoRL 2024, arXiv 2408.11812), HPT(NeurIPS 2024, arXiv 2409.20537) **[기억, 식별자 미확인]**. 행동 공간 처리 분류는 서베이 *Data Pyramid for Embodied Manipulation* arXiv [2607.24744](https://arxiv.org/pdf/2607.24744)가 "embodiment-specific projection, fixed-dimensional zero-padded interfaces, semantic action slots"로 정리; *Toward Unified Robot Learning* arXiv [2609.03927](https://arxiv.org/pdf/2609.03927)도 교차 형태 절 보유. **[검색 요약]**
- Human2Sim2Robot·SimToolReal은 "물체 중심 보상/궤적"이라는 몸체 무관 표현을 쓴다(B1 참조).

### Inferences
- **[추론]** 이 계열에서 "몸체 무관 표현"은 (i) 말단 자세(UMI·Phantom), (ii) 영상 편집(Mirage·Shadow·Masquerade), (iii) 물체 궤적(Human2Sim2Robot·SimToolReal)의 세 종류다. 접촉점·렌치 공간을 표현으로 삼는 것은 파지 합성 분야(A2)에만 있고, 다중 암 조작 정책 학습에는 아직 쓰이지 않았다.

### Gaps
- 2025–2026년 교차 형태 학습 전용 서베이(arXiv)는 검색에서 확정하지 못했다(검색어: "survey cross-embodiment robot learning 2025 arXiv embodiment-agnostic action space").

---

## B1. 사람 유래 파지·조작을 물리 시뮬레이션으로 검증/여과한 연구 — 실패율, 원인, 지표

### Takeaway
공통 프로토콜은 "관통 임계 → 6방향(±x,±y,±z) 외력 또는 중력 하 들어올리기 → 변위 임계"이며(DexGraspNet 2.0: 2 mm 관통, 0.2 마찰, 60스텝, 6방향; CEDex: 6방향 1초, 2 cm; Dexonomy: MuJoCo 6방향 2초, 관통 >1 cm 실패; Get-a-Grip: 손목 ±5 mm·±2.5° 섭동 5회 재시뮬레이션), **합성 파지의 통과율은 낮고(DexGraspNet 2.0: 16.07%), 사람 운동학 기록을 그대로 재생하면 성공률이 한 자릿수(접촉 고정 리타게팅: MANO 재생 7.3%; ManipTrans Retarget-Only 4.6%/0.0%; DexFlow DexRetarget 5.35%)**라는 점이 반복 확인된다. 실패 원인으로는 관통, 접촉 상실, 손 자세 추정 잡음, 물체 모델 부정확, 마찰 모델링 차이가 지목된다.

### Cited Findings

**데이터셋 여과 프로토콜과 통과율**
- *DexGraspNet 2.0* — arXiv [2410.23004](https://arxiv.org/pdf/2410.23004). **[검색 요약, 원문 문장 인용 포함]** "We perform grasp filtering in the IsaacGym simulator. First, we check for each grasp pose if the penetration between the hand mesh and object mesh is below 2 mm. For all collision-free grasps, we execute the grasp with a predefined heuristic and simulate for 60 timesteps at 60Hz. The grasp pose is validated as stable if it can deny gravity in all 6 axis-aligned directions." 마찰 0.2(저자 표현 "very strict"). 유효율 **약 16.07%**(280M 초기화 중 44.9M), 방법별 7.91%·24.19%. 실로봇 90.7%(초록). 2차 요약의 시뮬레이션 85.8%는 미검증.
- *DexGraspNet* (v1) — arXiv [2210.02697](https://arxiv.org/pdf/2210.02697): 100 시뮬레이션 스텝, 관통 0.1 cm 한계, 두 검사 통과만 보존. **[검색 요약]**
- *Dexonomy* **[원문 확인]**: MuJoCo; 접촉 인지 제어("zero-external-wrench QP"로 접촉력 결정 → 전치 Jacobian 토크); 국소 정제 시 1 mm 접촉 여유, 2 mm 이내를 접촉으로 간주; "the object remains stable under all six orthogonal external forces for 2 seconds"; 관통 >1 cm 실패; 물체 질량 100 g; 자세 성공 임계 5 cm·15°. 수치: Allegro 유형 무관 GSR 60.50%(BODex 49.23%, DexGraspNet 12.10%), 어려운 벤치마크 GSR 27.18%(DexGraspNet 물체)·18.34%(Objaverse); 유형별 정상 조건 GSR power 24.2%, intermediate 23.0%, precision 36.0%. 실로봇 82.3%(13 물체·12 유형·약 200회). 실패 원인: 물체 형상에 맞지 않는 유형, Tip Pinch(#24)는 "not very robust", 모션 플래닝. "random initialization"은 치명적으로 실패(고차원 템플릿 공간에서 유효 파지가 희소) → 사람 템플릿 필수.
- *CEDex* **[원문 확인]**: Isaac Gym 6방향 1초 2 cm. 통과율 미기재.
- *Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer* — T. G. W. Lum, A. H. Li, P. Culbertson, K. Srinivasan, A. D. Ames, M. Schwager, J. Bohg (Stanford·Caltech); arXiv [2410.23701](https://arxiv.org/html/2410.23701v1) (2024-10-31). **[원문 확인]** 3.5M 파지·4.3K 물체·Allegro+LEAP. 라벨: 충돌 y_coll, 들어올리기 y_pick(링크 3개 이상 접촉, 손바닥·근위 링크 비접촉, 20 cm 들어올리는 동안 10 cm·오일러각 45° 이내), 성공확률 y_PGS = 둘의 결합; **각 파지를 손목 ±5 mm·±2.5° 섭동으로 5회 재시뮬레이션해 [0,1] 소프트 라벨**. 212 미학습 물체 중앙 성공률 37%(충돌 없는 파지 조건부 80%). 실로봇(Allegro+Franka, 20 물체×5): 평가기 기반 76–81%, 평가기 없음 43%(확산 샘플러)·51%(FRoGGeR); 어려운 물체 55–65% vs 20%. 해석적 지표와 성공의 상관은 **보고하지 않음**.
- *DexVLG / DexGraspNet 3.0*: 170M 파지 각각 물리 시뮬레이션 검사(세부 미확인). **[검색 요약]**
- PAIR Lab 서베이 *A Survey on Dexterous Robot Grasping* ([페이지](https://www.pair.toronto.edu/DexRobotGraspSurvey.github.io), 2025, Advanced Robotics Research 심사 중, GraspEval 저장소). **[원문 페이지 확인]** 성공 기준 "max_height_achieved − z_table ≥ 0.25 m"; 메시 타당성 검사(관통 부피 ≤ 1×10⁻⁵ m³, 접촉 3개 이상); Allegro/LEAP로 재평가 시 GraspXL 부분집합 51%(200 YCB, 슬립 8.2%), MultiDex v1.0 48%, DexFuncGrasp 44%. 이는 논문 자체 수치(GraspXL 82.2%)와 큰 차이 — 기준·손이 다르므로 직접 비교 불가.
- *GraspXL* — ETH·MPI; arXiv [2403.19649](https://arxiv.org/abs/2403.19649): RL+물리 시뮬레이션, 58 물체로 학습해 500k+ 미학습 물체 82.2%. *Grasp'D* — Turpin 외(Toronto·NVIDIA·Samsung·Vector); ECCV 2022; arXiv 2208.12250 ([ECVA](https://ecva.net/papers/eccv_2022/papers_ECCV/html/5242_ECCV_2022_paper.php)): 미분 가능 접촉 시뮬레이션으로 사람 손·로봇 손 모델에 4배 이상 조밀한 접촉. **[검색 요약]**

**사람 기록을 시뮬레이션에서 재생·정제한 연구(실패율의 직접 증거)**
- 접촉 고정 리타게팅 (arXiv 2609.24093) **[원문 확인]**: 25,454개 단일 손 MANO 궤적을 시뮬레이션 MANO 손으로 재생하면 성공 **7.3%**, 잔차 RL 물리 정제 후 59.3%; 양손 16.0% → 62.4%.
- *ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning* — Li 외(BIGAI); CVPR 2025; arXiv [2503.21860](https://arxiv.org/html/2503.21860v1). **[원문 확인]** Isaac Gym 4096 환경; 성공 조건 손목 회전 <30°, 병진 <3 cm, 평균 관절 오차 <8 cm, 평균 손끝 오차 <6 cm(양손은 둘 다). Table 1 성공률(단일/양손): **Retarget-Only 4.6 / 0.0**, RL-Only 34.3/12.1, Retarget+Residual 47.8/13.9, ManipTrans 58.1/39.5. 문장: "Retargeting methods often struggle with transferring MoCap data to physically plausible motions"; Retarget-Only는 "hardly works"; 리타게팅은 "often causes collisions in contact-rich scenarios, resulting in instability during residual policy training". 데이터: FAVOR·OakInk-V2(DexManipNet 3.3K 에피소드·61과제·1.2K 물체), GRAB·ARCTIC은 정성 평가(ARCTIC은 부록의 관절형 물체). 한계: "some sequences cannot be transferred effectively" — 상호작용 자세 잡음 과다, 물체 모델 부정확(특히 관절형); 가우시안 잡음 1.5 cm까지 강건. 시도 대비 전이 성공 비율은 미기재.
- *DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation* — arXiv [2505.24853](https://arxiv.org/html/2505.24853v1) (2025-05; ICML 2026 포스터 목록 있음). **[원문 확인]** AnyTeleop 계열 운동학 리타게팅 재생은 "cannot complete the task", "the actions cannot achieve more than slightly lifting up each object", "results in frequent penetration with the object"(물체 인지 후처리로 보정). 가상 물체 제어기(PD, 이득 감쇠 커리큘럼). 6개 손(Inspire, Allegro, Xhand, Schunk, Ability, DexRobot) × ARCTIC 관절형 물체 5종, 지표 ADD-AUC. 한계: 특권 상태 입력, 실로봇 미평가. 수치는 Fig. 3에만 있어 본문으로 확인 불가.
- *Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration (Human2Sim2Robot)* — T. G. W. Lum, O. Y. Lee, C. K. Liu, J. Bohg (Stanford); CoRL 2025; arXiv [2504.12609](https://arxiv.org/html/2504.12609v3). **[원문 확인]** 사용 정보: FoundationPose 물체 6D 궤적(물체 중심 추종 보상, 앵커 3점) + 물체가 움직이기 약 1초 전의 사전 조작 손 자세(HaMeR + 깊이 ICP; cuRobo 팔 IK → PyBullet 손 IK). 비판 문장: "hand pose estimation noise and the human-robot embodiment gap often result in infeasible or suboptimal IK solutions"; "Simply replaying modifications of the single demonstration is unlikely to succeed in contact-rich settings". 7개 실과제에서 Replay 대비 +67%, Object-Aware Replay 대비 +55%, 행동 복제 대비 +68%(과제별 수치는 Fig. 6). 도메인 무작위화: 관측·행동 가우시안 잡음 σ=0.01, 중력 잡음 σ=0.3, **물체·테이블·로봇의 질량·마찰·스케일 [0.7, 1.3]**, 로봇 감쇠·강성 동일 범위, 5% 확률로 물체 질량 50배 무작위 힘. 한계: 단일 물체·단일 과제, 강체만, 대칭·반사 물체 자세 추정 실패, 시뮬레이션 마찰 모델 부정확.
- *SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation* — arXiv [2602.16863](https://arxiv.org/html/2602.16863v2) (2026-02; RSS 2026 목록 p151). 절차 생성 도구 프리미티브로 단일 물체 중심 RL 정책 학습 → RGB-D 사람 영상(SAM 3D + FoundationPose)에서 뽑은 자세 궤적을 추종. DexToolBench 120회·24과제·12 물체·6 도구 범주. NVIDIA 저자 여부는 검색 결과로 미확인. **[검색 요약]**
- *OmniRetarget* — arXiv [2509.26633](https://arxiv.org/html/2509.26633) (휴머노이드 전신, 유추용). **[원문 확인]** "scaling human hand keypoints to the robot's size often drives the robot's hands inside the object"; 물체 과제 관통(지속/최대 깊이): PHC 0.68/5.11 cm, GMR 0.83/8.50 cm, OmniRetarget 0.00/1.34 cm; 하류 RL 성공 71.28%/50.83%/82.20%. 상호작용 메시(Delaunay 사면체화, 물체 국소 좌표계, Laplacian 변형 최소화)로 접촉 관계 보존.
- *GNR* arXiv [2610.12440](https://alphaxiv.org/abs/2610.12440v1) (Meta Reality Labs 외, 2026-10): "inverse kinematics ignores dynamics and often yields infeasible motions"; 성공 기준에 물체 위치·자세 오차와 관통·접촉력 한계 포함; 56.2% vs MPC 27.2%. **[검색 요약]**
- *RobotFingerPrint* **[검색 요약]**: 사람→Fetch 실패 3/14의 대부분이 손 자세 추정 잡음.
- *Physically-based Grasp Quality Evaluation under Pose Uncertainty* — J. Kim, K. Iwamoto, J. Kuffner, Y. Ota, N. Pollard; IEEE T-RO 29(6):1424–1439, 2013 ([CMU RI](https://www.ri.cmu.edu/publications/physically-based-grasp-quality-evaluation-under-pose-uncertainty)). 기존 지표가 "파지 후 접촉"을 측정해 실로봇에서 재현이 어렵다고 보고, 확률분포에서 샘플링한 초기 자세들로 **동적 시뮬레이션**을 돌려 성공/실패를 추정; 운동학·힘 폐쇄 방법보다 안정적이고 자연스러운 파지 집합. **[검색 요약]**

### Inferences
- **[추론]** 위 수치를 모으면 "사람 운동학 기록의 물리 재생 성공률 ≈ 5–8%, 접촉·물리 정제 후 40–60%, 데이터셋 여과 통과율 ≈ 16–60%"라는 범위가 나온다. 프로젝트가 "사람 유래 접촉 앵커를 그대로 쓰면 다중 암에서도 비슷한 비율로 실패한다"는 식의 기준선을 보고하면 선행 연구와 직접 비교가 가능하다.
- **[추론]** 모든 프로토콜이 "관통 + 외력/중력 저항 + 변위"의 3단이지만, **접촉 위치 섭동**을 명시적으로 넣은 것은 Get-a-Grip(손목 ±5 mm·±2.5°)과 Kim et al. 2013(초기 자세 분포)뿐이다. 프로젝트의 "계획–실현 접촉 오프셋(mm, 수 도)" 측정은 이들과 가장 가깝지만, 다중 암·동적 궤적 설정에서 이를 성공 예측 변수로 분석한 선행 연구는 찾지 못했다.

### Gaps
- Dexonomy·CEDex·DexVLG는 "합성 후보 중 통과 비율"을 보고하지 않는다.
- ManipTrans·DexMachina의 리타게팅 기준선 세부(어떤 리타게팅 라이브러리, 접촉 후처리 유무)는 부록/그림에 의존하며 본문으로 전부 확정하지 못했다.
- "실제 사람 파지 데이터셋(GRAB, DexYCB, ARCTIC)을 시뮬레이션에서 여과했을 때 몇 %가 통과하는가"를 정면으로 보고한 논문은 검색에서 찾지 못했다(검색어: "GraspXL Grasp'D human grasp physics simulation validation success", "SynH2R ... physics simulation filter human grasps fraction").

---

## B2. 불확실성 하의 해석적 파지 품질, "실현 오차에 대한 강건성"을 최적화 대상으로 삼은 연구, 접촉 배치 조건부 학습형 성공 예측기

### Takeaway
"ε(Ferrari–Canny) 자체는 실행 성공을 잘 예측하지 못하고, 자세·접촉 오차에 대한 강건성이 성공을 가른다"는 프로젝트의 관찰은 Weisz & Allen (ICRA 2012)이 Barrett 손 실험으로 이미 보였고(εGWS가 0.2 이상인 구간에서 P(힘 폐쇄)와 상관 없음; 강건성 순위 파지 91% vs ε 순위 67%), Li–Culbertson–Ames (2024)는 ε와 최소 가중치 지표 ℓ*가 **접촉점·법선 섭동(기저 렌치 섭동)에 대한 내재적 강건성**을 하한한다는 정리를 제시했으며, FIRMGrasp (2026)는 마찰 불확실성만으로도 명목 ε가 인증한 파지의 53%가 힘 폐쇄를 잃음을 보였다. ℓ∞ 요구 렌치 여유 "w_req + ρ D v ∈ W_c" 꼴을 다중 암 접촉 배치 최적화에 쓴 선행 연구는 찾지 못했다.

### Cited Findings

**고전·해석적 강건성**
- Ferrari & Canny (ICRA 1992) ε 지표, Pollard (2004), Li & Sastry (1988) 과제 지향 품질, Zheng & Qian (IJRR 2005) "Coping with the grasping uncertainties in force-closure analysis" **[기억, 식별자 미확인]** — 배경 인용용.
- *Pose Error Robust Grasping from Contact Wrench Space Metrics* — Jonathan Weisz, Peter K. Allen (Columbia); ICRA 2012 ([PDF](https://www.cs.columbia.edu/~allen/PAPERS/weisz_icra12.pdf)). **[원문 PDF 텍스트 확인]**
  - 초록: "We find that in most cases the grasp with the most robust force closure with respect to pose error for a particular object is not the grasp with the highest epsilon quality ... grasps ranked best by this method are successful more often in physical experiments than grasps ranked best by the epsilon quality."
  - 오차 모델: 탁상 위 물체의 [x, y, θ] 3차원, 1 mm·1° 간격으로 균일 샘플링; 강건성 지표는 P(εGWS > δ)(0보다 큰 하한 δ 사용). Fig. 1 예: ε=0.17인 파지가 20° 회전·1 cm 병진 후 힘 폐쇄를 잃음.
  - 시뮬레이션: 480 파지에서 "over most of the range of εGWS, there is no correlation"; εGWS > 0.2 구간에서만 약한 상관. 강건성 기준 선택이 ε 기준보다 "19% percent more often" 힘 폐쇄 달성.
  - 물리 실험(Barrett 손, DARPA ARM-S 물체): P(fc) 순위 파지 85/93 성공(91%) vs ε 순위 67%; 더 크고 복잡한 물체 부분집합에서 35/43(81%) vs 16/43(37%) — "success rate is doubled".
- 독립 접촉 영역(Independent Contact Regions): Roa & Suárez, *Computation of Independent Contact Regions for Grasping 3-D Objects*, IEEE T-RO 25(4):839–850, 2009; 불확실성 확장 *Influence of Contact Types and Uncertainties in the Computation of Independent Contact Regions*, ICRA 2011 ([PDF](https://iocnet.upc.edu/usuaris/raulsuarez/papers/icra11-rs.pdf), [UPC](https://upcommons.upc.edu/entities/publication/83f8d9f0-c161-4ee3-8163-4cde3f347c94)). 시작 파지의 접촉점 주위로 "힘 폐쇄가 유지되는 영역"을 키우는 방식 — **접촉 위치 오차 허용 범위를 직접 기하로 표현**한 고전. **[검색 요약]**
- Kim et al. T-RO 2013(B1 참조): 자세 불확실성 하 동적 시뮬레이션 기반 품질.
- *Toward An Analytic Theory of Intrinsic Robustness for Dexterous Grasping* — Albert H. Li, Preston Culbertson, Aaron D. Ames (Caltech); arXiv [2403.07249](https://arxiv.org/html/2403.07249v2) (v2 2024-08-29; 검색 요약은 IROS 2024로 표기). **[원문 확인]**
  - 정의: 내재적(intrinsic) 강건성 = 실행 전 계획에 영향을 주는 물체 자세·기하 불확실성에 대한 강건성(외란 저항인 외재적 강건성과 구분). 선형(피라미드) 마찰원뿔; 접촉점 x와 법선 n의 불확실성이 기저 렌치를 섭동.
  - Theorem 1: 각 섭동 기저 렌치의 편차가 명목 렌치 껍질의 음(−)집합 안에 있으면 원점이 섭동 껍질 안에 남음. Corollary 1: **각 섭동 렌치가 명목값 반지름 ε 공 안에 있으면 참 파지는 힘 폐쇄** → Ferrari–Canny ε는 내재적 강건성도 측정. Theorem 2: 체비쇼프 반지름 하한 δ_min(정규성 조건) 아래 K·ℓ* ≤ ε, 즉 최소 가중치 지표 ℓ*(FRoGGeR) 최대화가 강건성 보장.
  - 실험: Allegro+Franka Research 3, 8 물체, NeRF 재구성; FRoGGeR(ℓ* ≥ 0.3/n_w) 34/40(85%) vs Wu et al. 힘 폐쇄 실현가능성 21/40(52.5%); 약 15% 시도는 사전 파지 중 물체 충돌로 제외(NeRF 자세·기하 오차).
  - PONG(Probabilistic Object Normals for Grasping): 접촉 위치는 알려진 것으로 두고 법선만 접평면 가우시안 섭동, 힘 폐쇄 확률 하한 L_FC를 다각형 위 가우시안 적분(Green 정리)으로 계산; 성공률이 ℓ*·L_FC와 함께 상승.
  - 한계(저자): 기하 믿음 분포 획득이 어려움, PONG은 접촉 위치 고정, 근사 힘 폐쇄 집합이 보수적, 하드웨어 8 물체. **ℓ∞/ℓ1 렌치 노름은 언급하지 않음**(ε의 노름은 "assumed constraints on a hand's control authority"에 의존한다고만 서술).
- *FIRMGrasp: A Friction-Informed Risk Margin for Robust Grasp Synthesis* — Clinton Enwerem, John S. Baras, Calin Belta (UMD); arXiv 2607.25049 ([alphaXiv v1](https://www.alphaxiv.org/abs/2607.25049v1), 2026-07-27; v2 2026-08-06). **[alphaXiv 페이지 확인]** 마찰 μ를 확률변수로 두고 CVaR_β(μ)로 위험 조정 마찰 v_β, 위험 조정 여유 ε(β) = v_β로 스케일한 렌치 공간 내접구 반지름; Q_β = 명목 여유의 CVaR. **"53% of the grasps that the nominal Ferrari-Canny margin certifies lose force closure in the adverse friction tail"**(LEAP+Allegro 1,599 파지; 보정된 마찰 분포에 의존), 머스터드 병 86% 마찰 민감. Drake에서 6축 흔들기·중력 하 들어올리기+측면 당김(μ∈{0.2,0.3,0.4,0.7}); μ=0.2 측면 당김에서 ε(β) 인증 파지 70% vs 명목만 인증 25%. **접촉 위치·법선 불확실성은 모델링하지 않음**(향후 과제로 언급). 실하드웨어 없음.
- *UNCLE-Grasp* arXiv [2601.14492](https://arxiv.org/pdf/2601.14492) (2026-01): 형상 완성 몬테카를로 샘플링 위에서 ε 지표로 접촉점 채점(딸기 수확 응용). **[검색 요약]**
- Dex-Net 1.0/2.0 (Mahler et al., 2016/2017): 자세·마찰 불확실성 하 힘 폐쇄 확률("robust epsilon")로 평행 그리퍼 파지를 라벨링하고 GQ-CNN으로 학습 **[기억, 식별자 미확인: Dex-Net 2.0 arXiv 1703.09312]**. 다중 접촉 확장은 Get-a-Grip 계보가 가장 가깝다 **[추론]**.

**접촉 배치/섭동 조건부 학습형 성공 예측기**
- Get-a-Grip(B1): 손목 ±5 mm·±2.5° 섭동 5회 평균 소프트 라벨로 평가기 학습; 평가기 유무가 실로봇 43–51% → 76–81%. 해석적 지표와의 상관은 미보고. **[원문 확인]**
- *EfficientGrasp* arXiv [2206.15159](https://arxiv.org/abs/2206.15159): 힘 폐쇄 추정기에서 빌린 품질 점수로 접촉점 집합 0.75 이상을 유효로 간주, 18 YCB 실로봇 83.3%. **[검색 요약]**
- Lu et al., *Multi-Fingered Grasp Planning via Inference in Deep Neural Networks* arXiv [2001.09242](https://arxiv.org/pdf/2001.09242): 복셀 3D CNN이 물체 기하+손 구성에서 성공 확률을 추정하고 이를 직접 최대화(접촉점이 아닌 손바닥 자세·프리셰이프 조건부). **[검색 요약]**
- GraspGen 판별기(A2), CollaBot의 완화 힘 폐쇄 지표(A1)도 평가기 역할.

### Inferences
- **[추론]** 프로젝트의 "렌치 공간 유사도는 정적 성공을 예측하지 못하고 실현 오차가 예측한다"는 발견은 Weisz & Allen(2012)의 "ε는 자세 오차 강건성의 빈약한 예측자"와 같은 방향이며, Li–Culbertson–Ames(2024)가 "ε·ℓ*는 접촉점·법선 섭동 허용량의 하한"임을 증명했으므로, 프로젝트의 ρ(ℓ∞ 여유)를 "접촉 실현 오차 허용 반경"으로 해석하는 이론적 연결이 가능하다(단, 그 논문은 ℓ∞ 노름을 명시하지 않았으므로 프로젝트가 직접 유도해야 한다).
- **[추론]** 독립 접촉 영역(ICR)은 "접촉 위치가 이 영역 안에만 있으면 힘 폐쇄 유지"라는 기하적 허용 범위이므로, 프로젝트의 "mm 수준 오프셋이 성공을 가른다"는 관찰을 "실현된 접촉이 ICR을 벗어났는가"로 재진술할 수 있다. 다중 암(구형 손끝) + 동적 궤적에서 ICR을 계산·추적한 연구는 찾지 못했다.

### Gaps
- "w_req + ρ D v ∈ W_c" 꼴의 ℓ∞ 요구 렌치 강건성 여유를 누가 처음 정식화했는지는 이번 검색으로 확정하지 못했다(검색어: "robust force closure contact position uncertainty grasp quality margin"). 프로젝트가 따르는 "published formulation"의 서지는 프로젝트 쪽 기록으로 보완 필요.
- Weisz & Allen의 "72 of ..." 문장은 줄바꿈으로 잘려 전체 문맥(72개 중 몇 개 물체에서 ε가 강건성을 예측하지 못했는지)을 확정하지 못했다.
- Dex-Net 계열의 다중 접촉 확장(예: 흡착·다지)과 접촉점 조건부 성공 예측기 중 2024–2026년 신규 연구는 검색에서 뚜렷이 나오지 않았다(검색어: "learned grasp success predictor contact points multi-fingered neural force closure 2024").

---

## B3. "운동학 리타게팅/IK/인페인팅은 물리적으로 유효한 접촉·파지 성공을 보장하지 않는다"는 명시적 비판과 정량 실패 수치

### Takeaway
이 비판은 2019년 ContactGrasp부터 2026년 접촉 고정 리타게팅까지 반복적으로 문장과 수치로 제시되어 있다. 정량적으로는 운동학 재생 성공률 4.6%/0.0%(ManipTrans), 5.35%(DexFlow의 DexRetarget), 7.3%(MANO 재생), DexPilot 접촉 유지율 2.0%(≤5 mm), GMR 관통 지속 0.83·최대 8.5 cm(OmniRetarget, 휴머노이드)가 인용 가능하다. 인페인팅 계열(Phantom, Masquerade, Shadow, DexUMI)은 "접촉 보존"을 주장하지 않으며 손 자세→말단 자세 변환에 의존한다.

### Cited Findings(인용 가능한 문장·수치 모음)
- ContactGrasp (2019): "deterministic mapping of grasps between hand models does not reliably reproduce contact" ([arXiv 1904.03754](https://arxiv.org/pdf/1904.03754)). **[검색 요약의 인용]**
- Du et al. (2022): 기존 리타게팅은 "handcrafted correspondences ... and they do not consider the object" ([arXiv 2211.07304](https://arxiv.org/pdf/2211.07304)). **[원문 확인]**
- DexFlow (2025): 전통적 리타게팅은 "suffer from penetration artifacts and unstable contact patterns"; DexRetarget 관통 지표 84.4 vs DexFlow 8.5, 성공률 5.35% vs 40.32% ([arXiv 2505.01083](https://arxiv.org/html/2505.01083v1)). **[원문 확인]**
- ManipTrans (CVPR 2025): "Retargeting methods often struggle with transferring MoCap data to physically plausible motions"; Retarget-Only 성공률 4.6%(단일)/0.0%(양손) ([arXiv 2503.21860](https://arxiv.org/html/2503.21860v1)). **[원문 확인]**
- DexMachina (2025): 운동학 리타게팅은 "results in frequent penetration with the object", "cannot complete the task" ([arXiv 2505.24853](https://arxiv.org/html/2505.24853v1)). **[원문 확인]**
- Human2Sim2Robot (CoRL 2025): "hand pose estimation noise and the human-robot embodiment gap often result in infeasible or suboptimal IK solutions"; 재생 대비 +67% ([arXiv 2504.12609](https://arxiv.org/html/2504.12609v3)). **[원문 확인]**
- 접촉 고정 리타게팅 (2026): "mapping joint angles or fingertip positions, treats hands with different kinematics as scaled copies of one another and fails precisely when morphology differs"; "Video alone ... recovers neither precise hand–object pose nor contact force"; MANO 재생 7.3%; DexPilot 접촉 유지율 2.0%(≤5 mm)·관절 한계 위반 21.4% ([arXiv 2609.24093](https://arxiv.org/pdf/2609.24093)). **[원문 확인]**
- OmniRetarget (2025, 휴머노이드): "scaling human hand keypoints to the robot's size often drives the robot's hands inside the object"; GMR 관통 0.83/8.50 cm ([arXiv 2509.26633](https://arxiv.org/html/2509.26633)). **[원문 확인]**
- GNR (2026-10): "inverse kinematics ignores dynamics and often yields infeasible motions" ([alphaXiv 2610.12440](https://alphaxiv.org/abs/2610.12440v1)). **[검색 요약]**
- CEDex (2025): 시연 기반 방법은 관절 보정·재매핑 필요, 의인화 손에 국한 ([arXiv 2509.24661](https://arxiv.org/html/2509.24661v1)). **[원문 확인]**
- GeoRT (2025): "the robot and human hands keypoint spaces are not directly related through a linear mapping as suggested by previous works" ([arXiv 2503.07541](https://arxiv.org/pdf/2503.07541)). **[검색 요약]**
- RobotFingerPrint (2024): 사람→그리퍼 실패 대부분이 손 자세 추정 잡음 ([arXiv 2409.14519](https://arxiv.org/pdf/2409.14519)). **[검색 요약]**
- 인페인팅 계열의 자기 서술: Shadow는 "keeps the gripper-object interaction and the surrounding scene intact"라고 하지만 이는 **영상 편집상의 보존**이지 물리 검증이 아니다 ([arXiv 2503.00774](https://arxiv.org/pdf/2503.00774)) **[검색 요약 + 추론]**; Phantom·Masquerade·DexUMI는 손 자세 추정→로봇 행동 변환을 전제로 하며 접촉 유효성 검사는 보고되지 않았다 **[검색 요약 범위에서의 부재; 원문 전수 확인은 못 함]**; RealDexUMI는 "retargeting을 거치며 배치 가능한 손재주를 잃는다"는 문제의식을 명시 ([arXiv 2606.06033](https://arxiv.org/html/2606.06033)) **[검색 요약]**.

### Inferences
- **[추론]** "물리적 성공 보장 부재"라는 비판은 이미 널리 공유된 문제의식이므로, 프로젝트 논문의 서론에서는 위 수치를 인용해 짧게 정리하고, 기여는 (1) 다중 암·대형 물체라는 새로운 몸체 간극, (2) 렌치 여유 ρ와 접촉 실현 오차의 관계를 폐루프 도메인 무작위화로 검증한 점, (3) 사람 접촉 그래프를 접촉 수가 줄어드는 방향으로 "수축"하는 정식화에 두는 것이 정확하다.
- **[추론]** 기존 해법은 거의 모두 "잔차 RL로 물리 보정"(ManipTrans, DexMachina, 접촉 고정 리타게팅, Human2Sim2Robot)이다. 프로젝트의 QP 힘 할당기 + 렌치 여유 최적화는 "학습 없는 해석적 보정"이라는 점에서 이 흐름과 대비된다.

### Gaps
- 실로봇에서 "IK 리타게팅된 사람 파지가 몇 % 실패하는가"를 직접 측정한 논문은 찾지 못했다. 모든 정량 수치는 시뮬레이션 재생(ManipTrans, DexFlow, 접촉 고정 리타게팅) 또는 간접 비교(Human2Sim2Robot의 실로봇 Replay 기준선 상대 차이, RobotFingerPrint 3/14 실패)다.
- 자기중심 영상(egocentric video) 전용으로 "접촉 유효성"을 다룬 비판 문헌은 HOWTransfer·MAPLE·EgoLoc 초록 수준에서만 확인했고, 정량 실패율은 없다.
