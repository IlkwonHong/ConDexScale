# Egocentric(1인칭) 인간 영상 → 로봇 조작 학습 파이프라인 조사 (2026-11-08 기준)

작성 원칙에 대한 안내입니다.
- 각 논문 항목은 "제목 / 저자·소속 / 발표처·날짜 / arXiv 식별자 / 전이 메커니즘 / 대상 로봇 / 데이터 출처 / 물리적 타당성(grasp가 실제로 성립하는지) 처리 방식 / 핵심 수치 / 저자가 밝힌 한계" 순서로 적었습니다.
- 표기 규칙: **[출처 확인]** 은 이번 조사에서 1차 출처(arXiv 전문 또는 초록, 공식 프로젝트 페이지)를 직접 읽고 확인한 내용입니다. **[초록만 확인]** 은 arXiv 초록 페이지만 확인한 내용입니다. **[추론]** 은 확인된 사실들로부터 제가 도출한 판단입니다. **[배경지식, 전문 미확인]** 은 2023~2024년의 잘 알려진 선행 연구로, 이번 조사에서 전문을 다시 읽지는 않았고 arXiv 식별자만 제시한 것입니다.
- 용어 안내: 아래에서 "리타겟팅(retargeting)"은 사람 손의 자세·움직임을 로봇의 관절 공간으로 옮기는 과정, "역기구학(inverse kinematics, IK)"은 원하는 말단 위치·자세로부터 관절 각도를 역산하는 계산, "인페인팅(inpainting)"은 영상에서 특정 영역(여기서는 사람 팔)을 지우고 배경으로 자연스럽게 메우는 기법, "힘 폐쇄(force closure)"는 접촉점들이 만드는 힘·토크(렌치, wrench)의 조합이 임의 방향의 외란을 상쇄할 수 있는 상태를 뜻합니다. "평행 조(parallel-jaw) 그리퍼"는 두 개의 판이 마주 닫히는 집게형 손, "다지 손(dexterous hand)"은 손가락이 여러 개인 로봇 손입니다.

---

## 핵심 질문 1. 메커니즘별 분류와 대표 논문: 각 방법은 무엇을 입력받아 어떻게 전이하며, 물리적 성립 여부를 확인하는가

### Takeaway
2025~2026년 egocentric 영상 기반 로봇 조작 학습은 (1) 손 자세 추정 → 운동학적 리타겟팅, (2) 사람 팔을 지우고 로봇을 합성하는 영상 편집, (3) 점 궤적·흐름·affordance 같은 객체 중심 표현, (4) 대규모 VLA(Vision-Language-Action 모델) 사전학습, (5) 영상 → 시뮬레이션 → 강화학습(RL) 또는 시뮬레이션 검증의 다섯 갈래로 정리됩니다. 앞의 네 갈래는 거의 예외 없이 "실제 로봇 롤아웃의 과제 성공률"만을 검증 수단으로 삼고 grasp 자체의 물리적 성립(관통 없음, 힘 폐쇄, 중력 하 유지)을 별도로 검사하지 않으며, 다섯 번째 갈래만이 시뮬레이션 안에서 grasp를 재수립하거나 들어올림·흔들기 테스트로 검증합니다.

### Cited Findings

#### 1-A. 손 자세 추정 → 운동학적 리타겟팅 / 역기구학 계열

- **EgoMimic: Scaling Imitation Learning via Egocentric Video** — Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Judy Hoffman, Danfei Xu (Georgia Tech), Chen Wang (Stanford). arXiv 2024-10-31, [arXiv:2410.24221](https://arxiv.org/html/2410.24221). **[출처 확인]**
  - 입력·추적: Project Aria 안경을 사람과 로봇 양쪽에 장착하고, Aria Machine Perception Services(MPS)의 손 추적으로 손 자세를 얻습니다. 로봇은 ViperX 300 S 6자유도 팔 두 대입니다.
  - 전이 메커니즘: 사람 손과 로봇 말단(end-effector)의 궤적을 카메라 중심 좌표계로 변환해 같은 네트워크가 행동 묶음(action chunk)으로 예측하도록 공동 학습합니다. 사람 영상과 로봇 영상 모두에서 팔을 SAM2로 분할하고 빨간 선(red line)으로 덮어 외형 차이를 줄입니다.
  - 물리적 타당성 처리: **사람 데이터에는 grasp 정보가 전혀 없습니다**(Aria는 손 자세만 기록). 그리퍼 개폐는 로봇 관절 손실로만 감독됩니다. grasp 성립 여부에 대한 별도 검사는 없습니다.
  - 핵심 수치: Laundry 성공률 ACT 55% / EgoMimic(사람 데이터 제외) 73% / EgoMimic 88%, Groceries 22% / 28% / 30%, Open Bag 54% / 60% / 70%. 로봇 데이터 2시간 + 사람 데이터 1시간이 로봇 데이터 3시간의 ACT를 능가했습니다(128 대 74점).
  - 저자가 밝힌 실패 사례: 장난감이나 그릇을 몇 인치 빗나감, 봉지 손잡이를 잡지 못함, 셔츠 한쪽만 잡음.

- **EgoZero: Robot Learning from Smart Glasses** — Vincent Liu, Ademi Adeniji, Haotian Zhan, Siddhant Haldar, Raunaq Bhirangi, Pieter Abbeel, Lerrel Pinto (NYU, UC Berkeley). arXiv 2025-05, [arXiv:2505.20290](https://arxiv.org/html/2505.20290). **[출처 확인]**
  - 입력·추적: Aria 안경이 6자유도 손 자세·카메라 내외부 파라미터를 제공하고, HaMeR로 21개 관절점을 추정한 뒤 Aria의 손바닥 자세로 보정합니다. 로봇 데이터는 전혀 쓰지 않습니다(zero-shot).
  - 전이 메커니즘: 행동을 엄지·검지 손끝의 3D 좌표 + 그리퍼 개폐로 정의합니다. grasp는 엄지·검지 거리의 임계값으로 판정합니다. Franka Panda에 역기구학으로 전달합니다.
  - 물리적 타당성 처리: 별도 검사 없음. 검증은 실제 로봇 롤아웃뿐입니다.
  - 핵심 수치: 과제당 사람 시연 100개(약 20분)로 7개 과제 평균 약 70%(74/105). 시각 기반 정책(Baku 변형) 0/15, 단안 깊이 추정으로 대체 시 0/15.
  - 저자가 밝힌 한계: 행동 라벨 오차 1~2 cm로 고정밀 과제 불가, 단안 metric depth 모델 오차 5 cm 초과, 평행 그리퍼에 한정(다지 손·양손은 향후 과제), 추론 시 iPhone-로봇 보정 필요.

- **HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos** — Zhi (Leo) Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, Yiannis Aloimonos (University of Maryland). arXiv v3 2026-09-14, [arXiv:2605.24934](https://arxiv.org/html/2605.24934). **[출처 확인]**
  - 입력·추적: Aria Gen1 안경 30 Hz, Aria MPS 손 추적(스테레오 SLAM 카메라 기반)을 Savitzky–Golay 필터와 지수이동평균으로 평활화합니다.
  - 전이 메커니즘: 사람 손을 **가상 평행 조 그리퍼**로 리타겟팅합니다. 위치는 엄지·검지 손끝의 중점, 자세는 MCP 관절로 만든 좌표계, grasp는 손끝 거리로 판정해 배치 시 이진화합니다. 흐름 정합(flow matching) 정책이 양손 6자유도 자세 + 이진 grasp 상태를 출력합니다. WidowX 팔, Franka·UR10에도 전이 주장.
  - 물리적 타당성 처리: 도달 가능성이나 충돌 검사 같은 별도 분석 없음. 과제당 40회 실제 롤아웃이 유일한 증거입니다.
  - 핵심 수치: 과제당 사람 데이터 30분으로 4개 과제 평균 92.5%, 15분이면 75%. 같은 시간의 원격조작 데이터로 학습한 ACT는 51.2%. zero-shot 비교군(EgoZero, Point Policy, ZeroMimic, Track2Act, SPOT)은 1.9%~45.0%.
  - 저자가 밝힌 실패 사례·한계: **점(point) 기반 비교군은 그리퍼가 무작위 배치된 물체에 어긋나 도착하면 grasp에서 실패**합니다. 준정적(quasi-static) 장면을 가정하므로 손안 재배치, 미끄러짐, 다시 잡기는 가정을 위반합니다. 평행 조 팔에만 전이가 확인되었고 다지 손은 아닙니다. 몇 번의 시연으로는 약 1 cm 정밀도에서 정체됩니다.

- **DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies** — Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, Deepak Pathak (CMU). arXiv 2025-05, [arXiv:2505.07813](https://arxiv.org/html/2505.07813v1). **[출처 확인]**
  - 입력·추적: 전자기장 센서 기반 모션캡처 장갑으로 손끝 위치를, 장갑 위 ArUco 마커를 외부 카메라로 추적해 손목 자세를 얻습니다(영상 기반 손 추정이 아님). 손바닥에 스테레오 카메라 두 대를 사람과 로봇에 대칭 배치합니다.
  - 전이 메커니즘: PyBullet에서 손끝 위치를 맞추는 역기구학으로 LEAP Hand V2 Advanced(xArm)에 리타겟팅하고, 사람 데이터 9,290개와 원격조작 로봇 데이터 1,395개를 공동 학습합니다.
  - 물리적 타당성 처리: 별도 검사 없음.
  - 핵심 수치: 로봇 데이터만 학습 vs 1:2 공동 학습 — 영역 내 64.7% vs 79.8%, 실외 환경 28.5% vs 75.1%, 극단적 환경 22.0% vs 62.7%. 사람 데이터만으로는 3.6%(영역 내), 7.3%(실외). 사람 데이터 비율을 1:5로 높이면 오히려 성능이 떨어졌습니다(54.5%, 50.9%).
  - 저자가 밝힌 한계: 사람-로봇 행동 간극을 메우려면 여전히 원격조작 로봇 데이터가 필요, 사람 시연에는 오류 복구가 거의 없음, 시각·운동학 정보만 쓰므로 접촉 집약적 과제에 한계(촉각·햅틱 제안).

- **EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos** — Ruihan Yang 외, Xiaolong Wang (MIT, NVIDIA, UC San Diego, UIUC). arXiv v3, [arXiv:2507.12440](https://arxiv.org/html/2507.12440). **[출처 확인]**
  - 입력·추적: HOI4D(4,000편), HOT3D(833분, 3D 손·카메라 자세 포함), HoloAssist(166시간 중 1/10 샘플, 손 자세를 MANO로 리타겟팅), TACO(2,317 시퀀스). 약 50만 이미지-행동 쌍.
  - 전이 메커니즘: 행동 공간을 카메라 좌표계의 손목 이동·회전 + MANO 손 파라미터 상위 15개 주성분으로 통일합니다. 로봇 손 자세는 손끝 위치 불일치를 최소화해 MANO로 대응시키고, 배치 시 4층 MLP가 손끝 좌표를 관절 명령으로 변환합니다. 팔은 역기구학.
  - 대상·검증: Isaac Lab의 Unitree H1 + Inspire 손 **시뮬레이션 전용**(Ego Humanoid Manipulation Benchmark, 12과제). 논문 스스로 "직접 sim-to-real 전이용으로 설계되지 않았다"고 명시.
  - 물리적 타당성 처리: grasp 단위 지표 없음. 리타겟팅 오차로는 시뮬레이션 손끝 위치 오차 5×10⁻⁵ m만 보고.
  - 핵심 수치: 보이는 환경 단기 과제 EgoVLA 77.78% / 사전학습 없음 64.55% / ACT 24.87%, 장기 과제 45.93% / 26.67% / 2.22%. **로봇 후속 학습 없이 zero-shot 배치하면 모든 과제 0%**.
  - 저자가 밝힌 한계: 외형·지각·운동학 불일치 때문에 zero-shot 전이 실패, 손·손목 자세 주석이 있는 사람 데이터가 필요.

- **Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos** — Hao Luo 외, Zongqing Lu (Peking University, Renmin University, BeingBeyond). arXiv 2025-07-21, [arXiv:2507.15597](https://arxiv.org/html/2507.15597v1). **[출처 확인]**
  - 입력·추적: UniHand 데이터셋 — 모션캡처(ARCTIC, H2O, HOI4D, OAKINK2, TACO, DexYCB), VR 기록(EgoDex), 의사 주석(Taste-Rob 등) 11개 출처, 약 44.4만 궤적·1,155시간. 3D 손 라벨은 MANO 직접 제공, 3D 관절에 MANO 피팅, 또는 **HaMeR 프레임별 추정**(좌우 보정·보간·관절 한계·시간 평활 포함) 세 경로로 얻습니다. Ego4D와 EPIC-Kitchens는 가림·카메라 문제로 제외했습니다.
  - 전이 메커니즘: MANO-D162 특징을 15 FPS로 부위별 그룹 잔차 양자화(8층)해 손목·손가락 토큰(초당 128개)으로 만들어 사전학습하고, 로봇에는 사전학습 인코더 위에 학습 가능한 행동 질의 + MLP 헤드를 얹어 Franka Research 3 + Inspire 손(6자유도)의 말단 자세·손 관절을 원격조작 데이터로 L1 모방 학습합니다. 즉 사람 손 → 로봇 손의 명시적 리타겟팅은 정책 단계에 없고, 로봇 데이터로 행동 공간을 다시 맞춥니다.
  - 물리적 타당성 처리: 없음. "physical space alignment"는 카메라 기하(약한 원근 투영·시점 균형)를 뜻하며 접촉 역학이 아닙니다. 저자 스스로 "이번 버전에서는 물체의 6D 자세 같은 상호작용 물체 모델링을 넣지 않았다"고 밝힘.
  - 핵심 수치(과제당 20회): Pick-Place-Toy 보이는/안 보이는/어수선 75/65/60 vs GR00T N1.5 75/40/50, Close-Toolbox 85 vs 80, Close-Lid 60 vs 50, Pour-Cup 100 vs 90, Unfold-Clothes 75 vs 60. 비교군(InternVL3)의 실패 사례로 "불안정한 grasp가 붓기를 망침", "뚜껑 가장자리를 놓쳐 충분한 힘을 못 가함"이 기록되어 있습니다.
  - 향후 과제: 깊이·촉각 피드백, 시뮬레이션·강화학습 통합.

- **EgoWild2Dex: Learning Dexterous Robotic Manipulation from In-the-Wild Human Experience** — Kunyang Lin 외, Ping Luo (University of Hong Kong, Kinetix AI). arXiv 2026-09-20, [arXiv:2609.23755](https://arxiv.org/html/2609.23755). **[출처 확인]**
  - 입력·추적: 직접 수집한 EgoWild(538.9시간, 179,049 에피소드, 1,282 물체 범주). PICO VR 헤드셋 손 추적 + 양손에 보정된 트래커. 트래커 없이는 손끝 오차 중앙값 4.21 cm, 있으면 0.66 cm.
  - 전이 메커니즘: 상대 손목 자세 매핑 + 제약 역기구학(팔) + 손가락 특징 → 손 모터 명령 리타겟팅으로 "로봇 고유 행동 공간"에 통일하고, 흐름 정합 행동 전문가를 가진 VLA를 사람→로봇 학습 → 공동 학습 → DAgger 보정의 3단계로 훈련합니다. 로봇은 AgileX Piper 팔 두 대 + BrainCo Revo2 손(6자유도).
  - 물리적 타당성 처리: 시뮬레이션이나 물리 검증 없음. 명령 필터링, 관절 제약, 순응 토크 제어만 언급. 리타겟팅 정합도는 주성분 공간 겹침(IoU 0.73)으로 평가.
  - 핵심 수치: 로봇 시연 100개/과제로 장기 양손 과제 3개 평균 96.7%; 같은 데이터로 직접 미세조정한 π0 3.3%, π0.5 6.7%, GR00T-N1.7 0.0%. 안 본 물체 zero-shot 33.3%.
  - 저자가 밝힌 한계: "사람 시연만으로 배우는 것은 여전히 어렵다"며 장기·다지 협응에는 실제 로봇 미세조정이 필요. 자유도가 더 높은 손은 안정성·내구성·발열·비용 조건을 만족하는 상용품이 없어 평가하지 못함.

- **Humanoid Policy ~ Human Policy** — Ri-Zhao Qiu 외, Xiaolong Wang (15인). arXiv 2025-03 (v3 2025-10), [arXiv:2503.13441](https://arxiv.org/abs/2503.13441). **[초록만 확인]** 사람과 휴머노이드가 공유하는 상태-행동 공간(Human Action Transformer, HAT)을 두고 "미분 가능하게 로봇 행동으로 리타겟팅"한다고 밝혔습니다. 물리적 검증 언급 없음. 수치는 초록에 없음.

- 배경 연구 **[배경지식, 전문 미확인]**: MimicPlay([arXiv:2302.12422](https://arxiv.org/abs/2302.12422)), HumanPlus([arXiv:2406.10454](https://arxiv.org/abs/2406.10454)), OKAMI([arXiv:2410.11792](https://arxiv.org/abs/2410.11792), CoRL 2024 구두발표; 단일 RGB-D 영상에서 물체 인식 리타겟팅, 폐루프 정책 평균 79.2% — 이 수치는 초록에서 확인), DexCap([arXiv:2403.07788](https://arxiv.org/abs/2403.07788)), ARCap([arXiv:2410.08464](https://arxiv.org/abs/2410.08464); 초록 확인: 증강현실 시각 피드백과 햅틱 경고로 "로봇 운동학에 맞고 장면과 충돌하지 않는" 데이터를 수집하도록 유도 — 즉 **수집 시점에 운동학적 실행 가능성을 사람에게 되먹임**하는 드문 사례이나 grasp 역학 검증은 아님), DexMV([arXiv:2108.05877](https://arxiv.org/abs/2108.05877)), DexVIP([arXiv:2202.00164](https://arxiv.org/abs/2202.00164)), VideoDex([arXiv:2212.04498](https://arxiv.org/abs/2212.04498)), Robotic Telekinesis([arXiv:2209.12993](https://arxiv.org/abs/2209.12993)).

#### 1-B. 영상 편집(인페인팅·마스킹·로봇 합성) 계열

- **Phantom: Training Robots Without Robots Using Only Human Videos** — Marion Lepert, Jiaying Fang, Jeannette Bohg (Stanford). CoRL 2025(PMLR 305), [arXiv:2503.00779](https://arxiv.org/html/2503.00779v2), [PMLR 페이지](https://proceedings.mlr.press/v305/lepert25a.html). **[출처 확인]**
  - 입력·추적: RGB-D 카메라. HaMeR로 21개 관절점과 778정점 손 메시를 얻고, SAM2로 손을 분할해 깊이에서 부분 점군을 만든 뒤 ICP로 절대 3D 자세를 맞춥니다. 가림 시 비현실적 자세를 줄이려고 엄지·검지 마지막 두 관절을 1자유도로 제한합니다.
  - 전이 메커니즘: 로봇 행동 = 엄지·검지 손끝 중점(위치), 엄지·검지 관절점 평면 + 주축(자세), 손끝 거리(그리퍼 개폐; **각 궤적의 하위 20% 값은 완전히 닫힘으로 강제해 미끄러짐을 줄이는 휴리스틱**). MuJoCo(Menagerie 모델)로 가상 로봇을 렌더링해 덮어쓰고, SAM2 + E2FGVI로 사람 팔을 인페인팅합니다. 추론 시에도 실제 로봇 위에 가상 로봇을 덮어 학습 분포에 맞춥니다. 정책은 Diffusion Policy.
  - 물리적 타당성 처리: **없음.** 20% 닫힘 규칙은 미끄러짐 휴리스틱이지 성립 검사가 아닙니다.
  - 핵심 수치(25회): Pick/Place Book 0.92, Stack Cups 0.72, Tie Rope 0.64, Rotate Box 0.72, Grasp Brush 0.88, Sweep 0.40. EgoMimic식 빨간 선 비교군과 원본 영상 비교군은 모든 과제 0%. 사람 데이터만 300개로 0.84 도달.
  - 저자가 밝힌 한계: 손 자세 추정기가 grasp 중 가림에 취약, **로봇이 사람의 전략을 그대로 따를 수 있어야 함**(사람이 피한 충돌을 로봇이 일으킬 수 있고, 손끝과 그리퍼 표면 물성 차이가 물체 움직임을 바꿈), **평행 조 그리퍼라서 pinch grasp만 사용**, 준정적 과제만 평가.

- **Masquerade: Learning from In-the-wild Human Videos using Data-Editing** — Marion Lepert, Jiaying Fang, Jeannette Bohg (Stanford). arXiv 2025-08-13, [arXiv:2508.09976](https://arxiv.org/html/2508.09976v1), [프로젝트 페이지](https://masquerade-robot.github.io). **[출처 확인]**
  - 입력·추적: EPIC-Kitchens 약 1만 클립(675,713 프레임), HaMeR 21 관절점. 단안이라 "정확한 절대 3D 자세는 불가"하여 3D 자세를 2D로 투영한 보조 라벨로만 씁니다. 호모그래피로 egocentric 카메라 움직임을 보정하고, 한 스텝에 5 cm 또는 0.5 rad 넘게 움직인 프레임은 제거합니다.
  - 전이 메커니즘: Phantom 방식으로 손 → 말단 자세·그리퍼 개폐를 계산하고, Kinova Gen3 두 대 + Robotiq 2F-85 가상 로봇을 합성, Detectron2 + SAM2 + E2FGVI로 팔 제거. 로봇 데이터는 과제당 50개 양손 시연과 공동 학습.
  - 물리적 타당성 처리: **없음.** 관절점 오류나 운동학 한계로 무효한 행동 프레임을 버리는 것이 전부이며, 이는 도달성 문제이지 grasp 품질 검사가 아닙니다.
  - 핵심 수치: 안 본 장면에서 평균 약 74% vs 최강 비교군 약 12%(초록의 "5~6배"). Stack Pots 데이터 비율 0%→2%, 10%→26%, 50%→47%, 100%→68%.
  - 저자가 밝힌 한계: 깊이가 없어 가림을 처리하지 못해 로봇 픽셀이 물체 위에 잘못 그려지며 "렌더링된 grasp의 현실감이 떨어짐", **"다지 손의 grasp를 평행 조 그리퍼로 리타겟팅하는 것은 불완전한 대응이며 모든 다지 grasp를 처리할 수 없음"**, 손이 가려지면 마지막 행동을 재사용.

- **Robotizing Human Videos with Physically Consistent Interactions** — Ching-Lam Cheng, Shengfeng He, Bin Zhu (Singapore Management University). arXiv 2026-10-05, [arXiv:2610.06137](https://arxiv.org/html/2610.06137v1). **[출처 확인]**
  - 입력·추적: EPIC-Kitchens(VISOR + CaRe-Ego 마스크), MANO 손 메시. 물체 메시나 6D 자세는 복원하지 않고 2D 물체 마스크의 중심만 사용합니다.
  - 전이 메커니즘: LatentAct가 778개 MANO 정점별 접촉 확률을 예측 → 엄지 접촉 중심과 엄지를 마주 보는 비엄지 접촉 중심(법선 임계 0.3)을 두 손가락 그리퍼의 두 접점으로 삼습니다(시간 평활 λ=0.3). 인페인팅 후 깊이를 추정해 로봇보다 가까운 장면 픽셀로 로봇을 가립니다.
  - 물리적 타당성 처리: 제목의 "physically consistent"는 **접촉 기반 grasp 배치와 깊이 기반 가림이라는 기하학적 일관성**을 뜻하며, 동역학이나 물리 시뮬레이션 검증은 없습니다.
  - 핵심 수치(RoboTwin 100회, 4과제 평균): DP-ViT 기준 베이스라인 42.25/16.50(영역 내/밖) → Masquerade 65.50/23.50 → 본 방법 68.00/45.75. 접촉 복원을 빼면 영역 밖 30.25로 하락. 실제 로봇 10회: Pick Up Sponge 9/8, Pick Up Box 4/2.
  - 저자가 밝힌 한계: 하향 grasp는 정밀한 깊이가 필요한데 단안이라 어려움, "더 좋은 시연도 기하학적 관측 가능성 부족을 상쇄할 수 없다", 다지 손은 향후 과제.

- **Mitty: Diffusion-based Human-to-Robot Video Generation** — Yiren Song, Cheng Liu, Weijia Mao, Mike Zheng Shou. arXiv 2025-12-19, [arXiv:2512.17253](https://arxiv.org/abs/2512.17253). **[초록만 확인]** 기존 방법이 "관절점이나 궤적 같은 중간 표현에 의존해 정보 손실과 누적 오차를 만든다"고 비판하고, 행동 라벨이나 중간 추상 없이 사람 시연을 로봇 실행 영상으로 번역합니다. 물리적 타당성 언급 없음.

- **MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training** — Haoyun Li 외 15인. arXiv 2025-09-26, [arXiv:2509.22199](https://arxiv.org/abs/2509.22199). **[초록만 확인]** 비디오 확산 모델(H2R Aligner)로 로봇 시연 영상을 생성하고, egocentric 영상을 호모그래피로 안정화(EgoStabilizer)하며, 손 궤적을 로봇 좌표계로 옮긴 뒤 제약 역기구학으로 "실행 가능하고 떨림이 적은 관절 명령"을 만듭니다. 6개 과제 평균 성공률 14.7% 향상. grasp 품질 검증 언급 없음.

- **Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data** — Ye Wang 외 15인. arXiv 2026-08-03, [arXiv:2608.02580](https://arxiv.org/abs/2608.02580). **[초록만 확인]** 행동 리타겟팅 → 로봇 팔 시각 합성 → 다단계 품질 선별의 3단계로 **15종 로봇 형태**에 대해 18,561시간의 로봇 학습 데이터를 합성했다고 주장. 물리·실행 가능성 검사는 초록에 없음.

- 배경 연구 **[배경지식, 전문 미확인]**: Shadow([arXiv:2503.00774](https://arxiv.org/abs/2503.00774); Phantom의 손 마스크 방식이 여기서 왔다는 점은 [검색 결과](https://arxiv.org/html/2503.00779v2)에서 확인), RoVi-Aug([arXiv:2409.03403](https://arxiv.org/abs/2409.03403)), Mirage([arXiv:2402.19249](https://arxiv.org/abs/2402.19249)), Giving Robots a Hand([arXiv:2307.05959](https://arxiv.org/abs/2307.05959)). 서베이 [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)은 이 계열에 WARPED(2026)와 H2R-Grounder(2025)도 포함시킵니다.

#### 1-C. 객체 중심 표현(점 궤적·흐름·affordance·잠재 행동) 계열

- **Track2Act** — ECCV 2024, [arXiv:2405.01527](https://arxiv.org/pdf/2405.01527). **[검색 요약 확인]** 인터넷 영상에서 학습한 형태 비의존 점 궤적 예측기 + 소량 로봇 데이터로 학습한 잔차 정책. 배치 로봇 전용 데이터 없이 기성 점 추적기로 정답 궤적을 생성합니다.
- **ATM (Any-point Trajectory Modeling)** — [arXiv:2401.00025](https://arxiv.org/abs/2401.00025) **[배경지식, 전문 미확인]**. 2D 점 궤적을 예측해 정책 조건으로 씁니다. 후속 연구들의 흐름 기반 비교군으로 쓰입니다.
- **MT-π (Motion Tracks)** — [arXiv:2501.06994](https://arxiv.org/html/2501.06994v2). **[검색 요약 확인]** 2D 영상 공간 운동 궤적을 교차 형태 행동 공간으로 사용. 몇 분의 사람 영상 + 제한된 로봇 시연.
- **Point Policy** — Siddhant Haldar, Lerrel Pinto. arXiv 2025-02-27, [arXiv:2502.20391](https://arxiv.org/abs/2502.20391). **[초록만 확인]** 사람 손 자세를 로봇 자세로 변환하고 물체 상태는 의미 있는 핵심점으로 표현. 실제 8개 과제에서 선행 대비 약 75% 절대 향상 주장. 그리퍼 개폐 도출 방식은 초록에 없음.
- **3PoinTr: 3D Point Tracks for Learning Manipulation from Unconstrained Human Videos** — Adam Hung, Bardienus Pieter Duisterhof, Jeffrey Ichnowski (CMU). arXiv v2 2026-06-03, [arXiv:2603.08485](https://arxiv.org/html/2603.08485v2). **[출처 확인]**
  - 입력: 3인칭 고정 ZED Mini 스테레오 영상(egocentric 아님), CoTracker3 2D 추적 → FoundationStereo로 3D 승격, SAM3로 "human arm"을 분할해 비가시 처리.
  - 전이: 초기 점군에서 궤적을 한 번 예측해 Perceiver-IO 토큰 4개로 압축, 확산 정책의 FiLM·교차 주의에 조건. 행동은 말단 위치·6D 회전·그리퍼 명령(xArm 7, 평행 조).
  - 물리적 타당성: 별도 grasp 계획이나 검사 없음. **Right Glass 과제에서 로봇은 사람의 줄기 쥐기를 흉내 내지 않고 그리퍼를 잔 안에 넣어 테두리를 들어올림** — 사람 grasp와 다른 전략을 정책이 택한 사례.
  - 수치: 실제 4과제 3PoinTr 75/80 vs DP3 55/80 vs ATM 16/80. 투명·반사·가림 물체에서 점 추적 실패.
- **ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos** — Junyao Shi 외, Dinesh Jayaraman. ICRA 2025, [arXiv:2503.23877](https://arxiv.org/abs/2503.23877). **[초록만 확인]** EPIC-Kitchens에서 "현대적 grasp affordance 검출기"로 그리퍼 grasp를 생성하고 이미지 목표 조건 기술 정책을 증류. 수치는 초록에 없음.
- **HOST: Robots Acquire Manipulation Skills in Seconds from a Single Human Video** — Guangyan Chen 외 12인. arXiv 2026-07-22, [arXiv:2607.20033](https://arxiv.org/abs/2607.20033). **[초록만 확인]** 로봇 궤적과 사람 영상을 "공유 과제 진행 다양체"에 올려 과제 진행 → 미래 관측 → 행동을 예측. 평균 29초 만에 기술 습득, 평균 62% 성공. 실행 가능성 검사 언급 없음.
- **Zero-WAM: In-Context World-Action Modeling from Human Videos** — Jiaming Zhou 외 12인. arXiv 2026-08-26, [arXiv:2608.26103](https://arxiv.org/abs/2608.26103). **[초록만 확인]** 사람 영상을 과제 명세로 삼는 인컨텍스트 비디오-행동 모델. HumanGen(7.42만 사람-로봇 쌍) 학습, RoboTwin 2.0 안 본 7과제 47.0%. 물리적 타당성 언급 없음.
- 잠재 행동(latent action) 계열: 서베이 [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)은 LAPA, IGOR, Moto, GO-1, UniSkill, DreamDojo, villa-X, CLAP 등을 열거하며 "이를 형태별 제어로 접지하려면 여전히 추가 로봇 데이터 감독이 필요하다"고 정리합니다. GR00T N1은 사람 egocentric 영상에 행동 라벨이 없어 연속 프레임 쌍으로 VQ-VAE 잠재 행동을 학습합니다([arXiv:2503.14734](https://arxiv.org/pdf/2503.14734), [검색 요약 확인]).
- 서베이 [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)의 정리: affordance를 정책 조건으로 쓰는 "대표 연구는 모두 평행 그리퍼를 대상"으로 하며, 이 계열은 "힘, 순응성, 접촉 안정성을 시각만으로 추론할 수 없을 때 어려움을 겪는다"고 명시합니다.
- 배경 연구 **[배경지식, 전문 미확인]**: Im2Flow2Act([arXiv:2407.15208](https://arxiv.org/abs/2407.15208)), General Flow([arXiv:2401.11439](https://arxiv.org/abs/2401.11439); 3PoinTr의 궤적 예측 비교군으로 등장), Gen2Act([arXiv:2409.16283](https://arxiv.org/abs/2409.16283)), VidBot([arXiv:2503.07135](https://arxiv.org/abs/2503.07135)), ORION([arXiv:2405.20321](https://arxiv.org/abs/2405.20321)), Vid2Robot([arXiv:2403.12943](https://arxiv.org/abs/2403.12943)), R+X([arXiv:2407.12957](https://arxiv.org/abs/2407.12957)), LAPA([arXiv:2410.11758](https://arxiv.org/abs/2410.11758)), UniVLA([arXiv:2505.06111](https://arxiv.org/abs/2505.06111)), SPOT([arXiv:2411.00965](https://arxiv.org/abs/2411.00965); HumanEgo의 비교군).

#### 1-D. 대규모 VLA 사전학습에 egocentric 인간 영상을 쓰는 계열

- **GR00T N1.5 + FLARE** — NVIDIA GEAR. [공식 연구 페이지](https://research.nvidia.com/labs/gear/gr00t-n1_5/index.html), FLARE 논문 [arXiv:2505.15659](https://arxiv.org/html/2505.15659v1). **[출처 확인]**
  - 메커니즘: FLARE(Future LAtent Representation Alignment)는 미래 프레임을 생성하는 대신 미래 임베딩에 정렬하는 보조 손실(계수 0.2)이며, 이것이 "사람 영상에서 배우는 능력을 열어준다"고 공식 페이지가 밝힙니다. 사람 영상에는 **행동 라벨이 없고 손 추적도 하지 않으며** 정렬 손실만 적용합니다(FLARE 논문: "명시적 자세 추정기나 점 추적 도구가 필요 없다").
  - 수치: 안 본 물체 10종 pick-and-place — GR00T N1 0-shot 0%, N1.5 0-shot 15.0%, 해당 물체가 포함된 사람 영상으로 FLARE 후속 학습 55.0%. FLARE 논문의 실험: 물체당 GoPro 머리 장착 사람 시연 150개 + 원격조작 10개 → 80%(행동 라벨 데이터만의 약 2배). 예시 과제는 "큰 파란 테이프는 위에서 내려잡아야 한다"처럼 **새로운 grasp 전략**을 배우는 것입니다.
  - 물리적 타당성 처리: 없음. grasp 검증은 바구니에 넣었는지(실패 시 0.5점 부분 점수)로만 측정.
  - 한계: 사람 영상은 통제된 환경에서 GoPro로 수집, 주로 pick-and-place, 강화학습 없음. 공식 페이지 표에는 사람 영상 없이 같은 조건으로 후속 학습한 N1.5 수치가 없어 **사람 영상의 기여가 분리되어 있지 않습니다**.

- **DreamGen** — [arXiv:2505.12705](https://arxiv.org/abs/2505.12705) **[배경지식, 전문 미확인]**. 검색 요약에 따르면 이미지→비디오 생성 모델을 대상 로봇 형태에 맞춰 합성 로봇 영상을 만들고 잠재 행동 모델 또는 역동역학 모델로 의사 행동을 복원하는 파이프라인이며, GR00T N1.5의 합성 데이터 생성(GR00T-Dreams)에 쓰였습니다([검색 요약](https://research.nvidia.com/labs/gear/gr00t-n1_5/index.html)). 사람 영상이 아니라 **로봇 영상 합성**이므로 본 조사의 핵심 대상은 아닙니다.

- **HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining** — Juncheng Ma 외 22인(DAGroup-PKU). arXiv 2026-06-18, [arXiv:2606.20521](https://arxiv.org/abs/2606.20521). **[초록만 확인]** 같은 양의 사전학습 데이터·같은 후속 학습 조건에서 egocentric 사람 영상이 원격조작 로봇 궤적보다 나았다고 주장: 검증 손실 24% 감소, 영역 내 성공률 52.5% 상대 향상, 영역 밖 90% 상대 향상. 여전히 "행동 공간 정렬"을 위한 소량의 라벨된 로봇 데이터가 필요.

- **EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video** — Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, Jian Zhang (Apple). ICLR 2026, [arXiv:2505.11709](https://arxiv.org/abs/2505.11709). **[초록만 확인]** Apple Vision Pro로 829시간, 194개 탁상 과제, 보정된 다중 카메라 + 온디바이스 SLAM으로 모든 손 관절 3D 추적. 논문의 자체 실험은 **손 궤적 예측 모방 학습 벤치마크**이며 로봇 실험은 초록에 언급되지 않습니다. H-RDT([arXiv:2507.23523](https://arxiv.org/pdf/2507.23523), [검색 요약 확인])가 EgoDex로 사전학습 후 로봇 데이터로 미세조정합니다.

- 기타 2026년 egocentric 데이터셋·사전학습 연구(제목·검색 요약만 확인): EgoVerse([arXiv:2604.07607](https://arxiv.org/pdf/2604.07607)), EgoLive([arXiv:2604.23570](https://arxiv.org/pdf/2604.23570)), ACE-Ego-0([arXiv:2606.17200](https://arxiv.org/pdf/2606.17200)), Wh0([arXiv:2606.22136](https://arxiv.org/pdf/2606.22136); 생성 세계 모델로 egocentric 손 조작 데이터 합성), MotionTrans([arXiv:2509.17759](https://arxiv.org/html/2509.17759v1); VR 데이터), Being-H0.7([arXiv:2605.00078](https://arxiv.org/html/2605.00078v1); egocentric 영상의 잠재 세계-행동 모델).

#### 1-E. 영상 → 시뮬레이션 → 강화학습 / 시뮬레이션 검증 계열

- **Human2Sim2Robot (Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration)** — Tyler Ga Wei Lum, Olivia Y. Lee, C. Karen Liu, Jeannette Bohg (Stanford). arXiv v3 2025-08, [arXiv:2504.12609](https://arxiv.org/html/2504.12609), [코드](https://github.com/tylerlum/human2sim2robot). **[출처 확인]**
  - 입력·복원: LiDAR 앱으로 장면, Kiri Engine으로 물체 스캔(약 10분). 단안 RGB-D 사람 영상 1편(약 30초). SAM2 마스크 + FoundationPose로 물체 6D 궤적, HaMeR + 깊이 ICP로 손 자세. 물체가 5 cm/s 이상 움직이기 1초 전을 "조작 전 자세"로 추출.
  - 전이: cuRobo(팔, 가운데 손가락 관절 정렬) + PyBullet(손끝 정렬) 2단계 역기구학으로 KUKA iiwa + Allegro 손에 리타겟팅하되, 이는 **RL의 초기 상태 분포로만** 쓰입니다. IsaacGym + PPO, 보상은 물체 앵커점과 목표 궤적 앵커점 거리의 지수 함수(형태 비의존), 관측은 영상이 아닌 물체 자세.
  - 물리적 타당성 처리: 명시적 grasp 성립 검사는 없고 **강체 시뮬레이터 안에서 RL이 grasp를 발견**하게 둡니다. 저자는 일부 정책이 마찰 등 시뮬레이션 부정확성을 악용했다고 적었습니다. 영역 무작위화: 크기·질량·마찰·감쇠·강성 [0.7, 1.3], 5% 확률로 물체 질량의 50배 외력.
  - **운동학적 리타겟팅이 실패하는 이유(논문 명시)**: HaMeR의 3D 오차가 손이 카메라에서 멀면 20~30 cm에 달함, 프레임별 역기구학은 시간 일관성이 없고 자주 도달 불가 목표에 부딪힘, 사람·로봇 형태 차이로 차선의 궤적이 나옴, 이런 잡음 라벨로 학습한 개루프 재생과 행동 복제는 접촉 집약 다단계 과제에서 실패.
  - 수치: 실제 7과제 10회씩, Replay 대비 +67%, 물체 인식 Replay 대비 +55%, 행동 복제 대비 +68%.
  - 한계: 강체 하나·과제 하나, 대칭·반사 물체에서 자세 추적 실패, 정책이 조작 전 자세 근처에서 시작해야 함, 얇은 접시는 쉽게 미끄러져 가장 어려움.

- **Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single Human Video** — Yunhai Han 외 17인(Kenneth Shaw, Matthew Gombolay, Zsolt Kira, Harish Ravichandar 포함). arXiv 2026-06-07, [arXiv:2606.08828](https://arxiv.org/html/2606.08828v1). **[출처 확인]**
  - 입력·복원: 영상 3프레임 + 조작 전 RGB-D 기준 이미지. Gemini가 물체·과제 유형을 JSON으로 파싱, SAM3 마스크, SAM 3D Objects로 메시 복원, 질량·관성은 메시 부피와 재질 라벨에서, 마찰·반발은 재질 표에서 추정. 손은 HaMeR → Mink 미분 가능 역기구학(관절 한계·충돌 제약)으로 Kinova + 16자유도 LEAP Hand에 리타겟팅, 물체는 CoTracker 2D 흐름을 깊이로 3D 승격.
  - **물리적 타당성 처리(이 계열에서 가장 명시적)**: 리타겟팅 궤적을 시뮬레이션에서 재생해 접촉 시 손-물체 변환을 얻고 상향/측면 grasp 규칙으로 손바닥 위치를 보정 → Lightning Grasp로 후보 생성 → 연속성 필터로 인공 접촉점 제거, 탁자 아래 손 링크는 기각 → **접근·사전 grasp·닫기·들어올림·흔들기를 시뮬레이션으로 재생해 물체가 6 cm 이상 들린 채 유지되고 손-탁자 접촉이 없으면 채택**. 논문은 이를 "힘 분석이 아닌 시뮬레이션 안정성 테스트"로 설명합니다. 밀기·당기기는 흐름 방향과 손끝 면 선택으로 처리.
  - sim-to-real: 물체 점군 → 손 자세 잔차를 회귀하는 모방 네트워크 + PPO 잔차 RL(손가락 관절 잔차, 흐름 추적 보상).
  - 수치: 시뮬레이션 보정 성공 91.4% ± 22.7%(안전 100%) vs 최강 비교군(잔차 RL + 보조 보상) 52.9%; 실제 10회씩 평균 95.7% vs FoundationPose 기반 15.7%.
  - 한계: 희소 키프레임에서만 보정하므로 연속 조정·복잡한 손안 재배치 불가, 키프레임 검출은 휴리스틱, 관절 물체 미지원.

- **Real2Render2Real (R2R2R)** — Justin Yu, Max Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus, Richard Cheng, Muhammad Zubair Irshad, Ken Goldberg (UC Berkeley, Toyota Research Institute). arXiv 2025-05-14, [arXiv:2505.09601](https://arxiv.org/html/2505.09601v1). **[출처 확인]**
  - 입력·복원: 스마트폰 다중 시점 스캔 → 3D Gaussian Splatting → GARField 부품 분할 → SuGaR 확장으로 메시. 단안 사람 시연 1편을 4D-DPM으로 부품별 6자유도 추적. HaMeR 손끝에 가장 가까운 부품을 "잡힌 부품"으로 판정.
  - 전이: Dex-Net식 **해석적 대척(antipodal) grasp 샘플러**를 거친 메시 위에서 돌려 후보를 만들고, PyRoki 미분 역기구학으로 관절 궤적 생성. **물체는 접촉 중 그리퍼와 강체로 함께 움직인다고 가정**하며 동역학·접촉 모델링은 없습니다. IsaacLab에서 운동학적 물체로 렌더링(RTX 4090 1장에서 분당 약 51개 시연, 원격조작 1.7개의 27배).
  - 물리적 타당성 처리: **물리 시뮬레이션 없음, 충돌 검사 없음.** 저자가 밝힌 한계: "마찰·순응·힘 피드백이 모델링되지 않음", "충돌 인식 계획이 없어 궤적이 장면과 교차할 수 있음", "대척 grasp 때문에 평행 조 그리퍼에만 적용, 다지 손 제외", 강체·관절 물체의 쥐는 grasp만 지원(변형체·밀기·넘어뜨리기 불가). 수도꼭지 과제에서 원격조작자는 손잡이를 비쥐는 방식으로 밀었지만 R2R2R은 쥐고 돌리는 것만 가능.
  - 수치: ABB YuMi 양팔, 5과제, 물리 평가 1,050회. Mug 과제 π0-FAST: 원격조작 150개 73.3% vs R2R2R 1,000개 80.0%; 같은 150개끼리는 73.3% vs 33.3%로 실제 데이터가 우세. 동등성 검정(TOST ±5%)은 어떤 과제도 통과하지 못해 "유사하지만 동등함을 입증하진 못함".

- **DemoBridge: A Simulation-in-the-Loop Toolkit for Single-View Human Demonstration Retargeting** — Zehao Wang, Fabien Despinoy, Sergey Zakharov, Tinne Tuytelaars, Rahaf Aljundi (GitLab 도메인상 KU Leuven, Toyota 관계자 포함으로 보이나 페이지에 소속 미기재). RSS 2026 RoboData Workshop, arXiv 2026-07-10, [arXiv:2607.09519](https://arxiv.org/html/2607.09519v1). **[출처 확인]**
  - 입력·복원: 단일 시점 RGB 스테레오(TRI-Stereo 또는 FoundationStereo), SAM 분할, RecGen 또는 SAM3D로 수밀(watertight) 메시, MediaPipe Hands 또는 HaMeR 손, Any6D 또는 FoundationPose 물체 6D 자세. 백엔드는 설정으로 교체 가능.
  - 전이·검증: grasp 시점을 손-물체 운동 동조로 자동 추론(rest/grasp/transport/release). **대척 grasp 후보를 시연 손 자세 순으로 정렬한 뒤 Isaac Sim에서 "그리퍼가 도달해 잡고, 물체가 시연 궤적을 따르고, 주변 물체가 움직이지 않음"을 확인해야 해당 단계를 확정**하며, 실패 시 이전 상태로 되돌려 다음 후보를 시도(backtracking)합니다. PyRoKi 기반 전체 궤적 최적화 + RRT-Connect로 충돌 없는 호모토피 클래스 확보.
  - 선행 방법에 대한 평가: RoboWheel은 프레임별 역기구학에 자기충돌만 검사, YOTO는 희소 키프레임 역기구학 보간이며 "둘 다 장면에 대해 팔 전체를 계획하지 않는다". 역기구학 일반에 대해 "대응된 말단 자세의 역기구학은 종종 충돌 없는 해가 없다"고 지적.
  - 수치(실제 시연 3과제, 10회 중 누적 성공): Mug-on-plate 본 방법 reach 8/grasp 8/transport 8/place 6, RoboWheel 재구현 7/3/3/2, YOTO 재구현 6/0/0/0; Cup-stack 9/9/9/7 vs 8/1/1/0 vs 7/3/3/0; Pour-milk 8/8/8/7 vs 10/1/1/1 vs 8/1/1/1. 즉 **비교군은 "도달"은 되지만 "grasp"에서 무너집니다.** 합성 벤치마크(Franka 50장면): 본 방법 98% 성공·충돌 0/50 vs Trajopt 26%·충돌 1/50.
  - 한계: 손 가림으로 물체 추적이 끊기면 기준이 오염됨, 단일 팔만 지원(양팔 계획 미구현), 시연 grasp가 팔 도달 한계 근처면 실패.

- **Generative Neural Retargeting for Human-to-Robot Dexterous Manipulation (GNR)** — Dechen Gao, Yue Yang, Ben Abbatematteo, Nathan Godwin, Pengcheng Wang, Roger Boldu, Steven Man, Zhiyang Dou, Chuan Qin, Sho Nakagome, Eric Whitmire (Meta Reality Labs Research, UC Davis, UNC, UC Berkeley, MIT, CMU). arXiv 2026-10-08, [arXiv:2610.12440](https://arxiv.org/html/2610.12440). **[출처 확인]**
  - 문제 제기(초록 인용): "Inverse kinematics (IK) retargets human motion to robots efficiently but ignores dynamics", 운동학적 궤적은 "often dynamically infeasible".
  - 파이프라인: SuperDex Physics 시뮬레이터(비볼록 충돌 기하, 분산 3D 접촉력). 사람 손 → 시뮬레이션 사람 손 → 관절점 역기구학으로 로봇 참조 생성. **이미 잡힌 상태로 시작하는 구간은 물체를 임시 고정하고 로봇이 손을 벌렸다 다시 쥐어 grasp를 재수립**. DIAL-MPC로 물체 궤적 추종 제어 최적화 후 접촉력 기록.
  - **채택 기준 Γ(명시적 물리 검증)**: (1) 물체 위치·자세 추종 오차 한계, (2) 사전 계산한 부호 거리장(SDF)으로 손가락 캡슐 관통 한계, (3) **최소 지속 쥐는 힘(minimum sustained squeeze force)**, (4) 손가락별 접촉력 상한. 네 가지를 모두 통과해야 궤적 채택.
  - 데이터: **HOT3D**(egocentric, 33 물체, 780 원천 모션)를 36만 개로 증강 → **실행 가능한 로봇 궤적 22.3만 개**(3.3천 기하, 접촉력 라벨 포함). Tesollo DG-5F 손(+Franka FR3). 정밀 과제(ShapeFilter, NIST)는 OptiTrack + 외골격으로 따로 수집했는데, 그 이유는 **"egocentric 영상에서 복원한 사람 동작은 밀리미터 수준 과제에 필요한 정밀도가 없기 때문"**.
  - 모델: 조건부 흐름 정합이 역기구학 참조 제어에 대한 잔차를 예측(시간 U-Net + FiLM, Point Transformer V3 기하 인코더). 후보 S개 중 하나라도 Γ를 통과하면 성공(pass@S).
  - 수치: HOT3D 전체 DIAL-MPC 27.2%(3,000 샘플) vs GNR 56.2%(256 샘플). **ShapeFilter: 역기구학 3.7% vs GNR 99.3%**. NIST(장기 접촉 집약): 개루프 GNR 4.7%, GNR 초기화 MPC 92.2%.
  - 한계: 전부 시뮬레이션(sim-to-real은 향후 과제), MPC 시드 데이터에 의존, 장기 과제는 MPC 보정 필요.

- **Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning (DexGEM)** — Zihao Yang, Chengyuan Liu 외, JieQ Sun (DexGEM Lab, Shanghai Jiao Tong University, Tongji University, DexRobot). arXiv v1 2026-09-21, v2 2026-11-03, [arXiv:2609.24093](https://arxiv.org/html/2609.24093v2). **[출처 확인]**
  - 입력: **영상이 아니라 자체 광학 모션캡처**(적외선 카메라 22대, 손 마커 14개, 120 Hz). 단일 손 25,454 궤적(14 물체), 양손 250 궤적(25 과제).
  - 1단계 물리 보정: Isaac Gym에서 **구동되는 MANO 손**(손가락 22관절 + 6자유도 PD 손목)으로 모션캡처를 재실행하고 잔차 RL로 보정해 프레임별 접촉 라벨(어느 손가락 부위), 물체 표면 접촉 위치, 복원된 힘 벡터를 저장합니다. 보상은 물체 궤적 추종 + 지정 손가락의 물체 표면 도달.
  - 2단계 접촉 앵커 리타겟팅: 지정된 로봇 손끝 표면점을 기록된 MANO 접촉 위치에 맞추는 1 cm 허용 힌지 벌점 + 관통(물체·자기·바닥)·손-물체 거리·관절 한계·시간 떨림 벌점. 접촉 패턴으로 프레임을 분할해 다중 접촉 프레임은 전체 목적함수, 희소 접촉 프레임은 손가락만 최적화. 손 형태별로는 "embodiment profile"(운동학, 제어 결합, 관절 한계, 후보 표면점)만 바뀝니다.
  - 3단계 잔차 RL: PPO, 과제 ID·물체 기하·참조 동작 조건, 손목 이동·손가락 제어 잔차. 성공 기준은 모든 물체가 모션캡처 기준에서 3 cm 이내(자세 제약 과제는 최종 30° 이내).
  - 수치: 단일 손 14물체 평균 **재생 7.26% → 잔차 후 59.31%**; 양손 16.0% → 62.4%; 네 손(DexHand 021, 021Pro, Shadow, Allegro) 평균 17.1% → 79.5%(DexHand 021은 0.5% → 79.0%); 접촉력 조건을 Diffusion Policy에 주면 34.7% → 46.5%. 운동학 리타겟팅 비교에서 벡터 방식은 정합도 10.6 mm로 가장 좋았지만 **관절 한계 위반 29.9%**, 본 방법은 14.7 mm·위반 3.2%. 실제 JAKA Mini 2 두 대 + DexHand 021: 종이컵 쌓기 7/10, 병 건네 세우기 5/10, 물 붓기 6/10, 손안 병 재배치 3/10(로봇 학습 데이터 없음).
  - 저자의 진단: "손가락 수·비율·관절 범위가 다르면 관절·손끝 대응이 무너진다", "운동학 기록에는 힘이 없고, 리타겟팅된 참조는 토크 한계·힘줄 결합·물체 동역학을 무시한다". 관통·미끄러짐 자체의 정량 분석은 없고 벌점 항으로만 다룸.

- **One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry (DemoMimic)** — Satvik Sharma, Samrat Sahoo, Huang Huang, Fei-Fei Li, Jiajun Wu, Dorsa Sadigh, Jeannette Bohg (Stanford). arXiv v2 2026-09-03, [arXiv:2609.01938](https://arxiv.org/html/2609.01938v2). **[출처 확인]**
  - 입력: 단일 사람 시연 — ARCTIC 모션캡처 또는 Meta Quest 손 추적 + FoundationPose 물체 추적. **"기준 물체 자세 궤적이 필요한데, 이를 원시 영상에서 신뢰성 있게 얻기는 어렵다"**고 명시.
  - 전이: 접촉 기하를 별도 표현으로 추출하지 않고 시뮬레이션(Genesis, IsaacLab) RL 보상으로 형성 — 손 링크 법선과 시연 접촉점의 물체 표면 법선 정렬 보상, 시연 접촉 구간 동안 지속 접촉 보상, 접촉 중인 관절점에 역기구학 가중치. 물체를 비등방 스케일링해 변형 생성. 추론 시 손목 깊이 카메라(0.4 m 클리핑) 저수준 정책이 국소 상호작용 기하를 관측.
  - 물리적 타당성: 명시적 grasp 품질 지표 없음. 시뮬레이션 RL 성공과 실제 낙하율로 간접 평가.
  - 수치: Tesollo DG-5F(15자유도)·Sharpa(22자유도) 양팔 Franka, 16물체 4과제 실제 평균 71%(Sharpa 76%, Tesollo 65%). 로봇 손 모양 상자는 39%(국소 접촉 기하가 다름). HERMES, DexMachina 비교군은 시뮬레이션은 좋지만 실기에서 낙하율 높음.

- **AdaDexGrasp: Learning Adaptive Dexterous Grasping from Single Demonstrations** — Liangzhi Shi 외, Hao Su. arXiv 2025-03 (v2 2025-08), [arXiv:2503.20208](https://arxiv.org/abs/2503.20208). **[초록만 확인]** 단일 사람 시연에서 grasp 기술을 배우고 RL로 보정, PSYONIC Ability Hand 실제 90%. 검색 요약([링크](https://arxiv.org/pdf/2503.20208))에 따르면 DexYCB 사람 영상의 손끝 위치 매핑 + 팔 역기구학 → 궤적 유도 RL.

- **V2P-Manip: Learning Dexterous Manipulation from Monocular Human Videos** — Kaihan Chen, Yanming Shao, Haifeng Ji, Xiaokang Yang, Yao Mu. arXiv 2026-06-15, [arXiv:2606.16436](https://arxiv.org/abs/2606.16436). **[초록만 확인]** 3D 자산 획득 → 궤적 추정 → 다지 정책 학습, "공간 정렬과 물리적 일관성을 강제하는 2단계 보정". TACO·OakInk 벤치마크, 합성 과제 평균 75% 이상. 접촉 물리 검사 방식은 초록에 없음.

- **ReForce: Learning Force-aware Retargeting for Dexterous Manipulation** — Yuhang Wu, Lingqi Zeng, Changwei Jing, Jianglong Ye, Xiaolong Wang (UC San Diego). arXiv 2026-08-16, [arXiv:2608.15560](https://arxiv.org/html/2608.15560v1). **[출처 확인]**
  - 주장(초록): "Today's retargeting is mostly kinematic, yet manipulation is decided by force."
  - 입력: **영상이 아님** — Quest 컨트롤러(손목) + Manus 장갑(손) + 손끝 FSR 센서 5개(사람 접촉력).
  - 방법: Dex1B 자산·grasp로 만든 시뮬레이션 상호작용에서 범용 힘 추적기를 학습, 손가락별 MLP가 운동학 리타겟팅 위에 관절 명령 잔차(스텝당 약 5° 제한)를 예측. 12자유도 XHand + xArm, 손끝 촉각으로 힘 측정.
  - 수치: 재생 힘 오차(N) 측면 grasp 0.309 → 0.247, 상향 grasp 0.736 → 0.474. 종이컵 "힘 안전 성공" 참조 정책 30% → 어드미턴스 60% → ReForce 70%, 심각한 접촉 누락 7/10 → 0/10, 평균 활성 손가락 0.17 → 2.61. 집게(tongs)는 어드미턴스가 더 좋았음(90% vs 54.5%).
  - 한계: 법선 방향 손끝 힘만 다루며 전단·미끄러짐·접촉 분포는 모델링하지 않음.

- **HOWTransfer: Hand-centric Human-to-Robot Trajectory Transfer from Video Demonstrations via Open-World Contact Localization** — Yitian Shi 외, Rania Rayyes (KIT). arXiv 2026-06-09, [arXiv:2606.10743](https://arxiv.org/html/2606.10743v1). **[출처 확인]** 저가 스테레오(3인칭) 영상, WiLoR 손 + 스테레오 삼각측량 + SE(3) 평활. SAM3 마스크로 물체 이름 없이 접촉 구간 추정. HOGraspFlow가 grasp 분류 사전 지식으로 평행 조 SE(3) grasp 후보를 샘플링 → DBSCAN 군집 → 손목 상대 변환으로 전파. **충돌 처리는 "전체 물리나 폐루프 재계획이 아닌 국소 여유 휴리스틱"**, 힘·동역학 검사 없음. 11과제 110영상 재생 성공 86%(템플릿 grasp 대비 +23점). 장기 과제(Pot Cooking, Breakfast)에서 접촉 전환마다 오차 누적.

- **You Only Teach Once (YOTO)** — Huayi Zhou 외, Kui Jia (CUHK-Shenzhen, HIT Weihai, DexForce). arXiv 2025-01-24, [arXiv:2501.14208](https://arxiv.org/html/2501.14208v1). **[출처 확인]** 고정 양안 카메라, WiLoR + MANO, 스테레오로 3D 승격. 약 10개 키프레임으로 축약하고 "motion mask"로 양팔(Aubo i5 + DH 평행 그리퍼) 동기·비동기 협응을 기록. 그리퍼 상태는 접촉 검출로 결정. **운동학·충돌 검사 없이 각 키프레임을 실제 로봇에서 약 3분간 재생해 수동 검증, 실패 키프레임 제거.** 1회 시연을 약 100배 증식해 BiDP 학습: 영역 내 76.8%(ACT 5.7%, DP 15.8%, DP3 19.4%), 안 본 물체 35.0%.

- 휴머노이드 전신 계열(대형 물체 관련, 핵심 질문 5에서 재인용): HDMI([arXiv:2509.16757](https://arxiv.org/abs/2509.16757), CMU; 단안 RGB → 사람·물체 궤적 → Unitree G1 RL 공동 추종, 문 통과 67회 연속) **[초록만 확인]**, SUGAR([arXiv:2605.20373](https://arxiv.org/abs/2605.20373), 2026-05; 영상 유래 사전 지식의 "가림, 접촉 인공물, 리타겟팅 오류"를 **특권 정보 기반 물리 보정기**로 실행 가능하게 만든 뒤 계층 정책으로 증류) **[초록만 확인]**.

- 배경 **[배경지식, 전문 미확인]**: Video2Policy([arXiv:2502.09886](https://arxiv.org/abs/2502.09886); R2R2R 본문이 "시각 불일치 때문에 시험 시 물체 검출이 필요"하다고 언급 — [출처](https://arxiv.org/html/2505.09601v1)).

### Inferences
- **[추론]** 1-A, 1-B, 1-C, 1-D 계열은 공통적으로 grasp를 "엄지-검지 거리 임계값"(EgoZero, HumanEgo, Phantom, Masquerade) 또는 "로봇 데이터로만 감독"(EgoMimic, Being-H0, EgoVLA, GR00T N1.5)으로 처리합니다. 즉 사람 grasp의 접촉 구조는 두 점(엄지·검지)으로 붕괴되고, 그 두 점이 실제로 물체를 지탱하는지는 학습 후 실기 롤아웃에서만 드러납니다. 이는 사용자의 가설("주류 방법은 리타겟팅된 grasp의 물리적 성공을 검사하지 않는다")을 지지합니다.
- **[추론]** 1-E 계열은 가설의 반례처럼 보이지만, 검증의 형태를 보면 (가) 시뮬레이션 롤아웃 안정성(Video2Sim2Real의 6 cm 들어올림 + 흔들기, DemoBridge의 "잡고 운반하며 주변이 안 움직임"), (나) RL 성공률(Human2Sim2Robot, DexGEM, DemoMimic), (다) 힘 임계 기반 채택 기준(GNR의 최소 지속 쥐는 힘·손가락별 힘 상한·SDF 관통)의 세 가지이며, **힘 폐쇄 여부나 렌치 공간 여유(ℓ∞ margin) 같은 해석적 grasp 품질 지표를 영상 유래 grasp에 적용해 보고한 논문은 이번 조사에서 찾지 못했습니다.** GNR의 Γ 기준이 가장 가깝지만 "쥐는 힘이 임계 이상"이라는 스칼라 검사이지 외란 방향 전체에 대한 보장은 아닙니다.
- **[추론]** DemoBridge의 비교군 수치(reach는 6~10/10이지만 grasp는 0~3/10)와 DexGEM의 재생 성공 7.26%, GNR의 역기구학 3.7%는 서로 다른 세 그룹이 독립적으로 "운동학적 리타겟팅만으로는 grasp가 거의 성립하지 않는다"는 정량 증거를 2026년에 내놓은 것으로, 가설의 전제를 강하게 뒷받침합니다.
- **[추론]** 2026년 하반기 들어 접촉을 명시적 표현으로 올리는 흐름(Robotizing Human Videos의 접촉 정점 → 그리퍼 접점, DexGEM의 접촉 앵커, HOWTransfer의 접촉 구간, DemoMimic의 접촉 법선 보상)이 뚜렷하지만, 이들 중 영상(특히 egocentric) 입력 + 다지 손/다중 팔 + 물리 검증을 동시에 만족하는 것은 없습니다(DexGEM은 모션캡처, Robotizing은 두 손가락 그리퍼·물리 없음, DemoMimic은 물체 자세 궤적 필요).

### Gaps
- OpenReview의 공식 리뷰 본문(Phantom CoRL 2025, EgoZero, Masquerade 등)은 검색으로 찾지 못했습니다. 리뷰어가 지적한 한계는 포함하지 못했으며, 위의 "한계"는 모두 저자 본인이 논문에 적은 것입니다.
- Point Policy, ZeroMimic, HOST, Zero-WAM, Mitty, Ego2Robot, V2P-Manip, HumanScale, AdaDexGrasp, HDMI, SUGAR는 초록 수준에서만 확인했으므로 grasp 처리 세부와 한계 절은 전문 확인이 필요합니다.
- "HOP", "RHyME", "Vid2Sim", "RoboVerse/Genesis 디지털 트윈", "NVIDIA video_to_data"는 이번 조사에서 1차 출처를 확인하지 못해 수록하지 않았습니다.
- π 시리즈(π0, π0.5)가 egocentric 사람 영상을 사전학습에 쓰는지는 조사하지 못했습니다. EgoWild2Dex 비교군으로 π0·π0.5가 등장하는 것만 확인했습니다.

---

## 핵심 질문 2. 각 메커니즘은 로봇 grasp의 물리적 실행 가능성을 어디서, 어떻게 확인하는가. 형태 불일치로 인한 실패를 명시적으로 논한 논문은 무엇인가

### Takeaway
확인 방식은 네 층위로 나뉩니다: (1) 실제 로봇 롤아웃의 과제 성공률만(대부분), (2) 수집·변환 단계의 운동학 검사(도달성·관절 한계·충돌; ARCap, Masquerade의 프레임 제거, MimicDreamer의 제약 역기구학, DemoBridge의 충돌 인식 계획), (3) 시뮬레이션 안에서의 grasp 재수립·안정성 재생(Video2Sim2Real, DemoBridge, GNR, DexGEM, SUGAR), (4) 힘 수준 검증(GNR의 쥐는 힘·접촉력 한계, ReForce의 힘 추적, DexGEM의 접촉력 복원). 학습된 성공 예측기로 grasp를 선별하는 방식은 이번 조사에서 발견하지 못했습니다.

### Cited Findings

아래 표는 위 1절의 출처를 그대로 재인용한 것입니다(각 행의 근거 링크는 1절 해당 항목 참조).

| 논문 | 검증이 이루어지는 곳 | 무엇을 검사하는가 | 형태 불일치(morphology mismatch) 실패를 명시했는가 |
|---|---|---|---|
| EgoMimic ([출처](https://arxiv.org/html/2410.24221)) | 실제 로봇 롤아웃만 | 과제 성공률 | 봉지 손잡이 grasp 실패, 셔츠 한쪽만 잡음 등 실패 사례 기록; 형태 차이 원인 분석은 없음 |
| EgoZero ([출처](https://arxiv.org/html/2505.20290)) | 실제 로봇 롤아웃만 | 과제 성공률 | 행동 라벨 오차 1~2 cm로 고정밀 과제 불가, 평행 그리퍼 한정 명시 |
| HumanEgo ([출처](https://arxiv.org/html/2605.24934)) | 실제 로봇 롤아웃만 | 과제 성공률(40회/과제) | 점 기반 비교군이 grasp에서 실패, 준정적 가정 위반(미끄러짐·재파지) 명시, 다지 손 미검증 |
| DexWild ([출처](https://arxiv.org/html/2505.07813v1)) | 실제 로봇 롤아웃만 | 과제 성공률 | 사람-로봇 행동 간극을 로봇 데이터로 메움, 접촉 집약 과제 한계 명시 |
| EgoVLA ([출처](https://arxiv.org/html/2507.12440)) | 시뮬레이션 과제 성공률만(실기 없음) | 과제 성공률·진행률 | zero-shot 0%의 원인으로 "외형·지각·운동학 불일치" 명시 |
| Being-H0 ([출처](https://arxiv.org/html/2507.15597v1)) | 실제 로봇 롤아웃만 | 과제 성공률(20회/과제) | 물체 모델링 없음, 힘·마찰·촉각 부재 명시 |
| EgoWild2Dex ([출처](https://arxiv.org/html/2609.23755)) | 실제 로봇 롤아웃만 | 과제 성공률 | 사람 시연만으로는 다지 협응 불충분 명시 |
| Phantom ([출처](https://arxiv.org/html/2503.00779v2)) | 실제 로봇 롤아웃만 | 과제 성공률(25회) | "로봇이 사람 전략을 따를 수 있어야 함", 손끝·그리퍼 표면 물성 차이, pinch grasp 한정 명시 |
| Masquerade ([출처](https://arxiv.org/html/2508.09976v1)) | 운동학 한계로 무효 프레임 제거 + 실기 롤아웃 | 도달성(프레임 단위), 과제 성공률 | "다지 grasp → 평행 조 리타겟팅은 불완전하며 모든 grasp를 처리 못 함" 명시 |
| Robotizing Human Videos ([출처](https://arxiv.org/html/2610.06137v1)) | 기하학적 일관성(접촉 기반 접점 배치, 깊이 가림) + 시뮬레이션/실기 과제 성공률 | 물리 시뮬레이션 검증 없음 | 하향 grasp의 깊이 관측 부족 명시 |
| GR00T N1.5 / FLARE ([출처](https://research.nvidia.com/labs/gear/gr00t-n1_5/index.html), [출처](https://arxiv.org/html/2505.15659v1)) | 실제 로봇 롤아웃만 | pick-and-place 성공(부분 점수) | 없음 |
| 3PoinTr ([출처](https://arxiv.org/html/2603.08485v2)) | 실제 로봇 롤아웃만 | 과제 성공률 | 정책이 사람과 다른 grasp 전략(잔 안쪽 들기)을 택한 사례 기록 |
| Real2Render2Real ([출처](https://arxiv.org/html/2505.09601v1)) | 없음(물리 시뮬레이션·충돌 검사 모두 없음) + 실기 롤아웃 | 대척 grasp 샘플만 | "마찰·순응·힘 피드백 미모델링", "궤적이 장면과 교차 가능", 평행 조 한정 명시 |
| Human2Sim2Robot ([출처](https://arxiv.org/html/2504.12609)) | 시뮬레이션 RL이 grasp를 발견 + 영역 무작위화 + 실기 | RL 성공(물체 궤적 추종) | 역기구학 리타겟팅의 20~30 cm 오차·도달 불가·형태 차이 명시, 시뮬레이션 마찰 악용 사례 |
| Video2Sim2Real ([출처](https://arxiv.org/html/2606.08828v1)) | 시뮬레이션 안정성 테스트(6 cm 들어올림 + 흔들기, 손-탁자 접촉 없음) + 잔차 RL + 실기 | 들어올림 유지 여부(힘 분석 아님), 탁자 아래 링크 기각 | 상향/측면 grasp 규칙으로 손바닥 보정 |
| DemoBridge ([출처](https://arxiv.org/html/2607.09519v1)) | Isaac Sim 단계별 검증 + backtracking + 충돌 인식 전체 궤적 최적화 | 그리퍼가 잡고 유지, 물체가 시연 궤적 추종, 주변 물체 정지 | "역기구학은 종종 충돌 없는 해가 없음", 비교군은 grasp 단계에서 붕괴(0~3/10) |
| GNR ([출처](https://arxiv.org/html/2610.12440)) | SuperDex Physics에서 grasp 재수립 + 채택 기준 Γ | 추종 오차, SDF 관통, 최소 지속 쥐는 힘, 손가락별 힘 상한 | 역기구학 3.7% vs 99.3%; "운동학은 동역학을 무시해 실행 불가" 명시 |
| DexGEM ([출처](https://arxiv.org/html/2609.24093v2)) | Isaac Gym 구동 MANO 손 재실행 + 접촉 앵커 최적화(관통 벌점) + 잔차 RL | 물체 3 cm/30° 추종, 관통·관절 한계·떨림 | 손가락 수·비율·관절 범위 차이로 대응 붕괴, 토크 한계·힘줄 결합 무시 명시; 운동학 비교군 관절 한계 위반 29.9% |
| DemoMimic ([출처](https://arxiv.org/html/2609.01938v2)) | 시뮬레이션 RL(접촉 법선 정렬·지속 접촉 보상) + 실기 낙하율 | RL 성공, 낙하율 | 로봇 손 모양 상자에서 39%로 하락(국소 접촉 기하 차이) |
| ReForce ([출처](https://arxiv.org/html/2608.15560v1)) | 시뮬레이션 학습 힘 추적기 + 실기 촉각 | 힘 추적 오차, 접촉 누락, 과도 힘 | "운동학은 힘을 무시" 명시; 전단·미끄러짐 미모델링 |
| HOWTransfer ([출처](https://arxiv.org/html/2606.10743v1)) | 국소 여유 휴리스틱 + 실기 재생 | 재생 성공 | 평행 조 한정, 접촉 전환마다 오차 누적 |
| YOTO ([출처](https://arxiv.org/html/2501.14208v1)) | 실제 로봇에서 키프레임을 수동 재생 검증 | 재생 가능 여부 | 시각 손 추출 오차로 수동 검증 필요, 평행 그리퍼 한계 |
| ARCap ([초록](https://arxiv.org/abs/2410.08464)) | 수집 시점 증강현실 피드백 | 로봇 운동학 적합, 장면 충돌 회피 | — |

추가로 서베이가 정리한 실패 유형(출처 확인):
- 서베이 "Robot Learning from Human Videos: A Survey"는 행동에 가까운 방법은 "mis-grounding"으로 실패한다고 규정합니다 — 영상에서는 그럴듯한 궤적·잠재 행동이 "대상 로봇의 운동학, 제어 주기, 충돌 제약, 접촉 역학을 위반"한다는 뜻입니다. 또 "RL remains indispensable when the embodiment gap is large", 잠재 행동은 "암묵적 성격 때문에 제어 가능성과 물리적 접지를 검증하기 어렵다", 관측 정렬 방법은 "정렬된 이미지나 임베딩이 정렬된 행동을 뜻하지 않는 거짓 시각 동등성"을 겪는다고 적었습니다. — [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)
- 같은 서베이는 ManipTrans가 "시연된 손이 물체 근처일 때 접촉력 보상을 추가해 더 안정적인 grasp를 유도"하고, Hsieh 외가 접촉 사전 보상으로 리타겟팅 동작을 보정한다고 소개합니다. — [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)
- 서베이 "From Human Videos to Robot Manipulation(IJCAI 2026 Survey Track)"은 리타겟팅이 "본질적으로 과소 제약(intrinsically under-constrained)"이며 "서로 다른 사람 동작이 같은 로봇 명령으로 대응될 수 있고 일부는 실행 가능한 로봇 실현이 없다"고 적고, "단순 그리퍼는 형태 불일치를 넓힌다"고 명시합니다. 그러나 이 서베이는 **체계적인 grasp·접촉 실행 가능성 검사를 기술하지 않으며**, 힘·토크 감지는 논의하지 않습니다. — [arXiv:2606.00054](https://arxiv.org/html/2606.00054v1)
- HOI 서베이 "Hand-Object Interaction in the Age of Large Foundation Models"는 "접촉 범위와 관통 벌점은 실행 안정성의 불완전한 대리 지표이며, 이 때문에 체화된 검증(embodied validation)이 필요하다", "의미적 그럴듯함은 물리적 타당성을 뜻하지 않으므로 시뮬레이터/로봇 검증 루프로 접지해야 한다"고 적었습니다. — [arXiv:2607.28394](https://arxiv.org/html/2607.28394)

### Inferences
- **[추론]** 검증이 "있다"고 분류되는 논문들도 대부분 **과제 수준의 결과(물체가 궤적을 따랐는가, 들려 있는가)**를 검사하지, grasp 자체의 품질(접촉점 배치가 외란에 얼마나 강건한가)을 독립 지표로 평가하지 않습니다. 따라서 "grasp가 성공했지만 여유가 거의 없는" 경우와 "안정적인" 경우를 구분하지 못합니다. 사용자의 α(힘 폐쇄)·ρ(ℓ∞ 강건성 여유) 지표는 이 공백을 메우는 위치에 있습니다.
- **[추론]** 형태 불일치의 양적 증거가 가장 명확한 곳은 "같은 사람 시연 → 여러 로봇 손"을 비교한 DexGEM(DexHand 021 재생 0.5% vs Allegro 50.0%)과 DemoMimic(Sharpa 76% vs Tesollo 65%)입니다. 즉 손 형태가 결과를 지배하지만, 어떤 접촉 배치가 어느 형태에서 성립하는지를 사전에 예측하는 장치는 두 논문 모두 두지 않고 RL에 맡깁니다.
- **[추론]** 학습된 grasp 성공 예측기(learned success prediction)로 영상 유래 grasp를 선별하는 사례는 없었습니다. ZeroMimic과 HOWTransfer가 grasp 검출기·생성기(affordance detector, HOGraspFlow)를 쓰지만, 이는 "그럴듯한 그리퍼 자세 제안"이지 성공 여부 예측이 아닙니다.

### Gaps
- GNR의 Γ 임계값(Table 4)과 NIST 결과의 불일치(본문 92.2% vs Table 2 89.1%)는 전문 후반부를 읽지 못해 해소하지 못했습니다.
- SUGAR·HDMI의 물리 보정기가 손 접촉(grasp)까지 다루는지, 전신 접촉(안기·밀기)만 다루는지는 초록으로는 판단할 수 없습니다.
- Human2Sim2Robot 과제별 성공률(Figure 6)은 본문 텍스트에 없어 수록하지 못했습니다.

---

## 핵심 질문 3. 2025~2026년 논문과 서베이가 꼽는 미해결 문제: 특히 grasp·접촉의 embodiment gap과 "그리퍼 대상 vs 다지 손·다중 팔 대상"의 차이

### Takeaway
서베이와 개별 논문이 공통으로 꼽는 미해결 문제는 (1) 리타겟팅의 과소 제약성과 실행 불가능한 대응, (2) 시각만으로 추론할 수 없는 힘·순응성·접촉 안정성, (3) 로봇 데이터 없이는 닫히지 않는 행동 공간 간극, (4) 다지 손을 위한 고차원 표현과 그리퍼를 위한 희소 표현 사이의 분열, (5) 벤치마크·에피소드화 부재입니다. 다중 팔 협응은 어느 서베이도 과제로 분석하지 않았습니다.

### Cited Findings
- "Robot Learning from Human Videos: A Survey"(Junyi Ma 외, Shanghai Jiao Tong University·University of Cambridge, arXiv 2026-04-30)의 5장 제목: Physically Grounded World Models, Physics-Aware Functional Affordance Extraction, Continual learning with human video data, Multi-agent interaction, Multimodal signals accompanying human videos, Utilization of low-quality human videos, Standardized benchmarking, Egocentric data ecosystems. 본문 핵심 문장: "the LfHV community has not fully eliminated the need for real-world robot data", 사람 영상은 "cannot yet completely replace real robot data in general for the development of generalist agents", affordance 기반 전이는 "struggles when force, compliance, or contact stability cannot be inferred from vision alone", "egocentric usage significantly dominates the recent LfHV studies in 2026". — [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)
- 같은 서베이의 그리퍼 대 다지 손 구분: "다지 손 방법은 고차원 표현(MANO 파라미터, 관절각, 손끝 위치)을, 그리퍼 방법은 희소한 궤적·흐름·접촉점·마스크를 쓰는 경향"이 있으며, 이를 "형태 복잡도와 affordance 간결성 사이의 트레이드오프"로 규정합니다. 정책 조건용 affordance 연구는 "모두 평행 그리퍼 대상". — [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)
- "From Human Videos to Robot Manipulation: A Survey on Scalable VLA Learning with Human-Centric Data"(Zhiyuan Feng 외, Tsinghua·HKUST·Microsoft Research Asia 등, IJCAI 2026 Survey Track)의 5장 미해결 문제: (1) 에피소드화 — 웹 영상은 학습용 에피소드로 포장되어 있지 않고 egocentric 영상은 "다듬어지지 않았고 의도가 섞여 있으며 이산 토큰화를 불안정하게 함"; (2) 이질성 — 형태 불일치에는 "형태 불변 감독(상호작용 의도, 물체 중심 효과)과 형태 조건 접지", 시점 불일치에는 "물체 상태 변화 같은 결과에 고정된 행동 표현" 제안; (3) 벤치마크 — LIBERO·CALVIN·SIMPLER는 인터넷 규모 다양성을 대표하지 못하며, "로봇 데이터 예산 대비 성능, 안 본 물체·장면, 시점·형태 변화 강건성"을 보고하고 사람 영상 사전학습 유무의 짝지은 비교를 권고. 양손 연구(H-RDT, ARCTIC, TACO)는 등장하지만 **다중 팔 협응은 과제로 분석되지 않음**, 힘·토크는 논의되지 않음. — [arXiv:2606.00054](https://arxiv.org/html/2606.00054v1)
- HOI 서베이의 8장 미해결 과제 제목: integrated and verifiable HOI foundation systems; geometric to interaction correctness; long-horizon, dynamic-camera, world-space HOI; prior reliability, routing, conflict resolution; HOI analysis to dynamic embodied memory; robot-centric understanding of human HOI. — [arXiv:2607.28394](https://arxiv.org/html/2607.28394)
- 개별 논문이 꼽은 향후 과제(1절 출처 재인용): Being-H0는 "깊이·촉각 피드백, 시뮬레이션·RL 통합"([출처](https://arxiv.org/html/2507.15597v1)); DexWild는 "리타겟팅 개선 또는 온라인 정책 적응, 촉각·햅틱"([출처](https://arxiv.org/html/2505.07813v1)); EgoZero는 "다지 손·양손 확장"([출처](https://arxiv.org/html/2505.20290)); Robotizing Human Videos는 "다지 손, 복잡한 물체 기하, 비제약 영상"([출처](https://arxiv.org/html/2610.06137v1)); GNR은 "sim-to-real"([출처](https://arxiv.org/html/2610.12440)); DemoBridge는 "양팔 계획"([출처](https://arxiv.org/html/2607.09519v1)); FLARE는 "자연 환경 egocentric 데이터, 세밀한 다지 조작, RL"([출처](https://arxiv.org/html/2505.15659v1)).
- 성능 수치가 보여주는 간극: DexWild는 사람 데이터만으로는 3.6%, 로봇 데이터를 섞어야 79.8%([출처](https://arxiv.org/html/2505.07813v1)); EgoVLA는 로봇 후속 학습 없이는 0%([출처](https://arxiv.org/html/2507.12440)); EgoWild2Dex는 같은 데이터로 π0 3.3%, π0.5 6.7%, GR00T-N1.7 0.0%([출처](https://arxiv.org/html/2609.23755)); Phantom은 사람 데이터만 100개로 0.64 vs 로봇 100개 0.88([출처](https://arxiv.org/html/2503.00779v2)).

### Inferences
- **[추론]** 두 서베이 모두 "접촉 역학"을 불일치 원천으로 거명하면서도 해결책을 "형태 불변 감독(물체 상태 변화)"이나 "RL 결합"으로 제시하지, grasp 품질을 직접 계산하는 접촉 역학적 중간 표현(접촉점 집합 + 렌치 공간)을 제안하지는 않습니다. 이는 사용자의 접근(사람 grasp → 접촉 앵커 → 렌치 품질로 로봇 배치 채점)이 서베이가 비워 둔 자리를 노린다는 해석을 뒷받침합니다.
- **[추론]** 그리퍼 대상 연구가 다수인 구조적 이유는 "엄지-검지 두 점"으로 사람 grasp를 붕괴시키면 평행 조에는 바로 대응되기 때문이며, 다지 손이나 다중 팔로 가는 순간 "어느 접촉을 어느 손가락/팔에 배정하는가"라는 조합 문제가 생깁니다. DexGEM이 이를 접촉 앵커 최적화로, GNR이 생성 모델 + 물리 채택으로 풀기 시작했지만 둘 다 단일 손 또는 양손 수준이며 팔 여러 대로의 배정은 다루지 않습니다.

### Gaps
- 서베이 2604.27621의 5장 본문(특히 Physically Grounded World Models, Physics-Aware Functional Affordance Extraction 단락)은 세 차례 범위를 나눠 읽었음에도 추출되지 않아 제목만 수록했습니다. PDF 직접 열람이 필요합니다.
- "learning from human videos" 관련 2025~2026 서베이 중 "From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data"([arXiv:2604.04974](https://arxiv.org/html/2604.04974v3))와 "Learning Human-Intention Priors from Large-Scale Human Demonstrations"([arXiv:2604.24681](https://arxiv.org/pdf/2604.24681))는 제목만 확인했습니다.

---

## 핵심 질문 4. 어떤 방법이 물체 메시·3D 자세 없이 자연 환경 egocentric 영상에서 바로 작동하며, 어떤 방법이 3D 복원을 요구하는가. 손 3D는 어디서 얻는가

### Takeaway
물체 3D 없이 작동하는 방법(EgoZero, HumanEgo, EgoMimic, Masquerade, Robotizing Human Videos, GR00T N1.5/FLARE, 3PoinTr, HOST 등)은 예외 없이 **평행 조 그리퍼 대상이거나 물리 검증이 없으며**, 물리 검증을 하는 방법(Human2Sim2Robot, Video2Sim2Real, DemoBridge, GNR, DexGEM, DemoMimic, R2R2R)은 예외 없이 **물체 메시 또는 6D 자세 궤적을 요구**합니다. 손 3D는 HaMeR(단안 RGB)·WiLoR·Aria MPS·Apple Vision Pro·VR 헤드셋 추적·장갑·모션캡처로 얻으며, 단안 추정의 절대 깊이 오차(수 cm~수십 cm)가 반복적으로 병목으로 지목됩니다.

### Cited Findings

물체 3D 복원이 필요 없는 방법과 그 대가:
- EgoZero: 정지 물체를 다중 시점 삼각측량으로 점을 얻으므로 "물체 추적이 불가능"하고, 단안 metric depth로 대체하면 0/15. — [출처](https://arxiv.org/html/2505.20290)
- HumanEgo: 물체 자세 없이 egocentric 손 궤적만으로 정책 학습, 단 준정적 가정. — [출처](https://arxiv.org/html/2605.24934)
- Masquerade: 단안 EPIC-Kitchens, 깊이 없음 → 가림 미처리로 "렌더링된 grasp의 현실감 저하". — [출처](https://arxiv.org/html/2508.09976v1)
- Robotizing Human Videos: 물체 메시·6D 자세 없이 2D 마스크 중심만 사용, 인페인팅 후 깊이를 추정해 가림 처리. — [출처](https://arxiv.org/html/2610.06137v1)
- GR00T N1.5/FLARE: 손 추적·점 추적 없이 미래 임베딩 정렬만. — [출처](https://arxiv.org/html/2505.15659v1)
- 3PoinTr: 스테레오 점 궤적만, 물체 모델 없음(단 3인칭 고정 카메라). — [출처](https://arxiv.org/html/2603.08485v2)
- Being-H0: 물체 6D 자세를 "이번 버전에서는 넣지 않음". — [출처](https://arxiv.org/html/2507.15597v1)

물체 3D(메시 또는 자세 궤적)를 요구하는 방법:
- Human2Sim2Robot: Kiri Engine 물체 스캔 + FoundationPose 6D 궤적. — [출처](https://arxiv.org/html/2504.12609)
- Video2Sim2Real: SAM 3D Objects로 메시 복원, CoTracker 흐름 → 물체 자세(Kabsch + RANSAC). — [출처](https://arxiv.org/html/2606.08828v1)
- R2R2R: 스마트폰 다중 시점 스캔(3DGS → 메시) + 4D-DPM 부품 추적; "복원이 수밀 기하가 아닌 경우가 많음". — [출처](https://arxiv.org/html/2505.09601v1)
- DemoBridge: RecGen/SAM3D 수밀 메시 + Any6D/FoundationPose; 손 가림 시 물체 추적 손실이 주 약점. — [출처](https://arxiv.org/html/2607.09519v1)
- GNR: HOT3D의 물체 메시·6D 자세(모션캡처 정답)를 사용. — [출처](https://arxiv.org/html/2610.12440); HOT3D 자체는 Aria·Quest 3로 833분, 33개 강체, UmeTrack·MANO 손 자세, 6DoF 물체 자세, PBR 메시, 광학 마커 모션캡처 정답(CVPR 2025) — [arXiv:2411.19167](https://arxiv.org/abs/2411.19167)
- DemoMimic: "기준 물체 자세 궤적이 필요하며 원시 영상에서 신뢰성 있게 얻기 어렵다". — [출처](https://arxiv.org/html/2609.01938v2)
- DexGEM: 영상이 아닌 모션캡처(물체 포함). — [출처](https://arxiv.org/html/2609.24093v2)

손 3D 획득 경로(확인된 것만):
- HaMeR(단안 RGB → MANO): Phantom(깊이 ICP 보정), EgoZero(Aria 손바닥 자세로 보정), Masquerade(절대 3D 불가하여 2D 보조 라벨로만), Human2Sim2Robot(깊이 ICP; 멀면 20~30 cm 오차), R2R2R(잡힌 부품 판정용), Video2Sim2Real, Being-H0(의사 라벨), DemoBridge(선택지). — 각 1절 출처
- WiLoR: HOWTransfer(스테레오 삼각측량 병용), YOTO(스테레오 승격). — [출처](https://arxiv.org/html/2606.10743v1), [출처](https://arxiv.org/html/2501.14208v1)
- Aria Machine Perception Services 손 추적(스테레오 SLAM 카메라): EgoMimic, EgoZero(HaMeR와 병용), HumanEgo(단안 대체 시 성공률 급락). — [출처](https://arxiv.org/html/2410.24221), [출처](https://arxiv.org/html/2605.24934)
- Apple Vision Pro 온디바이스 추적: EgoDex(829시간). — [출처](https://arxiv.org/abs/2505.11709)
- VR 헤드셋 + 손 트래커: EgoWild2Dex(PICO; 트래커 없으면 4.21 cm, 있으면 0.66 cm). — [출처](https://arxiv.org/html/2609.23755)
- 장갑·컨트롤러·힘 센서: DexWild(전자기 장갑 + ArUco), ReForce(Manus 장갑 + Quest 컨트롤러 + FSR). — [출처](https://arxiv.org/html/2505.07813v1), [출처](https://arxiv.org/html/2608.15560v1)
- 모션캡처: DexGEM(광학 22대), HOT3D 정답, GNR의 정밀 과제(OptiTrack + 외골격; "egocentric 영상은 밀리미터 정밀도가 없음"). — [출처](https://arxiv.org/html/2609.24093v2), [출처](https://arxiv.org/html/2610.12440)
- Meta Quest 손 추적 + FoundationPose: DemoMimic. — [출처](https://arxiv.org/html/2609.01938v2)
- HOI 서베이는 HaMeR, WiLoR, HaWoR(egocentric SLAM + Metric3D 스케일로 세계 좌표계 손 복원, CVPR 2025), Dyn-HaMR를 "HOI 파이프라인의 전단 모듈"로 분류합니다. — [arXiv:2607.28394](https://arxiv.org/html/2607.28394)

### Inferences
- **[추론]** 사용자의 목표(HOT3D, EgoDex, Ego4D 등으로 확장)에 대해: HOT3D는 물체 메시·6D 자세·MANO 손이 모두 모션캡처 정답으로 제공되어 접촉·렌치 계산이 바로 가능하며 GNR이 이미 이를 물리 리타겟팅 데이터 원천으로 썼습니다. EgoDex는 손 3D는 정밀하지만 물체 3D가 없고, Ego4D는 둘 다 없어(Being-H0도 가림·카메라 문제로 제외) 접촉 복원을 위한 별도 물체 복원 단계가 필요합니다. 따라서 ARCTIC 다음 단계로는 HOT3D가 가장 마찰이 적은 선택으로 보입니다.
- **[추론]** 단안 손 깊이 오차(EgoZero 1~2 cm 라벨 오차·5 cm 이상 깊이 오차, Human2Sim2Robot 20~30 cm, EgoWild2Dex 4.21 cm)는 접촉점 기반 방법에 더 치명적입니다. 접촉은 표면에서 수 mm 이내에서 결정되기 때문입니다. Robotizing Human Videos가 접촉을 "확률 예측(LatentAct)"으로 우회하고 DexGEM이 "시뮬레이션 재실행으로 접촉 복원"을 택한 것은 이 문제의 두 가지 대응 전략으로 읽힙니다.

### Gaps
- Ego4D·Ego-Exo4D를 직접 로봇 정책 학습에 쓴 2025~2026 사례는 이번 조사에서 확인하지 못했습니다(Being-H0와 EgoVLA는 다른 데이터셋을 선택했고, 제외 사유만 확인).
- HaWoR·Dyn-HaMR를 로봇 파이프라인에 실제로 채택한 논문은 찾지 못했습니다.

---

## 핵심 질문 5. 한 사람의 조작을 여러 로봇 팔 또는 로봇 팀으로 대응시킨 연구, 그리고 한 손 규모를 넘는 대형·무거운 물체를 다룬 연구가 있는가

### Takeaway
단일 인간의 손 조작을 **여러 로봇 팔(또는 로봇 팀)의 접촉 배치**로 변환한 연구는 이번 조사에서 발견되지 않았습니다. 가장 가까운 것은 (가) 사람 양손 → 로봇 양팔의 1:1 대응(Masquerade, HumanEgo, YOTO, EgoMimic, EgoWild2Dex, DexGEM 양손, R2R2R), (나) 사람 전신 영상 → 휴머노이드 전신으로 상자 등 큰 물체를 안아 옮기는 연구(HDMI, SUGAR, ResMimic)이며, 물체를 **스케일업**하거나 접촉을 **팔 여러 대에 재배정**하는 사례는 없었습니다.

### Cited Findings
- 양손 1:1 대응 사례: Masquerade는 Kinova Gen3 두 대로 3개 장기 양손 주방 과제([출처](https://arxiv.org/html/2508.09976v1)); HumanEgo는 양손 6자유도 + 이진 grasp 묶음을 출력하며 "일부 과제는 두 팔이 순차적으로 작업해야 함"을 언급([출처](https://arxiv.org/html/2605.24934), [검색 요약](https://www.alphaxiv.org/abs/2605.24934)); YOTO는 motion mask로 양팔 동기·비동기 협응 기록([출처](https://arxiv.org/html/2501.14208v1)); R2R2R은 "양손 과제는 손마다 따로 처리"하며 ABB YuMi로 "두 손으로 꾸러미 집기" 과제 포함([출처](https://arxiv.org/html/2505.09601v1)); DexGEM은 양손 25과제 16.0% → 62.4%와 실제 JAKA 두 대 + DexHand 두 개 양손 과제([출처](https://arxiv.org/html/2609.24093v2)); H-RDT는 EgoDex로 사전학습한 양손 모델([검색 요약](https://arxiv.org/pdf/2507.23523)); DemoBridge는 "양팔 계획은 미구현"([출처](https://arxiv.org/html/2607.09519v1)).
- 서베이 2606.00054는 "양손 연구는 등장하지만 다중 팔 협응은 과제로 분석되지 않는다"고 정리합니다. — [arXiv:2606.00054](https://arxiv.org/html/2606.00054v1)
- 서베이 2604.27621은 Ag2x2가 "완전한 사람 제거는 양손 협응 정보를 잃을 수 있어 손-물체 상호작용 점을 유지"한다고 소개하고, 5장에 "Multi-agent interaction" 절을 둡니다(본문 미확보). — [arXiv:2604.27621](https://arxiv.org/html/2604.27621v1)
- 대형·무거운 물체(휴머노이드 전신): HDMI(CMU)는 단안 RGB 사람-물체 영상에서 Unitree G1의 전신 로코-매니퓰레이션을 RL로 학습, 실기 6과제·시뮬레이션 14과제, 문 통과 67회 연속; 검색 요약에는 상자 옮기기·폼 매트 운반 데모가 언급됨([초록](https://arxiv.org/abs/2509.16757), [검색 요약](https://arxiv.org/pdf/2509.16757)). SUGAR(2026-05)는 비구조 영상의 사람-물체 궤적·접촉 라벨을 추출하고 물리 기반 보정기로 실행 가능하게 만들며, 검색 요약에 Carry Box 과제가 언급됨([초록](https://arxiv.org/abs/2605.20373), [검색 요약](https://arxiv.org/html/2605.20373)). ResMimic은 4.5 kg 상자를 전신 접촉으로 운반하지만 **영상이 아닌 모션캡처**에서 학습([검색 요약](https://arxiv.org/html/2510.05070v1)). Pro-HOI([arXiv:2603.01126](https://pro-hoi.github.io))도 상자 운반 수집. 검색 요약은 "영상 기반 조작 연구는 대부분 상체·탁상 상호작용에 한정되어 있었고, 사람과 물체를 함께 추종하는 전신 방법이 새 방향"이라고 요약했습니다.
- 다중 로봇 쪽: "Multi-Humanoid Robot Arm Motion Imitation and Collaboration Based on Improved Retargeting"은 RGB-D 카메라 한 대로 **여러 모방자**를 찍어 여러 로봇에 대응시키는 것으로, 한 사람의 시연을 팀으로 나누는 것이 아닙니다([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11939925/), 검색 요약). "From One Hand to Multiple Hands"(2022)는 한 시연을 **여러 종류의 로봇 손**(여러 형태) 데이터로 바꾸는 것이지 여러 대가 협동하는 것이 아닙니다([alphaxiv](https://www.alphaxiv.org/abs/2204.12490), 검색 요약). 사람-로봇 협동 운반 학습(Rozo 외 2015의 운동감각 시연 기반 힘 제약 GMM; PROACT 2026의 사람-사람 운반 데이터 기반 예측)은 로봇 한 대 + 사람 한 명 구도입니다([calinon.ch](https://calinon.ch/paper3046.htm), [arXiv:2609.25351](https://arxiv.org/pdf/2609.25351), 검색 요약).

### Inferences
- **[추론]** 기존 양손 연구는 "사람 왼손 = 로봇 왼팔, 사람 오른손 = 로봇 오른팔"이라는 고정 대응을 전제합니다. 사용자의 설정(5손가락 → 2~3개 구형 손끝 팔, 스케일업 물체)은 이 고정 대응을 깨고 접촉 배정을 최적화 문제로 바꾸는 것이어서, 조사된 어느 계열과도 직접 겹치지 않습니다. 가장 가까운 기술 조각은 DexGEM의 "접촉 라벨 → 로봇 손끝 후보 표면점 배정"(단일 손 내부 배정)과 GNR의 "grasp 재수립 + 힘 채택 기준"이며, 이 둘을 팔 여러 대로 확장한 사례는 없습니다.
- **[추론]** 휴머노이드 전신 연구는 큰 물체를 다루지만 접촉이 몸통·팔 전체에 분산된 "안기" 형태라서 손끝 접촉의 렌치 공간 분석과는 문제 구조가 다릅니다. 다만 SUGAR가 "접촉 라벨 + 물리 보정기"를 쓴다는 점은 영상 유래 접촉을 물리로 보정한다는 면에서 참고 가치가 있습니다.

### Gaps
- "단일 인간 시연 → 다중 로봇 팀" 또는 "스케일업 물체"를 명시한 논문은 세 차례의 검색(일반, 양팔·대형 물체, 다중 로봇 협동 운반)에서 모두 나오지 않았습니다. 존재하지 않는다고 단정할 수는 없으나, 주요 검색어로는 표면화되지 않는다는 것은 신호입니다.
- HDMI·SUGAR의 상자 운반 과제가 실제로 포함되어 있는지는 초록에서 확인되지 않았고 검색 요약(2차 출처)에만 의존했습니다. 전문 확인이 필요합니다.
- Ag2x2, H-RDT, Pro-HOI, ResMimic은 검색 요약 수준에서만 확인했습니다.
