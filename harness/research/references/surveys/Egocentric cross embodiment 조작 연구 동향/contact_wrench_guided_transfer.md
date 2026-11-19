# 접촉·렌치 기반 사람→로봇 조작 전이 연구 동향: CHORD, DexMachina, 계보와 경쟁 연구 (2026-11-08 기준)

표기 규칙
- 각 항목 끝의 [출처](URL)는 해당 문장을 확인한 1차 출처입니다. "(검증)"은 원문(논문 PDF/HTML, 공식 저장소, arXiv 초록)에서 직접 확인한 사실, "(추론)"은 확인된 사실들로부터 제가 끌어낸 해석입니다.
- CHORD 본문은 로컬에 저장된 v1 PDF(2026-06-22, `harness/research/references/papers/2607.00033v1.pdf`)를 쪽 단위로 읽었고, v2(2026-08-14)는 arXiv HTML로 표·한계 절을 대조했습니다. 두 판본의 수치가 다른 곳은 따로 표시했습니다.
- 처음 나오는 용어는 풀어 씁니다. 강화학습(reinforcement learning, RL), 역기구학(inverse kinematics, IK), 가상 물체 제어기(virtual object controller, VOC), 곡선 아래 면적(area under the curve, AUC), 성공률(success rate, SR), 힘 폐쇄(force closure), 지지 함수(support function), 접촉 렌치 지지 보상(contact wrench support reward, CWS 보상).

---

## 핵심 질문 1. CHORD는 정확히 무엇을 하는가: 전체 파이프라인, 데이터, 구현체, 수치, 성공 검증 방식, 한계

### Takeaway
CHORD(NVIDIA, arXiv 2607.00033, v1 2026-06-22, v2 2026-08-14, 프로젝트 페이지는 CoRL 2026 표기)는 "사람 손과 로봇 손의 접촉을 물체에 유발할 수 있는 힘·토크(렌치) 공간의 지지 함수로 비교"하는 보상을 강화학습 추적 정책 학습에 넣은 프레임워크입니다. 입력은 모션캡처 7개 데이터셋 + 자체 재구성 영상에서 만든 4,739개 양손 과제이고, 로봇은 Sharpa 손(시뮬·실기), Unitree G1 + Dex3 세 손가락 손(전신)입니다. 파지 성공의 "검증"은 시뮬레이션 물체 궤적 추적(완료율 0.7 이상)과 실기 5~7회 시도이며, 정적 힘 폐쇄 검사는 보조 보상·데이터 통계로만 쓰입니다. 평행 그리퍼·다중 팔·점 손끝은 다루지 않습니다.

### Cited Findings

**(1) 서지·저자·공개물**
- 제목 Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration. 저자 Xinghao Zhu*, Zixi Liu*, Shalin Jain*, Chenran Li†, Milad Noori†, Michael Andres Lin, Huihua Zhao, John Welsh, (v2에서 Mrinal Verghese 추가), Wei Liu, Tingwu Wang, Xingye Da, Zhengyi Luo, Vishal Kulkarni, Naema Bhatti, Yuke Zhu, Linxi Fan, Bowen Wen, Danfei Xu, Soha Pouya, Yan Chang‡(교신). 전원 NVIDIA. v1 2026-06-22, v2 2026-08-14. (검증) — [arXiv abs](https://arxiv.org/abs/2607.00033), [v1 PDF 1쪽](https://arxiv.org/pdf/2607.00033v1)
- 프로젝트 페이지는 학회를 Conference on Robot Learning(CoRL) 2026으로 표기하고, 코드는 GitHub `nvidia-isaac/video_to_data`의 `robotic_grounding` 하위, 데이터는 Hugging Face `nvidia/video-to-data` 컬렉션, 4,739개 시뮬레이션 과제가 공개되었다고 밝힙니다. (검증) — [프로젝트 페이지](https://nvidia-isaac.github.io/video_to_data/chord/)
- 초록 수치: 4,739개 양손 과제 벤치마크, 1,831개 평가에서 평균 성공률 82.12%, 전신 조작 90.77%, 실기 개루프·폐루프 전이. (검증) — [v1 PDF 1쪽](https://arxiv.org/pdf/2607.00033v1)

**(2) 파이프라인과 보상 정의 (3장)**
- 시연은 τ_ref = {x_human_t, x_object_t}로, 사람 손 3D 키포인트와 K개 강체 부위의 SE(3) 자세 궤적입니다. 먼저 키포인트→로봇 관절로 역기구학 리타게팅을 한 뒤, 정책 π(a_t | o_robot, o_object; x_ref_robot, x_ref_object)를 학습해 물체 자세가 기준 궤적을 추적하게 합니다. (검증) — [v1 PDF 3쪽](https://arxiv.org/pdf/2607.00033v1)
- 보상 r = r_task + r_imit + r_contact이며 DexMachina의 가상 물체 제어기(VOC)를 함께 씁니다. r_task는 물체 부위별 SE(3) 오차의 지수 커널 + 물체 간 상대 자세 보상 r_relative(물체-물체 상호작용 구간에서만 활성, 시연의 물체 간 거리로 게이트), r_imit는 리타게팅된 사람 동작에 대한 지수 커널입니다. (검증) — [v1 PDF 4쪽, 18쪽](https://arxiv.org/pdf/2607.00033v1)
- 접촉 렌치: 매 시점, 각 강체 부위 k에 대해 사람 시연에서 물체 좌표계의 접촉 위치 p와 법선 n을 뽑고, 접촉 i의 힘 f에 대한 원시 렌치 w = [f, p×f] ∈ R⁶를 정의합니다. 쿨롱 마찰 원뿔을 d개 모서리 다면체로 근사하여 렌치 행렬 W_{h,k} ∈ R^{6×(c·d)}를 만들고, 미리 뽑은 b개 6차원 단위 방향 B에 대한 지지 함수 σ_{h,k} = max_col(Bᵀ W_{h,k}) ∈ R^b를 계산합니다. 로봇 쪽 σ_{r,k}도 같은 방식입니다. (검증) — [v1 PDF 4쪽, 식 (1)(2)](https://arxiv.org/pdf/2607.00033v1)
- 접촉 렌치 보상(식 3): r^k_cws = exp(−(‖max(0,(1−β)σ_h − σ_r)‖² + ‖max(0, σ_r − (1+β)σ_h)‖²)/v_cws). 상대 허용오차 β의 하한보다 로봇 지지가 큰 것, 상한을 넘지 않는 것을 각각 보상합니다. 추가로 의도치 않은 접촉 r_unintend(σ_h=0인데 σ_r>0)와 누락 접촉 r_miss(σ_h>0인데 σ_r=0)를 벌점으로 둡니다. (검증) — [v1 PDF 4~5쪽](https://arxiv.org/pdf/2607.00033v1)
- 논문이 스스로 밝힌 위치: "Contact wrench rewards in prior RL work optimize for static stability, making them effective for grasping but too rigid for general robot manipulation. In contrast, we leverage the wrench space, for the first time, to measure the similarity between a human demonstration and a robot execution in terms of the induced motion", 그리고 "this metric accommodates transient, non-force-closure phases of long-horizon manipulation such as pushing, levering, and sliding". (검증) — [v1 PDF 3쪽](https://arxiv.org/pdf/2607.00033v1)
- 학습 안정화 장치(3.2절): 기준 궤적의 임의 시점으로 리셋 후 짧은 구간 VOC 완전 활성, 사람 렌치 행렬에서 뽑은 렌치로 물체 부위를 교란, 리타게팅 동작을 사전분포로 쓰는 잔차 행동공간, VOC 커리큘럼 감쇠. (검증) — [v1 PDF 5쪽](https://arxiv.org/pdf/2607.00033v1)
- 잡음이 큰 접촉(RGB 영상 재구성)에서는 직접 매칭 대신 "축소된 접촉 렌치 지침" r^k_fc = (1/B) Σ_b 1[σ_{r,k,b} > ε]로 전환하며, 논문은 "When maximized, this is equivalent to force closure"라고 씁니다. (검증) — [v1 PDF 5쪽](https://arxiv.org/pdf/2607.00033v1)
- 전신 확장: 손만 있는 기준(자기중심 재구성 등)은 끝단(end-effector) 궤적→전신 동작을 채우는 인페인팅 모듈(MotionBricks 구조, 참고문헌 [42])을 학습해 쓰고, 3인칭 전신 기준(Cari4D [45])은 손가락 재구성이 부정확해 r_fc(힘 폐쇄 목적)로 학습합니다. 전신 정책은 사전학습된 전신 제어기(Sonic [21]) 위의 잔차입니다. (검증) — [v1 PDF 5쪽, 21~22쪽](https://arxiv.org/pdf/2607.00033v1)
- 관측(부록 B): 손목 상태, 손가락 관절, 물체 상태, 명령(목표 델타), 행동 이력, 접촉 관측(손목 좌표계의 접촉 위치·힘 방향). 행동: 양손 잔차(손목 위치 0.05 m, 손목 자세 0.15 rad, 손가락 0.15 rad 스케일), 지수이동평균 0.3. (검증) — [v1 PDF 17쪽](https://arxiv.org/pdf/2607.00033v1)

**(3) 데이터셋·벤치마크 (3.3절, 부록 D)**
- 벤치마크는 모션캡처 7개 데이터셋 ARCTIC, TACO, HOT3D, OakInk2, DexYCB, GRAB, H2O와 자체 영상 재구성에서 만들며, Isaac Lab으로 가져옵니다. 각 로더는 "wrist pose, 21 MANO joint positions, per-link hand-object contacts, and the object's full 6-DoF pose"의 통일 표현으로 내보냅니다. 출력은 Sharpa-Wave 로봇 궤적입니다. (검증) — [v1 PDF 5쪽, 20쪽](https://arxiv.org/pdf/2607.00033v1)
- HOT3D(자기중심 장시간 녹화)는 "Contact is detected when the minimum fingertip-to-object distance falls below 2 cm", 15프레임 이하 끊김은 이어 붙이고, 프레임별 다수결로 지배 물체를 정해 약 294개 원본→약 4,045개 원자 클립으로 나눕니다. (검증) — [v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1)
- 리타게팅: 2차계획법(quadratic programming) 기반 미분 역기구학, 손목 자세 + 손끝 잔차 추적, 관절 위치·속도 한계, 프레임당 최대 200회 반복, 잔차 변화 1e-6 이하면 조기 종료. (검증) — [v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1)
- 품질 검사: 캡슐 모델로 손-물체(볼록 껍질 기준)·손-손 관통을 재고 어느 프레임이든 2 cm 초과면 거부, 볼록껍질/메시 부피비 3 초과인 오목 물체(컵·AR 안경)는 손-물체 검사 제외; Isaac Lab에서 300스텝 재생 검사. (검증) — [v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1)
- 데이터 통계 지표: 접촉 이벤트 수, 그리고 Ferrari–Canny ε = min_b h_{h,k}(512개 기저 방향의 지지값 최솟값), "ε > 0 certifies force closure". 벤치마크 분포 그림은 시간 길이·접촉 이벤트 수·ε에서 DexMachina/ManipTrans/SPIDER 과제군보다 넓습니다. (검증) — [v1 PDF 5쪽 그림 3, 20쪽](https://arxiv.org/pdf/2607.00033v1)

**(4) 로봇 구현체**
- 시뮬레이션 주 구현체: Sharpa(Sharpa-Wave) 양손 떠 있는 손. 전신: Unitree G1 + Dex3 세 손가락 손. 실기: Dexmate 로봇 + Sharpa 손 2개(손당 22 구동 관절, 팔 7관절, 몸통 3관절 → 61차원 명령), Vicon 6대 모션캡처, ONNX 20 Hz 추론. (검증) — [v1 PDF 10쪽, 23쪽](https://arxiv.org/pdf/2607.00033v1)
- 저장소 README의 과제 이름: 떠 있는 Sharpa `Sharpa-V2D-v0`, Vega Sharpa(전신 휴머노이드 형태 Sharpa), G1 전신 `SonicG1-ReconBody-v0`·`SonicG1-ReconHand-*`, `g1_dex3` 분할(arctic_to_dex3.py, taco_to_dex3.py). 그리퍼나 다중 팔 구현체는 없습니다. (검증) — [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)

**(5) 정량 결과**
- 성공 정의: 위치 오차 15 cm 초과 또는 회전 오차 40° 초과면 종료, 완료율(종료 없이 완료한 비율) > 0.7이면 성공. 1,831개 과제에 같은 하이퍼파라미터·커리큘럼. (검증) — [v1 PDF 6쪽](https://arxiv.org/pdf/2607.00033v1)
- 데이터셋별 성공률(v1 그림 5): HOT3D-1Obj n=1131 0.772±0.012, OakInk2-1Obj n=147 0.786±0.132, OakInk2-2Obj n=130 0.746±0.038, ARCTIC n=196 0.814±0.028, TACO n=77 0.935±0.028, HOT3D-2Obj n=150 0.753±0.035. 전신(17개 하위 과제): ARCTIC 강체 n=4 0.994, TPV(3인칭) n=5 0.867, ARCTIC 관절 n=4 0.914, TACO n=4 0.866. (검증) — [v1 PDF 7쪽](https://arxiv.org/pdf/2607.00033v1)
- v1 표 1(각 베이스라인의 원래 과제군, 모두 Sharpa 손으로 재구현): DexMachina 과제군 AUC 0.232±0.214 → CHORD 0.687±0.358; ManipTrans 과제군 MT-SR 0.428 → 0.639; SPIDER 과제군 SP-SR 0.333±0.488 → 0.359±0.482; CHORD 자체 단일 물체군(Ours-1) AUC 0.211 → 0.895, SP-SR 0.133 → 0.999; 두 물체군(Ours-2) SP-SR 0.533 → 0.982. 과제 수: ManipTrans 8, SPIDER 3, DexMachina 7, 자체 9. (검증) — [v1 PDF 6~7쪽](https://arxiv.org/pdf/2607.00033v1)
- v2 표 1(60개 과제, 3 시드, 모두 Isaac Lab 재구현; 새 베이스라인 Human2Sim2Robot(H2S2R)과 새 지표 JPE·RPE·MPPE 추가): DexMachina AUC 0.737±0.053 / SP-SR 0.690 / MP-SR 0.569 / JPE 13.10; ManipTrans 0.506 / 0.423 / 0.323 / 30.07; SPIDER 0.804 / 0.911 / 0.789 / 41.35; H2S2R 0.730 / 0.686 / 0.569 / 11.80; CHORD 0.918±0.018 / 0.926±0.022 / 0.894±0.032 / 4.34±0.25. SPIDER의 SP-SR 0.911은 CHORD 0.926에 근접합니다. (검증) — [v2 HTML](https://arxiv.org/html/2607.00033v2), [프로젝트 페이지](https://nvidia-isaac.github.io/video_to_data/chord/)
- 보상 소거 실험(v1 표 2, ARCTIC `box grab`·`mixer use`, 3 시드 × 4096 환경): box grab CHORD 0.702±0.257 / 위치만(Position Only, DexMachina식 접촉 위치 보상) 0.334±0.141 / 접촉 없음 0.384±0.206; mixer use 0.894±0.023 / 0.624±0.166 / 0.423±0.273. (검증) — [v1 PDF 7~8쪽](https://arxiv.org/pdf/2607.00033v1)
- 보상 소거 실험(v2, 60개 과제): CHORD AUC 0.918 / Position Only 0.862 / No Contact 0.774. (검증) — [v2 HTML](https://arxiv.org/html/2607.00033v2)
- 전신 교차 구현체 소거(v1 표 3, G1+Dex3, 12 과제): 강체 0.994 vs Position Only 0.460, 관절 물체 0.914 vs 0.000, 다중 물체 0.866 vs 0.192, 전체 0.925±0.104 vs 0.217±0.333. 저자 해석: "contact position rewards being tightly coupled with the demonstrated embodiment, whereas wrench space alignment guides learning independently of embodiment constraints". (검증) — [v1 PDF 10쪽](https://arxiv.org/pdf/2607.00033v1)
- CWS 보상과 성공의 상관(4.3절): 1,831회 실행에서 시퀀스별 정규화한 달성 CWS 보상 대 완료율의 피어슨 r ≈ 0.80(ARCTIC 0.89, HOT3D 0.81, OakInk2 0.76, TACO 0.84), 선형 적합 y = 1.112x + 0.057, 관계는 "monotonic but saturating". (검증) — [v1 PDF 8쪽](https://arxiv.org/pdf/2607.00033v1)
- 장기 과제(그림 7): 40~48초 시퀀스에서도 ADD-AUC ≈ 0.85~0.98 유지, DexMachina·ManipTrans는 길어질수록 크게 하락. (검증) — [v1 PDF 9쪽](https://arxiv.org/pdf/2607.00033v1)
- 실기(v2 4.6절): bowl handover 5/5, in-hand plate reorientation 3/5, bowl-plate stacking 5/5, mixer open-close 5/5, capsule machine alternating pick-place 5/5, box pick-place 개루프 6/7·폐루프 7/7. (검증) — [v2 HTML](https://arxiv.org/html/2607.00033v2)
- 자기중심 영상 종단간 검증(부록 C): 단안 RGB 영상 + 물체 이름 텍스트 → MoGe 단안 깊이를 미터 기준으로, Grounding DINO + SAM2 + SAM3D(메시) + FoundationPose(6-DoF 추적), ViPE + Dyn-HaMR(MANO) → DROID-SLAM 초기화 가우시안 스플랫 정제. 브러시 과제에서 접지 품질 AUC 0.647. (검증) — [v1 PDF 18~20쪽](https://arxiv.org/pdf/2607.00033v1)
- 베이스라인 재현(부록 G): ManipTrans를 Inspire 손으로 원 코드 재현 시 OakInk-v2 77개 원시 과제에서 단손 64.1%, 양손 29.7%; 이후 Sharpa로 이식. SPIDER는 CHORD의 역기구학 해를 써서 Sharpa 동작 생성. (검증) — [v1 PDF 24쪽](https://arxiv.org/pdf/2607.00033v1)

**(6) 잡음 민감도(부록 A)**
- 잡음 스케일 1~6: 손목 위치 10~60 mm, 손목 자세 5~30°, 접촉 위치 5~30 mm, 접촉 법선 5~30°. 세 설정: 손 자세 잡음만 + CWS, 손+접촉 잡음 + CWS, 손+접촉 잡음 + 힘 폐쇄 보상. 결론: "CHORD remains robust to moderate hand-pose noise", 접촉 지침에 잡음이 더해지면 "performance decreases more noticeably", 매우 큰 잡음에서는 "force-closure guidance can outperform noisy wrench-space guidance", 단 "relying on force-closure guidance reduces dexterity, as it prioritizes stable grasps over faithful imitation". (검증) — [v1 PDF 16~17쪽](https://arxiv.org/pdf/2607.00033v1)

**(7) 저자가 밝힌 한계(6장, v1·v2 동일)**
- "First, real-world deployment currently relies on state-based observations. Second, effective contact guidance requires relatively clean human demonstrations, as noisy contacts can degrade reward quality and will require a force-closure assumption. Third, object pose error is an imperfect evaluation metric: exact object placement may not be essential in some tasks, while a small pose error can lead to functional failure in others." (검증) — [v1 PDF 11쪽](https://arxiv.org/pdf/2607.00033v1), [v2 HTML](https://arxiv.org/html/2607.00033v2)
- 원격조작 비교 파일럿(4.7절): 상자 들기가 특히 어려웠고, 역기구학이 손가락 구성을 보존 못함, 햅틱 피드백 부재, 관절 토크 직접 제어 불가 → "the resulting grasps were fragile and sensitive to small errors in contact location, timing, or applied force". (검증) — [v1 PDF 11쪽](https://arxiv.org/pdf/2607.00033v1)

**(8) 공식 코드 `nvidia-isaac/video_to_data` (V2D)**
- README: "An end-to-end pipeline that converts human demonstration videos into simulation-ready assets and physics-grounded robot training data". 세 단계: `video_ingestion_agent/`(LangGraph 분할·장면 그래프·SigLIP-2 임베딩), `reconstruction/`("18 containerized modules + multi-view pipelines", 깊이·마스크·메시·6D 자세·SMPL), `robotic_grounding/`(Isaac Lab 2.3.1 + RSL-RL PPO + 리타게팅), `robotic_grounding/flash_chord/`(Newton/Warp/JAX 컨테이너의 PPO·FlashSAC 가속 학습), `robotic_grounding/groot_finetune/`(LeRobot 변환, GR00T N1.7 미세조정). 약 949 스타. (검증) — [GitHub README](https://github.com/nvidia-isaac/video_to_data)
- `reconstruction/modules/`에는 37개 모듈이 있고, 자기중심 관련은 `v2d_ego_hand_reconstruction`, 손 관련 `v2d_hamer`, `v2d_hawor`, `v2d_wilor`, `v2d_hand_alignment`, 물체 `v2d_foundation_pose`, `v2d_sam3d`, `v2d_moge`, `v2d_sam2`, `v2d_grounding_dino`, `v2d_bundlesdf`, 몸 `v2d_sam3d_body`, `v2d_cari4d`, SLAM `v2d_droid_slam`, 정제 `v2d_gsplat_refinement`, 다시점 `v2d_hoi_object_reconstruction`, `v2d_mv_*`, `v2d_cusfm`. 사용자가 물어본 "ego_recon"이라는 이름은 모듈명이 아니라 robotic_grounding README가 "monocular egocentric reconstruction" 데이터 원천을 가리키는 이름(`ego_recon`, 스크립트 `ego_recon_to_sharpa.py`)으로 쓰입니다. (검증) — [modules 디렉터리](https://github.com/nvidia-isaac/video_to_data/tree/main/reconstruction/modules), [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- robotic_grounding README의 보상 설명: `contact_wrench_support_reward`는 실시간 접촉 기하를 기준 기하와 매칭하고, `force_closure` 보상은 "gates on the reference 'contact expected' label and rewards the fraction of live wrench-basis directions the hand actually supports". 관통한 MANO 키포인트는 물체의 방향 경계상자(oriented bounding box)에 투영해 손 강성을 보존(`retarget/object_collision.py`). (검증) — [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- robotic_grounding README가 밝힌 자기중심 영상 재구성의 한계: "Fingers routinely start inside the object", "Contact positions and normals are noisy"여서 기본 접촉 보상이 "a poor signal"이 되며, 중력 정렬 잔차(예: tissue_box 8.4° 보정)가 남습니다. 또 TACO 체크포인트는 원시 도형 URDF(`--use_primitive_urdfs`)가 필요하고 전체 메시 URDF로 녹화하면 파지가 미끄러집니다. (검증) — [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- 렌치 보상 구현 파일: `robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/` 안의 `utils.py`(16,587 B), `utils_jit.py`(27,463 B), `rewards.py`(26,145 B), `curriculum.py`, `observations.py`, `terminations.py`, `events.py`, `actions/`, `commands/`. (검증) — [mdp 디렉터리](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp)
- 사용자가 보관한 참조 사본(`extlib/chord_ref/wrench_support_ref.py`, 커밋 cbee067, 2026-08-14)에서 확인한 구현 상수: SPDX 헤더 Apache-2.0, `sample_wrench_space_basis_scaled`는 6차원 가우시안을 토크 성분만 rc(물체 경계구 반지름)로 나눈 뒤 정규화, `compute_friction_cone_edges`의 기본 `friction_coefficients=0.5`, `append_normal` 옵션으로 순수 법선 모서리를 추가, 토크를 rc로 나눠 렌치 구성. (검증, 로컬 사본 기준) — [원본 경로](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp)
- 종단간 워크플로 문서: 입력은 "a raw egocentric human demonstration captured on an iPhone"(예제 tissue_box.mp4) → HaMeR 손 추적(+선택적 가우시안 스플랫 정제, DROID-SLAM, 중력 정렬) → Vega Sharpa(58관절) 리타게팅(Parquet·USD·역기구학 사이드카·QA 보고서) → RSL-RL 전문가 학습(공개 레시피 20,000 반복) → 전문가 롤아웃 수집 → GR00T N1.7 미세조정 → 폐루프 `lift_hold` 평가. 실기: 시뮬 200 롤아웃 사전학습 + 실기 재생 10회 적응 정책 4/5 성공, 실기 재생 10·30회만 쓴 GR00T 0/5, 저자 표현 "a task-specific validation rather than a general sim-to-real benchmark". 48 GB GPU 1장. 이 문서는 CHORD 보상이나 렌치 계산을 언급하지 않습니다. (검증) — [e2e_workflow.md](https://github.com/nvidia-isaac/video_to_data/blob/main/docs/e2e_workflow.md)

### Inferences
- (추론) CHORD의 렌치 유사도는 "정적 파지 성공 예측기"로 설계되지 않았습니다. 저자 스스로 정적 안정성용 렌치 보상과 구별했고, 상관 r≈0.80은 RL 정책의 롤아웃 중 달성된 보상과 완료율 사이(시퀀스별 정규화 후)의 상관이지, 주어진 정적 접촉 배치가 성공할지에 대한 예측력이 아닙니다. 사용자의 "정적 파지 성공 예측 AUC ≈ 0.56" 결과는 CHORD의 주장과 모순되지 않고, 오히려 CHORD가 다루지 않은 질문입니다.
- (추론) CHORD가 "교차 구현체"를 입증한 범위는 사람 5손가락 → Sharpa 5손가락(22 DoF)과 G1+Dex3 3손가락까지입니다. 그리퍼·다중 팔·구형 점 손끝으로의 전이 증거는 논문·코드 어디에도 없습니다. 다만 접촉 수·위치·손 형태가 달라도 비교 가능하다는 표현 자체(지지 함수는 접촉 수에 무관)는 점 접촉 집합에도 그대로 적용 가능하므로, 수식 차원에서는 다중 팔 점 접촉으로 확장하는 데 장애가 없습니다.
- (추론) 자기중심 영상은 CHORD의 "보조 입력"입니다. 대규모 결과(1,831개)는 모션캡처 데이터셋(HOT3D는 자기중심이지만 모션캡처 정답 포함)이고, 순수 단안 영상 경로는 부록 C의 1개 과제(AUC 0.647)와 저장소의 iPhone 예제뿐이며, 저장소 README는 이 경로의 접촉이 잡음이 커서 기본 접촉 보상이 "poor signal"이라고 인정합니다. 즉 "영상 → 접촉/렌치 보상"은 아직 CHORD에서도 미해결 구간입니다.
- (추론) "물리적 성공 검증"은 시뮬레이션 RL 추적 성공과 실기 5~7회 시도로만 이루어집니다. 힘 폐쇄(ε>0)는 데이터 통계와 잡음 대비 보조 보상으로만 쓰이고, 리타게팅된 파지를 배포 전에 걸러내는 용도로는 쓰이지 않습니다.

### Gaps
- 논문 본문에서 접촉 추출의 세부(손 부위 수, 거리 문턱, 마찰 원뿔 모서리 수 d, 방향 수 b, β, v_cws 값)는 명시되어 있지 않았습니다(부록 D의 2 cm는 HOT3D 클립 분할용 문턱이고, 512는 데이터 통계의 기저 방향 수). 사용자가 코드 분석으로 파악한 "MANO 16부위·1 cm·모서리 8+1·512방향·rc=1"은 본 세션에서 코드 원문을 다시 열어 재확인하지는 못했습니다(로컬 사본에는 렌치 함수 5개만 있음).
- v1(6월)과 v2(8월)의 표 1은 과제군·베이스라인·지표가 달라 직접 비교가 되지 않으며, 어떤 실험이 v2에서 새로 추가됐는지 저자 변경 기록은 찾지 못했습니다.
- 저장소 최상위 LICENSE 파일의 종류는 가져오지 못했습니다(참조 사본의 SPDX 헤더만 Apache-2.0으로 확인).

---

## 핵심 질문 2. DexMachina는 정확히 무엇을 하는가: 기능적 리타게팅, 가상 물체 제어기, 손·데이터, 접촉 대 운동학 보상, 한계, 후속

### Takeaway
DexMachina(Stanford·NVIDIA, arXiv 2505.24853, 2025-05-30, ICML 2026 게재)는 "손 자세가 아니라 물체 상태 궤적을 목표로 추적"하는 기능적 리타게팅(functional retargeting)을 제안하고, 세기가 줄어드는 가상 물체 제어기 커리큘럼으로 ARCTIC 관절 물체 7클립을 6종 손에 학습시켰습니다. 접촉 보상은 "접촉 위치"(MANO 메시와 물체 정점 거리 1 cm, 최대 50점 → 가장 가까운 로봇 링크 평균)이며, 모션캡처 전용·시뮬레이션 전용·상태 기반이고, 실기 실험·영상·그리퍼는 없습니다.

### Cited Findings
- 서지: Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar*, Shuran Song*(공동 지도), arXiv 2505.24853 v1 2025-05-30, 프로젝트 페이지 project-dexmachina.github.io, 코드 github.com/MandiZhao/dexmachina. (검증) — [arXiv abs](https://arxiv.org/abs/2505.24853), [GitHub README](https://github.com/MandiZhao/dexmachina)
- 학회: CHORD 참고문헌 [49]가 "In Proceedings of the Forty-Third International Conference on Machine Learning, 2026"으로 인용. (검증) — [CHORD v1 PDF 15쪽](https://arxiv.org/pdf/2607.00033v1)
- 초록 요지: 기능적 리타게팅 = 사람 손-물체 시연에서 물체 상태를 추적하는 정책 학습; 장기·양손·관절 물체 과제; "virtual object controllers with decaying strength"로 처음엔 물체를 목표 상태로 끌어주고 점차 정책이 넘겨받음; 다양한 과제·손의 시뮬레이션 벤치마크 공개. (검증) — [arXiv abs](https://arxiv.org/abs/2505.24853)
- 손: 벤치마크 6종. 주 실험 Inspire, Allegro, XHand, Schunk; 손 형태 분석에서 Ability, DexRobot 추가. 로컬 자산 폴더에도 ability_hand, allegro_hand, dex3_hand, dexrobot_hand, inspire_hand, schunk_hand, xhand, mano_hand가 있습니다. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853), 로컬 `extlib/dexmachina/dexmachina/assets/`
- 데이터: ARCTIC의 관절 물체 5종, 시연 7클립(Notebook-300, Mixer-300, Waffleiron-300, Ketchup-100, Mixer-170 등). 문서는 "Current dexmachina codebase assumes only using the use_01, use_02 clips"라고 명시하며 `grab_` 클립 지원은 큰 수정이 필요하다고 합니다. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853), [docs 2_process_demo](https://mandizhao.github.io/dexmachina-docs/2_process_demo.html)
- 보상: 과제 보상 = 물체 위치·회전·관절각 오차의 지수항 곱; 모션 모방 = 키포인트 매칭 + 리타게팅 기준 관절각에 대한 행동복제 보상; 접촉 보상 = 물체 메시 정점과 MANO 손 메시 사이 거리 0.01(m) 문턱, 최대 50 접촉, 가장 가까운 로봇 링크에 배정해 평균, 유효성 불일치에 최대 벌점 거리. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 코드의 접촉 보상 기본값(`envs/rewards.py`): `contact_rew_weight` 0.0(기본 꺼짐), `contact_rew_function` 'exp' 또는 'sigmoid'(exp(−β·dist), β=30), `wrist_frame_contact`, `mask_zero_contact`(양쪽 다 접촉 없으면 0), 좌우 손 평균. `retargeting/map_contacts.py` 주석: 물체 표면 정점 접촉(`contacts.left`, `valid_contacts.left`, 모양 (T, N=50, 4))을 리타게팅된 손의 충돌 링크 중심(AABB 근사)에 가장 가까운 것으로 배정해 평균. (검증, 로컬 사본) — [GitHub 저장소](https://github.com/MandiZhao/dexmachina)
- 가상 물체 제어기: 물체의 6개 1자유도 기저 관절 + 관절 1개에 스프링-댐퍼 PD, 특권 상태로 시연 궤적을 따라 끌어줌; 이득은 모든 보상 이동평균이 문턱을 넘을 때만 지수 감쇠, kp ≤ 0.01이면 0으로. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 베이스라인·결과: 운동학만, ObjDex 재구현, 커리큘럼 없는 과제+보조 보상, ManipTrans 커리큘럼 재구현. 원 ObjDex 논문의 Ketchup-100 41.2%, Mixer-170 57.6% 대비 재구현은 Ketchup-100 모든 손 >90%, Mixer-170 4종 중 3종 >70%. 수치 AUC는 그림 3에만 있어 본문에서 추출되지 않았습니다. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 접촉 보상 대 운동학 보상에 대한 입장: 본문은 접촉 보상과 운동학 보상을 직접 대결시키지 않고, "보조 보상(모방·접촉)이 과제 보상 단독 대비 주는 이득은 inconsistent"하며 "운동학만으로는 과제를 완료 못 한다"고 보고합니다. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 손 형태 결론: 크고 완전 구동되는 손이 성능·효율 모두 높음; Allegro는 사람과 덜 닮았는데도 긴 손가락 덕에 잘함; 크기 유사성보다 자유도가 중요; Schunk > Inspire, Ability; 구동이 적은 손은 덜 사람다운 전략을 택함. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 한계(저자): 특권 시뮬레이터 상태 기반 정책(비전 기반 학습/증류 제안); 고품질 손-물체 시연 가정(3D 생성 모델·재구성으로 확장 제안); 공개 손 모델의 물성 추정치가 실기와 다를 수 있어 "more careful tuning with real reference hardware will be needed"; 하드웨어 접근 부재로 "have not yet been evaluated in real-world settings". 영상·자기중심 데이터는 다루지 않습니다. (검증) — [arXiv HTML](https://arxiv.org/html/2505.24853)
- 리타게팅 도구: 외부 `dex-retargeting` 패키지 필요, 손마다 MANO 손끝 매핑 `retarget_config.yaml` 수작업, 2단계(병렬 운동학 리타게팅 → 단일 물리 롤아웃 평활화), 접촉 리타게팅은 `map_contacts.py`. 시뮬레이터는 Genesis 포크, RL은 rl_games 포크. (검증) — [docs 3_retargeting](https://mandizhao.github.io/dexmachina-docs/3_retargeting.html), [GitHub README](https://github.com/MandiZhao/dexmachina)
- CHORD가 Sharpa로 재구현한 DexMachina의 성능: v1 자체 과제군 AUC 0.232±0.214, v2 60개 과제 AUC 0.737±0.053(CHORD 0.918). (검증) — [CHORD v1 PDF 7쪽](https://arxiv.org/pdf/2607.00033v1), [CHORD v2 HTML](https://arxiv.org/html/2607.00033v2)
- 인용 현황: Semantic Scholar에 97건의 인용 기록(2025~2026)이 있고 거의 전부 2026년 arXiv 논문입니다. 저자들의 별도 2026 후속 논문은 검색에서 나오지 않았습니다. (검증) — [Semantic Scholar 인용 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2505.24853/citations?fields=title,year,externalIds,authors&limit=100)
- (주의) 로컬 사본 `extlib/dexmachina/dexmachina/envs/hand_cfgs/point2.py`, `point4.py`, `point_common.py`의 "point robot(떠 있는 구 k개, 팔 하나당 접촉 1개)" 설정은 docstring에 "URDFs are generated by condex scripts/gen_point_urdf.py"라고 적혀 있어 사용자 프로젝트가 추가한 것이지 공식 DexMachina 배포물이 아닙니다. (검증, 로컬 사본)

### Inferences
- (추론) DexMachina의 핵심 기여는 보상 설계보다 "커리큘럼(가상 물체 제어기)"입니다. CHORD, SPIDER 등 후속작이 공통으로 가져간 것도 이 커리큘럼이며, 접촉 위치 보상은 CHORD가 "Position Only" 소거 조건으로 비판하는 대상이 되었습니다.
- (추론) 접촉 보상 가중치의 코드 기본값이 0인 점, 그리고 본문에서 보조 보상의 효과가 일관되지 않다고 보고한 점을 보면, DexMachina 자체도 접촉 매칭 보상을 필수로 보지 않았습니다.
- (추론) 손 형태 분석(자유도가 크기보다 중요, 덜 구동된 손은 다른 전략을 택함)은 사용자의 "2~3대 팔·6개 이하 점 손끝" 설정에 직접 함의가 있습니다. 즉 사람 손 전략을 그대로 옮기기보다 로봇 측 자유도에 맞는 다른 접촉 전략을 허용해야 한다는 근거가 됩니다.

### Gaps
- DexMachina 논문 자체의 손별 AUC 수치(그림 3)는 본문 텍스트에서 추출되지 않아 기록하지 못했습니다. 프로젝트 페이지는 자바스크립트 렌더링이라 본문을 가져오지 못했습니다.
- 접촉 보상 가중치(λ 값)와 "최대 벌점 거리"의 실제 값은 부록에 있다고만 확인했습니다.

---

## 핵심 질문 3. 두 연구가 다루지 않는 것: 사람 손과 매우 다른 형태(그리퍼·다중 팔·점 손끝), 자기중심/야생 영상, 물리적 성공 검증 — 종합

### Takeaway
두 연구 모두 (i) 다지 손(5손가락 또는 3손가락) 사이의 전이만 보였고 평행 그리퍼·다중 팔·점 손끝은 없으며, (ii) 자기중심 영상은 CHORD만 보조적으로(HOT3D 모션캡처 + 단안 영상 1과제 + iPhone 예제) 다뤘고 DexMachina는 모션캡처 전용이며, (iii) 성공 검증은 시뮬레이션 RL 추적 성공(물체 자세 오차 문턱) 중심이고 실기는 CHORD만 소수 시도, 힘 폐쇄 검사는 어느 쪽도 배포 전 필터로 쓰지 않습니다.

### Cited Findings
- 구현체 범위: CHORD는 Sharpa(22 DoF 다지 손)와 G1+Dex3(3손가락), DexMachina는 Inspire·Allegro·XHand·Schunk·Ability·DexRobot(모두 다지 손). 두 저장소 모두 그리퍼·다중 팔 구성 없음. (검증) — [CHORD v1 PDF 10·23쪽](https://arxiv.org/pdf/2607.00033v1), [DexMachina HTML](https://arxiv.org/html/2505.24853), [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- CHORD가 "형태 차이"를 논증한 유일한 실험은 5손가락 사람 → 3손가락 Dex3 전신 소거(표 3)입니다. (검증) — [CHORD v1 PDF 10쪽](https://arxiv.org/pdf/2607.00033v1)
- 데이터 원천: CHORD 벤치마크의 HOT3D는 Project Aria·Quest 3 자기중심 다중 시점 스트림 + 광학 모션캡처 정답(833분, 370만 장, 33 물체, 19 피험자, MANO 손 자세, 물체 6-DoF; 접촉 주석 없음)입니다. 즉 "자기중심이지만 정답이 있는" 데이터입니다. (검증) — [HOT3D arXiv](https://arxiv.org/abs/2406.09598), [CHORD v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1)
- 순수 단안 자기중심 영상은 CHORD 부록 C의 1개 과제(AUC 0.647)와 저장소의 iPhone 티슈 상자 예제뿐이며, 저장소는 이 경로의 접촉이 잡음이 커서 기본 접촉 보상이 나쁜 신호라고 적시합니다. (검증) — [CHORD v1 PDF 18~20쪽](https://arxiv.org/pdf/2607.00033v1), [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- DexMachina는 ARCTIC 모션캡처만 쓰고 영상·자기중심 데이터는 논의하지 않습니다. (검증) — [DexMachina HTML](https://arxiv.org/html/2505.24853)
- 성공 검증: CHORD는 완료율>0.7(위치 15 cm·회전 40° 문턱), 실기 과제당 5~7회; DexMachina는 물체 추적 오차 AUC, 실기 없음. 힘 폐쇄(ε>0)는 CHORD에서 데이터 통계·잡음 대비 보조 보상으로만 등장. (검증) — [CHORD v1 PDF 6·20쪽](https://arxiv.org/pdf/2607.00033v1), [CHORD v2 HTML](https://arxiv.org/html/2607.00033v2), [DexMachina HTML](https://arxiv.org/html/2505.24853)
- 평가 지표의 한계를 CHORD 저자가 인정: "object pose error is an imperfect evaluation metric". (검증) — [CHORD v1 PDF 11쪽](https://arxiv.org/pdf/2607.00033v1)

### Inferences
- (추론) 사용자의 가설("손 자세 추적→역기구학 리타게팅/인페인팅 경로는 리타게팅된 파지의 물리적 성공을 보장하지 않는다")은 CHORD 자신의 원격조작 파일럿(역기구학이 손가락 구성을 보존 못해 파지가 취약)과 저장소의 자기중심 재구성 한계(손가락이 물체 안에서 시작, 접촉·법선 잡음)로 뒷받침됩니다. 다만 CHORD의 해법은 "RL로 고친다"이지 "배포 전에 물리 타당성으로 거른다"가 아니므로, 사용자의 접근(렌치 품질 α·ρ + 시뮬레이션 검증으로 앵커를 선별)은 CHORD와 다른 위치를 차지합니다.
- (추론) "자기중심 + 다중 팔 + 물리 검증"의 교집합을 다룬 연구는 본 조사에서 발견되지 않았습니다(핵심 질문 6 참조).

### Gaps
- CHORD 실기 실험의 실패 사례 분석(어떤 접촉 오류로 실패했는지)은 논문에 없습니다.

---

## 핵심 질문 4. 계보: 접촉·렌치·물체 중심 표현으로 사람 시연을 로봇 파지·조작에 옮긴 선행 연구(2019~2025, 배경 포함)

### Takeaway
계보는 세 갈래입니다. (a) 접촉 지도 전이(ContactDB/ContactPose → GenDexGrasp 등, 정적 파지, 힘 폐쇄 최적화로 합성), (b) 물체 궤적 추적형 RL(ObjDex·DexTrack·ManipTrans·DexMachina·Human2Sim2Robot·SimToolReal, 동적 조작, 시뮬 추적 성공으로 검증), (c) 물리 기반 리타게팅(SPIDER, 샘플링 최적화). 사람 접촉을 "렌치 공간"에서 비교하는 것은 CHORD가 처음이라고 스스로 주장하며, 본 조사에서 그 주장을 반박하는 선행 연구는 찾지 못했습니다.

### Cited Findings
**(a) 접촉 지도·정적 파지 계열 (배경)**
- ContactDB(CVPR 2019 구두): 열화상으로 3D 프린트 가정용 물체 50종의 접촉 지도 수집(37.5만 프레임 RGB-D+열화상, 3,750 메시), 의도(사용/건네기)·크기가 접촉에 미치는 영향 분석, 형상→접촉 예측 모델. (검증) — [arXiv 1904.06830](https://arxiv.org/abs/1904.06830)
- ContactPose(ECCV 2020): 접촉 + 손 자세 + 물체 자세 + RGB-D를 묶은 데이터셋(2,306 파지, 25 물체, 50 피험자, 290만 장). (검증) — [arXiv 2007.09545](https://arxiv.org/abs/2007.09545)
- GenDexGrasp(ICRA 2023): 접촉 지도를 손 무관 중간 표현으로 써서 다지 손 간 전이; 학습 데이터 MultiDex는 사람 데이터가 아니라 힘 폐쇄 최적화로 합성. (검증) — [arXiv 2210.00722](https://arxiv.org/abs/2210.00722)
- Dexonomy(RSS 2025, PKU): GRASP 분류 31종 전부를 임의 물체·임의 관절 손에 합성(10.7k 물체, 950만 파지), 손마다 유형별 사람 주석 템플릿 1개, 물체→템플릿 맞춤 후 시뮬레이션에서 접촉 인지 제어로 검증, 단일 시점 점군 조건 생성 모델 실기 82.3%. (검증) — [arXiv 2504.18829](https://arxiv.org/abs/2504.18829)
- UniDexGrasp++(2023): 점군 조건 물체 무관 파지 정책, 기하 커리큘럼 + 일반가-전문가 반복, 학습 85.4%/시험 78.2%. (검증) — [arXiv 2304.00464](https://arxiv.org/abs/2304.00464)
- GraspXL(ECCV 2024): 3D 손-물체 데이터 없이 58개 물체로 학습한 정책이 50만+ 미지 물체 82.2%, 다른 손으로 전이. (검증) — [arXiv 2403.19649](https://arxiv.org/abs/2403.19649)
- Dex-Net 2.0(RSS 2017, 배경): 해석적 강건 파지 지표로 라벨한 합성 670만 파지로 GQ-CNN 학습, YuMi 평행 그리퍼 실기 적대 물체 93%. (검증) — [arXiv 1703.09312](https://arxiv.org/abs/1703.09312)

**(b) 물체 궤적 추적형 RL 계열**
- ObjDex = Object-Centric Dexterous Manipulation from Human Motion Data(Chen, Wang, Yang, Liu, 2024-11; CHORD 참고문헌 [5], DexMachina 베이스라인): 대규모 사람 손 모션캡처로 학습한 상위 생성 모델이 물체 목표 상태 조건 손목 궤적을 내고, 하위 RL 손가락 제어기가 실행; 10개 가정용 물체, 시뮬→실기 양손 로봇 전이. (검증) — [arXiv 2411.04005](https://arxiv.org/abs/2411.04005)
- DexH2R(Zhao, Zhu 외 2024-11; IEEE/ASME T-Mech 2025): 운동학 리타게팅 + 과제 지향 잔차 정책, 시험 시 사람 손·물체 궤적을 지침으로 입력, 시뮬·실기에서 선행 대비 40% 향상 주장. CHORD 1저자 Xinghao Zhu가 공저. (검증) — [arXiv 2411.04428](https://arxiv.org/abs/2411.04428)
- DexTrack(ICLR 2025): 사람 참조 추적 신경 제어기, 성공 데이터 플라이휠 + 호모토피 최적화, 선행 대비 10%+ 향상, 시뮬·실기. (검증) — [arXiv 2502.09614](https://arxiv.org/abs/2502.09614)
- ManipTrans(CVPR 2025): 일반 궤적 모방기 사전학습 → "interaction constraints" 하의 잔차 모듈 미세조정, DexManipNet 3.3K 에피소드 공개. CHORD는 이를 "rewards contact forces near demonstrated hand-object interactions"로 요약하고 Inspire 손 원 코드 재현에서 OakInk-v2 단손 64.1%·양손 29.7%를 얻었습니다. (검증) — [arXiv 2503.21860](https://arxiv.org/abs/2503.21860), [CHORD v1 PDF 2·24쪽](https://arxiv.org/pdf/2607.00033v1)
- Human2Sim2Robot(Lum, Lee, Liu, Bohg, 2025-04): 사람 RGB-D 영상 1편으로 실→시뮬→실 RL; 물체 자세 궤적이 "object-centric, embodiment-agnostic reward", 영상에서 뽑은 사전 조작 손 자세로 탐색 유도; 물체 인지 재생 대비 55%+, 모방학습 대비 68%+ 향상. CHORD v2가 베이스라인(H2S2R, AUC 0.730)으로 추가. (검증) — [arXiv 2504.12609](https://arxiv.org/abs/2504.12609), [CHORD v2 HTML](https://arxiv.org/html/2607.00033v2)
- SimToolReal(Kedia, Lum, Bohg 외, 2026-02): 절차 생성 도구형 원시 도형으로 단일 시뮬→실 RL 정책, 리타게팅·고정 파지 대비 37% 향상, 실기 120 롤아웃(24 과제·12 물체·6 도구 범주) 제로샷. (검증) — [arXiv 2602.16863](https://arxiv.org/abs/2602.16863)
- OmniGrasp(2024): 시뮬 휴머노이드가 1,200+ 물체를 집어 임의 궤적으로 운반, 메시+궤적만 필요, 시뮬 평가. PhysHOI(2023): 접촉 그래프로 몸-물체 접촉을 명시한 전신 모방, 과제 보상 없음. (검증) — [arXiv 2407.11385](https://arxiv.org/abs/2407.11385), [arXiv 2312.04393](https://arxiv.org/abs/2312.04393)

**(c) 물리 기반 리타게팅·운동학 리타게팅**
- SPIDER(Pan, Wang, Qi 외, 2025-11; CHORD 베이스라인): 운동학 전용 사람 동작을 샘플링 기반 해법으로 동역학적으로 실행 가능한 로봇 궤적으로 변환, 9개 휴머노이드/손 구현체·6 데이터셋·3 시뮬레이터, RL 대비 약 10배 빠름, 6,520개 시뮬 시연(손 4종·191 물체) 공개, 개루프 실기 또는 RL 참조로 사용. (검증) — [arXiv 2511.09484](https://arxiv.org/abs/2511.09484)
- DexPilot(2019)·AnyTeleop(RSS 2023): 비전 기반 손 추적 원격조작(23 구동 자유도 손-팔; 다양한 팔·손·카메라 일반화). 배경. (검증) — [arXiv 1910.03135](https://arxiv.org/abs/1910.03135), [arXiv 2307.04577](https://arxiv.org/abs/2307.04577)
- DexFlow(2025-05): 다중 출처 손-물체 데이터 변환, 시간 일관성 손실, 접촉 지도로 관통 억제; 수치 없음. (검증) — [arXiv 2505.01083](https://arxiv.org/abs/2505.01083)
- Progressive Kinematic-Dynamic Alignment(2025-11): 사람 손 영상만으로 손 무관 3단계(운동학 매칭 → 잔차 정책 → 손목 궤적), 평균 전이 성공 73%, 시뮬. (검증) — [arXiv 2511.10987](https://arxiv.org/abs/2511.10987)
- DreamGen(2025-05): 영상 세계 모델로 "신경 궤적" 생성, 휴머노이드가 1개 과제 원격조작 데이터로 22개 새 행동; 접촉·렌치 표현 없음. (검증) — [arXiv 2505.12705](https://arxiv.org/abs/2505.12705)
- EgoMimic(ICRA 2025): Project Aria 자기중심 영상 + 3D 손 추적을 로봇 시연과 동등하게 공동학습, 양손 저가 매니퓰레이터. (검증) — [arXiv 2410.24221](https://arxiv.org/abs/2410.24221)

### Inferences
- (추론) 사용자가 쓰는 Dexonomy는 "정적 파지 합성·시뮬 검증" 계열, ARCTIC 동적 단계는 "물체 궤적 추적" 계열에 속합니다. 두 계열을 잇는 공개 연구는 CHORD(정적 통계 ε + 동적 렌치 보상)가 유일하게 가깝지만, 배포 전 정적 검증은 하지 않습니다.
- (추론) 선행 연구 중 "평행 그리퍼"를 명시 대상으로 둔 것은 Dex-Net 계열(사람 시연 없음)과 GenHand(핵심 질문 6)뿐이고, 사람 시연 기반 동적 조작 연구는 거의 전부 다지 손을 가정합니다.

### Gaps
- 사용자가 언급한 "ContactGrasp"(Brahmbhatt 외, IROS 2019), "GeoRT", "Dexterous grasp transfer via contact maps"는 arXiv 식별자를 확정하지 못해 1차 출처로 확인하지 못했습니다(시도한 1906.11148, 1906.07440은 다른 논문이었음).

---

## 핵심 질문 5. 2026년 후속·경쟁 연구: CHORD와 DexMachina를 인용한 논문 중 접촉/힘/렌치/교차 구현체 관련

### Takeaway
2026년 6~11월에만 CHORD를 인용한 논문 23편, DexMachina 97편이 있고, 흐름은 (1) 접촉 구조를 리타게팅 목표로 삼는 "contact-anchored" 계열(CARD, Contact-Anchored Retargeting, TopoRetarget, DexOAK, Morphometric Imitation), (2) 힘·촉각을 명시 목표로 두는 계열(ReForce, FoLD, Kinematic-Tactile Aligned Retargeting with Contact Wrench Space Alignment, HACo, TacOT), (3) 영상→접촉 복원(ReContact, C2Dex, EgoCPT, CARD의 Vid-HACO, Do as I Do), (4) 그리퍼까지 포함한 교차 구현체(KITE, X-Reset, LUCID, GOTT)로 나뉩니다. CHORD의 렌치 공간 정렬을 직접 계승한 것은 Kinematic-Tactile Aligned Retargeting(2610.39582) 한 편입니다.

### Cited Findings
**CHORD를 인용한 23편(2026-11-08 기준 Semantic Scholar)**: ReContact(2611.02968), Kinematic-Tactile Aligned Dexterous Retargeting with Contact Wrench Space Alignment(2610.39582), CARD(2610.36539), DONIC(2610.29997), Gen2Dex(2610.24392), EgoCPT(2610.17572), Dex-One2Many(2610.12470), Generative Neural Retargeting(2610.12440), Contact-Aware Imitation Learning Through Contact Factorization(2610.09533), FlashDexRetarget(2610.01849), HACo(2609.36596), X-Reset(2609.35715), FoLD(2609.33551), When Does Touch Matter?(2609.24068), Touch2Robot(2609.24660), Weave(2609.16683), Grounding Generated Video Plans(2609.10050), RoboTok(2609.03199), ReForce(2608.15560), Hand-Object Interaction in the Age of Large Foundation Models 서베이(2607.28394), BORA(2605.30226), OATH(2610.14379), 기타 1편(척추 안정화, 무관). (검증) — [Semantic Scholar CHORD 인용](https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)

**(1) 접촉 구조를 리타게팅 목표로**
- CARD: Contact-Anchored Retargeting for Dexterous Manipulation(Dai, He, Hogan 외, 2026-10-27): 물체 모델 없이 공유 손-접촉 표현으로 리타게팅, 운동학과 함께 명시적 접촉력 목표 생성; 영상·모션캡처·촉각 장갑·GRAB 입력, 영상용 접촉 예측 모델 Vid-HACO; 촉각 추적 제어기로 실기 개루프 재생. 접촉 프레임의 43~78%에서 접촉 유지(운동학 방법 <10%, 물체 인지 방법 22.5%); 물리 기반 리타게팅에 추가 시 GRAB 엄격 성공 12%→32%. (검증) — [arXiv 2610.36539](https://arxiv.org/abs/2610.36539)
- Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning(Yang 외, 2026-09-21, v2 11-03): "어떤 손가락 부위가 어떤 물체 위치에 어떤 순서로 닿는가"가 관절 동작보다 손 형태를 넘어 잘 전이된다고 주장; 시뮬 MANO 손 물리 정제 → 접촉 앵커 리타게팅(손 형태 무관 목적) → 단일 잔차 RL 정책; 25,454 단손 궤적(성공 7.3%→59.3%), 25 양손 과제(16.0%→62.4%), 서로 다른 손 4종(+62.4 pp), 실기 양손 4과제 제로샷. 입력은 모션캡처. (검증) — [arXiv 2609.24093](https://arxiv.org/abs/2609.24093)
- TopoRetarget(Wu 외, 2026-06-15): 손 중심 리타게팅 목적이 과제 관련 접촉 구조를 보존하지 못한다고 지적, 손-물체 키포인트 희소 상호작용 그래프 + 거리 가중 라플라시안 변형; ContactPose에서 접촉 정밀도 최고, Pen-Spin 학습 성공 +40.6 pp, Wuji Hand 실기 제로샷. (검증) — [arXiv 2606.16272](https://arxiv.org/abs/2606.16272)
- DexOAK(Spinola, Katzschmann, Schmid, 2026-10-19): 손-물체 결합 그래프의 라플라시안으로 물체 상대 접촉 기하 보존, Inspire 손, OakInk-v2 성공 39.5→48.4%, ARCTIC ADD-AUC 62.7→77.9%, 실기 개루프 재생 46/56. (검증) — [arXiv 2610.26762](https://arxiv.org/abs/2610.26762)
- Morphometric Imitation(Sadjadpour 외, 2026-09-23): 접촉 보존 리타게팅 → 잔차 RL → 시각운동 정책 증류, 손 3종·상호작용 10개에서 접촉 F1 +8~28점, Sharpa 손 실기 300회 89.3%. (검증) — [arXiv 2609.28660](https://arxiv.org/abs/2609.28660)
- ReContact(Gu 외, 2026-11-02): 사람 영상에서 "접촉 단계·접촉 쌍·앵커·누르는 방향·고착/미끄럼 모드"의 물체 중심 접촉 계획을 만들고, 고정된 전신 추적기 위 잔차 정책이 계획된 앵커에 지정 방향으로 힘을 내도록 보상; 시뮬 허벅지 지지 재파지 83.5(기준 0.16%), 튜브 전달 20.4→97.8%, Unitree G1 실기 9과제. (검증) — [arXiv 2611.02968](https://arxiv.org/abs/2611.02968)
- One Demonstration, Many Objects(DemoMimic, 2026-09-01): 접촉점 주변 국소 기하에 집중한 접촉 중심 보상, 실기 16 물체·4 과제·손 2종 71%, 시뮬→실 하락 최소. (검증) — [arXiv 2609.01938](https://arxiv.org/abs/2609.01938)
- Weave(2026-09-15): 접촉 인지 리타게팅 + 접근 동작 보완으로 휴머노이드(29 몸 관절 + 12 손가락 관절) 단일 정책, 학습 92.5%·미지 65.0%, 접촉 주석 9,000 롤아웃 공개. (검증) — [arXiv 2609.16683](https://arxiv.org/abs/2609.16683)

**(2) 힘·촉각·렌치를 명시 목표로**
- Kinematic-Tactile Aligned Dexterous Retargeting with Contact Wrench Space Alignment(He, Xu, Chen 외, 2026-10-29): 물체 중심 접촉 렌치 공간을 사람-로봇 촉각 정보 정렬의 물리 표현으로 사용; 3단계(운동학·접촉 기하 정렬 → 촉각 힘으로 사람 렌치 지지 프로파일 구축 → 로봇 손끝 힘 목표 최적화 후 관절 위치 보정); 붓기(pouring) 3조건 실기 재생 성공 96.7%, 렌치 전이 충실도 베이스라인 상회. 손 이름 미기재. (검증) — [arXiv 2610.39582](https://arxiv.org/abs/2610.39582)
- ReForce(2026-08-16): 운동학 리타게팅 행동에 대규모 시뮬 상호작용으로 학습한 일반 힘 추적기의 잔차를 더해 목표 힘 달성; 온라인 힘 인지 원격조작·오프라인 사람 데이터 변환; 종이컵·집게 실기. (검증) — [arXiv 2608.15560](https://arxiv.org/abs/2608.15560)
- FoLD(2026-09-27): 사람 시연과 현재 상호작용 상태에서 보상 힘장(force field)을 계산해 시연 물체 운동을 선호하는 힘 사전분포로 쓰고 잔차 정책이 리타게팅 동작을 조정; 관절 물체 벤치마크·실기. (검증) — [arXiv 2609.33551](https://arxiv.org/abs/2609.33551)
- SoftAct(2026-04-01): VR에서 손 운동학·물체 운동·접촉 패치·접촉력을 캡처, 사람 접촉력을 손가락에 배분해 비의인화 공압 소프트 손의 손가락에 비례 배분, 손끝 추적 RMSE 최대 55% 감소. (검증) — [arXiv 2604.01224](https://arxiv.org/abs/2604.01224)
- HACo(2026-09-29): 손끝 촉각 + 관절 토크로 힘 조절 행동 학습, 실기 5과제 평균 83%(최강 베이스라인 35%). TacOT(2026-10-03): 촉각 유도 최적수송으로 사람-로봇 시연 정렬, 실기 4과제 +17~20점. When Does Touch Matter?(2026-09-21): LEAP Hand에서 손가락별 렌치 추정 + 택셀 결합 정책 24/25 vs 비전만 14/25. (검증) — [arXiv 2609.36596](https://arxiv.org/abs/2609.36596), [arXiv 2610.04363](https://arxiv.org/abs/2610.04363), [arXiv 2609.24068](https://arxiv.org/abs/2609.24068)
- Transferring Contact, Not Just Motion(Atar, Huang, Yip, 2026-06-14): 손 자세 잠재 + 손별 노력 신호를 관절 토크(N·m)로 보정해 손끝 힘·손가락별 하중 기술자로 매핑, 구조가 다른 손 사이 순응 파지 전이. 수치 없음. (검증) — [arXiv 2606.15516](https://arxiv.org/abs/2606.15516)

**(3) 영상(자기중심 포함) → 접촉 복원 → 로봇**
- EgoCPT(2026-10-13): 자기중심 RGB 영상 + 참조 손 자세 궤적만으로 접촉점을 반복 정제하는 접촉 인지 최적화; 5과제에서 잡음 많은 EgoDex 재구성 67.3%, 모션캡처 기반 TACO 재구성 92.0%; 실기 직접 배포. (검증) — [arXiv 2610.17572](https://arxiv.org/abs/2610.17572)
- C2Dex(2026-08-07): 단안 영상의 프레임별 잡음 관측을 물체 정준 공간에 모아 안정적 물체 측 접촉을 복원, 라플라시안 상호작용 최적화 + 잔차 RL; DexYCB 57.78%·TACO 26.67%(최강 베이스라인 17.78%·10.00%); 실기는 재생 기반 타당성만. (검증) — [arXiv 2608.07045](https://arxiv.org/abs/2608.07045)
- DexMan(2025-10-09): 3인칭·생성 영상에서 보정·깊이·모델 없이 잡음 있는 손-물체 자세 추정, 양손 휴머노이드 시뮬, "contact-based rewards"; TACO ADD-S +0.08, OakInk-v2 RL 성공 +19%; 시뮬만. (검증) — [arXiv 2510.08475](https://arxiv.org/abs/2510.08475)
- Do as I Do(2026-06-17): 야생 단안(자기중심·외부) 영상에서 손-물체 재구성 → 다지 손 리타게팅. EgoEngine(2026-06-10): 자기중심 영상에서 사람을 로봇으로 치환한 관측 영상 + 타당성 제약 행동 궤적, 실기 제로샷 주장. (검증) — [arXiv 2606.19333](https://arxiv.org/abs/2606.19333), [arXiv 2606.12604](https://arxiv.org/abs/2606.12604)
- HumanEgo(2026-05-24): 자기중심 시연을 개체 수준 손-물체 상호작용 표현으로 올려 흐름 매칭 정책 학습, 과제당 30분 영상으로 실기 4과제 평균 92.5%, 하드웨어 무관 주장. (검증) — [arXiv 2605.24934](https://arxiv.org/abs/2605.24934)
- AcrossDEX 1.0(2026-10-29): HOT3D Aria 영상으로 양손 동작 사전학습 후 Inspire(6채널)·Allegro(16관절)에 적응, Allegro 재배향 93.3%. (검증) — [arXiv 2610.40210](https://arxiv.org/abs/2610.40210)
- 관련 데이터셋: EgoDex(Apple, 2025-05) 829시간·194 과제·Apple Vision Pro·3D 손 관절 쌍 주석, 물체 자세 주석 언급 없음. (검증) — [arXiv 2505.11709](https://arxiv.org/abs/2505.11709)

**(4) 그리퍼·복합 구현체까지 포함한 교차 구현체**
- KITE(Wang, Jin, Fang, 2026-06-20): 조작을 구현체 무관 "상호작용 예측(접촉 패턴 기반 의도 표현)"과 구현체별 "운동학 디코딩"으로 분리; 원천 로봇 1대 시연으로 학습 후 평행 그리퍼·다지 손·복합 구현체에 제로샷, 시뮬·실기 4과제. (검증) — [arXiv 2606.22113](https://arxiv.org/abs/2606.22113)
- X-Reset(Dan, Ma, Zhan, 2026-09-28): 리타게팅 동작을 모방·추적하지 말고 "리셋 분포"로만 쓰자고 주장; 22 DoF 손을 단 팔 2대와 평행 그리퍼, 20 물체, 물체 상태·목표만 보는 정책, 시뮬→실 제로샷. (검증) — [arXiv 2609.35715](https://arxiv.org/abs/2609.35715)
- LUCID(Gupta, Shi, Yuan, 2026-06-10): 인터넷 영상(+스마트폰 1시간)으로 구현체 무관 단기 의도 모델 학습, 3D 재구성 없음, 다지 손과 평행 그리퍼 제어기에 공통 인터페이스, 실기 5과제. (검증) — [arXiv 2606.11628](https://arxiv.org/abs/2606.11628)
- GOTT(2026-10-02): 접근-접촉 획득-이동 틀에서 재사용 가능한 교차 구현체 접촉 획득 프리미티브, 로봇 무관 물체 궤적 + 접근 사양 입력, 여러 팔-손 플랫폼·미지 손 형태 실기. (검증) — [arXiv 2610.03861](https://arxiv.org/abs/2610.03861)
- Object-centric 3D Motion Field(Yin, Yang, Abbeel, NeurIPS 2025): 사람 영상의 물체 3D 운동장을 행동 표현으로, 손 동작 대신 물체 운동으로 제로샷 제어, 실기 평균 55%(선행 ≤10%). (검증) — [arXiv 2506.04227](https://arxiv.org/abs/2506.04227)
- UniBYD(2025-12): 통일 형태 표현 + 모방→탐색으로 어닐링되는 PPO, UniManip 벤치마크, 평균 +44.08%. (검증) — [arXiv 2512.11609](https://arxiv.org/abs/2512.11609)

**(5) 기타 비판·대안**
- Generative Neural Retargeting(2026-10-08): 역기구학은 동역학을 무시하고 RL·샘플링 MPC는 비효율적이라 비판, 흐름 매칭으로 동역학적 실행 가능 궤적 생성, 56.20% vs MPC 27.20%, 접촉력 라벨이 있는 223k 시연 데이터 엔진. (검증) — [arXiv 2610.12440](https://arxiv.org/abs/2610.12440)
- OATH(2026-10-09): 물리 기반 리타게팅을 운동학 참조 위 잔차 학습으로 재정의, 링크별 국소 접촉 오차 관측, DexYCB 보류 궤적 58%·미지 물체 54%(2 cm 이내), 시뮬. (검증) — [arXiv 2610.14379](https://arxiv.org/abs/2610.14379)
- Dex-One2Many(2026-10-08): 엄격한 동작 매칭이 일반화를 제한한다고 비판, 장면 그래프로 추상화, 미지 상황 +71%. (검증) — [arXiv 2610.12470](https://arxiv.org/abs/2610.12470)
- A Minimalist Retargeting-Guided RL Recipe(REGRIND, 2026-07-13): 물체 중심 키포인트 추적 + 잔차 RL + 시스템 동정으로 가위·드라이버 실기 제로샷; 전이 요인 체계 분석. (검증) — [arXiv 2607.11874](https://arxiv.org/abs/2607.11874)
- Dexterous Robotic Hands in the Foundation-Model Era 서베이(Pei, 2026-11-04): 2024-01~2026-11 문헌을 하드웨어·데이터·교차 구현체 전이·학습·평가 5층으로 정리; 핵심 주장은 "사람 데이터의 가치는 양이 아니라 운동학·시각·접촉 정렬에 달렸다", "공유 행동 표현은 힘·지연·접촉의 구현체 한계를 없애지 못한다", "형태가 다른 손 사이 접촉 집약 기술의 제로샷 전이 증거는 빈약하다". (검증) — [arXiv 2611.06724](https://arxiv.org/abs/2611.06724)

### Inferences
- (추론) 2026년 하반기 합의는 "운동학(관절·키포인트) 매칭만으로는 접촉이 보존되지 않는다"는 것이고, 대안은 접촉 구조(어디·언제·순서)를 1급 목표로 올리는 것입니다. 렌치 공간까지 올라간 것은 CHORD와 Kinematic-Tactile Aligned Retargeting뿐이며 후자는 촉각 센서가 있는 사람 데이터를 전제합니다.
- (추론) 그리퍼까지 가는 연구(KITE, X-Reset, LUCID, GOTT)는 하나같이 "사람 손 접촉을 그리퍼 접촉으로 매핑"하지 않고, 물체 운동/의도/리셋 분포처럼 접촉보다 더 추상적인 표현으로 우회합니다. 사용자의 "접촉 그래프를 축약(contraction)해 그리퍼·팔 수에 맞춘다"는 발상은 이 중간 지대를 메우는 위치입니다.

### Gaps
- DexMachina 인용 97편 중 위에 다루지 않은 상당수(예: PatternDex, EigenDEXplore, PiHand, MDPO, DexStitch, Gen2Dex, FlashDexRetarget, EaDex, Video2Sim2Real, DexAgent)는 제목·초록 수준만 확인했거나 확인하지 못했습니다.
- 대부분 2026년 arXiv 선공개라 학회 심사(OpenReview) 기록은 찾지 못했습니다.

---

## 핵심 질문 6. 영상/시연에서 리타게팅한 파지를 "렌치 공간/힘 폐쇄 타당성"으로 검증·필터링한 선행 연구가 있는가, 한 사람 손을 여러 로봇 팔로 사상한 연구가 있는가

### Takeaway
"리타게팅 과정에 힘 폐쇄를 명시 제약/목적으로 넣어 그리퍼까지 커버한" 연구는 GenHand(npj Robotics 2026, 글래스고)가 가장 가깝고, 단일 이미지의 정적 파지를 Robotiq 2지·Barrett 3지·Allegro·Shadow로 옮기며 PyBullet에서 성공률을 검증합니다. 동적 궤적에 대해 렌치 지지 프로파일을 전이하는 것은 Kinematic-Tactile Aligned Retargeting(촉각 데이터 필요)이 있습니다. 한 사람 손을 여러 로봇 팔(손가락을 독립 에이전트로)에 사상한 연구는 본 조사에서 발견하지 못했습니다.

### Cited Findings
- GenHand: generalised human grasp kinematic retargeting(Qi, Popoola, Imran, Ahmad; James Watt School of Engineering, University of Glasgow; npj Robotics 2026, 4:19, DOI 10.1038/s44182-026-00076-1): 사람 파지 이미지 → 두 분기 인코더로 MANO 손 + 물체 SDF 추정 → HDBSCAN으로 접촉 영역 군집 → 접촉 앵커 근방에서 힘 폐쇄 최적화로 최적 접촉 x* → 그리퍼 구성(q, R, T) 운동학 최적화(루프 내 선형 배정 + 반복 최근접점 매칭) → 시뮬 검증. 초록: "optimising for force closure and kinematic similarity", "outperforms the key-vector based baseline by 39.8% in simulated experiments across four grippers and twenty objects". (검증) — [npj Robotics DOI](https://doi.org/10.1038/s44182-026-00076-1), [글래스고 사본 PDF](https://eprints.gla.ac.uk/377598/2/377598.pdf)
- GenHand 그리퍼: Robotiq 2F(2지 평행), Barrett(3지 12관절), Allegro(4지 16관절), Shadow(5지 24관절, 20 구동). 재구성 평가는 DexYCB(29,656 학습/5,928 시험 샘플). 마찰계수 μ=0.1~0.9에서 순 렌치 잔차: Shadow 베이스라인 ~26.77 → GenHand 0.45, Robotiq 4.44 → 0.12; 잔차는 손가락 수가 적을수록 작음. SDF 잔차: GenHand Shadow·Allegro·Barrett <0.35 cm, 베이스라인 1.34~1.82 cm, Robotiq는 GenHand도 가장 큼. (검증) — [글래스고 사본 PDF 3~6쪽](https://eprints.gla.ac.uk/377598/2/377598.pdf)
- GenHand 시뮬 성공률(PyBullet, 들어서 2초 유지; 표 4, μ=0.9): GenHand+HO(전체 파이프라인) Shadow 0.695 / Allegro 0.740 / Barrett 0.786 / Robotiq 0.634; GenHand+GT(정답 메시) 0.813 / 0.814 / 0.817 / 0.703; Baseline+GT 0.625 / 0.701 / 0.622 / 0.338. 결론 문장: "preserving contact intent while explicitly enforcing equilibrium feasibility produces higher task-level success than hand-pose matching alone, especially for low-DOF grippers". (검증) — [글래스고 사본 PDF 2·7쪽](https://eprints.gla.ac.uk/377598/2/377598.pdf)
- GenHand 가정·실패: 사람 시연이 이미 힘 폐쇄를 이루고 접촉이 주로 손끝으로 이뤄진다고 가정(검색 요약); 실패 유형은 불완전 메시, 과도 평활, 잘못된 손 자세, 접촉 배정 모호성, 관통, 섬세한 기하에서 불안정; "it does not explicitly preserve functional ..." (해당 문장 뒷부분은 8쪽에서 잘림). 실기 실험 없음, 정적 파지만. (검증) — [글래스고 사본 PDF 8쪽](https://eprints.gla.ac.uk/377598/2/377598.pdf)
- Kinematic-Tactile Aligned Dexterous Retargeting with Contact Wrench Space Alignment: 사람 촉각 힘으로 렌치 지지 프로파일을 만들고 로봇 손끝 힘 목표를 최적화, 실기 재생 96.7%. 동적 궤적에 대한 렌치 수준 전이이지만 촉각 장갑 데이터가 필요합니다. (검증) — [arXiv 2610.39582](https://arxiv.org/abs/2610.39582)
- CHORD의 데이터 통계 ε(Ferrari–Canny)는 과제 난이도 기술용이며, 필터링 기준으로 쓰였다는 서술은 없습니다. 품질 검사는 관통(2 cm)과 300스텝 재생뿐입니다. (검증) — [CHORD v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1)
- PhysV2A(2026-07-10): 영상 유도 6D 물체 운동 + RGB-D 파지 후보 쌍을 "로봇 중심 운동학 검사(도달 가능성)"로 걸러 순위화·조작성 정제, 실기 4과제; 힘 폐쇄는 언급 없음. (검증) — [arXiv 2607.09365](https://arxiv.org/abs/2607.09365)
- X-Reset: 리타게팅 상태 중 "unstable simulation states are filtered out"하여 리셋 분포로만 사용(물리 안정성 필터). (검증) — [arXiv 2609.35715](https://arxiv.org/abs/2609.35715)
- 한 손 → 여러 로봇 손(팔 아님): From One Hand to Multiple Hands(Qin 외, 2022): 조작자 MANO 형상으로 맞춤 손 모델을 만들고 손끝 키포인트 최적화로 Allegro·Schunk·Adroit에 오프라인 리타게팅. (검증) — [arXiv 2204.12490](https://arxiv.org/abs/2204.12490)
- 다중 팔 큰 물체 협동 파지(사람 시연 아님): Learning Cooperative Bimanual Dexterous Grasp of Large Objects with Tactile Sensing(2026-08-11): 실기 다중모달 데이터셋(관절각·시각·힘), 단일 시점 점군에서 확산 모델로 관절 수준 파지 구성 생성, 이중 팔 실기 미지 물체 고성공. (검증) — [arXiv 2608.10383](https://arxiv.org/abs/2608.10383)
- 검색에서 나온 간접 선행: 2011년 URAI 학회 영상 논문(사람 시연으로 다중 로봇 파지 계획 학습; 초록 일부만 확인), 2017년 Aalto 석사논문(2 에이전트가 하중 분담으로 파지 렌치 최소화), RSS XII 다중 팔 가상 물체 동역학계 조정. 어느 것도 "사람 손가락을 팔에 배정"하지는 않습니다. (검증, 요약 수준) — [unpaywall URAI 2011](https://unpaywall.org/10.1109%2FURAI.2011.6145992), [Aalto 2017](https://aaltodoc.aalto.fi/items/08f604ef-76df-4c22-8baf-4011e2b489f8), [RSS XII p19](https://www.roboticsproceedings.org/rss12/p19.html)

### Inferences
- (추론) 사용자의 설정(사람 접촉 그래프 → 2~3대 팔의 점 접촉 앵커, 렌치 척도 α·ρ로 선별, 시뮬 검증, QP 힘 배분으로 재생)은 GenHand의 "힘 폐쇄 최적화 후 운동학 맞춤" 구조를 (a) 정적 이미지에서 동적 궤적으로, (b) 단일 그리퍼에서 다중 팔 점 접촉으로 확장한 위치에 있습니다. 두 확장을 동시에 한 공개 연구는 확인되지 않았습니다.
- (추론) GenHand가 평행 그리퍼에서 SDF 잔차·거리 잔차가 가장 컸고 관절 수가 적을수록 순 렌치 잔차가 작았다는 결과는, 점 접촉 수가 적은 구현체일수록 "접촉 위치 정확도"가 성패를 좌우한다는 사용자 관찰(계획-실현 접촉 오프셋이 성공을 더 잘 예측)과 방향이 같습니다.

### Gaps
- GenHand 코드 공개 여부, 실기 실험 유무(본문 8쪽까지는 없음), 9쪽 이후의 방법 세부(힘 폐쇄 목적함수의 정확한 형태, 마찰 원뿔 근사)는 읽지 못했습니다.
- "손가락을 독립 에이전트로 다루는" 다중 에이전트 RL 조작 연구는 별도 검색을 하지 않았습니다.

---

## 핵심 질문 7. 렌치 공간 유사도·접촉 매칭 보상에 대한 알려진 비판: 성공 예측력, 재구성 오차 민감도, 마찰계수 가정

### Takeaway
세 층의 비판이 문헌에 있습니다. (1) 고전 파지 품질 지표(ε 등)는 자세 오차·실세계 성공을 잘 예측하지 못한다(Weisz & Allen 2012, Rubert 외 2019). (2) 접촉 "위치" 매칭은 법선·힘 방향을 무시해 물리적으로 틀린 접촉을 보상할 수 있다(CHORD의 Position Only 소거, TopoRetarget). (3) 렌치 유사도 자체도 접촉 위치·법선 잡음에 민감해 잡음이 크면 단순 힘 폐쇄 목적이 더 낫고, 그 경우 사람다움(dexterity)을 잃는다(CHORD 부록 A, 저장소 README). 렌치 유사도가 "정적 파지 성공을 예측하지 못한다"는 직접 보고는 CHORD 외부 문헌에서 찾지 못했습니다.

### Cited Findings
- Weisz & Allen, Pose Error Robust Grasping from Contact Wrench Space Metrics(ICRA 2012, pp. 557–562): 대부분의 경우 자세 오차에 가장 강건한 힘 폐쇄 파지는 ε 품질이 가장 높은 파지가 아니며, ε의 안정성 추정으로 재순위화하면 물리 실험 성공이 더 높았습니다. (검증, 검색 요약 + 서지) — [Columbia PDF](https://www.cs.columbia.edu/~allen/PAPERS/weisz_icra12.pdf), [dblp](https://dblp.org/rec/html/conf/icra/WeiszA12)
- Rubert 외, Predicting grasp success in the real world – A study of quality metrics and human assessment(Robotics and Autonomous Systems, 2019): 물리 시뮬레이션 기반 지표가 고전 ε 지표보다 실세계 성공을 더 일관되게 예측하며, 사람 평가와의 상관도 ε보다 높았습니다. (검증, 초록 수준) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921889019300247)
- CHORD의 위치 기반 보상 비판: "the contact location-based reward encourages the robot to touch object regions similar to those touched by the human demonstrator. However, contact location alone does not determine the effect of a contact... The same object region can produce different object motions depending on the contact surface normal and force direction." 상자 열기 예시에서 위치 보상이 고른 접촉은 "the contact normal is nearly perpendicular to that of the demonstration". (검증) — [CHORD v1 PDF 3~4쪽](https://arxiv.org/pdf/2607.00033v1)
- CHORD의 잡음 민감도 자인: 접촉 지침 잡음(위치 5~30 mm, 법선 5~30°)에서 성능 하락이 더 뚜렷하고, 매우 큰 잡음에서는 "heavily corrupted wrench supports may provide misleading supervision"이라 힘 폐쇄 보상이 더 낫지만 "reduces dexterity". 한계 절: "noisy contacts can degrade reward quality and will require a force-closure assumption". (검증) — [CHORD v1 PDF 11·17쪽](https://arxiv.org/pdf/2607.00033v1)
- CHORD 저장소의 자인: 자기중심 영상 재구성에서 "Contact positions and normals are noisy", 기본 접촉 보상이 "a poor signal". (검증) — [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- CHORD 공개 코드의 마찰계수 기본값은 0.5로 고정 상수이며(`friction_coefficients: float = 0.5`), 논문 본문은 마찰계수 추정이나 민감도 분석을 제시하지 않습니다. (검증, 로컬 참조 사본) — [mdp 디렉터리](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp)
- CHORD 상관 분석의 자기 한정: 상관 r≈0.80은 "saturating" 관계이고 선형 적합은 분산의 약 2/3만 설명하며, 저자도 이를 "a first-order summary"라 부릅니다. (검증) — [CHORD v1 PDF 8쪽](https://arxiv.org/pdf/2607.00033v1)
- TopoRetarget: 손 중심 리타게팅 목적은 접촉 구조를 보존하지 못해 "contact and feasibility errors will hurt policy performance". (검증) — [arXiv 2606.16272](https://arxiv.org/abs/2606.16272)
- Pei 서베이: 형태가 다른 손 사이 접촉 집약 기술의 제로샷 전이 증거가 빈약하며, 공유 행동 표현이 힘·지연·접촉의 구현체 한계를 제거하지 못한다. (검증) — [arXiv 2611.06724](https://arxiv.org/abs/2611.06724)
- 운동학 리타게팅 전반에 대한 비판: GenHand("reduced grasp quality when adapting to grippers with different morphologies", 키벡터 방법은 물체 기하를 무시), Generative Neural Retargeting(역기구학은 동역학 무시), X-Reset(리타게팅 동작을 추적하지 말 것), Dex-One2Many(엄격 동작 매칭은 일반화 저해). (검증) — [npj Robotics](https://doi.org/10.1038/s44182-026-00076-1), [arXiv 2610.12440](https://arxiv.org/abs/2610.12440), [arXiv 2609.35715](https://arxiv.org/abs/2609.35715), [arXiv 2610.12470](https://arxiv.org/abs/2610.12470)
- 관련 간접 근거: 변형체 파지에서 "a central issue with most quality measures is their dependence on contact points"(2022). (검증) — [arXiv 2203.12420](https://arxiv.org/abs/2203.12420)

### Inferences
- (추론) 문헌의 비판은 모두 "지표가 접촉 위치·법선·마찰의 정확성에 의존한다"로 수렴합니다. 사용자의 결과(렌치 유사도 AUC ≈ 0.56, 계획-실현 접촉 오프셋 AUC 0.78)는 Weisz & Allen(자세 오차 강건성이 ε보다 중요)과 같은 결을 가진 새 증거이며, 정적 성공 예측 맥락에서 CHORD식 지지 함수 유사도를 평가한 공개 결과는 없으므로 보고 가치가 있습니다.
- (추론) CHORD가 잡음 상황에서 힘 폐쇄 보상으로 후퇴한다는 사실은, 영상 기반 파이프라인에서는 "사람과의 유사도"보다 "로봇 측 물리 타당성(힘 폐쇄·강건 여유)"이 더 쓸모 있는 신호일 수 있음을 시사하며, 이는 사용자의 α·ρ 설계와 일치합니다.

### Gaps
- Rubert 2019·Weisz 2012의 구체 수치(상관계수, 성공률 차이)는 원문 접근 실패(ScienceDirect 403, Semantic Scholar 429)로 기록하지 못했습니다.
- "렌치 유사도 보상이 성공을 예측하지 못한다"는 직접 반례 연구는 찾지 못했습니다(없을 가능성이 큼).

---

## 핵심 질문 8. 재사용 가능한 구체 구성요소: 접촉 추출, 보상 설계, 데이터 파이프라인, 코드

### Takeaway
즉시 재사용 가능한 것은 CHORD 저장소의 렌치 지지 함수 구현(Apache-2.0 헤더)·7개 데이터셋 통일 로더·QP 역기구학 리타게팅·품질 검사·자기중심 재구성 모듈 37종, DexMachina의 ARCTIC 접촉 근사(`process_arctic.py`)·접촉→링크 매핑(`map_contacts.py`)·가상 물체 제어기 커리큘럼(`curriculum.py`)·손 자산 6종입니다. 영상→접촉 모델로는 CARD의 Vid-HACO, C2Dex, ReContact의 접촉 계획이 후보이나 코드 공개 여부는 미확인입니다.

### Cited Findings
- CHORD 렌치 함수(사용자 참조 사본 기준 함수 5개): `sample_wrench_space_basis_scaled`(6차원 방향 샘플, 토크를 rc로 스케일), `compute_tangent_basis`(Frisvad 2012 접선 기저), `compute_friction_cone_edges`(μ 기본 0.5, 법선 모서리 추가 옵션), `compute_wrench_space`(힘·토크/rc 결합), `compute_wrench_space_support_function`. 원 경로 `robotic_grounding/.../tasks/v2d/mdp/utils.py`(16.6 KB), 함께 `rewards.py`(26 KB), `curriculum.py`, `observations.py`, `terminations.py`. (검증) — [mdp 디렉터리](https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp)
- CHORD 데이터 파이프라인: ARCTIC·TACO·HOT3D·OakInk2·DexYCB·GRAB·H2O 로더 → 통일 표현(손목 자세, MANO 21 관절, 링크별 접촉, 물체 6-DoF) → QP 미분 역기구학 → 관통·재생 품질 검사 → Isaac Lab. HOT3D 원자 클립 분할 규칙(2 cm, 15프레임). (검증) — [CHORD v1 PDF 20쪽](https://arxiv.org/pdf/2607.00033v1), [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- CHORD 재구성 모듈(컨테이너): `v2d_ego_hand_reconstruction`, `v2d_hamer`, `v2d_hawor`, `v2d_wilor`, `v2d_hand_alignment`, `v2d_foundation_pose`, `v2d_sam3d`, `v2d_sam3d_body`, `v2d_moge`, `v2d_sam2`, `v2d_grounding_dino`, `v2d_droid_slam`, `v2d_gsplat_refinement`, `v2d_hoi_object_reconstruction`, `v2d_bundlesdf`, `v2d_cari4d`, `v2d_cusfm`, 다시점 `v2d_mv_*`. 종단간 스크립트 `run_e2e.sh`, Blackwell(compute capability 12.0) 미지원(cuSfM 경로). (검증) — [modules 디렉터리](https://github.com/nvidia-isaac/video_to_data/tree/main/reconstruction/modules), [GitHub README](https://github.com/nvidia-isaac/video_to_data)
- CHORD 벤치마크 자산: 4,739 과제, Hugging Face `nvidia/video-to-data`. MANO와 각 데이터셋은 라이선스 게이트로 원 출처에서 받아야 함. (검증) — [프로젝트 페이지](https://nvidia-isaac.github.io/video_to_data/chord/), [robotic_grounding README](https://github.com/nvidia-isaac/video_to_data/blob/main/robotic_grounding/README.md)
- DexMachina 코드 구성: `retargeting/process_arctic.py`(MANO 모델로 접촉 근사, `--farthest_sample`), `retargeting/map_contacts.py`(물체 정점 접촉 → 로봇 링크), `retargeting/parallel_retarget.py`(dex-retargeting, 2단계), `envs/rewards.py`(접촉 보상 exp/sigmoid, β=30), `envs/curriculum.py`(VOC 감쇠), `hand_proc/add_wrist_dof.py`·`tune_gains.py`·`inspect_raw_urdf.py`(새 손 추가), 자산: 손 6종 + ARCTIC 물체 5종(box, ketchup, laptop, mixer, notebook, waffleiron; V-HACD 볼록 분해 URDF). Genesis·rl_games 포크 필요. (검증) — [GitHub README](https://github.com/MandiZhao/dexmachina), [docs](https://mandizhao.github.io/dexmachina-docs/)
- DexMachina 제약: use_01/use_02 클립만, 손마다 `retarget_config.yaml` 수작업, 리타게팅은 손-클립 쌍마다 재실행. (검증) — [docs 2_process_demo](https://mandizhao.github.io/dexmachina-docs/2_process_demo.html), [docs 3_retargeting](https://mandizhao.github.io/dexmachina-docs/3_retargeting.html)
- 영상→접촉 후보 모듈: CARD의 Vid-HACO(촉각 없는 영상용 접촉 예측), C2Dex(물체 정준 공간 접촉 집계), ReContact(접촉 단계·쌍·앵커·누름 방향·고착/미끄럼 계획), EgoCPT(접촉점 반복 정제). (검증) — [arXiv 2610.36539](https://arxiv.org/abs/2610.36539), [arXiv 2608.07045](https://arxiv.org/abs/2608.07045), [arXiv 2611.02968](https://arxiv.org/abs/2611.02968), [arXiv 2610.17572](https://arxiv.org/abs/2610.17572)
- 물리 기반 리타게팅 데이터: SPIDER 6,520 시연(손 4종·191 물체), Weave 접촉 주석 9,000 롤아웃, Generative Neural Retargeting 접촉력 라벨 223k 시연. (검증) — [arXiv 2511.09484](https://arxiv.org/abs/2511.09484), [arXiv 2609.16683](https://arxiv.org/abs/2609.16683), [arXiv 2610.12440](https://arxiv.org/abs/2610.12440)
- Contact-Anchored Retargeting 코드: github.com/DexGEM-Lab/real2sim2real. (검증) — [arXiv 2609.24093](https://arxiv.org/abs/2609.24093)

### Inferences
- (추론) 사용자 프로젝트와의 접점: ARCTIC `box_grab`·`box_use`는 CHORD도 소거 실험 시퀀스(`box grab`, `mixer use`)로 썼고 DexMachina는 `use_01/02`만 지원하므로, `grab` 클립을 쓰는 사용자 설정은 DexMachina 코드 경로 밖이며 CHORD 로더 쪽이 더 맞습니다.
- (추론) 사용자 포트(`contraction_gt/wrench_support.py`)는 기본 μ=0.6, 8모서리, 법선 성분 1 정규화를 쓰고, CHORD 참조는 μ=0.5·합력 크기 1·순수 법선 모서리 추가입니다. 유사도 비교 실험을 보고할 때 이 규약 차이(균일 배율 √(1+μ²) 이내)를 명시하면 됩니다.

### Gaps
- `rewards.py`·`curriculum.py` 본문(가중치, β, v_cws, 커리큘럼 일정)은 열어 보지 못했습니다.
- GenHand·Kinematic-Tactile Aligned Retargeting·CARD·C2Dex·ReContact의 코드 공개 여부는 확인하지 못했습니다.
