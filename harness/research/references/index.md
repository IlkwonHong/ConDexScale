# Paper index

Search this file first during discussions. Add new papers at the bottom using the template.

<!-- Entry template
## key: firstauthor-year-keyword
- **Title:**
- **Authors / venue / year:**
- **Link:** (arXiv, DOI, project page)
- **Local file:** papers/xxxx.pdf (or none)
- **Status:** verified | unverified (how it was checked)
- **Found by / date:** researcher | Claude, YYYY-MM-DD
- **One-line summary:**
- **Relevance to our work:** (what it does that we build on, compare to, or must differ from)
- **Related ideas:** IDEA-NNN, ...
- **Notes:**
-->

## key: zhu-2026-chord
- **Title:** Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration (CHORD)
- **Authors / venue / year:** Xinghao Zhu, Zixi Liu, Shalin Jain 외 (NVIDIA), 교신 Yan Chang / 프로젝트 페이지 기준 CoRL 2026 / arXiv v1 2026-06-22, v2 2026-08-14
- **Link:** https://arxiv.org/abs/2607.00033 (프로젝트 https://nvidia-isaac.github.io/video_to_data/chord/, 코드 https://github.com/nvidia-isaac/video_to_data)
- **Local file:** papers/2607.00033v1.pdf
- **Status:** verified (v1 PDF를 쪽 단위로 읽음, v2는 arXiv HTML로 표·한계 절 대조, 공식 저장소 README 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손과 로봇 손의 접촉을 렌치 공간 지지 함수로 비교하는 보상을 강화학습 물체 추적 정책에 넣어, 4,739개 양손 과제에서 평균 성공률 82.12%를 보고합니다.
- **Relevance to our work:** 우리 렌치 유사도 계산의 직접 원천이며, 렌치를 보상으로만 쓰고 배포 전 필터·정적 성공 예측·다중 팔·점 손끝은 다루지 않으므로 우리가 갈라져야 하는 핵심 비교 대상입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md(언급), egocentric_datasets_and_reconstruction.md, 종합 보고서. v1과 v2의 표 1은 과제군·지표가 달라 직접 비교 불가. 마찰계수 0.5(참조 사본 함수 기본값)와 0.1(보상 설정값)은 서로 다른 층의 값이라고 종합 보고서가 정리.

## key: mandi-2025-dexmachina
- **Title:** DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation
- **Authors / venue / year:** Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song (Stanford·NVIDIA) / ICML 2026 / arXiv 2025-05-30
- **Link:** https://arxiv.org/abs/2505.24853 (코드 https://github.com/MandiZhao/dexmachina)
- **Local file:** none
- **Status:** verified (arXiv 초록·HTML 본문, GitHub README, 공식 문서, 로컬 코드 사본 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 자세 대신 물체 상태 궤적을 추적 목표로 삼고, 세기가 줄어드는 가상 물체 제어기 커리큘럼으로 ARCTIC 관절 물체 7클립을 다지 손 6종에 학습시킵니다.
- **Relevance to our work:** 우리 점 로봇 설정이 이 코드 위에 구축되어 있고, 접촉 위치 보상(기본 가중치 0)과 가상 물체 제어기는 보정 방식 비교의 기준선이 됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_datasets_and_reconstruction.md, 종합 보고서. 로컬 사본의 point2/point4 설정은 우리 프로젝트가 추가한 것. 코드 경로는 use_01/use_02 클립만 지원.

## key: chen-2024-objdex
- **Title:** Object-Centric Dexterous Manipulation from Human Motion Data (ObjDex)
- **Authors / venue / year:** Chen, Wang, Yang, Liu / arXiv 2024-11
- **Link:** https://arxiv.org/abs/2411.04005
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손 모션 캡처로 학습한 상위 생성 모델이 물체 목표 조건 손목 궤적을 내고 하위 강화학습 손가락 제어기가 실행합니다.
- **Relevance to our work:** 물체 궤적 추적형 강화학습 계열의 선행이며 DexMachina·CHORD의 비교 기준선입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: zhao-2024-dexh2r
- **Title:** DexH2R
- **Authors / venue / year:** Zhao, Zhu 외 / IEEE/ASME T-Mech 2025 / arXiv 2024-11
- **Link:** https://arxiv.org/abs/2411.04428
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 운동학 리타게팅 위에 과제 지향 잔차 정책을 학습하고, 시험 시 사람 손·물체 궤적을 지침으로 입력합니다.
- **Relevance to our work:** 잔차 강화학습으로 리타게팅을 보정하는 계열로, 우리의 학습 없는 2차 계획법 힘 배분과 대비됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md. CHORD 1저자가 공저.

## key: unknown-2025-dextrack
- **Title:** DexTrack
- **Authors / venue / year:** unknown / ICLR 2025
- **Link:** https://arxiv.org/abs/2502.09614
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 참조 궤적을 추적하는 신경 제어기를 성공 데이터 플라이휠과 호모토피 최적화로 학습합니다.
- **Relevance to our work:** 물체 궤적 추적형 강화학습 계열의 선행입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: li-2025-maniptrans
- **Title:** ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning
- **Authors / venue / year:** Li 외 (BIGAI) / CVPR 2025
- **Link:** https://arxiv.org/abs/2503.21860 (HTML https://arxiv.org/html/2503.21860v1)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 2 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 일반 궤적 모방기를 사전학습한 뒤 상호작용 제약 하의 잔차 모듈로 미세조정하고, 운동학 리타게팅만으로는 성공률이 4.6%(한 손)/0.0%(양손)임을 보고합니다.
- **Relevance to our work:** 운동학 리타게팅이 파지를 거의 성립시키지 못한다는 서론 인용 수치의 출처이자 CHORD의 비교 기준선입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, 종합 보고서

## key: lum-2025-human2sim2robot
- **Title:** Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration (Human2Sim2Robot)
- **Authors / venue / year:** Tyler Ga Wei Lum, Olivia Y. Lee, C. Karen Liu, Jeannette Bohg (Stanford) / CoRL 2025 / arXiv 2025-04
- **Link:** https://arxiv.org/abs/2504.12609 (코드 https://github.com/tylerlum/human2sim2robot)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 2 '원문 확인', 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 RGB-D 영상 한 편의 물체 자세 궤적을 형태 무관 보상으로 삼아 시뮬레이션 강화학습 후 실기로 옮기며, 재생 대비 +67%를 보고합니다.
- **Relevance to our work:** 역기구학 리타게팅의 실패 원인(손 추정 오차 20~30 cm, 도달 불가)을 명시한 인용원이고, 질량·마찰 [0.7, 1.3] 무작위화 범위는 우리 검증 프로토콜의 비교 축입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: kedia-2026-simtoolreal
- **Title:** SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
- **Authors / venue / year:** Kedia, Lum, Bohg 외 / RSS 2026 목록 / arXiv 2026-02
- **Link:** https://arxiv.org/abs/2602.16863
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약 수준으로만 기록)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 절차 생성 도구형 원시 도형으로 단일 물체 중심 강화학습 정책을 학습해 사람 영상에서 뽑은 자세 궤적을 실기에서 제로샷으로 추종합니다.
- **Relevance to our work:** 물체 궤적을 형태 무관 표현으로 쓰는 계열의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md. NVIDIA 저자 여부는 노트 2에서 미확인.

## key: unknown-2024-omnigrasp
- **Title:** OmniGrasp
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2407.11385
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 시뮬레이션 휴머노이드가 1,200개 이상의 물체를 집어 임의 궤적으로 운반하며, 물체 메시와 궤적만 필요합니다.
- **Relevance to our work:** 물체 궤적 추적형 배경 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2023-physhoi
- **Title:** PhysHOI
- **Authors / venue / year:** unknown / arXiv 2023
- **Link:** https://arxiv.org/abs/2312.04393
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 그래프로 몸-물체 접촉을 명시한 전신 모방이며 과제 보상이 없습니다.
- **Relevance to our work:** 접촉 그래프를 모방 목표로 쓴 선행으로, 우리의 접촉 그래프 수축과 표현 측면에서 비교됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: pan-2025-spider
- **Title:** SPIDER
- **Authors / venue / year:** Pan, Wang, Qi 외 / arXiv 2025-11
- **Link:** https://arxiv.org/abs/2511.09484
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 운동학 전용 사람 동작을 샘플링 기반 해법으로 동역학적으로 실행 가능한 로봇 궤적으로 바꾸고 6,520개 시뮬레이션 시연을 공개합니다.
- **Relevance to our work:** 물리 기반 리타게팅 기준선이자 데이터 원천이며 CHORD의 비교 기준선입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md(Spider 서지는 노트 2에서 미확인으로 표기)

## key: handa-2019-dexpilot
- **Title:** DexPilot
- **Authors / venue / year:** Handa et al. (노트 2 표기) / ICRA 2020 / arXiv 2019
- **Link:** https://arxiv.org/abs/1910.03135
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 '기억, 식별자 미확인'으로 표기)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 비전 기반 손 추적으로 23 구동 자유도 손-팔을 원격조작합니다.
- **Relevance to our work:** 운동학 리타게팅의 대표로, DexGEM 비교에서 접촉 유지율 2.0%가 보고되어 서론 인용 수치의 대상입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: qin-2023-anyteleop
- **Title:** AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System
- **Authors / venue / year:** Yuzhe Qin 외 (UCSD/NVIDIA) / RSS 2023
- **Link:** https://arxiv.org/abs/2307.04577 (노트 2는 v2 링크)
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 다양한 팔·손·카메라에 일반화되는 비전 기반 원격조작 시스템이며 dex-retargeting 라이브러리가 여기서 파생되었습니다.
- **Relevance to our work:** 운동학 리타게팅 기준선(dex-retargeting)의 출처입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: unknown-2025-dexflow
- **Title:** DexFlow: A Unified Approach for Dexterous Hand Pose Retargeting and Interaction
- **Authors / venue / year:** unknown / arXiv 2025-05
- **Link:** https://arxiv.org/abs/2505.01083 (노트 2는 https://arxiv.org/html/2505.01083v1)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 2 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 지도와 시간 일관성 손실로 관통을 억제하는 리타게팅이며, 기존 DexRetarget의 시뮬레이션 성공률이 5.35%임을 보고합니다.
- **Relevance to our work:** 운동학 리타게팅 실패 수치의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md. 노트 1은 '수치 없음'이라 적었으나 노트 2는 원문 Table II 수치를 기록(상충).

## key: unknown-2025-pkda
- **Title:** Progressive Kinematic-Dynamic Alignment
- **Authors / venue / year:** unknown / arXiv 2025-11
- **Link:** https://arxiv.org/abs/2511.10987
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손 영상만으로 운동학 매칭, 잔차 정책, 손목 궤적의 손 무관 3단계 전이를 하여 시뮬레이션 평균 73%를 얻습니다.
- **Relevance to our work:** 영상 기반 손 무관 전이의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2025-dreamgen
- **Title:** DreamGen
- **Authors / venue / year:** unknown / arXiv 2025-05
- **Link:** https://arxiv.org/abs/2505.12705
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 4는 '배경지식, 전문 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 영상 세계 모델로 신경 궤적을 생성해 휴머노이드가 한 과제 원격조작 데이터로 22개 새 행동을 배웁니다.
- **Relevance to our work:** 로봇 영상 합성 계열로 접촉·렌치 표현이 없어 우리와 직접 겹치지 않습니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md

## key: kareer-2024-egomimic
- **Title:** EgoMimic: Scaling Imitation Learning via Egocentric Video
- **Authors / venue / year:** Simar Kareer, Dhruv Patel 외, Danfei Xu (Georgia Tech), Chen Wang (Stanford) / ICRA 2025 / arXiv 2024-10-31
- **Link:** https://arxiv.org/abs/2410.24221 (노트 4는 https://arxiv.org/html/2410.24221)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 4 '출처 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Project Aria 안경의 손 추적 궤적과 로봇 데이터를 공동 학습하며, 사람 데이터에는 파지 정보가 없고 파지 성립 검사도 없습니다.
- **Relevance to our work:** egocentric 영상 학습이 파지를 검사하지 않는다는 분야 현황의 대표 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md

## key: brahmbhatt-2019-contactdb
- **Title:** ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging
- **Authors / venue / year:** S. Brahmbhatt, C. Ham, C. C. Kemp, J. Hays / CVPR 2019 Oral
- **Link:** https://arxiv.org/abs/1904.06830 (노트 2는 PDF 링크)
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 열화상으로 50개 가정용 물체의 사람 접촉 지도를 수집한 데이터셋입니다.
- **Relevance to our work:** 사람 접촉 지도의 표준 출처로 접촉 기반 전이 계보의 시작점입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: unknown-2020-contactpose
- **Title:** ContactPose
- **Authors / venue / year:** unknown (ContactDB 그룹으로 추정되나 노트에 저자 미기재) / ECCV 2020
- **Link:** https://arxiv.org/abs/2007.09545
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 arXiv 식별자 미확인으로 표기)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉, 손 자세, 물체 자세, RGB-D를 묶은 파지 데이터셋(2,306 파지, 25 물체, 50 피험자)입니다.
- **Relevance to our work:** 접촉 주석이 있는 정적 파지 데이터로 TopoRetarget 평가에도 쓰였습니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md. 식별자 확인 여부가 노트 사이에 상충.

## key: li-2023-gendexgrasp
- **Title:** GenDexGrasp: Generalizable Dexterous Grasping
- **Authors / venue / year:** Puhao Li, Tengyu Liu, Yuyang Li, Yiran Geng, Yixin Zhu, Yaodong Yang, Siyuan Huang / ICRA 2023
- **Link:** https://arxiv.org/abs/2210.00722 (노트 2는 PDF 링크)
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약, 손별 수치는 2차 집계)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 무관 접촉 지도를 생성한 뒤 각 손이 자기 운동학으로 맞추며, 학습 데이터 MultiDex는 힘 폐쇄 최적화로 합성합니다.
- **Relevance to our work:** 접촉 지도를 공유해 손가락 수 차이를 넘는 방식으로, 어떤 접촉이 남는지가 최적화 부산물이라는 점에서 우리 수축 정식화와 대비됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: chen-2025-dexonomy
- **Title:** Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy
- **Authors / venue / year:** Jiayi Chen, Yubin Ke, Lin Peng, He Wang (PKU) / RSS 2025
- **Link:** https://arxiv.org/abs/2504.18829 (노트 2는 https://arxiv.org/html/2504.18829v2)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 2 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** GRASP 분류 31종을 유형당 사람 템플릿 하나로 임의 물체에 합성하고 MuJoCo 6방향 외력으로 검증합니다(950만 파지).
- **Relevance to our work:** 우리 앵커 히트맵 학습의 사람 파지 데이터 원천이며 6방향 2초 외력·관통 1 cm 프로토콜은 우리 검증과 같은 축입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2023-unidexgraspplusplus
- **Title:** UniDexGrasp++
- **Authors / venue / year:** unknown / ICCV 2023
- **Link:** https://arxiv.org/abs/2304.00464
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 점군 조건 물체 무관 파지 정책을 기하 커리큘럼과 일반가-전문가 반복으로 학습합니다.
- **Relevance to our work:** 단일 ShadowHand 파지 배경 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: unknown-2024-graspxl
- **Title:** GraspXL
- **Authors / venue / year:** unknown (ETH·MPI) / ECCV 2024
- **Link:** https://arxiv.org/abs/2403.19649
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 3D 손-물체 데이터 없이 58개 물체로 학습한 정책이 50만 개 이상의 미지 물체에서 82.2%를 얻고 다른 손으로 전이됩니다.
- **Relevance to our work:** 파지 성공률이 평가 기준에 따라 크게 달라진다는 사례(PAIR Lab 재평가 51%)로 인용 가능합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: mahler-2017-dexnet2
- **Title:** Dex-Net 2.0
- **Authors / venue / year:** Mahler et al. (노트 2 표기) / RSS 2017
- **Link:** https://arxiv.org/abs/1703.09312
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 2는 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 해석적 강건 파지 지표로 라벨한 합성 670만 파지로 GQ-CNN을 학습해 평행 그리퍼 실기 93%를 얻습니다.
- **Relevance to our work:** 자세·마찰 불확실성 하 힘 폐쇄 확률로 라벨링하는 고전으로, 우리의 강건성 여유와 비교되는 배경입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md

## key: qin-2022-onehandmultiplehands
- **Title:** From One Hand to Multiple Hands
- **Authors / venue / year:** Qin 외 / 2022
- **Link:** https://arxiv.org/abs/2204.12490 (노트 4는 alphaxiv 링크)
- **Local file:** none
- **Status:** verified (노트 1 '검증'; 노트 4는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 조작자 MANO 형상으로 맞춤 손 모델을 만들고 손끝 키포인트 최적화로 Allegro·Schunk·Adroit에 오프라인 리타게팅합니다.
- **Relevance to our work:** 한 시연을 여러 종류의 로봇 손으로 옮기지만 여러 대의 협동은 아니어서, 우리의 다중 팔 배정과 구별됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2024-hot3d
- **Title:** HOT3D
- **Authors / venue / year:** unknown (Meta) / CVPR 2025
- **Link:** https://arxiv.org/abs/2411.19167 (노트 1은 https://arxiv.org/abs/2406.09598 을 인용, 홈페이지 https://facebookresearch.github.io/hot3d/, 코드 https://github.com/facebookresearch/hot3d)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 3 공식 페이지·본문 확인, 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Project Aria 안경과 Quest 3로 녹화한 egocentric 손-물체 데이터셋으로, 광학 마커 모션 캡처 정답의 UmeTrack·MANO 손, 강체 33개의 스캔 메시와 6자유도 자세를 제공합니다(833분, 깊이 없음, 접촉 주석 없음).
- **Relevance to our work:** egocentric 영상 + 미터 메시 + 6자유도 궤적 + MANO 손을 모두 공식 제공하는 유일한 데이터셋이라 우리 egocentric 확장의 1순위 후보입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_datasets_and_reconstruction.md, egocentric_video_to_robot_pipelines.md, 종합 보고서. arXiv 식별자가 노트 사이에 다름(2406.09598 대 2411.19167). 정답 mm 정확도·물체 질량·라이선스 상업 조항은 미확인.

## key: dai-2026-card
- **Title:** CARD: Contact-Anchored Retargeting for Dexterous Manipulation
- **Authors / venue / year:** Dai, He, Hogan 외 / arXiv 2026-10-27
- **Link:** https://arxiv.org/abs/2610.36539
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물체 모델 없이 공유 손-접촉 표현으로 리타게팅하고 명시적 접촉력 목표를 만들며, 영상용 접촉 예측 모델 Vid-HACO를 포함합니다.
- **Relevance to our work:** 접촉 구조를 리타게팅 1급 목표로 올린 2026년 흐름의 대표이며, Vid-HACO는 영상→접촉 후보 모듈입니다(코드 공개 미확인).
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: yang-2026-contactanchored
- **Title:** Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning (DexGEM)
- **Authors / venue / year:** Zihao Yang, Chengyuan Liu 외, JieQ Sun (DexGEM Lab, 상하이교통대, 동제대, DexRobot) / arXiv v1 2026-09-21, v2 2026-11-03
- **Link:** https://arxiv.org/abs/2609.24093 (노트 2·4는 PDF/HTML v2 링크, 코드 https://github.com/DexGEM-Lab/real2sim2real)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 2 '원문 PDF 텍스트 확인', 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 모션 캡처를 시뮬레이션 MANO 손으로 재실행해 접촉을 복원한 뒤 지정 손끝 표면점을 접촉 위치에 맞추고 잔차 강화학습으로 보정하여 단손 성공률을 7.3%에서 59.3%로 올립니다.
- **Relevance to our work:** 한 손 안의 접촉 앵커 배정이라는 점에서 우리 다중 팔 앵커 배정에 가장 가까운 기술 조각이며, DexPilot 접촉 유지율 2.0% 등 서론 인용 수치의 출처입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: wu-2026-toporetarget
- **Title:** TopoRetarget
- **Authors / venue / year:** Wu 외 / arXiv 2026-06-15
- **Link:** https://arxiv.org/abs/2606.16272
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손-물체 키포인트 희소 상호작용 그래프와 거리 가중 라플라시안 변형으로 과제 관련 접촉 구조를 보존하는 리타게팅입니다.
- **Relevance to our work:** 손 중심 리타게팅이 접촉 구조를 잃는다는 비판의 인용원이며 그래프 기반 접촉 표현으로 우리 접촉 그래프와 비교됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: spinola-2026-dexoak
- **Title:** DexOAK
- **Authors / venue / year:** Spinola, Katzschmann, Schmid / arXiv 2026-10-19
- **Link:** https://arxiv.org/abs/2610.26762
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손-물체 결합 그래프의 라플라시안으로 물체 상대 접촉 기하를 보존해 ARCTIC ADD-AUC를 62.7에서 77.9%로 올립니다.
- **Relevance to our work:** 같은 ARCTIC 데이터에서 접촉 기하를 보존하는 경쟁 방법입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: sadjadpour-2026-morphometric
- **Title:** Morphometric Imitation
- **Authors / venue / year:** Sadjadpour 외 / arXiv 2026-09-23
- **Link:** https://arxiv.org/abs/2609.28660
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 보존 리타게팅, 잔차 강화학습, 시각운동 정책 증류로 Sharpa 손 실기 89.3%를 보고합니다.
- **Relevance to our work:** 접촉 보존 리타게팅 계열의 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: gu-2026-recontact
- **Title:** ReContact
- **Authors / venue / year:** Gu 외 / arXiv 2026-11-02
- **Link:** https://arxiv.org/abs/2611.02968
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 영상에서 접촉 단계·쌍·앵커·누르는 방향·고착/미끄럼 모드의 물체 중심 접촉 계획을 만들고 Unitree G1 잔차 정책이 계획된 앵커에 힘을 내도록 보상합니다.
- **Relevance to our work:** 영상에서 접촉 앵커를 계획하는 점에서 우리의 앵커 정식화와 가장 가까운 영상 쪽 경쟁 연구입니다(코드 공개 미확인).
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: sharma-2026-demomimic
- **Title:** One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry (DemoMimic)
- **Authors / venue / year:** Satvik Sharma, Samrat Sahoo, Huang Huang, Fei-Fei Li, Jiajun Wu, Dorsa Sadigh, Jeannette Bohg (Stanford) / arXiv v2 2026-09-03
- **Link:** https://arxiv.org/abs/2609.01938 (노트 4는 https://arxiv.org/html/2609.01938v2)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 법선 정렬과 지속 접촉 보상으로 단일 시연을 여러 물체에 일반화하여 실기 평균 71%를 얻지만 로봇 손 모양 상자에서는 39%로 떨어집니다.
- **Relevance to our work:** 국소 접촉 기하가 결과를 지배한다는 근거로, 2~3접촉에서 사람 전략을 그대로 옮기지 않는 설계의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2026-weave
- **Title:** Weave
- **Authors / venue / year:** unknown / arXiv 2026-09-15
- **Link:** https://arxiv.org/abs/2609.16683
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 인지 리타게팅과 접근 동작 보완으로 휴머노이드 단일 정책을 학습하고 접촉 주석 9,000 롤아웃을 공개합니다.
- **Relevance to our work:** 접촉 인지 리타게팅 계열이자 접촉 주석 데이터 원천입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: he-2026-kinematictactile
- **Title:** Kinematic-Tactile Aligned Dexterous Retargeting with Contact Wrench Space Alignment
- **Authors / venue / year:** He, Xu, Chen 외 / arXiv 2026-10-29
- **Link:** https://arxiv.org/abs/2610.39582
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 촉각 힘으로 렌치 지지 프로파일을 만들고 로봇 손끝 힘 목표를 최적화하는 3단계 리타게팅으로 붓기 실기 96.7%를 얻습니다.
- **Relevance to our work:** CHORD 외에 렌치 공간까지 올라간 유일한 연구이지만 촉각 장갑 데이터를 전제하므로, 영상 입력·다중 팔인 우리와 입력 조건이 다릅니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md. 손 이름 미기재, 코드 공개 미확인.

## key: wu-2026-reforce
- **Title:** ReForce: Learning Force-aware Retargeting for Dexterous Manipulation
- **Authors / venue / year:** Yuhang Wu, Lingqi Zeng, Changwei Jing, Jianglong Ye, Xiaolong Wang (UC San Diego) / arXiv 2026-08-16
- **Link:** https://arxiv.org/abs/2608.15560 (노트 4는 https://arxiv.org/html/2608.15560v1)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 운동학 리타게팅 행동에 시뮬레이션으로 학습한 힘 추적기의 잔차를 더해 목표 손끝 힘을 달성합니다(입력은 장갑·FSR 센서).
- **Relevance to our work:** 힘을 명시 목표로 두는 계열로, 법선 방향 힘만 다룬다는 한계가 우리 렌치 여유와 대비됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2026-fold
- **Title:** FoLD
- **Authors / venue / year:** unknown / arXiv 2026-09-27
- **Link:** https://arxiv.org/abs/2609.33551
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 시연과 현재 상태에서 보상 힘장을 계산해 시연 물체 운동을 선호하는 힘 사전분포로 쓰고 잔차 정책이 리타게팅 동작을 조정합니다.
- **Relevance to our work:** 힘 기반 리타게팅 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-softact
- **Title:** SoftAct
- **Authors / venue / year:** unknown / arXiv 2026-04-01
- **Link:** https://arxiv.org/abs/2604.01224
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** VR에서 사람 접촉력을 캡처해 비의인화 공압 소프트 손의 손가락에 비례 배분합니다.
- **Relevance to our work:** 사람 접촉력을 형태가 다른 몸체로 배분한 사례로, 접촉 배정 문제의 인접 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-haco-force
- **Title:** HACo
- **Authors / venue / year:** unknown / arXiv 2026-09-29
- **Link:** https://arxiv.org/abs/2609.36596
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손끝 촉각과 관절 토크로 힘 조절 행동을 학습해 실기 5과제 평균 83%를 얻습니다.
- **Relevance to our work:** 힘·촉각 계열 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md. 노트 3의 손 접촉 추정기 HACO(arXiv 2505.11152)와는 다른 논문.

## key: unknown-2026-tacot
- **Title:** TacOT
- **Authors / venue / year:** unknown / arXiv 2026-10-03
- **Link:** https://arxiv.org/abs/2610.04363
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 촉각 유도 최적수송으로 사람-로봇 시연을 정렬합니다.
- **Relevance to our work:** 힘·촉각 계열 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-whentouchmatters
- **Title:** When Does Touch Matter?
- **Authors / venue / year:** unknown / arXiv 2026-09-21
- **Link:** https://arxiv.org/abs/2609.24068
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** LEAP Hand에서 손가락별 렌치 추정과 택셀 결합 정책이 비전만 쓴 정책보다 낫다는 것(24/25 대 14/25)을 보입니다.
- **Relevance to our work:** 손가락별 렌치 추정 사례로 참고할 수 있습니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: atar-2026-transferringcontact
- **Title:** Transferring Contact, Not Just Motion
- **Authors / venue / year:** Atar, Huang, Yip / arXiv 2026-06-14
- **Link:** https://arxiv.org/abs/2606.15516
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 자세 잠재와 손별 노력 신호를 관절 토크로 보정해 손끝 힘·손가락별 하중 기술자로 매핑하여 구조가 다른 손 사이 순응 파지를 전이합니다.
- **Relevance to our work:** 접촉 하중을 형태 간에 전이하는 인접 연구입니다(수치 없음).
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-egocpt
- **Title:** EgoCPT
- **Authors / venue / year:** unknown / arXiv 2026-10-13
- **Link:** https://arxiv.org/abs/2610.17572
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric RGB 영상과 참조 손 자세만으로 접촉점을 반복 정제해 EgoDex 재구성 67.3%, TACO 92.0%를 보고합니다.
- **Relevance to our work:** egocentric 영상에서 접촉을 복원하는 후보 모듈입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-c2dex
- **Title:** C2Dex
- **Authors / venue / year:** unknown / arXiv 2026-08-07
- **Link:** https://arxiv.org/abs/2608.07045
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단안 영상의 프레임별 잡음 관측을 물체 정준 공간에 모아 안정적 물체 측 접촉을 복원하고 잔차 강화학습을 합니다.
- **Relevance to our work:** 영상→접촉 복원 후보 모듈입니다(코드 공개 미확인).
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2025-dexman
- **Title:** DexMan
- **Authors / venue / year:** unknown / arXiv 2025-10-09
- **Link:** https://arxiv.org/abs/2510.08475
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 3인칭·생성 영상에서 잡음 있는 손-물체 자세를 추정해 접촉 기반 보상으로 양손 휴머노이드를 시뮬레이션에서 학습합니다.
- **Relevance to our work:** 영상→접촉 보상 경로의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-doasido
- **Title:** Do as I Do
- **Authors / venue / year:** unknown / arXiv 2026-06-17
- **Link:** https://arxiv.org/abs/2606.19333
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 야생 단안 영상에서 손-물체를 재구성한 뒤 다지 손으로 리타게팅합니다.
- **Relevance to our work:** 영상→다지 손 경로의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-egoengine
- **Title:** EgoEngine
- **Authors / venue / year:** unknown / arXiv 2026-06-10
- **Link:** https://arxiv.org/abs/2606.12604
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric 영상의 사람을 로봇으로 치환한 관측 영상과 타당성 제약 행동 궤적을 만듭니다.
- **Relevance to our work:** 영상 편집 계열의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: wang-2026-humanego
- **Title:** HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos
- **Authors / venue / year:** Zhi (Leo) Wang, Botao He 외, Yiannis Aloimonos (University of Maryland) / arXiv 2026-05-24 (v3 2026-09-14)
- **Link:** https://arxiv.org/abs/2605.24934 (노트 4는 https://arxiv.org/html/2605.24934)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 4 '출처 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손을 가상 평행 조 그리퍼로 환원해 과제당 30분 egocentric 영상으로 실기 4과제 평균 92.5%를 얻지만 파지 성립 검사는 없습니다.
- **Relevance to our work:** 엄지-검지 두 점으로 파지를 붕괴시키는 주류 방식의 사례로, 우리의 접촉 그래프 수축과 대비됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2026-acrossdex
- **Title:** AcrossDEX 1.0
- **Authors / venue / year:** unknown / arXiv 2026-10-29
- **Link:** https://arxiv.org/abs/2610.40210
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** HOT3D Aria 영상으로 양손 동작을 사전학습한 뒤 Inspire·Allegro 손에 적응합니다.
- **Relevance to our work:** HOT3D를 로봇 학습에 쓴 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: hoque-2025-egodex
- **Title:** EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **Authors / venue / year:** Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, Jian Zhang (Apple) / ICLR 2026 / arXiv 2025-05
- **Link:** https://arxiv.org/abs/2505.11709 (저장소 https://github.com/apple/ml-egodex)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 3 공식 확인, 노트 4는 '초록만 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Apple Vision Pro로 녹화한 829시간·194과제 egocentric 데이터셋으로 3D 손 관절 추적을 제공하지만 물체 자세·메시는 없습니다(CC-by-NC-ND).
- **Relevance to our work:** 규모가 큰 손 모션 사전분포로는 쓸 수 있으나 물체 3D가 없어 접촉 정답 공급원으로는 부적합합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_datasets_and_reconstruction.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: wang-2026-kite
- **Title:** KITE
- **Authors / venue / year:** Wang, Jin, Fang / arXiv 2026-06-20
- **Link:** https://arxiv.org/abs/2606.22113
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 조작을 형태 무관 상호작용 예측(접촉 패턴 기반 의도)과 형태별 운동학 디코딩으로 분리해 평행 그리퍼·다지 손에 제로샷 전이합니다.
- **Relevance to our work:** 그리퍼까지 가는 교차 형태 연구로, 사람 접촉을 그리퍼 접촉으로 직접 매핑하지 않고 추상화로 우회한다는 점에서 우리와 다릅니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: dan-2026-xreset
- **Title:** X-Reset
- **Authors / venue / year:** Dan, Ma, Zhan / arXiv 2026-09-28
- **Link:** https://arxiv.org/abs/2609.35715
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 리타게팅 동작을 추적하지 않고 불안정 상태를 걸러낸 리셋 분포로만 쓰며, 22자유도 손을 단 팔 2대와 평행 그리퍼에서 시뮬레이션→실기 제로샷을 보입니다.
- **Relevance to our work:** 물리 안정성 필터를 쓰는 교차 형태 연구로, 우리의 배포 전 선별과 비교됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: gupta-2026-lucid
- **Title:** LUCID
- **Authors / venue / year:** Gupta, Shi, Yuan / arXiv 2026-06-10
- **Link:** https://arxiv.org/abs/2606.11628
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 인터넷 영상으로 형태 무관 단기 의도 모델을 학습해 다지 손과 평행 그리퍼 제어기에 공통 인터페이스로 씁니다.
- **Relevance to our work:** 접촉 대신 의도로 우회하는 교차 형태 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-gott
- **Title:** GOTT
- **Authors / venue / year:** unknown / arXiv 2026-10-02
- **Link:** https://arxiv.org/abs/2610.03861
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접근-접촉 획득-이동 틀에서 재사용 가능한 교차 형태 접촉 획득 프리미티브를 학습합니다.
- **Relevance to our work:** 교차 형태 접촉 획득의 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: yin-2025-motionfield
- **Title:** Object-centric 3D Motion Field
- **Authors / venue / year:** Yin, Yang, Abbeel / NeurIPS 2025
- **Link:** https://arxiv.org/abs/2506.04227
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 영상의 물체 3D 운동장을 행동 표현으로 써서 손 동작 대신 물체 운동으로 제로샷 제어합니다.
- **Relevance to our work:** 물체 운동으로 우회하는 교차 형태 표현의 사례입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2025-unibyd
- **Title:** UniBYD
- **Authors / venue / year:** unknown / arXiv 2025-12
- **Link:** https://arxiv.org/abs/2512.11609
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 통일 형태 표현과 모방에서 탐색으로 어닐링되는 PPO로 여러 손을 다루고 UniManip 벤치마크를 제안합니다.
- **Relevance to our work:** 교차 형태 학습 배경입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: gao-2026-gnr
- **Title:** Generative Neural Retargeting for Human-to-Robot Dexterous Manipulation (GNR)
- **Authors / venue / year:** Dechen Gao, Yue Yang 외, Eric Whitmire (Meta Reality Labs Research 외) / arXiv 2026-10-08
- **Link:** https://arxiv.org/abs/2610.12440 (노트 2는 https://alphaxiv.org/abs/2610.12440v1, 노트 4는 https://arxiv.org/html/2610.12440)
- **Local file:** none
- **Status:** verified (노트 1 '검증', 노트 4 '출처 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** HOT3D 모션을 물리 시뮬레이터에서 재수립하고 추종 오차·관통·최소 지속 쥐는 힘·손가락별 힘 상한의 네 기준을 통과한 22.3만 궤적을 만들며, 역기구학 3.7% 대 99.3%를 보고합니다.
- **Relevance to our work:** 물리 채택 기준을 쓰는 가장 가까운 선행으로, 우리 ρ는 스칼라 쥐는 힘이 아니라 모든 외란 방향에 대한 여유라는 점에서 갈라집니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md, 종합 보고서. NIST 결과가 본문 92.2%와 표 89.1%로 불일치(노트 4).

## key: unknown-2026-oath
- **Title:** OATH
- **Authors / venue / year:** unknown / arXiv 2026-10-09
- **Link:** https://arxiv.org/abs/2610.14379
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물리 기반 리타게팅을 운동학 참조 위 잔차 학습으로 재정의하고 링크별 국소 접촉 오차를 관측합니다.
- **Relevance to our work:** 물리 기반 리타게팅 경쟁 연구입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-dexone2many
- **Title:** Dex-One2Many
- **Authors / venue / year:** unknown / arXiv 2026-10-08
- **Link:** https://arxiv.org/abs/2610.12470
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 엄격한 동작 매칭이 일반화를 제한한다고 비판하고 장면 그래프로 추상화합니다.
- **Relevance to our work:** 운동학 리타게팅 비판의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-regrind
- **Title:** A Minimalist Retargeting-Guided RL Recipe (REGRIND)
- **Authors / venue / year:** unknown / arXiv 2026-07-13
- **Link:** https://arxiv.org/abs/2607.11874
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물체 중심 키포인트 추적, 잔차 강화학습, 시스템 동정으로 가위·드라이버 실기 제로샷을 보이고 전이 요인을 분석합니다.
- **Relevance to our work:** 잔차 강화학습 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: pei-2026-dexhandsurvey
- **Title:** Dexterous Robotic Hands in the Foundation-Model Era (서베이)
- **Authors / venue / year:** Pei / arXiv 2026-11-04
- **Link:** https://arxiv.org/abs/2611.06724
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2024~2026 다지 손 문헌을 하드웨어·데이터·교차 형태 전이·학습·평가로 정리하고, 형태가 다른 손 사이 접촉 집약 기술의 제로샷 전이 증거가 빈약하다고 결론짓습니다.
- **Relevance to our work:** 교차 형태 접촉 전이가 미해결이라는 서베이 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: qi-2026-genhand
- **Title:** GenHand: generalised human grasp kinematic retargeting
- **Authors / venue / year:** Qi, Popoola, Imran, Ahmad (University of Glasgow) / npj Robotics 2026, 4:19
- **Link:** https://doi.org/10.1038/s44182-026-00076-1 (사본 PDF https://eprints.gla.ac.uk/377598/2/377598.pdf)
- **Local file:** none
- **Status:** verified (글래스고 사본 PDF 8쪽까지 읽음; '사람 시연이 이미 힘 폐쇄'라는 가정 부분은 검색 요약 수준, 9쪽 이후 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 파지 이미지 한 장에서 MANO 손과 물체 부호 거리장을 추정하고 접촉 앵커 근방 힘 폐쇄 최적화 후 2지·3지·4지·5지 그리퍼의 운동학을 맞추며, 키 벡터 기준선보다 39.8% 앞섭니다.
- **Relevance to our work:** 우리 프로젝트의 직접 계보이며, 정적→동적, 단일 그리퍼→다중 팔 점 접촉, 들어올리기→물성 변화 폐루프로의 확장으로 우리 위치를 서술하는 기준점입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, 종합 보고서. 코드 공개 여부·실기 실험 유무 미확인.

## key: unknown-2026-physv2a
- **Title:** PhysV2A
- **Authors / venue / year:** unknown / arXiv 2026-07-10
- **Link:** https://arxiv.org/abs/2607.09365
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 영상 유도 6D 물체 운동과 RGB-D 파지 후보 쌍을 도달 가능성 검사로 거르고 순위화합니다(힘 폐쇄 언급 없음).
- **Relevance to our work:** 운동학 검사로만 거르는 사례로, 우리의 렌치 기반 선별과 대비됩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-cooperativebimanual
- **Title:** Learning Cooperative Bimanual Dexterous Grasp of Large Objects with Tactile Sensing
- **Authors / venue / year:** unknown / arXiv 2026-08-11
- **Link:** https://arxiv.org/abs/2608.10383
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 시점 점군에서 확산 모델로 큰 물체의 이중 팔 관절 수준 파지를 생성하며 실기 다중모달 데이터셋을 공개합니다.
- **Relevance to our work:** 큰 물체 다중 팔 협동 파지의 인접 연구이지만 사람 시연 기반이 아닙니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2011-urai-multirobot
- **Title:** URAI 2011 학회 논문(사람 시연으로 다중 로봇 파지 계획 학습, 제목 미기재)
- **Authors / venue / year:** unknown / URAI 2011
- **Link:** https://unpaywall.org/10.1109%2FURAI.2011.6145992
- **Local file:** none
- **Status:** unverified (노트 표기 '검증, 요약 수준': 초록 일부만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 시연으로 다중 로봇 파지 계획을 학습한 것으로 보이는 간접 선행입니다.
- **Relevance to our work:** 사람 시연→다중 로봇 파지의 드문 간접 선행이지만 손가락을 팔에 배정하지는 않는다고 노트가 판단합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2017-aalto-thesis
- **Title:** Aalto 2017 석사논문(2 에이전트 하중 분담으로 파지 렌치 최소화, 제목 미기재)
- **Authors / venue / year:** unknown / Aalto University 2017
- **Link:** https://aaltodoc.aalto.fi/items/08f604ef-76df-4c22-8baf-4011e2b489f8
- **Local file:** none
- **Status:** unverified (노트 표기 '검증, 요약 수준')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 두 에이전트가 하중을 분담해 파지 렌치를 최소화합니다.
- **Relevance to our work:** 다중 에이전트 힘 분배의 간접 선행입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2016-rss12-p19
- **Title:** RSS XII p19(다중 팔 가상 물체 동역학계 조정, 제목 미기재)
- **Authors / venue / year:** unknown / RSS 2016
- **Link:** https://www.roboticsproceedings.org/rss12/p19.html
- **Local file:** none
- **Status:** unverified (노트 표기 '검증, 요약 수준')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 다중 팔을 가상 물체 동역학계로 조정합니다.
- **Relevance to our work:** 다중 팔 협동 제어의 간접 선행으로, 우리 2차 계획법 힘 배분과 역할이 겹칩니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md. 연도는 RSS XII 회차에서 따온 것으로 노트에 연도 직접 기재 없음.

## key: weisz-2012-poseerrorrobust
- **Title:** Pose Error Robust Grasping from Contact Wrench Space Metrics
- **Authors / venue / year:** Jonathan Weisz, Peter K. Allen (Columbia) / ICRA 2012, pp. 557–562
- **Link:** https://www.cs.columbia.edu/~allen/PAPERS/weisz_icra12.pdf (dblp https://dblp.org/rec/html/conf/icra/WeiszA12)
- **Local file:** none
- **Status:** verified (노트 2 '원문 PDF 텍스트 확인'; 노트 1은 '검색 요약 + 서지')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 자세 오차에 대한 힘 폐쇄 강건성으로 순위화한 파지가 Ferrari–Canny ε 순위 파지보다 실기에서 더 자주 성공함(91% 대 67%)을 보입니다.
- **Relevance to our work:** 우리 발견(렌치 유사도가 아니라 접촉 실현 오차가 성공을 예측)을 같은 계보의 새 증거로 자리매김하는 핵심 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, cross_morphology_and_physical_verification.md, 종합 보고서. '72 of ...' 문장은 줄바꿈으로 잘려 전체 문맥 미확정.

## key: rubert-2019-graspsuccess
- **Title:** Predicting grasp success in the real world – A study of quality metrics and human assessment
- **Authors / venue / year:** Rubert 외 / Robotics and Autonomous Systems 2019
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0921889019300247
- **Local file:** none
- **Status:** unverified (노트 표기 '초록 수준'; 원문 접근 실패로 구체 수치 미기록)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물리 시뮬레이션 기반 지표가 고전 ε 지표보다 실세계 파지 성공을 더 일관되게 예측한다고 보고합니다.
- **Relevance to our work:** 해석적 지표의 성공 예측력 한계를 보이는 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, 종합 보고서

## key: unknown-2022-deformablegraspquality
- **Title:** 변형체 파지 품질 지표 연구(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/abs/2203.12420
- **Local file:** none
- **Status:** verified (노트 표기 '검증': 원문 또는 arXiv 초록 페이지를 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대부분의 파지 품질 지표가 접촉점에 의존한다는 문제를 지적합니다.
- **Relevance to our work:** 품질 지표의 접촉 위치 의존성에 대한 간접 근거입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-donic
- **Title:** DONIC
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2610.29997, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-gen2dex
- **Title:** Gen2Dex
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2610.24392, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-contactfactorization
- **Title:** Contact-Aware Imitation Learning Through Contact Factorization
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2610.09533, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-flashdexretarget
- **Title:** FlashDexRetarget
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2610.01849, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-touch2robot
- **Title:** Touch2Robot
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2609.24660, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-groundinggenvideoplans
- **Title:** Grounding Generated Video Plans
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2609.10050, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-robotok
- **Title:** RoboTok
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2609.03199, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-bora
- **Title:** BORA
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** none in notes (arXiv 2605.30226, Semantic Scholar 인용 목록 https://api.semanticscholar.org/graph/v1/paper/arXiv:2607.00033/citations?fields=title,year,externalIds,authors&limit=100)
- **Local file:** none
- **Status:** unverified (CHORD 인용 목록에서 제목과 식별자만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** CHORD를 인용한 2026년 논문으로 제목만 확인되었습니다.
- **Relevance to our work:** CHORD 후속·경쟁 연구 후보이며 내용 확인이 필요합니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md

## key: unknown-2026-exs2d
- **Title:** One-to-Two Acting (ExS2D)
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2606.19897
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 팔 데이터를 두 팔로 확장합니다.
- **Relevance to our work:** 사람 한 손→여러 팔 검색에서 나온 가장 가까운 결과 중 하나이지만 손가락→팔 배정은 아닙니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-odil
- **Title:** One-Shot Dual-Arm Imitation Learning (ODIL)
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/pdf/2503.06831
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 한 번의 양팔 시연으로 양팔 조작을 학습합니다.
- **Relevance to our work:** 양팔 모방 학습의 인접 연구입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: iodice-unknown-sawing
- **Title:** Iodice et al. 두 사람 톱질 영상 → 두 대의 Panda 재현(제목 미기재)
- **Authors / venue / year:** Iodice et al. / 연도·발표처 미기재
- **Link:** https://re.public.polimi.it/retrieve/b5e6f0bf-a283-4bdd-a251-7c660c9f07f1/11311-1228206_Jodice.pdf
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 두 사람의 톱질 영상에서 리더–팔로워 역할과 임피던스를 추출해 두 대의 로봇 팔에 재현합니다.
- **Relevance to our work:** 두 사람→두 로봇 구도로, 한 손의 손가락→여러 팔인 우리와 다릅니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md. 연도가 노트에 없어 키에 unknown 사용.

## key: gioioso-2014-flyinghand
- **Title:** The Flying Hand: a Formation of UAVs for Cooperative Aerial Tele-Manipulation
- **Authors / venue / year:** G. Gioioso, A. Franchi, G. Salvietti, S. Scheggi, D. Prattichizzo / ICRA 2014, pp. 4335–4341, DOI 10.1109/ICRA.2014.6907490
- **Link:** https://hal.archives-ouvertes.fr/hal-01990572 (UTwente, dblp 링크도 노트에 있음)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약: 초록 문장 수준')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손끝 하나로 접촉점 하나를 담당하는 쿼드로터 편대를 사람 손끝 운동으로 원격조작하며, 검증은 사람이 루프에 든 시뮬레이션뿐입니다.
- **Relevance to our work:** '손가락 하나 = 로봇 하나 = 접촉점 하나' 개념을 선점한 선행이므로, 우리 참신성은 접촉 앵커 예측·렌치 여유 최적화·물리 검증·동적 궤적에 두어야 합니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2013-flyinghand-workshop
- **Title:** Hand Driven UAV Formation for Cooperative Grasping and Transportation: the Flying Hand
- **Authors / venue / year:** unknown / 워크숍 2013
- **Link:** https://homepages.laas.fr/afranchi/robotics/?q=node/167
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** The Flying Hand의 선행 워크숍판입니다.
- **Relevance to our work:** The Flying Hand와 같은 개념의 초기판입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: salvietti-2013-objectbilateral
- **Title:** Object-based bilateral telemanipulation between dissimilar kinematic structures
- **Authors / venue / year:** Salvietti, Meli, Gioioso, Malvezzi, Prattichizzo / IROS 2013
- **Link:** https://sirslab.dii.unisi.it/papers/2013/Salvietti.IROS.2013.Med.Fin.pdf
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 엄지·검지 햅틱 장치로 운동학이 다른 로봇 손을 조작하며 마스터·슬레이브 가상 물체에 같은 렌치가 작용하도록 합니다.
- **Relevance to our work:** 가상 물체 매개로 형태가 다른 몸체 사이를 잇는 선행으로, 협동 파지 확장 언급이 있습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2017-objectbased-tmech
- **Title:** Salvietti 그룹의 다중 접촉 객체 기반 원격조작 (IEEE/ASME T-Mech 2017, 제목 미기재)
- **Authors / venue / year:** unknown / IEEE/ASME Transactions on Mechatronics 2017
- **Link:** https://usiena-air.unisi.it/handle/11365/1009820
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 다중 접촉판으로 확장하고 여러 로봇이 같은 물체를 원격조작하는 협동 파지 시나리오로 확장할 수 있다고 언급합니다.
- **Relevance to our work:** 다중 로봇 협동 확장 언급의 출처이나 실험 유무는 미확인입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: gioioso-2012-synergymapping
- **Title:** An Object-Based Approach to Map Human Hand Synergies onto Robotic Hands with Dissimilar Kinematics
- **Authors / venue / year:** Gioioso et al. / RSS 2012
- **Link:** https://www.roboticsproceedings.org/rss08/p13.pdf
- **Local file:** none
- **Status:** unverified (검색 결과 제목만 확인, 내용은 기억)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손 시너지를 가상 물체 변형을 통해 운동학이 다른 로봇 손으로 사상하는 물체 중심 매핑의 고전입니다.
- **Relevance to our work:** 손가락 수에 의존하지 않는 물체 중심 매핑의 원조로 배경 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: song-2025-collabot
- **Title:** CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation
- **Authors / venue / year:** Kun Song 외 7인 / arXiv 2025-08-05
- **Link:** https://arxiv.org/abs/2508.03526v1
- **Local file:** none
- **Status:** verified (노트 표기 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 임의 대수의 이동 매니퓰레이터가 대형 물체를 함께 잡는데, 파지점은 시각-언어 모델이 고르고 렌치 지표는 평가에만 쓰며 전체 성공률 52%입니다.
- **Relevance to our work:** 대형 물체 다중 로봇 파지점 선택이라는 같은 문제 설정의 가장 가까운 선행이며, 실패 절반이 파지점 선택에서 나왔다는 분석은 렌치 여유를 선택 단계에 넣는 우리 동기로 인용됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: shaoul-2025-flowmatchingcogen
- **Title:** Collaborative Multi-Robot Non-Prehensile Manipulation via Flow-Matching Co-Generation
- **Authors / venue / year:** Yorai Shaoul, Zhe Chen, M. N. Gul Mohamed, F. Pecora, M. Likhachev, J. Li / arXiv v1 2025-11-14, v2 2026-02-17
- **Link:** https://arxiv.org/abs/2511.10874v1
- **Local file:** none
- **Status:** verified (노트 표기 '원문 초록 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 시각 관측으로부터 접촉 형성과 조작 궤적을 공동 생성하는 비파지(밀기) 다중 로봇 조작입니다.
- **Relevance to our work:** 다중 로봇 접촉 배치 생성의 인접 연구이지만 사람 시연·파지가 아닙니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-conpose
- **Title:** ConPoSe
- **Authors / venue / year:** unknown / arXiv 2025-10
- **Link:** https://arxiv.org/abs/2510.08705?context=cs
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** LLM과 국소 탐색으로 협동 밀기의 로봇–물체 접촉점을 선택합니다.
- **Relevance to our work:** 다중 로봇 접촉점 선택의 인접 연구(밀기)입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2026-formationpushing
- **Title:** Cooperative Optimal Multi-Robot Formation Arrangement for Object Pushing
- **Authors / venue / year:** unknown / ICR 2026
- **Link:** https://ia.spcras.ru/index.php/icr2026/article/view/17354
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 입자 군집 최적화로 물체 경계의 접촉점을 고르며 기준을 파지 행렬 해석에서 가져옵니다.
- **Relevance to our work:** 렌치 해석 기반 다중 로봇 접촉점 선택의 인접 연구(밀기)입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: liu-2024-interactionwrench
- **Title:** Distributed Motion Control of Multiple Mobile Manipulators for Reducing Interaction Wrench in Object Manipulation
- **Authors / venue / year:** Wenhang Liu, Meng Ren, Kun Song, Gaoming Chen, Michael Yu Wang, Zhenhua Xiong / arXiv (v4 2025-04-07)
- **Link:** https://arxiv.org/abs/2406.05613
- **Local file:** none
- **Status:** verified (노트 표기 '원문 초록 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 관절 속도 제어만으로 통신 지연을 고려해 불필요한 상호작용 렌치를 줄이는 분산 운동 제어입니다.
- **Relevance to our work:** 접촉점이 주어진 뒤의 힘 분배 문제로, 우리 2차 계획법 힘 할당기와 역할이 겹칩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: culbertson-2021-decentralizedadaptive
- **Title:** Decentralized Adaptive Control for Collaborative Manipulation of Rigid Bodies
- **Authors / venue / year:** P. Culbertson, J.-J. Slotine, M. Schwager / IEEE T-RO 37(6):1906–1920, 2021
- **Link:** https://arxiv.org/pdf/2005.03153
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물체 질량·마찰·부착점을 모르는 상태에서 각 로봇이 적응 제어로 SE(3) 궤적을 추종합니다.
- **Relevance to our work:** 접촉점이 주어진 상태의 협동 제어로, 우리의 기여를 접촉점 유도에 두어야 하는 근거입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: pereira-2003-caging
- **Title:** Decentralized Algorithms for Multirobot Manipulation via Caging
- **Authors / venue / year:** Pereira, Kumar, Campos / 2003
- **Link:** https://repository.upenn.edu/meam_papers/36/
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 힘 폐쇄보다 약한 물체 갇힘 조건을 분산 제어로 유지하며 운반합니다.
- **Relevance to our work:** 다중 로봇 운반의 고전 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: wang-2002-caging
- **Title:** Wang & Kumar 2002 평면 케이징(제목 미기재)
- **Authors / venue / year:** Wang, Kumar / 2002
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 평면 다중 로봇 케이징 운반입니다.
- **Relevance to our work:** 다중 로봇 운반의 고전 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: wang-2005-deformablecaging
- **Title:** Deformable caging formation control for cooperative object transportation
- **Authors / venue / year:** Wang, Hirata, Kosuge / 2005
- **Link:** https://scholars.cityu.edu.hk/en/publications/deformable-caging-formation-control-for-cooperative-object-transp/
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 변형 가능한 케이징 편대로 협동 운반합니다.
- **Relevance to our work:** 다중 로봇 운반의 고전 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: brahmbhatt-2019-contactgrasp
- **Title:** ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact
- **Authors / venue / year:** Brahmbhatt, Handa, Hays, Fox / IROS 2019
- **Link:** https://arxiv.org/pdf/1904.03754 (프로젝트 https://contactdb.cc.gatech.edu/contactgrasp.html)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 접촉 지도에서 사람 손·Allegro·Barrett의 기능적 파지를 합성하며, 손 모델 간 결정적 매핑은 접촉을 재현하지 못한다고 결론짓습니다.
- **Relevance to our work:** 접촉 지도 공유 방식의 원조이자 '운동학 매핑은 접촉을 보존하지 못한다'는 인용 문장의 출처입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서. 노트 1은 ContactGrasp의 arXiv 식별자를 확정하지 못했다고 적었고 노트 2는 1904.03754를 검색 요약으로 제시(상충).

## key: lakshmipathy-2021-contacttransfer
- **Title:** Contact Transfer: A Direct, User-Driven Method for Human to Robot Transfer of Grasps and Manipulations
- **Authors / venue / year:** A. Lakshmipathy, D. Bauer, C. Bauer, N. S. Pollard / arXiv (v2 2022-06-02)
- **Link:** https://arxiv.org/abs/2110.15532
- **Local file:** none
- **Status:** verified (노트 표기 '원문 초록 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 면적을 보존하며 사람 손 파지를 형태 가정 없이 다른 손으로 몇 초 안에 옮깁니다.
- **Relevance to our work:** 접촉 보존 전이의 선행이며 물리 검증이 없다는 점에서 우리와 대비됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: du-2022-graspinglikehumans
- **Title:** Multi-Finger Grasping Like Humans
- **Authors / venue / year:** Yuming Du, Philippe Weinzaepfel, Vincent Lepetit, Romain Brégier / IROS 2022(검색 결과 표기) / arXiv 2022-11-14
- **Link:** https://arxiv.org/pdf/2211.07304
- **Local file:** none
- **Status:** verified (노트 표기 '원문 PDF 텍스트 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람과 로봇 파지의 접촉 면적·전역 자세 유사성과 관통 벌점으로 Allegro·Barrett에 그리퍼별 조정 없이 리타게팅합니다.
- **Relevance to our work:** 손가락 수가 줄면 접촉이 최적화 부산물로 버려지는 방식의 사례로, 힘 폐쇄 기준 수축인 우리와 대비됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: shao-2020-unigrasp
- **Title:** UniGrasp: Learning a Unified Model to Grasp with Multifingered Robotic Hands
- **Authors / venue / year:** Lin Shao 외 / IEEE RA-L 5(2), 2020
- **Link:** https://arxiv.org/pdf/1910.10900 (프로젝트 https://sites.google.com/view/unigrasp)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** N지 그리퍼에 대해 힘 폐쇄이면서 도달 가능한 N개 접촉점을 점군에서 직접 고릅니다(사람 시연 미사용).
- **Relevance to our work:** 접촉 수에 맞춰 접촉점을 힘 폐쇄로 고르는 방식의 대표로, 사람 접촉 그래프에서 출발하는 우리와 비교 기준선입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: khargonkar-2024-robotfingerprint
- **Title:** RobotFingerPrint: Unified Gripper Coordinate Space for Multi-Gripper Grasp Synthesis and Transfer
- **Authors / venue / year:** N. Khargonkar, L. F. Casas, B. Prabhakaran, Y. Xiang / arXiv 2024
- **Link:** https://arxiv.org/pdf/2409.14519 (프로젝트 https://irvlutd.github.io/RobotFingerPrint)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 모든 그리퍼 손바닥 표면에 통일 좌표계를 두어 사람 손에서 Fetch 2지 그리퍼로 파지를 전이(YCB 11/14)합니다.
- **Relevance to our work:** 사람→2지 전이가 가능하다는 반대 증거로, 접촉 수에 따른 ρ 변화를 정량화해야 하는 이유가 됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2025-cedex
- **Title:** CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations
- **Authors / venue / year:** unknown / arXiv 2025-09
- **Link:** https://arxiv.org/html/2509.24661v1
- **Local file:** none
- **Status:** verified (노트 표기 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 접촉 맵·부위 맵을 생성하고 약지+새끼손가락 같은 사람 부위를 규칙으로 병합해 3~5지 손에 약 2천만 파지를 합성하며, 2지 그리퍼는 다방향 안정 힘 작용이 불가능하다며 제외합니다.
- **Relevance to our work:** 규칙 기반 접촉 병합의 대표 기준선이며, 2지 배제 사유는 팔 2대일 때 ρ가 작아지는 이유의 외부 근거입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2024-drograsp
- **Title:** D(R,O) Grasp
- **Authors / venue / year:** unknown / arXiv 2024-10
- **Link:** https://arxiv.org/html/2410.01702v4
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 로봇–물체 상호작용의 통일 거리 행렬 표현을 점군에서 예측합니다.
- **Relevance to our work:** 교차 형태 파지 표현 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-machagrasp
- **Title:** MachaGrasp
- **Authors / venue / year:** unknown / arXiv 2025-10
- **Link:** https://arxiv.org/pdf/2510.06068v3
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** URDF에서 eigengrasp와 형태 임베딩을 뽑아 새 손에 few-shot 적응합니다.
- **Relevance to our work:** 교차 형태 파지 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2026-affordancefield
- **Title:** 손가락별 어포던스 장 기반 교차 손 전이(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2026-01
- **Link:** https://arxiv.org/pdf/2601.08246v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 시각-언어 사전지식으로 손가락별 어포던스 장을 만들어 하드웨어별 파지 데이터 없이 손 간 전이합니다.
- **Relevance to our work:** 교차 형태 파지 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: yuan-2024-crossdex
- **Title:** Cross-Embodiment Dexterous Grasping with Reinforcement Learning (CrossDex)
- **Authors / venue / year:** Haoqi Yuan, Bohan Zhou, Yuhui Fu, Zongqing Lu / ICLR 2025
- **Link:** https://arxiv.org/pdf/2410.02479 (코드 https://github.com/PKU-RL/CrossDex)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손 기반 eigengrasp 공간을 공통 행동 공간으로 써서 4개 손을 학습하고 미학습 손에 제로샷 전이합니다.
- **Relevance to our work:** 행동 공간 공유형 교차 형태 파지 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: fang-2025-anydexgrasp
- **Title:** AnyDexGrasp
- **Authors / venue / year:** Hao-Shu Fang 외 / arXiv 2025-02
- **Link:** https://arxiv.org/abs/2502.16420v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 무관 접촉 중심 파지 표현과 손마다 실세계 시행착오로 학습한 파지 결정 모델을 씁니다.
- **Relevance to our work:** 접촉 중심 표현과 학습형 성공 판단의 인접 연구입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md. 두 실세계 수치의 설정 대응은 원문 확인 필요.

## key: unknown-2025-graspgen
- **Title:** GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training
- **Authors / venue / year:** unknown (NVIDIA) / arXiv 2025-07
- **Link:** https://arxiv.org/pdf/2507.13097 (코드 https://github.com/NVlabs/GraspGen)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 생성기가 만든 파지로 판별기를 학습해 생성기의 전형적 오류를 낮게 채점하는 그리퍼 파지 생성 프레임워크입니다.
- **Relevance to our work:** 학습형 파지 평가기의 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: zhong-2025-dexgraspanything
- **Title:** DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness
- **Authors / venue / year:** Zhong, Jiang, Yu, Ma / CVPR 2025
- **Link:** https://arxiv.org/abs/2503.08257v2
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 확산 모델의 학습과 샘플링 양쪽에 관통·접촉 부족 물리 제약을 넣어 340만 파지를 생성합니다.
- **Relevance to our work:** 단일 손 물리 인지 파지 생성 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-dexvlg
- **Title:** DexGraspNet 3.0 / DexVLG
- **Authors / venue / year:** unknown (Galbot·BAAI 외) / ICCV 2025
- **Link:** https://arxiv.org/abs/2507.02747v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Objaverse 17.4만 물체에 1.7억 파지를 각각 물리 시뮬레이션으로 검사하고 시각-언어 모델로 파지 자세를 생성합니다.
- **Relevance to our work:** 대규모 시뮬레이션 여과 데이터의 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-unigrasptransformer
- **Title:** UniGraspTransformer
- **Authors / venue / year:** unknown (Microsoft Research Asia) / CVPR 2025
- **Link:** https://arxiv.org/pdf/2412.02699
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물체별 강화학습 정책을 Transformer로 증류합니다.
- **Relevance to our work:** 단일 손 파지 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: huang-2024-fungrasp
- **Title:** FunGrasp: Functional Grasping for Diverse Dexterous Hands
- **Authors / venue / year:** Linyi Huang, Hui Zhang, Zijian Wu, Sammy Christen, Jie Song / arXiv 2024-11
- **Link:** https://arxiv.org/pdf/2411.16755
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 기능적 파지 RGBD 한 장에서 접촉 위치와 사람다운 자세를 보존하도록 최적화한 뒤 강화학습으로 여러 손에 배치합니다.
- **Relevance to our work:** 사람 파지→여러 손 전이의 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: shi-2026-howtransfer
- **Title:** HOWTransfer: Hand-centric Human-to-Robot Trajectory Transfer from Video Demonstrations via Open-World Contact Localization
- **Authors / venue / year:** Yitian Shi 외, Rania Rayyes (KIT) / arXiv 2026-06-09
- **Link:** https://arxiv.org/pdf/2606.10743 (노트 4는 https://arxiv.org/html/2606.10743v1)
- **Local file:** none
- **Status:** verified (노트 4 '출처 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 시작 시점을 찾아 사람 파지 의도를 평행 조 파지 후보로 바꾸고 손목 궤적을 따라 전파하여 재생 성공 86%를 얻습니다(충돌은 국소 휴리스틱).
- **Relevance to our work:** 사람→평행 조 전이 사례로 물리·힘 검사가 없다는 점에서 우리와 대비됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2025-maple
- **Title:** MAPLE
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2504.06084v2
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 한 장의 이미지에서 접촉 순간의 손–물체 접촉점과 3D 손 자세를 예측합니다(라벨은 VISOR-HOS).
- **Relevance to our work:** 영상→접촉점 예측 후보입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: iberall-1986-virtualfinger
- **Title:** Iberall, Bingham, Arbib 가상 손가락 논문(제목 미기재)
- **Authors / venue / year:** Iberall, Bingham, Arbib / Experimental Brain Research 1986
- **Link:** https://www.cs.cmu.edu/afs/cs/usr/sbk/www/thesis/virtual-finger.html (CMU 노트)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대부분의 파지를 대립하는 힘을 내는 두 개의 가상 손가락으로 정의할 수 있다고 봅니다.
- **Relevance to our work:** 사람 5접촉을 2~3접촉으로 수축하는 정식화의 생물학적 근거로 인용할 수 있습니다(원문 확인 후).
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: iberall-1990-oppositionspace
- **Title:** Opposition Space and Human Prehension
- **Authors / venue / year:** Iberall & MacKenzie / 1990
- **Link:** https://www.springerprofessional.de/opposition-space-and-human-prehension/13787620
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대립 공간을 가상 손가락 표면 사이에서 안정 파지를 위한 대립 힘을 낼 수 있는 영역으로 정의합니다.
- **Relevance to our work:** 접촉 수축의 생물학적 타당성 검사(종합 보고서 7-3)의 이론 근거입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2001-tripodopposition
- **Title:** J. Neurophysiol. 2001 삼각 파지 대립 공간 논문(제목 미기재)
- **Authors / venue / year:** unknown / Journal of Neurophysiology 2001
- **Link:** https://www.sciencegate.app/document/10.1152/jn.2001.86.2.604
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 삼각 파지에서 엄지와 두 손가락 사이 가상 손가락 사이에 대립 공간이 형성된다고 봅니다.
- **Relevance to our work:** 가상 손가락 이론의 보조 근거입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: yang-2024-ace
- **Title:** ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation
- **Authors / venue / year:** Shiqi Yang 외 / arXiv 2024-08
- **Link:** https://arxiv.org/html/2408.11805v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 저가 시각 외골격 원격조작 시스템으로 말단 운동 전달에 집중합니다.
- **Relevance to our work:** 원격조작 리타게팅 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-geort
- **Title:** Geometric Retargeting (GeoRT)
- **Authors / venue / year:** unknown / arXiv 2025-03-10
- **Link:** https://arxiv.org/pdf/2503.07541
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람과 로봇 손 키포인트 공간이 선형으로 연결되지 않는다며 1 kHz 비지도 신경 리타게팅을 제안합니다.
- **Relevance to our work:** 운동학 리타게팅 비판 인용원입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md. 노트 1은 사용자가 언급한 GeoRT의 식별자를 확정하지 못했다고 적음.

## key: unknown-2026-realdexumi
- **Title:** RealDexUMI
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/html/2606.06033
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 수집과 실행에 같은 말단 모듈을 써서 리타게팅으로 손재주를 잃는 것을 피합니다.
- **Relevance to our work:** 리타게팅 손실 문제의식의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: chen-2024-mirage
- **Title:** Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting
- **Authors / venue / year:** L. Y. Chen, K. Hari, K. Dharmarajan, C. Xu, Q. Vuong, K. Goldberg / RSS 2024
- **Link:** https://arxiv.org/abs/2402.19249
- **Local file:** none
- **Status:** unverified (노트 2 검색 요약, 노트 4 '배경지식, 전문 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 목표 로봇을 가리고 원 로봇을 그려 넣어 교차 형태 정책을 제로샷 전이합니다.
- **Relevance to our work:** 영상 편집 계열 배경으로 접촉·렌치 조건을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md

## key: lepert-2024-shadow
- **Title:** Shadow: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer
- **Authors / venue / year:** M. Lepert, R. Doshi, J. Bohg / CoRL 2024
- **Link:** https://arxiv.org/pdf/2503.00774
- **Local file:** none
- **Status:** unverified (노트 2 검색 요약, 노트 4 '배경지식, 전문 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 원·목표 로봇을 합성 분할 마스크로 대체해 교차 형태 정책을 전이합니다.
- **Relevance to our work:** 그리퍼-물체 상호작용 보존 주장이 영상 편집상 보존일 뿐 물리 검증이 아니라는 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md

## key: lepert-2025-phantom
- **Title:** Phantom: Training Robots Without Robots Using Only Human Videos
- **Authors / venue / year:** Marion Lepert, Jiaying Fang, Jeannette Bohg (Stanford) / CoRL 2025 (PMLR 305)
- **Link:** https://arxiv.org/html/2503.00779v2 (노트 2는 https://arxiv.org/pdf/2503.00779, PMLR https://proceedings.mlr.press/v305/lepert25a.html)
- **Local file:** none
- **Status:** verified (노트 4 '출처 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 자세 추정→로봇 행동 변환과 사람 팔 인페인팅 후 가상 로봇 합성으로 로봇 없이 정책을 학습하며, 파지 성립 검사 없이 하위 20% 닫힘 휴리스틱만 씁니다.
- **Relevance to our work:** 영상 편집 계열이 pinch grasp에 한정되고 파지를 검사하지 않는다는 저자 자인의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: lepert-2025-masquerade
- **Title:** Masquerade: Learning from In-the-wild Human Videos using Data-Editing
- **Authors / venue / year:** Marion Lepert, Jiaying Fang, Jeannette Bohg (Stanford) / arXiv 2025-08-13
- **Link:** https://arxiv.org/html/2508.09976v1 (프로젝트 https://masquerade-robot.github.io)
- **Local file:** none
- **Status:** verified (노트 4 '출처 확인'; 노트 2는 검색 요약)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** EPIC-Kitchens 영상을 Phantom 방식으로 편집해 양팔 로봇 정책을 학습하며, 다지 파지를 평행 조로 옮기는 것은 불완전한 대응이라고 적습니다.
- **Relevance to our work:** 다지→평행 조 대응의 불완전성 자인이 우리 접촉 수축 문제의 동기로 인용됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2025-h2rgrounder
- **Title:** H2R-Grounder
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2512.09406v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 상호작용 영상을 짝 데이터 없이 물리적으로 접지된 로봇 영상으로 변환합니다.
- **Relevance to our work:** 영상 편집 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, egocentric_video_to_robot_pipelines.md(서베이 언급)

## key: xu-2025-dexumi
- **Title:** DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation
- **Authors / venue / year:** Mengda Xu 외 (Shuran Song 그룹) / CoRL 2025
- **Link:** https://arxiv.org/html/2505.21864v3
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 목표 로봇 손에 맞춘 착용형 외골격과 로봇 손 인페인팅으로 다지 조작 데이터를 수집합니다.
- **Relevance to our work:** 손 착용형 수집 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-umi
- **Title:** UMI
- **Authors / venue / year:** unknown / 2024
- **Link:** none in notes (arXiv 2402.10329는 기억으로 적힌 식별자)
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 손 착용형 그리퍼 데이터 수집 인터페이스입니다.
- **Relevance to our work:** 말단 자세 공유 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2023-openx
- **Title:** Open X-Embodiment
- **Authors / venue / year:** unknown / 2023
- **Link:** none in notes (arXiv 2310.08864는 기억으로 적힌 식별자)
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대규모 교차 형태 로봇 정책입니다.
- **Relevance to our work:** 행동 공간 공유형 교차 형태 학습 배경이며 접촉·렌치를 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-octo
- **Title:** Octo
- **Authors / venue / year:** unknown / 2024
- **Link:** none in notes (arXiv 2405.12213는 기억으로 적힌 식별자)
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대규모 교차 형태 로봇 정책입니다.
- **Relevance to our work:** 행동 공간 공유형 교차 형태 학습 배경이며 접촉·렌치를 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-crossformer
- **Title:** CrossFormer
- **Authors / venue / year:** unknown / 2024
- **Link:** none in notes (arXiv 2408.11812는 기억으로 적힌 식별자)
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대규모 교차 형태 로봇 정책입니다.
- **Relevance to our work:** 행동 공간 공유형 교차 형태 학습 배경이며 접촉·렌치를 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-hpt
- **Title:** HPT
- **Authors / venue / year:** unknown / 2024
- **Link:** none in notes (arXiv 2409.20537는 기억으로 적힌 식별자)
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대규모 교차 형태 로봇 정책입니다.
- **Relevance to our work:** 행동 공간 공유형 교차 형태 학습 배경이며 접촉·렌치를 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2026-datapyramid
- **Title:** Data Pyramid for Embodied Manipulation (서베이)
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2607.24744
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 교차 형태 행동 공간 처리를 형태별 투영, 고정 차원 영 채움, 의미 행동 슬롯으로 분류합니다.
- **Relevance to our work:** 교차 형태 학습 서베이 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2026-unifiedrobotlearning
- **Title:** Toward Unified Robot Learning (서베이)
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2609.03927
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 교차 형태 절을 가진 로봇 학습 서베이입니다.
- **Relevance to our work:** 교차 형태 학습 서베이 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2024-dexgraspnet2
- **Title:** DexGraspNet 2.0
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/pdf/2410.23004
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약, 원문 문장 인용 포함')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** IsaacGym에서 관통 2 mm, 마찰 0.2, 6축 방향 중력 거부로 합성 파지를 거르며 통과율은 약 16.07%입니다.
- **Relevance to our work:** 데이터셋 여과 프로토콜과 통과율의 비교 기준입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2022-dexgraspnet
- **Title:** DexGraspNet
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/pdf/2210.02697
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 100 시뮬레이션 스텝과 관통 0.1 cm 한계로 합성 파지를 거릅니다.
- **Relevance to our work:** 여과 프로토콜 비교 기준입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: lum-2024-getagrip
- **Title:** Get a Grip: Multi-Finger Grasp Evaluation at Scale Enables Robust Sim-to-Real Transfer
- **Authors / venue / year:** T. G. W. Lum, A. H. Li, P. Culbertson, K. Srinivasan, A. D. Ames, M. Schwager, J. Bohg (Stanford·Caltech) / arXiv 2024-10-31
- **Link:** https://arxiv.org/html/2410.23701v1
- **Local file:** none
- **Status:** verified (노트 표기 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 각 파지를 손목 ±5 mm·±2.5° 섭동으로 5회 재시뮬레이션해 소프트 라벨을 만들고 평가기를 학습하여 실기 성공률을 43~51%에서 76~81%로 올립니다.
- **Relevance to our work:** 접촉 위치 섭동을 명시적으로 넣은 드문 검증 프로토콜이라 우리 섭동 범위·학습형 성공 예측의 비교 기준입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2025-pairdexsurvey
- **Title:** A Survey on Dexterous Robot Grasping (PAIR Lab)
- **Authors / venue / year:** unknown (PAIR Lab, Toronto) / 2025, Advanced Robotics Research 심사 중
- **Link:** https://www.pair.toronto.edu/DexRobotGraspSurvey.github.io
- **Local file:** none
- **Status:** verified (노트 표기 '원문 페이지 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** GraspEval 기준으로 기존 파지 데이터를 Allegro/LEAP에서 재평가해 GraspXL 부분집합 51% 등 원 논문보다 낮은 수치를 얻습니다.
- **Relevance to our work:** 평가 기준에 따라 성공률이 크게 달라진다는 근거로, 우리 검증 프로토콜을 명시해야 하는 이유입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: turpin-2022-graspd
- **Title:** Grasp'D
- **Authors / venue / year:** Turpin 외 (Toronto·NVIDIA·Samsung·Vector) / ECCV 2022
- **Link:** https://ecva.net/papers/eccv_2022/papers_ECCV/html/5242_ECCV_2022_paper.php (arXiv 2208.12250)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 미분 가능 접촉 시뮬레이션으로 사람 손·로봇 손 모델에 더 조밀한 접촉의 파지를 합성합니다.
- **Relevance to our work:** 물리 기반 파지 합성 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-omniretarget
- **Title:** OmniRetarget
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2509.26633
- **Local file:** none
- **Status:** verified (노트 표기 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 상호작용 메시로 접촉 관계를 보존하는 휴머노이드 전신 리타게팅으로, 기존 방법의 최대 관통 8.50 cm를 1.34 cm로 줄입니다.
- **Relevance to our work:** 키포인트 스케일링이 손을 물체 안으로 밀어 넣는다는 인용 문장과 관통 수치의 출처입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: kim-2013-physicalgraspquality
- **Title:** Physically-based Grasp Quality Evaluation under Pose Uncertainty
- **Authors / venue / year:** J. Kim, K. Iwamoto, J. Kuffner, Y. Ota, N. Pollard / IEEE T-RO 29(6), 2013
- **Link:** https://www.ri.cmu.edu/publications/physically-based-grasp-quality-evaluation-under-pose-uncertainty
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 자세 분포에서 샘플링한 초기 자세로 동적 시뮬레이션을 돌려 파지 성공을 추정합니다.
- **Relevance to our work:** 자세 불확실성 하 물리 기반 품질 평가로 우리 섭동 검증의 선행입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: ferrari-1992-epsilon
- **Title:** Ferrari & Canny ε 지표
- **Authors / venue / year:** Ferrari, Canny / ICRA 1992
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 고전 파지 품질 해석 문헌입니다.
- **Relevance to our work:** ε 및 불확실성 하 힘 폐쇄 분석의 배경 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: pollard-2004-graspquality
- **Title:** Pollard 2004 (제목 미기재)
- **Authors / venue / year:** Pollard / 2004
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 고전 파지 품질 해석 문헌입니다.
- **Relevance to our work:** ε 및 불확실성 하 힘 폐쇄 분석의 배경 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: li-1988-taskoriented
- **Title:** Li & Sastry 과제 지향 품질
- **Authors / venue / year:** Li, Sastry / 1988
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 고전 파지 품질 해석 문헌입니다.
- **Relevance to our work:** ε 및 불확실성 하 힘 폐쇄 분석의 배경 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: zheng-2005-graspuncertainty
- **Title:** Coping with the grasping uncertainties in force-closure analysis
- **Authors / venue / year:** Zheng, Qian / IJRR 2005
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 고전 파지 품질 해석 문헌입니다.
- **Relevance to our work:** ε 및 불확실성 하 힘 폐쇄 분석의 배경 인용 대상입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: roa-2009-icr
- **Title:** Computation of Independent Contact Regions for Grasping 3-D Objects
- **Authors / venue / year:** Roa, Suárez / IEEE T-RO 25(4), 2009
- **Link:** none in notes (2011 논문 링크만 있음)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 시작 파지의 접촉점 주위로 힘 폐쇄가 유지되는 독립 접촉 영역을 계산합니다.
- **Relevance to our work:** 접촉 위치 오차 허용 범위를 기하로 표현한 고전으로 ρ 해석과 연결됩니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: roa-2011-icruncertainty
- **Title:** Influence of Contact Types and Uncertainties in the Computation of Independent Contact Regions
- **Authors / venue / year:** Roa, Suárez / ICRA 2011
- **Link:** https://iocnet.upc.edu/usuaris/raulsuarez/papers/icra11-rs.pdf
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 독립 접촉 영역 계산을 접촉 유형과 불확실성까지 확장합니다.
- **Relevance to our work:** 접촉 실현 오차 허용 범위 해석의 근거입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: li-2024-intrinsicrobustness
- **Title:** Toward An Analytic Theory of Intrinsic Robustness for Dexterous Grasping
- **Authors / venue / year:** Albert H. Li, Preston Culbertson, Aaron D. Ames (Caltech) / IROS 2024(검색 요약 표기) / arXiv v2 2024-08-29
- **Link:** https://arxiv.org/html/2403.07249v2
- **Local file:** none
- **Status:** verified (노트 표기 '원문 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** ε와 최소 가중치 지표 ℓ*가 접촉점·법선 섭동에 대한 내재적 강건성을 하한한다는 정리를 제시하고 Allegro 실기에서 85% 대 52.5%를 보고합니다(ℓ∞ 노름은 언급 없음).
- **Relevance to our work:** 우리 ρ를 접촉 실현 오차 허용 반경으로 해석하는 이론적 연결의 출처이며, ℓ∞ 버전은 우리가 직접 유도해야 합니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: enwerem-2026-firmgrasp
- **Title:** FIRMGrasp: A Friction-Informed Risk Margin for Robust Grasp Synthesis
- **Authors / venue / year:** Clinton Enwerem, John S. Baras, Calin Belta (UMD) / arXiv 2607.25049 (v1 2026-07-27, v2 2026-08-06)
- **Link:** https://www.alphaxiv.org/abs/2607.25049v1
- **Local file:** none
- **Status:** verified (노트 표기 'alphaXiv 페이지 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 마찰을 확률변수로 두면 명목 ε가 인증한 파지의 53%가 불리한 마찰 꼬리에서 힘 폐쇄를 잃음을 보이고 위험 조정 여유를 제안합니다.
- **Relevance to our work:** ρ 계산에 마찰 범위를 명시적으로 넣어야 한다는 근거입니다(접촉 위치·법선 불확실성은 미모델링).
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md, 종합 보고서

## key: unknown-2026-unclegrasp
- **Title:** UNCLE-Grasp
- **Authors / venue / year:** unknown / arXiv 2026-01
- **Link:** https://arxiv.org/pdf/2601.14492
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 형상 완성 몬테카를로 샘플링 위에서 ε 지표로 접촉점을 채점합니다.
- **Relevance to our work:** 불확실성 하 해석적 품질 사용 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: mahler-2016-dexnet1
- **Title:** Dex-Net 1.0
- **Authors / venue / year:** Mahler et al. / 2016
- **Link:** none in notes
- **Local file:** none
- **Status:** unverified (노트 표기 '기억, 식별자 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 자세·마찰 불확실성 하 힘 폐쇄 확률로 평행 그리퍼 파지를 라벨링합니다.
- **Relevance to our work:** 강건 ε 라벨링의 고전 배경입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2022-efficientgrasp
- **Title:** EfficientGrasp
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/abs/2206.15159
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 힘 폐쇄 추정기에서 빌린 품질 점수로 접촉점 집합을 판정해 실로봇 83.3%를 얻습니다.
- **Relevance to our work:** 접촉점 기반 학습형 성공 판정의 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: lu-2020-multifingerinference
- **Title:** Multi-Fingered Grasp Planning via Inference in Deep Neural Networks
- **Authors / venue / year:** Lu et al. / arXiv 2020
- **Link:** https://arxiv.org/pdf/2001.09242
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 복셀 3D CNN으로 물체 기하와 손 구성에서 성공 확률을 추정하고 이를 직접 최대화합니다.
- **Relevance to our work:** 학습형 성공 예측기의 사례입니다.
- **Related ideas:** none yet
- **Notes:** cross_morphology_and_physical_verification.md

## key: unknown-2025-bop2024
- **Title:** BOP Challenge 2024
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2504.02812v2
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 깊이 없는 HOT3D에서 미지 물체 6자유도 검출 최고 평균 정밀도가 26.8에 그치며 2D 검출이 주 병목이라고 보고합니다.
- **Relevance to our work:** egocentric RGB만으로 물체 자세를 얻기 어렵다는 근거로, 접촉 앵커 정밀도 위험의 인용 수치입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2023-adt
- **Title:** Aria Digital Twin (ADT)
- **Authors / venue / year:** unknown (Meta) / ICCV 2023
- **Link:** https://arxiv.org/html/2306.06362v2 (문서 https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 두 실내 공간의 egocentric 녹화에 물체 6자유도 자세·메시·깊이를 제공하지만 손 메시는 없습니다.
- **Relevance to our work:** 손이 없어 접촉 계산에는 바로 쓸 수 없는 egocentric 데이터입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md. 시퀀스·물체 수가 출처마다 다름(초록 200, 본문 350+50, 문서 236).

## key: unknown-2023-egoexo4d
- **Title:** Ego-Exo4D
- **Authors / venue / year:** unknown (Meta 등) / 2023
- **Link:** https://arxiv.org/pdf/2311.18259
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 수작업 3D 손 자세 주석 68K를 제공하지만 물체 3D 자세는 없습니다.
- **Relevance to our work:** 물체 3D가 없어 접촉 정답 공급원으로 부적합합니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-hdepic
- **Title:** HD-EPIC
- **Authors / venue / year:** unknown / CVPR 2025
- **Link:** https://arxiv.org/abs/2502.04144v2 (페이지 https://hd-epic.github.io)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 41시간 egocentric 주방 영상에 손 마스크·물체 트랙·디지털 트윈을 주지만 3D 손·물체 메시는 없습니다(비상업).
- **Relevance to our work:** 물체 3D가 없는 보조 데이터입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-unknown-epicfields
- **Title:** EPIC-Fields
- **Authors / venue / year:** unknown / 연도 미기재
- **Link:** https://hd-epic.github.io
- **Local file:** none
- **Status:** unverified (HD-EPIC 페이지에 별도 데이터로 표기된 것만 확인, 논문 자체는 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** EPIC-KITCHENS 영상 671개의 카메라 자세·점군(등록 프레임 18.7M)을 제공합니다.
- **Relevance to our work:** egocentric 카메라 자세 보조 데이터입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-epiccontact
- **Title:** EPIC-Contact (HOPformer 포함)
- **Authors / venue / year:** unknown (Bristol + MPI) / ECCV 2026
- **Link:** https://arxiv.org/html/2606.30598 (데이터 https://huggingface.co/datasets/Sid2697/epic-contact)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 야외 egocentric에서 자세를 맞춘 MANO 손 + 물체 메시 + 접촉 대응을 주는 최초급 데이터(2,272 클립)이며, 같은 논문의 HOPformer 접촉 편차는 EPIC-Contact 20.7 mm, ARCTIC egocentric 31.9 mm입니다.
- **Relevance to our work:** 접촉 위치 오차를 mm로 보고한 유일한 출처이며, egocentric 접촉이 mm 정밀도가 없다는 우리 2단계 설계의 핵심 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서. 물체 클래스 수가 본문 '99'와 검색 요약 '9'로 상충. HOPformer를 별도 논문으로 볼지는 노트상 불명확(같은 arXiv 링크).

## key: unknown-2022-hoi4d
- **Title:** HOI4D
- **Authors / venue / year:** unknown / CVPR 2022
- **Link:** https://arxiv.org/abs/2203.01577v4
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric RGB-D 240만 프레임에 3D 손 자세, 카테고리 수준 물체 자세, 재구성 메시를 강체·관절 물체에 대해 제공하며 손 주석은 반자동입니다.
- **Relevance to our work:** 관절 물체 대안이지만 접촉 품질이 낮을 가능성이 큽니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-egohandposebench
- **Title:** 2D egocentric 손 자세 벤치마킹 연구(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/html/2409.07337v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** HOI4D 등에서 예측과 정답 주석 사이 정렬 불일치 증거를 보고합니다.
- **Relevance to our work:** 데이터셋 정답 품질 한계의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2021-h2o
- **Title:** H2O
- **Authors / venue / year:** unknown / ICCV 2021
- **Link:** https://arxiv.org/abs/2104.11181 (페이지 https://taeinkwon.com/projects/h2o/)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric + 다중 RGB-D로 양손 3D 자세, 6자유도 물체 자세, 물체 메시(8종)를 마커 없이 주석합니다.
- **Relevance to our work:** CHORD 7종 데이터셋 중 하나입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2023-assemblyhands
- **Title:** AssemblyHands
- **Authors / venue / year:** unknown / CVPR 2023
- **Link:** https://arxiv.org/abs/2304.12301v1 (페이지 https://assemblyhands.github.io)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric 49만 장을 포함한 손 키포인트 데이터로 자동 주석 오차 4.20 mm를 보고하며 물체 자세는 없습니다.
- **Relevance to our work:** 실험실 정답 품질 수치의 출처입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2022-arctic
- **Title:** ARCTIC
- **Authors / venue / year:** unknown (저장소 zc-alexfan) / CVPR 2023
- **Link:** https://www.arxiv.org/pdf/2204.13662 (코드 https://www.github.com/zc-alexfan/arctic)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 고정 8뷰 + egocentric 1뷰로 양손과 관절 물체 메시를 제공합니다(210만 장).
- **Relevance to our work:** 현재 우리가 쓰는 데이터셋이며 관절 물체 + MANO + egocentric 뷰를 동시에 주는 유일한 선택지입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서. 저자명은 노트에 없음(저장소 이름 zc-alexfan만 있음).

## key: unknown-2024-taco
- **Title:** TACO
- **Authors / venue / year:** unknown / CVPR 2024
- **Link:** https://arxiv.org/html/2401.08399v2 (페이지 https://taco2024.github.io)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 도구-동작-물체 양손 2.5K 시퀀스에 egocentric 뷰와 정밀 손-물체 메시를 제공합니다.
- **Relevance to our work:** 관절·도구 과제 2순위 확장 데이터입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2024-oakink2
- **Title:** OakInk2
- **Authors / venue / year:** unknown / CVPR 2024
- **Link:** https://arxiv.org/abs/2403.19417 (툴킷 https://github.com/oakink/OakInk2)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 다중뷰 영상에 몸·손·물체 자세를 세 추상화 수준의 양손 복합 작업으로 제공합니다.
- **Relevance to our work:** CHORD 7종 데이터셋 중 하나이며 확장 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md. 627 시퀀스 수치는 검색 요약 수준.

## key: unknown-2024-hograspnet
- **Title:** HOGraspNet
- **Authors / venue / year:** unknown (KAIST) / ECCV 2024
- **Link:** https://arxiv.org/abs/2409.04033
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 피험자 99명의 파지 분류학 라벨과 접촉 맵, MANO 피팅을 제공합니다(egocentric 여부 미확인).
- **Relevance to our work:** 접촉 맵이 있는 파지 데이터 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-gigahands
- **Title:** GigaHands
- **Authors / venue / year:** unknown (Brown + ETH) / CVPR 2025
- **Link:** https://arxiv.org/html/2412.04244v3 (저장소 https://github.com/brown-ivl/GigaHands)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 외부 카메라 51대로 34시간·417 물체의 양손 MANO와 6자유도 물체 자세를 제공합니다(egocentric 뷰 언급 없음).
- **Relevance to our work:** 대규모 양손-물체 데이터 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-hocap
- **Title:** HO-Cap
- **Authors / venue / year:** unknown (UT Dallas + NVIDIA) / NeurIPS 2025 D&B
- **Link:** https://arxiv.org/html/2406.06843v4 (페이지 https://irvlutd.github.io/HOCap/)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** RGB-D 8대 + HoloLens로 마커 없이 손·물체 자세를 공동 정제한 데이터입니다.
- **Relevance to our work:** 메시 있는 손-물체 데이터 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-humoto
- **Title:** HUMOTO
- **Authors / venue / year:** unknown (Adobe + UT Austin) / ICCV 2025
- **Link:** https://arxiv.org/html/2504.10414v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 전신 모션 캡처와 관절 부품 물체 데이터로, 관통·접촉 엔트로피 품질 지표를 제안합니다.
- **Relevance to our work:** 데이터 품질 지표의 참고입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-egopressure
- **Title:** EgoPressure
- **Authors / venue / year:** unknown (ETH + Microsoft) / CVPR 2025
- **Link:** https://arxiv.org/html/2409.02224v2 (페이지 https://yiming-zhao.github.io/EgoPressure/)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric + 외부 카메라로 터치패드 압력과 손 메시를 제공하고 PressureFormer로 단일 영상에서 압력을 예측합니다(실제 물체 없음).
- **Relevance to our work:** 영상→접촉 압력 추정의 참고입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-egoexohands
- **Title:** EgoExo-Hands
- **Authors / venue / year:** unknown (Meta Reality Labs) / arXiv 2025
- **Link:** https://arxiv.org/html/2510.02601v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 이동 리그 정답의 손-물체 상호작용 구간 키포인트 오차가 30대 돔 대비 중앙값 7.90 mm, 90퍼센타일 12.53 mm임을 보고합니다.
- **Relevance to our work:** 실험실 정답조차 mm 정밀도에 한계가 있다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2026-show3d
- **Title:** SHOW3D
- **Authors / venue / year:** unknown / CVPR 2026
- **Link:** https://arxiv.org/abs/2603.28760
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 야외를 포함한 손·물체 3D 주석 데이터입니다(세부는 초록에 없음).
- **Relevance to our work:** 향후 egocentric 확장 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-openego
- **Title:** OpenEgo
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/abs/2509.05513
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 공개 데이터 6종을 통합한 1,107시간 egocentric 손 자세 데이터입니다(공개 예정).
- **Relevance to our work:** 손 모션 사전분포 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-efm3d
- **Title:** Aria Everyday Objects / EFM3D
- **Authors / venue / year:** unknown (Meta) / 2024
- **Link:** https://arxiv.org/pdf/2406.10224 (문서 https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_objects)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Aria 26 시퀀스의 3D 경계상자 584개를 제공합니다.
- **Relevance to our work:** 손 정보가 없는 보조 데이터입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-dtc
- **Title:** Digital Twin Catalog (DTC)
- **Authors / venue / year:** unknown (Meta) / 2025
- **Link:** https://arxiv.org/abs/2504.08541
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 밀리미터 정확도의 스캔 물체 2,000개와 Aria 시퀀스 200개를 제공합니다.
- **Relevance to our work:** 정밀 물체 메시 출처 후보입니다(질량 여부 미확인).
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-hawor
- **Title:** HaWoR
- **Authors / venue / year:** unknown / CVPR 2025
- **Link:** https://arxiv.org/html/2501.02973v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric 영상에서 세계 좌표계 손을 복원해 HOT3D W-MPJPE 33.20 mm를 얻습니다(코드 공개, 실시간 아님).
- **Relevance to our work:** egocentric 손 재구성의 현재 최선 수치로, 접촉 오차 전이 분석의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2024-dynhamr
- **Title:** Dyn-HaMR
- **Authors / venue / year:** unknown / CVPR 2025
- **Link:** https://arxiv.org/html/2412.12861v2
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 움직이는 카메라에서 4D 양손을 복원하며 CHORD 저장소가 모듈로 포함합니다.
- **Relevance to our work:** CHORD 단안 영상 경로의 손 모듈입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, egocentric_video_to_robot_pipelines.md

## key: unknown-2026-hand3r
- **Title:** Hand3R
- **Authors / venue / year:** unknown / arXiv 2026-02
- **Link:** https://arxiv.org/abs/2602.03200v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 순전파 온라인 4D 손-장면 재구성입니다.
- **Relevance to our work:** 손 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-egograsp
- **Title:** EgoGrasp
- **Authors / venue / year:** unknown / arXiv 2026-01
- **Link:** https://arxiv.org/abs/2601.01050
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric 영상에서 세계 좌표계 손-물체 상호작용을 확산 모델로 복원합니다(초록에 수치 없음).
- **Relevance to our work:** egocentric 손-물체 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-stablehand
- **Title:** StableHand
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2605.18553
- **Local file:** none
- **Status:** unverified (노트 표기 '상세 미확인'; 손 PA-MPJPE 취합 수치의 출처 매핑도 불확실)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 손 재구성 연구로 상세는 확인되지 않았습니다.
- **Relevance to our work:** 손 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-unihand
- **Title:** UniHand
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2602.21631
- **Local file:** none
- **Status:** unverified (노트 표기 '상세 미확인'; 손 PA-MPJPE 취합 수치의 출처 매핑도 불확실)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 손 재구성 연구로 상세는 확인되지 않았습니다.
- **Relevance to our work:** 손 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-videodiffusionhand
- **Title:** 비디오 확산 모델 기반 손 재구성(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2606.30308
- **Local file:** none
- **Status:** unverified (노트 표기 '상세 미확인'; 손 PA-MPJPE 취합 수치의 출처 매핑도 불확실)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 손 재구성 연구로 상세는 확인되지 않았습니다.
- **Relevance to our work:** 손 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2023-hold
- **Title:** HOLD
- **Authors / venue / year:** unknown / CVPR 2024
- **Link:** https://arxiv.org/abs/2311.18448v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단안 상호작용 영상에서 카테고리 무관 손+물체를 3D 주석 없이 공동 재구성합니다.
- **Relevance to our work:** 손-물체 공동 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2026-ghost
- **Title:** GHOST
- **Authors / venue / year:** unknown / arXiv 2026-03
- **Link:** https://arxiv.org/abs/2603.18912
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2D 가우시안 스플래팅으로 단안 영상의 동적 손·물체를 빠르게 재구성합니다.
- **Relevance to our work:** 손-물체 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-hort
- **Title:** HORT
- **Authors / venue / year:** unknown / ICCV 2025
- **Link:** https://arxiv.org/html/2503.21313
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 RGB에서 손 기하를 사전지식으로 물체 점군을 재구성합니다.
- **Relevance to our work:** 생성 메시의 과도한 평활화 지적이 법선 품질 위험의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-mccho
- **Title:** MCC-HO + RAR
- **Authors / venue / year:** unknown / 2024
- **Link:** https://arxiv.org/abs/2404.06507
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 RGB와 추정 손에서 손·물체 기하를 얻고 텍스트→3D 생성 템플릿을 정렬합니다.
- **Relevance to our work:** 물체 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2024-easyhoi
- **Title:** EasyHOI
- **Authors / venue / year:** unknown / CVPR 2025
- **Link:** https://arxiv.org/html/2411.14280v4
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 영상에서 분할·인페인팅·3D 재구성과 물리 제약 최적화로 손-물체를 복원합니다.
- **Relevance to our work:** 손-물체 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-zeroshotscenehoi
- **Title:** 장면 내 조작 영상 제로샷 재구성(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2025-12
- **Link:** https://arxiv.org/abs/2512.19684
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단안 영상에서 물체 메시·자세, 장면 점군, 손 자세를 물리 타당성 최적화로 복원합니다.
- **Relevance to our work:** 재구성 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2023-foundationpose
- **Title:** FoundationPose
- **Authors / venue / year:** unknown / CVPR 2024
- **Link:** https://arxiv.org/html/2312.08344v2
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** RGB-D 입력으로 모델 기반·모델 프리 물체 6자유도 추정과 추적을 합니다.
- **Relevance to our work:** CHORD·Human2Sim2Robot 등이 쓰는 물체 추적기로, 깊이를 요구한다는 점이 egocentric 확장의 제약입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2025-bundlesdfocclusion
- **Title:** BundleSDF 가림 실패 보고 논문(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/pdf/2505.02335
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** BundleSDF가 부분·완전 가림 후 추적에 실패하는 사례를 보고합니다.
- **Relevance to our work:** 손 가림 하 물체 추적 한계의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-foundationposeho3d
- **Title:** FoundationPose HO3D 부정확 보고 논문(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/pdf/2506.07996
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** FoundationPose가 HO3D의 강한 회전에서 부정확함을 보고합니다.
- **Relevance to our work:** 손 가림 하 물체 추적 한계의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-any6d
- **Title:** Any6D
- **Authors / venue / year:** unknown / CVPR 2025
- **Link:** https://arxiv.org/html/2503.18673v2
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** RGB-D 참조 한 장으로 미지 물체 자세와 미터 크기를 추정합니다.
- **Relevance to our work:** 물체 자세·스케일 추정 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, egocentric_video_to_robot_pipelines.md(DemoBridge 백엔드)

## key: unknown-2023-deco
- **Title:** DECO
- **Authors / venue / year:** unknown / ICCV 2023
- **Link:** https://arxiv.org/abs/2309.15273v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 영상에서 SMPL 몸 표면 전체의 조밀 접촉을 추정합니다.
- **Relevance to our work:** 영상→접촉 추정 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-haco
- **Title:** HACO
- **Authors / venue / year:** unknown / NeurIPS 2025
- **Link:** https://arxiv.org/abs/2505.11152v2 (코드 github.com/dqj5182/HACO_RELEASE)
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 클래스 불균형 대응 손실로 조밀 손 접촉을 추정합니다(코드 공개).
- **Relevance to our work:** 손 메시 위 접촉 사전확률로 결합할 수 있는 후보 모듈입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2026-graspandlift
- **Title:** Grasp-and-Lift
- **Authors / venue / year:** unknown / arXiv 2026-01
- **Link:** https://arxiv.org/abs/2601.18121
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** DexYCB·HO3D 정답이 시뮬레이터 재생 시 물리적으로 타당하지 않음을 지적하고 CMA-ES로 실행 가능한 궤적으로 정제합니다.
- **Relevance to our work:** 데이터 정답도 시뮬레이터 재확정이 필요하다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: hasson-2019-obman
- **Title:** Hasson et al. 2019 (제목 미기재)
- **Authors / venue / year:** Hasson et al. / 2019
- **Link:** https://arxiv.org/pdf/1904.05767
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 제약 없는 학습 재구성의 평균 관통이 ObMan 9 mm, FHB 19 mm이며 관통과 시뮬레이션 안정성 사이의 상충을 보고합니다.
- **Relevance to our work:** 단안 재구성 관통 수치의 출처입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2023-deepsimho
- **Title:** DeepSimHO
- **Authors / venue / year:** unknown / arXiv 2023
- **Link:** https://arxiv.org/pdf/2310.07206
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물리 시뮬레이터로 손-물체 자세 안정성을 평가·학습합니다.
- **Relevance to our work:** 시뮬레이터 내 물리 재최적화의 선행입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2024-hands2024
- **Title:** 2024 HANDS 벤치마크
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/html/2403.16428v2
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric에서 빠른 손 동작, 좁은 시야의 물체 재구성, 양손-물체 근접 접촉이 난점이라고 결론짓습니다.
- **Relevance to our work:** egocentric 재구성 한계의 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2025-videophysprops
- **Title:** 비디오 파운데이션 모델 물성 추론(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/pdf/2510.02311v1
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 비디오 파운데이션 모델로 마찰·탄성·점성을 추론하지만 실물 마찰에서 정확도가 떨어집니다.
- **Relevance to our work:** 질량·마찰을 데이터에서 얻기 어렵다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2026-phypush
- **Title:** PhyPush
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2605.26284
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 한 번 밀기 영상으로 질량·마찰을 추정하며 질량 과소·마찰 과대 경향이 있습니다.
- **Relevance to our work:** 능동 상호작용이 필요해 오프라인 전처리에 쓸 수 없다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md, 종합 보고서

## key: unknown-2023-frictionfiltering
- **Title:** RGB-D 강체 추적 + 마찰 계수 필터링(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2023
- **Link:** https://arxiv.org/html/2309.15703v3
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** RGB-D 추적과 불확실성을 포함한 마찰 계수 필터링입니다.
- **Relevance to our work:** 물성 추정 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2021-containermass
- **Title:** 용기 질량 추정(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2021
- **Link:** https://arxiv.org/pdf/2107.01309
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 내용물 높이·부피·밀도 가정으로 용기 질량을 계산합니다.
- **Relevance to our work:** 물성 추정 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2020-liftingkinematics
- **Title:** 사람 들기 동작 운동학의 질량 구분 연구(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2020
- **Link:** https://ar5iv.arxiv.org/html/2003.12045
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람이 드는 동작의 운동학만으로는 가벼운/무거운 물체 구분이 부적합하다고 보고합니다.
- **Relevance to our work:** 영상에서 질량을 얻기 어렵다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: unknown-2021-internetforce
- **Title:** 인터넷 영상 사람-물체 힘 추정(제목 미기재)
- **Authors / venue / year:** unknown / arXiv 2021
- **Link:** https://arxiv.org/pdf/2111.01591
- **Local file:** none
- **Status:** verified (노트 기준 공식 페이지·논문 본문에서 직접 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 물체 유형별 공통 질량 분포를 가정해 인터넷 영상에서 사람-물체 힘을 추정합니다.
- **Relevance to our work:** 물성 추정 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_datasets_and_reconstruction.md

## key: liu-2025-egozero
- **Title:** EgoZero: Robot Learning from Smart Glasses
- **Authors / venue / year:** Vincent Liu, Ademi Adeniji 외, Lerrel Pinto (NYU, UC Berkeley) / arXiv 2025-05
- **Link:** https://arxiv.org/html/2505.20290
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Aria 안경 손 추적의 엄지·검지 손끝 좌표를 행동으로 삼아 로봇 데이터 없이 Franka에 전이하며 7과제 평균 약 70%를 얻습니다.
- **Relevance to our work:** 행동 라벨 오차 1~2 cm와 평행 그리퍼 한정이 egocentric 정밀도 한계의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: tao-2025-dexwild
- **Title:** DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies
- **Authors / venue / year:** Tony Tao 외, Deepak Pathak (CMU) / arXiv 2025-05
- **Link:** https://arxiv.org/html/2505.07813v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 모션캡처 장갑 데이터를 LEAP Hand에 역기구학으로 리타게팅해 로봇 데이터와 공동 학습하며, 사람 데이터만으로는 3.6%입니다.
- **Relevance to our work:** 사람 데이터만으로는 행동 공간 간극이 닫히지 않는다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: yang-2025-egovla
- **Title:** EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **Authors / venue / year:** Ruihan Yang 외, Xiaolong Wang / arXiv 2025 (v3)
- **Link:** https://arxiv.org/html/2507.12440
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** egocentric 사람 영상으로 시각-언어-행동 모델을 사전학습해 시뮬레이션 휴머노이드에 쓰지만 로봇 후속 학습 없이는 모든 과제 0%입니다.
- **Relevance to our work:** 사람 영상 단독 전이 실패의 인용 수치입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: luo-2025-beingh0
- **Title:** Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos
- **Authors / venue / year:** Hao Luo 외, Zongqing Lu / arXiv 2025-07-21
- **Link:** https://arxiv.org/html/2507.15597v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 11개 출처 1,155시간 손 데이터로 MANO 토큰을 사전학습하고 원격조작 데이터로 로봇 행동 공간을 맞추며 물체 6D 모델링은 넣지 않았습니다.
- **Relevance to our work:** 물리 검사 없는 대규모 사전학습 계열의 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: lin-2026-egowild2dex
- **Title:** EgoWild2Dex: Learning Dexterous Robotic Manipulation from In-the-Wild Human Experience
- **Authors / venue / year:** Kunyang Lin 외, Ping Luo (HKU, Kinetix AI) / arXiv 2026-09-20
- **Link:** https://arxiv.org/html/2609.23755
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** VR 헤드셋과 트래커로 모은 538.9시간 데이터로 양팔 다지 손 VLA를 3단계 학습하며 물리 검증은 없습니다.
- **Relevance to our work:** 사람 시연만으로 다지 협응이 어렵다는 근거이고, 트래커 유무에 따른 손끝 오차(4.21 cm 대 0.66 cm)는 정밀도 참고 수치입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: qiu-2025-humanoidpolicy
- **Title:** Humanoid Policy ~ Human Policy
- **Authors / venue / year:** Ri-Zhao Qiu 외, Xiaolong Wang / arXiv 2025-03 (v3 2025-10)
- **Link:** https://arxiv.org/abs/2503.13441
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람과 휴머노이드가 공유하는 상태-행동 공간으로 미분 가능하게 로봇 행동으로 리타게팅합니다.
- **Relevance to our work:** 행동 공간 공유 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2023-mimicplay
- **Title:** MimicPlay
- **Authors / venue / year:** unknown / arXiv 2023
- **Link:** https://arxiv.org/abs/2302.12422
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-humanplus
- **Title:** HumanPlus
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2406.10454
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-dexcap
- **Title:** DexCap
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2403.07788
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2021-dexmv
- **Title:** DexMV
- **Authors / venue / year:** unknown / arXiv 2021
- **Link:** https://arxiv.org/abs/2108.05877
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2022-dexvip
- **Title:** DexVIP
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/abs/2202.00164
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2022-videodex
- **Title:** VideoDex
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/abs/2212.04498
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2022-telekinesis
- **Title:** Robotic Telekinesis
- **Authors / venue / year:** unknown / arXiv 2022
- **Link:** https://arxiv.org/abs/2209.12993
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-roviaug
- **Title:** RoVi-Aug
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2409.03403
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2023-givingrobotsahand
- **Title:** Giving Robots a Hand
- **Authors / venue / year:** unknown / arXiv 2023
- **Link:** https://arxiv.org/abs/2307.05959
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-atm
- **Title:** ATM (Any-point Trajectory Modeling)
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2401.00025
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-im2flow2act
- **Title:** Im2Flow2Act
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2407.15208
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-generalflow
- **Title:** General Flow
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2401.11439
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-gen2act
- **Title:** Gen2Act
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2409.16283
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-vidbot
- **Title:** VidBot
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/abs/2503.07135
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-orion
- **Title:** ORION
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2405.20321
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-vid2robot
- **Title:** Vid2Robot
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2403.12943
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-rx
- **Title:** R+X
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2407.12957
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-lapa
- **Title:** LAPA
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2410.11758
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-univla
- **Title:** UniVLA
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/abs/2505.06111
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-spot
- **Title:** SPOT
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2411.00965
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-video2policy
- **Title:** Video2Policy
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/abs/2502.09886
- **Local file:** none
- **Status:** unverified (노트 표기 '배경지식, 전문 미확인': arXiv 식별자만 제시)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2023~2025년 사람 영상→로봇 학습의 배경 연구입니다.
- **Relevance to our work:** egocentric·사람 영상 학습 계열의 배경 인용 대상이며 접촉·렌치 검증을 다루지 않습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-okami
- **Title:** OKAMI
- **Authors / venue / year:** unknown / CoRL 2024 구두발표
- **Link:** https://arxiv.org/abs/2410.11792
- **Local file:** none
- **Status:** verified (arXiv 초록에서 수치만 확인, 노트 분류는 '배경지식, 전문 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 RGB-D 영상에서 물체 인식 리타게팅으로 휴머노이드 폐루프 정책 평균 79.2%를 얻습니다.
- **Relevance to our work:** 물체 인식 리타게팅 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-arcap
- **Title:** ARCap
- **Authors / venue / year:** unknown / arXiv 2024
- **Link:** https://arxiv.org/abs/2410.08464
- **Local file:** none
- **Status:** verified (arXiv 초록 확인, 노트 분류는 '배경지식, 전문 미확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 증강현실 시각 피드백과 햅틱 경고로 로봇 운동학에 맞고 장면과 충돌하지 않는 데이터를 수집하게 합니다.
- **Relevance to our work:** 수집 시점에 운동학 실행 가능성을 되먹임하는 드문 사례이나 파지 역학 검증은 아닙니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: cheng-2026-robotizinghumanvideos
- **Title:** Robotizing Human Videos with Physically Consistent Interactions
- **Authors / venue / year:** Ching-Lam Cheng, Shengfeng He, Bin Zhu (SMU) / arXiv 2026-10-05
- **Link:** https://arxiv.org/html/2610.06137v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** MANO 정점별 접촉 확률에서 엄지 접촉 중심과 대향 접촉 중심을 그리퍼 두 접점으로 삼아 사람 영상을 로봇 영상으로 바꾸며, 물리 시뮬레이션 검증은 없습니다.
- **Relevance to our work:** 사람 접촉을 두 접점으로 수축하는 기하적 규칙의 사례로, 렌치 여유로 수축하는 우리와 직접 비교됩니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: song-2025-mitty
- **Title:** Mitty: Diffusion-based Human-to-Robot Video Generation
- **Authors / venue / year:** Yiren Song, Cheng Liu, Weijia Mao, Mike Zheng Shou / arXiv 2025-12-19
- **Link:** https://arxiv.org/abs/2512.17253
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 중간 표현 없이 사람 시연을 로봇 실행 영상으로 번역합니다.
- **Relevance to our work:** 영상 편집 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: li-2025-mimicdreamer
- **Title:** MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training
- **Authors / venue / year:** Haoyun Li 외 15인 / arXiv 2025-09-26
- **Link:** https://arxiv.org/abs/2509.22199
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 비디오 확산 모델로 로봇 시연 영상을 생성하고 제약 역기구학으로 실행 가능한 관절 명령을 만듭니다.
- **Relevance to our work:** 운동학 검사만 있는 영상 편집 계열 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: wang-2026-ego2robot
- **Title:** Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data
- **Authors / venue / year:** Ye Wang 외 15인 / arXiv 2026-08-03
- **Link:** https://arxiv.org/abs/2608.02580
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 행동 리타게팅·로봇 팔 시각 합성·품질 선별로 15종 로봇 형태의 학습 데이터 18,561시간을 합성했다고 주장합니다.
- **Relevance to our work:** egocentric→다형태 데이터 합성 사례로 물리 검사는 초록에 없습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2024-track2act
- **Title:** Track2Act
- **Authors / venue / year:** unknown / ECCV 2024
- **Link:** https://arxiv.org/pdf/2405.01527
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 인터넷 영상으로 학습한 형태 비의존 점 궤적 예측기와 잔차 정책을 씁니다.
- **Relevance to our work:** 물체 중심 표현 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-mtpi
- **Title:** MT-π (Motion Tracks)
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2501.06994v2
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2D 영상 공간 운동 궤적을 교차 형태 행동 공간으로 씁니다.
- **Relevance to our work:** 물체 중심 표현 계열 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: haldar-2025-pointpolicy
- **Title:** Point Policy
- **Authors / venue / year:** Siddhant Haldar, Lerrel Pinto / arXiv 2025-02-27
- **Link:** https://arxiv.org/abs/2502.20391
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 손 자세를 로봇 자세로 바꾸고 물체 상태를 의미 있는 핵심점으로 표현합니다.
- **Relevance to our work:** 점 기반 표현 계열 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: hung-2026-3pointr
- **Title:** 3PoinTr: 3D Point Tracks for Learning Manipulation from Unconstrained Human Videos
- **Authors / venue / year:** Adam Hung, Bardienus Pieter Duisterhof, Jeffrey Ichnowski (CMU) / arXiv v2 2026-06-03
- **Link:** https://arxiv.org/html/2603.08485v2
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 3인칭 스테레오 영상의 3D 점 궤적을 확산 정책 조건으로 써서 실기 4과제 75/80을 얻으며, 정책이 사람과 다른 파지 전략을 택한 사례가 있습니다.
- **Relevance to our work:** 로봇이 사람 파지를 그대로 따르지 않아도 된다는 사례로 인용할 수 있습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: shi-2025-zeromimic
- **Title:** ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos
- **Authors / venue / year:** Junyao Shi 외, Dinesh Jayaraman / ICRA 2025
- **Link:** https://arxiv.org/abs/2503.23877
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** EPIC-Kitchens에서 파지 어포던스 검출기로 그리퍼 파지를 생성해 기술 정책을 증류합니다.
- **Relevance to our work:** 파지 제안기를 쓰지만 성공 예측은 하지 않는 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: chen-2026-host
- **Title:** HOST: Robots Acquire Manipulation Skills in Seconds from a Single Human Video
- **Authors / venue / year:** Guangyan Chen 외 12인 / arXiv 2026-07-22
- **Link:** https://arxiv.org/abs/2607.20033
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 로봇 궤적과 사람 영상을 공유 과제 진행 다양체에 올려 평균 29초 만에 기술을 습득합니다.
- **Relevance to our work:** 물체 3D 없이 작동하는 계열 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: zhou-2026-zerowam
- **Title:** Zero-WAM: In-Context World-Action Modeling from Human Videos
- **Authors / venue / year:** Jiaming Zhou 외 12인 / arXiv 2026-08-26
- **Link:** https://arxiv.org/abs/2608.26103
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 영상을 과제 명세로 삼는 인컨텍스트 비디오-행동 모델입니다.
- **Relevance to our work:** 물리 검사 없는 계열 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-gr00tn1
- **Title:** GR00T N1
- **Authors / venue / year:** unknown (NVIDIA) / arXiv 2025
- **Link:** https://arxiv.org/pdf/2503.14734
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 행동 라벨이 없는 사람 egocentric 영상에서 연속 프레임 쌍으로 잠재 행동을 학습합니다.
- **Relevance to our work:** 잠재 행동 계열 배경이며 CHORD 저장소가 GR00T N1.7 미세조정을 포함합니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-flare
- **Title:** FLARE (Future LAtent Representation Alignment) / GR00T N1.5
- **Authors / venue / year:** unknown (NVIDIA GEAR) / arXiv 2025
- **Link:** https://arxiv.org/html/2505.15659v1 (공식 페이지 https://research.nvidia.com/labs/gear/gr00t-n1_5/index.html)
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 미래 임베딩 정렬 보조 손실로 행동 라벨 없는 사람 영상에서 학습하며, 파지 검증은 바구니 투입 여부뿐입니다.
- **Relevance to our work:** 물리 검사 없는 대규모 사전학습 계열 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: ma-2026-humanscale
- **Title:** HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining
- **Authors / venue / year:** Juncheng Ma 외 22인 (DAGroup-PKU) / arXiv 2026-06-18
- **Link:** https://arxiv.org/abs/2606.20521
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 같은 양이면 egocentric 사람 영상 사전학습이 원격조작 로봇 궤적보다 낫다고 주장합니다.
- **Relevance to our work:** egocentric 데이터 가치의 근거이지만 소량 로봇 데이터가 여전히 필요합니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-hrdt
- **Title:** H-RDT
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/pdf/2507.23523
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** EgoDex로 사전학습한 뒤 로봇 데이터로 미세조정하는 양손 모델입니다.
- **Relevance to our work:** EgoDex 활용 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-egoverse
- **Title:** EgoVerse
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2604.07607
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 egocentric 데이터셋·사전학습 연구로 제목·검색 요약만 확인되었습니다.
- **Relevance to our work:** egocentric 확장 후보 데이터·모델입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-egolive
- **Title:** EgoLive
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2604.23570
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 egocentric 데이터셋·사전학습 연구로 제목·검색 요약만 확인되었습니다.
- **Relevance to our work:** egocentric 확장 후보 데이터·모델입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-aceego0
- **Title:** ACE-Ego-0
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2606.17200
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 egocentric 데이터셋·사전학습 연구로 제목·검색 요약만 확인되었습니다.
- **Relevance to our work:** egocentric 확장 후보 데이터·모델입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-wh0
- **Title:** Wh0
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2606.22136
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 egocentric 데이터셋·사전학습 연구로 제목·검색 요약만 확인되었습니다.
- **Relevance to our work:** egocentric 확장 후보 데이터·모델입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-beingh07
- **Title:** Being-H0.7
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/html/2605.00078v1
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 2026년 egocentric 데이터셋·사전학습 연구로 제목·검색 요약만 확인되었습니다.
- **Relevance to our work:** egocentric 확장 후보 데이터·모델입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-motiontrans
- **Title:** MotionTrans
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2509.17759v1
- **Local file:** none
- **Status:** unverified (제목·검색 요약만 확인, 상세 미확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** VR 데이터 기반 사람→로봇 학습입니다.
- **Relevance to our work:** 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: han-2026-video2sim2real
- **Title:** Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single Human Video
- **Authors / venue / year:** Yunhai Han 외 17인 / arXiv 2026-06-07
- **Link:** https://arxiv.org/html/2606.08828v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 영상 한 편에서 물체 메시·물성을 추정하고 후보 파지를 들어올림·흔들기 시뮬레이션으로 채택해 실기 평균 95.7%를 얻습니다.
- **Relevance to our work:** 시뮬레이션 안정성 테스트로 파지를 채택하는 가장 명시적인 영상 계열 사례이며, 힘 분석이 아니라는 점에서 우리 ρ와 갈라집니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: yu-2025-r2r2r
- **Title:** Real2Render2Real (R2R2R)
- **Authors / venue / year:** Justin Yu, Max Letian Fu 외, Ken Goldberg (UC Berkeley, TRI) / arXiv 2025-05-14
- **Link:** https://arxiv.org/html/2505.09601v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 스마트폰 스캔과 사람 시연 한 편에서 대척 파지 샘플러와 역기구학으로 시연을 대량 렌더링하며 물리 시뮬레이션·충돌 검사는 없습니다.
- **Relevance to our work:** 물리 검사 없는 데이터 합성의 한계 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: wang-2026-demobridge
- **Title:** DemoBridge: A Simulation-in-the-Loop Toolkit for Single-View Human Demonstration Retargeting
- **Authors / venue / year:** Zehao Wang, Fabien Despinoy, Sergey Zakharov, Tinne Tuytelaars, Rahaf Aljundi / RSS 2026 RoboData Workshop / arXiv 2026-07-10
- **Link:** https://arxiv.org/html/2607.09519v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** Isaac Sim에서 단계별로 파지·운반을 확인하고 실패하면 되돌려 다음 후보를 시도하며, 비교군은 도달은 되지만 파지 단계에서 0~3/10으로 무너집니다.
- **Relevance to our work:** 운동학 리타게팅이 파지에서 실패한다는 서론 인용 수치이자 시뮬레이션 루프 검증의 비교 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서. 소속은 페이지에 미기재.

## key: shi-2025-adadexgrasp
- **Title:** AdaDexGrasp: Learning Adaptive Dexterous Grasping from Single Demonstrations
- **Authors / venue / year:** Liangzhi Shi 외, Hao Su / arXiv 2025-03 (v2 2025-08)
- **Link:** https://arxiv.org/abs/2503.20208
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단일 사람 시연에서 파지 기술을 배우고 강화학습으로 보정해 Ability Hand 실기 90%를 얻습니다.
- **Relevance to our work:** 시연 기반 파지 학습 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: chen-2026-v2pmanip
- **Title:** V2P-Manip: Learning Dexterous Manipulation from Monocular Human Videos
- **Authors / venue / year:** Kaihan Chen, Yanming Shao, Haifeng Ji, Xiaokang Yang, Yao Mu / arXiv 2026-06-15
- **Link:** https://arxiv.org/abs/2606.16436
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 3D 자산 획득, 궤적 추정, 다지 정책 학습에 2단계 보정을 둡니다.
- **Relevance to our work:** 단안 영상→다지 손 경로 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: zhou-2025-yoto
- **Title:** You Only Teach Once (YOTO)
- **Authors / venue / year:** Huayi Zhou 외, Kui Jia / arXiv 2025-01-24
- **Link:** https://arxiv.org/html/2501.14208v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 양안 영상의 손 동작을 키프레임으로 축약해 양팔 평행 그리퍼에 옮기고, 실기에서 키프레임을 수동 재생 검증합니다.
- **Relevance to our work:** 양손→양팔 1:1 대응 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-hdmi
- **Title:** HDMI
- **Authors / venue / year:** unknown (CMU) / arXiv 2025
- **Link:** https://arxiv.org/abs/2509.16757
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 단안 RGB 사람-물체 영상에서 Unitree G1 전신 로코-매니퓰레이션을 강화학습으로 학습합니다.
- **Relevance to our work:** 대형 물체를 다루는 전신 계열 사례이며 상자 운반 포함 여부는 검색 요약에만 의존합니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2026-sugar
- **Title:** SUGAR
- **Authors / venue / year:** unknown / arXiv 2026-05
- **Link:** https://arxiv.org/abs/2605.20373
- **Local file:** none
- **Status:** verified (arXiv 초록 페이지만 확인, 노트 표기 '초록만 확인')
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 영상 유래 사전지식의 가림·접촉 인공물·리타게팅 오류를 특권 정보 기반 물리 보정기로 실행 가능하게 만든 뒤 계층 정책으로 증류합니다.
- **Relevance to our work:** 영상 유래 접촉을 물리로 보정한다는 점에서 참고 가치가 있습니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: ma-2026-lfhvsurvey
- **Title:** Robot Learning from Human Videos: A Survey
- **Authors / venue / year:** Junyi Ma 외 (SJTU·Cambridge) / arXiv 2026-04-30
- **Link:** https://arxiv.org/html/2604.27621v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인'; 5장 본문은 추출 실패로 제목만 수록)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람 영상 학습의 실패 유형(mis-grounding)과 어포던스 전이가 힘·순응·접촉 안정성을 시각만으로 추론하지 못한다는 한계를 정리합니다.
- **Relevance to our work:** 분야 진단의 인용원입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: feng-2026-humanvideovlasurvey
- **Title:** From Human Videos to Robot Manipulation: A Survey on Scalable VLA Learning with Human-Centric Data
- **Authors / venue / year:** Zhiyuan Feng 외 (Tsinghua·HKUST·MSRA 등) / IJCAI 2026 Survey Track
- **Link:** https://arxiv.org/html/2606.00054v1
- **Local file:** none
- **Status:** verified (노트 표기 '출처 확인': arXiv 전문 또는 초록 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 리타게팅이 본질적으로 과소 제약이라고 쓰면서도 체계적 파지 실행 가능성 검사는 기술하지 않고 다중 팔 협응은 분석하지 않습니다.
- **Relevance to our work:** 다중 팔 협응이 서베이에서 비어 있다는 근거입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2026-hoifmsurvey
- **Title:** Hand-Object Interaction in the Age of Large Foundation Models (서베이)
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/html/2607.28394
- **Local file:** none
- **Status:** verified (노트 4 '출처 확인'; 노트 1은 CHORD 인용 목록에서 제목만)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 접촉 범위와 관통 벌점은 실행 안정성의 불완전한 대리 지표이므로 체화된 검증이 필요하다고 정리합니다.
- **Relevance to our work:** 물리 검증이 필요하다는 서베이 인용원입니다.
- **Related ideas:** none yet
- **Notes:** contact_wrench_guided_transfer.md, egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2026-videotocontrolsurvey
- **Title:** From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/html/2604.04974v3
- **Local file:** none
- **Status:** unverified (제목만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 영상 기반 조작 인터페이스 서베이입니다.
- **Relevance to our work:** 배경 서베이 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-intentionpriors
- **Title:** Learning Human-Intention Priors from Large-Scale Human Demonstrations
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2604.24681
- **Local file:** none
- **Status:** unverified (제목만 확인)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 대규모 사람 시연에서 의도 사전분포를 학습하는 연구입니다.
- **Relevance to our work:** 배경 후보입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2025-resmimic
- **Title:** ResMimic
- **Authors / venue / year:** unknown / arXiv 2025
- **Link:** https://arxiv.org/html/2510.05070v1
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 모션 캡처에서 학습해 4.5 kg 상자를 전신 접촉으로 운반합니다.
- **Relevance to our work:** 대형 물체 전신 조작 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md, 종합 보고서

## key: unknown-2026-prohoi
- **Title:** Pro-HOI
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://pro-hoi.github.io (arXiv 2603.01126)
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 상자 운반을 포함한 사람-물체 상호작용 수집입니다.
- **Relevance to our work:** 대형 물체 전신 조작 사례입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-unknown-multihumanoidretarget
- **Title:** Multi-Humanoid Robot Arm Motion Imitation and Collaboration Based on Improved Retargeting
- **Authors / venue / year:** unknown / PMC 게재
- **Link:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11939925/
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** RGB-D 카메라 한 대로 여러 모방자를 찍어 여러 로봇에 대응시킵니다.
- **Relevance to our work:** 한 사람 시연을 팀으로 나누는 것이 아니어서 우리와 다릅니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md. 연도 미기재.

## key: rozo-2015-cooperativetransport
- **Title:** Rozo 외 2015 운동감각 시연 기반 힘 제약 GMM(제목 미기재)
- **Authors / venue / year:** Rozo 외 / 2015
- **Link:** https://calinon.ch/paper3046.htm
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 운동감각 시연으로 사람-로봇 협동 운반을 학습합니다.
- **Relevance to our work:** 로봇 한 대 + 사람 한 명 구도의 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: unknown-2026-proact
- **Title:** PROACT
- **Authors / venue / year:** unknown / arXiv 2026
- **Link:** https://arxiv.org/pdf/2609.25351
- **Local file:** none
- **Status:** unverified (노트 표기 '검색 요약': 검색 결과 요약에서만 확인, 수치는 원문 재확인 필요)
- **Found by / date:** Claude (literature survey), 2026-11-08
- **One-line summary:** 사람-사람 운반 데이터로 협동 운반을 예측합니다.
- **Relevance to our work:** 사람-로봇 협동 운반 배경입니다.
- **Related ideas:** none yet
- **Notes:** egocentric_video_to_robot_pipelines.md

## key: kappler-2015-bigdatagrasp
- **Title:** Leveraging Big Data for Grasp Planning
- **Authors / venue / year:** D. Kappler, J. Bohg, S. Schaal / ICRA 2015
- **Link:** https://is.mpg.de/publications/2015_icra_kbs
- **Local file:** none
- **Status:** unverified (2026-11-13 웹 검색 결과 요약에서만 확인, 원문은 읽지 않음)
- **Found by / date:** Claude, 2026-11-13
- **One-line summary:** 시뮬레이션으로 만든 파지에 고전 ε 지표와 물리 시뮬레이션 기반 지표를 함께 붙이고, 사람이 매긴 평가와 대조해 물리 기반 지표가 파지 성공을 더 일관되게 예측하고 라벨 잡음이 적다고 보고합니다.
- **Relevance to our work:** "선형계획으로 계산한 α만 높은 배치"보다 "시뮬레이션 유지 시험에 합격한 배치"가 실제 성공을 더 잘 예측하는가라는 질문의 가장 가까운 근거입니다. 대조 기준이 실물 로봇이 아니라 사람의 평가이고 대상이 손 파지라는 점이 우리와 다릅니다.
- **Related ideas:** IDEA-001
- **Notes:** rubert-2019-graspsuccess 항목의 한 줄 요약("물리 시뮬레이션 기반 지표가 ε 지표보다 더 일관되게 예측")은 검색 요약으로 보면 이 논문의 결론에 해당하는 것으로 보입니다. Rubert 2019는 검색 요약상 품질 지표 7개를 실물 로봇 두 대에서 대조하고 지표를 조합한 분류기로 약 85%(arXiv판 요약은 76%)의 예측 정확도를 보고합니다. 두 항목 모두 원문 재확인이 필요합니다.
