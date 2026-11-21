# Egocentric 데이터셋과 단안·egocentric 3D 재구성 도구 인벤토리 (접촉 기반 사람→로봇 전이 입력 관점, 2026-11-08 기준)

작성 기준일: 2026-11-08. 각 항목 끝의 [Source] 링크가 해당 수치의 출처입니다. "검증"은 공식 페이지·논문 본문에서 직접 읽은 사실, "추론"은 그 사실들로부터 제가 판단한 내용입니다. 로컬 파일을 출처로 쓴 경우 절대 경로를 적었습니다.

용어 설명(처음 나오는 곳에서 풀어 씀): MANO = 손 파라메트릭 메시 모델(778 꼭짓점, 21 관절). UmeTrack = Meta의 손 추적용 손 모델·형식. MPJPE = 관절 위치 평균 오차(mm). PA-MPJPE = 프로크루스테스(강체+스케일) 정렬 후 관절 오차. W-MPJPE / G-MPJPE = 세계(전역) 좌표계에서의 관절 오차. ADD-S = 대칭 허용 평균 모델점 거리 기반 물체 자세 지표. AP = 평균 정밀도. CDev(Contact Deviation) = 접촉 편차, 즉 정답 접촉 대응점 쌍 사이의 예측 거리(mm). SLAM = 동시 위치 추정 및 지도 작성. PBR = 물리 기반 렌더링 재질. RGB-D = 색상+깊이 영상.

---

## 핵심 질문 1. Egocentric(또는 egocentric 시점을 포함한) 3D 주석 데이터셋 인벤토리 — 각각 무엇을 제공하는가

### Takeaway
2026-11 현재 "egocentric 영상 + 미터 단위 물체 메시 + 6자유도 물체 궤적 + MANO 손"을 모두 공식 제공하는 데이터셋은 사실상 HOT3D 하나이며(강체 33개, 관절 물체 없음), EgoDex·Ego-Exo4D·HD-EPIC·Aria Digital Twin 은 물체 메시+자세나 손 메시 중 한쪽이 빠져 있습니다. 관절(articulated) 물체까지 필요하면 여전히 ARCTIC(egocentric 1뷰 포함)과 HOI4D(카테고리 수준 자세, 주석 품질 낮음)가 선택지이고, 2025~2026년 신규(EPIC-Contact, EgoExo-Hands, SHOW3D, OpenEgo)는 접촉 또는 야외성을 더했지만 물체 메시·질량 정보는 여전히 부족합니다.

### Cited Findings

**HOT3D (Meta, CVPR 2025) — 가장 먼저 쓸 후보**
- 833분 이상, 3.7M+ 영상(1.5M+ 다중뷰 프레임), 피험자 19명, 강체 물체 33개, Project Aria 안경 + Quest 3 헤드셋, 손·물체에 광학 마커를 붙인 모션 캡처로 정답 자세 생성, 손은 UmeTrack 과 MANO 두 형식, 물체는 사내 스캐너로 만든 PBR 재질 3D 메시, 장면 점군과 시선도 포함 — [arXiv 2411.19167 abstract](https://arxiv.org/abs/2411.19167)
- 카메라: Aria는 RGB 1408×1408 1대 + 흑백 640×480 2대, Quest 3는 흑백 1280×1024 전면 2대, 모두 30 fps. 깊이 영상은 없음 — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167); BOP 2024 논문은 "VSD 지표는 깊이가 필요한데 HOT3D 와 HANDAL 에는 깊이가 없어 제외"라고 명시 — [BOP 2024](https://arxiv.org/html/2504.02812v2)
- 녹화 425개(Aria 198, Quest 3 226), 각 약 2분. 학습 13명(약 1M 다중뷰 프레임), 테스트 6명(약 0.5M). 테스트 정답은 비공개(평가 서버). 큐레이션된 150프레임 클립(HOT3D-Clips) 3,832개(학습 2,804, 테스트 1,028) — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167); 공식 페이지는 클립 구성을 Aria 1,983 + Quest 3 1,849로 표기 — [HOT3D 홈페이지](https://facebookresearch.github.io/hot3d/)
- 마커: 3 mm 접착 마커, 손당 19개, 물체당 약 10개, OptiTrack 적외선 카메라로 추적. 손 자세는 피험자별 UmeTrack 손 모델을 마커에 피팅해서 생성. 논문 원문: "UmeTrack is more accurate while MANO is more standard" — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167)
- 알려진 한계(논문 자체 명시): "Annotations in some frames may be missing or be of a lower quality"; 1.5M 프레임 중 1.16M 만 완전 주석 + 육안 검수 통과; 동적 온보딩 시퀀스는 첫 프레임만 물체 자세 정답 — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167)
- 정답 정확도(mm)는 논문·홈페이지 어디에도 수치가 없음 — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167), [HOT3D 홈페이지](https://facebookresearch.github.io/hot3d/). 참고로 HOT3D로 학습한 UmeTrack 추적기의 HOT3D 테스트 평균 키포인트 오차는 10.95 mm(이는 모델 오차이지 주석 오차가 아님) — [arXiv 2510.02601](https://arxiv.org/html/2510.02601v1)
- 라이선스: 데이터는 "HOT3D license agreement"(조건 본문은 페이지에 없음), API 코드는 Apache 2.0, MANO 주석 사용에는 MANO 별도 라이선스 동의 필요. 다운로드 링크는 14일 유효(이메일 재입력으로 갱신). 테스트 피험자 ID P0004, P0005, P0006, P0008, P0016, P0020에는 정답 없음 — [facebookresearch/hot3d](https://github.com/facebookresearch/hot3d)
- 연혁: 2025-08-19 물체 온보딩 시퀀스(새 Aria 녹화 포함) Hugging Face 추가, 2026-06-03 CVPR 2026 EgoVis Distinguished Paper Award — [HOT3D 홈페이지](https://facebookresearch.github.io/hot3d/)
- 물체 질량·치수 등 물리량은 언급 없음 — [arXiv 2411.19167 html](https://arxiv.org/html/2411.19167)

**EgoDex (Apple, 2025; ICLR 2026)** — 손만 있고 물체 3D 없음
- 829시간, 30 fps, 9천만 프레임, 33.8만 작업 시연, 탁상 작업 194종, Apple Vision Pro 로 녹화 — [arXiv 2505.11709](https://arxiv.org/abs/2505.11709), [ICLR 2026 포스터](https://iclr.cc/virtual/2026/poster/10010617)
- 초록은 "paired 3D hand and finger tracking data" 만 언급하며 물체 자세·메시·깊이는 언급하지 않음 — [arXiv 2505.11709](https://arxiv.org/abs/2505.11709)
- 저장소 README: HDF5에 카메라·머리·상체·양손(손끝·마디 포함) SE(3) 변환, 관절별 신뢰도(일부 파일에 없음), 3×3 내부 파라미터; 영상 1080p 30 Hz MP4; 학습 약 725 h, 테스트 7 h, 추가 97 h; 다운로드 학습 300 GB × 5 + 테스트 16 GB + 추가 200 GB; 라이선스 "CC-by-NC-ND"; 한계: ARKit 원점이 세션마다 다름, Vision Pro 가 여러 카메라에서 RGB를 합성하므로 2D 재투영이 영상 속 손과 어긋날 수 있음, 설명문은 GPT-4/VLM 자동 생성 — [apple/ml-egodex](https://github.com/apple/ml-egodex)

**Aria Digital Twin (ADT, Meta, ICCV 2023)** — 물체 6자유도·메시는 있으나 손 메시 없음
- 초록: 두 실내 공간, 398개 물체 인스턴스(정적 324 + 동적 74), 물체 6자유도 자세·기기 자세·시선·사람 자세·분할·깊이·합성 렌더링 제공; 모션 캡처와 디지털 트윈 공간을 통합해 주석 — [arXiv 2306.06362](https://arxiv.org/html/2306.06362v2)
- 공식 문서: 시퀀스 236개(1인·2인 활동), 동적 물체 74개(두 공간 공유), 아파트 정적 물체 281종(324 인스턴스), 사무실 15종(20 인스턴스), 물체 3D 모델은 .glb 로 2024-09 공개, 몸 자세는 모션 캡처(일부 바디슈트). 손 자세·손 메시는 언급 없음 — [projectaria_tools ADT](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)
- 시퀀스 수와 물체 수가 출처(초록 200개/본문 350+50/문서 236개)마다 달라 인용 시 버전 명시 필요 — [arXiv 2306.06362](https://arxiv.org/html/2306.06362v2), [projectaria_tools ADT](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)

**Ego-Exo4D (Meta 등, 2023~)** — 3D 손 키포인트는 있으나 물체 3D 자세 없음
- 수작업 손 자세 주석 68K(3D)/340K(2D), 몸 자세 376K(3D)/2M(2D); 손과 몸 주석은 별도 세트 — [arXiv 2311.18259](https://arxiv.org/pdf/2311.18259)
- 후속 논문이 "Ego-Exo4D does not provide 3D annotations for object pose"라고 명시 — [arXiv 2510.02601](https://arxiv.org/pdf/2510.02601)

**EPIC-KITCHENS 계열**
- HD-EPIC(CVPR 2025): 41시간, 4.4M 프레임, 손 마스크 7.7M, 물체 트랙 19.9K, VQA 26.6K, 디지털 트윈·시선 프라이밍 포함; 3D 손 자세·손 메시·물체 메시는 없음(움직인 물체는 수작업 마스크를 3D 경계상자로 올림) — [hd-epic.github.io](https://hd-epic.github.io); 9개 주방, 고정물 413개 디지털 트윈, 37K 물체 마스크 3D 승격, 비상업 라이선스, 마스크·물체 연관 2025-04, 시선 2025-05 공개 — [arXiv 2502.04144](https://arxiv.org/abs/2502.04144v2)
- EPIC-Fields: 등록 프레임 18.7M, 영상 671개, 주방 45곳(카메라 자세·점군; HD-EPIC 페이지에 별도 데이터로 표기) — [hd-epic.github.io](https://hd-epic.github.io)
- **EPIC-Contact(ECCV 2026, Bristol + MPI)** — 야외 egocentric 에서 자세 맞춘 MANO 손 + 물체 메시 + 접촉 대응을 제공하는 최초급 데이터: 2,272 클립(약 2.3K), 62.3K 프레임, 학습 2,035 / 테스트 237, EPIC-Grasps 의 "안정 파지 구간"에서 추출. 접촉 주석은 3,106 꼭짓점으로 세분한 MANO 위에 칠하고 물체 쪽은 2자유도 접촉축(클립당 최대 3축, 6클릭). 자세 맞춤(EC-fit)은 WiLoR 로 중앙 프레임 손 초기화 → 접촉·가림 인지 마스크·관통·정규화 손실로 물체 자세 최적화 → 사람이 검수·수정, 물체 자세는 클립 전체에 복사. 물체 메시는 EPIC-Grasps 것을 "standard scale"로 사용하고 Gemini 2.5 로 부품별 스케일 추정: 물체 30개에서 MAE 0.94 cm(상대 5.9%), ±10% 이내 82.5%. 주석자 일치도 κ 손 0.61, 물체 0.62 — [arXiv 2606.30598 html](https://arxiv.org/html/2606.30598); 데이터·코드는 Hugging Face 미러(datasheet, 로더, 접촉 시각화 포함) — [Sid2697/epic-contact](https://huggingface.co/datasets/Sid2697/epic-contact)
- 물체 클래스 수는 본문 fetch 결과 "99 object classes", 검색 요약은 "9 object classes"로 상충 — [arXiv 2606.30598 html](https://arxiv.org/html/2606.30598) vs [검색 요약](https://huggingface.co/papers/2606.30598) (추후 확인 필요)

**HOI4D (CVPR 2022)** — egocentric RGB-D, 관절 물체 포함, 주석은 반자동
- 2.4M egocentric RGB-D 프레임, 4,000+ 시퀀스, 피험자 4명, 16개 카테고리 800개 물체 인스턴스, 610개 실내, 프레임별 파놉틱·모션 분할, 3D 손 자세, 카테고리 수준 물체 자세, 손 동작, 재구성 물체 메시, 장면 점군; 강체+관절 물체 — [arXiv 2203.01577](https://arxiv.org/abs/2203.01577v4)
- 손 주석 방법: 프레임의 20%만 2D 키포인트 수작업 → MANO 피팅(형상 파라미터는 피험자 실측으로 고정), 실패 프레임은 수동 수정; 마커 방식은 손 외관 때문에 배제 — [arXiv 2203.01577 pdf](https://arxiv.org/pdf/2203.01577)
- 2D egocentric 손 자세 벤치마킹 연구가 HOI4D 등에서 "예측과 정답 주석 사이 정렬 불일치" 증거를 보고 — [arXiv 2409.07337](https://arxiv.org/html/2409.07337v1)

**H2O (ICCV 2021)** — egocentric + 다중 RGB-D, 양손, 물체 8종
- 동기화 다중뷰 RGB-D, 양손 3D 자세, 6자유도 물체 자세, 물체 메시(BADSLAM 으로 생성), 손 관절은 OpenPose 검출 후 형상 최적화, 물체 자세는 DenseFusion + Mask R-CNN 으로 추정 후 카메라 간 최고 신뢰도 선택, 마커리스 주석 — [H2O 프로젝트 페이지](https://taeinkwon.com/projects/h2o/), [arXiv 2104.11181](https://arxiv.org/abs/2104.11181)
- 물체 인스턴스 8개뿐(HOI4D 논문의 비교 서술) — [arXiv 2203.01577](https://arxiv.org/abs/2203.01577v4)

**Assembly101 / AssemblyHands (CVPR 2023)** — 손 키포인트만, 물체 자세 없음
- 3.0M 주석 영상 중 egocentric 490K; 자동 주석 모델의 평균 키포인트 오차 4.20 mm(수작업 대비), Assembly101 원 주석보다 85% 낮음; CC BY-NC 4.0 — [arXiv 2304.12301](https://arxiv.org/abs/2304.12301v1), [assemblyhands.github.io](https://assemblyhands.github.io)

**ARCTIC (CVPR 2023)** — 현재 사용 중, 관절 물체 + egocentric 1뷰
- 2.1M RGB 영상, 고정 allocentric 8뷰 + 움직이는 egocentric 1뷰, 양손+관절 물체 메시; v1.0 2023-05 — [zc-alexfan/arctic](https://www.github.com/zc-alexfan/arctic), [arXiv 2204.13662](https://www.arxiv.org/pdf/2204.13662)
- 2025~2026년 버전 갱신("ARCTIC-v2")은 검색에서 확인되지 않음 — [검색 결과](https://www.github.com/zc-alexfan/arctic)

**TACO (CVPR 2024)** — 도구-동작-물체 양손, egocentric 뷰 포함
- 2.5K 모션 시퀀스, 3인칭 + egocentric 뷰, 정밀 손-물체 3D 메시, 동작 15종, 삼중항 131종, 고유 물체 모델 196개, 다중뷰 + 광학 모션 캡처 자동 파이프라인 — [arXiv 2401.08399](https://arxiv.org/html/2401.08399v2), [taco2024.github.io](https://taco2024.github.io)
- 데이터 라이선스·용량은 페이지에 미기재(웹사이트 자체는 CC BY-SA 4.0) — [taco2024.github.io](https://taco2024.github.io)

**OakInk2 (CVPR 2024)** — 양손 복합 작업, MANO + SMPL-X
- 다중뷰 영상 + 몸·손·물체 자세, 세 추상화 수준(Affordance, Primitive Task, Complex Task) — [arXiv 2403.19417](https://arxiv.org/abs/2403.19417); 툴킷은 MANO v1.2 별도 필요, 원시 데이터에 MANO + SMPL-X 포함 — [oakink/OakInk2](https://github.com/oakink/OakInk2); 검색 요약은 627 시퀀스라고 하나 원문에서 직접 확인 못함 — [검색 요약](https://arxiv.org/pdf/2403.19417)
- 공식 페이지는 "currently in preview", 주석이 바뀔 수 있음; 2024-12-31 물체 affordance 주석 추가; Hugging Face 호스팅; 라이선스·용량 미기재 — [oakink.net/v2](https://oakink.net/v2/)

**HOGraspNet (ECCV 2024, KAIST)** — 파지 분류학 + 접촉 맵, 다중뷰(egocentric 여부 미확인)
- 피험자 99명(10~74세), YCB 강체 22개 + 복합 물체 8개, 약 1.5M RGB-D 희소 프레임, MANO + HALO 메시 피팅(다중뷰 RGB-D), 물체 자세는 모션 캡처, 접촉 맵·파지 라벨 제공 — [arXiv 2409.04033](https://arxiv.org/abs/2409.04033), [hograspnet2024.github.io](https://hograspnet2024.github.io/)
- 접촉 맵 정의(거리 임계값)·라이선스·용량은 페이지에 없음, 다운로드는 신청서, 데이터 출처로 AI-Hub 명시 — [hograspnet2024.github.io](https://hograspnet2024.github.io/)

**GigaHands (CVPR 2025 Highlight, Brown + ETH)** — 대규모 양손 + 물체 6자유도, 외부 카메라 51대
- 34시간, 피험자 56명, 물체 417개, 모션 클립 14k, 183M 프레임, 텍스트 84k — [arXiv 2412.04244](https://arxiv.org/html/2412.04244v3)
- 저장소: 카메라 51뷰, MANO 파라미터(`params/`), 3D 키포인트(삼각측량·MANO·물체 좌표계 정렬판), 6자유도 물체 자세 3.3k 시퀀스(2025-09-04 전체 공개), 물체 메시는 "scanned and generated"(2025-07-09), CC BY-NC 4.0, 다중뷰 영상 약 1.06 TB, 영상 제외 약 20 GB, 2026-11-07 Hugging Face 전체 이관; egocentric 뷰 언급 없음 — [brown-ivl/GigaHands](https://github.com/brown-ivl/GigaHands)

**HO-Cap (NeurIPS 2025 D&B, UT Dallas + NVIDIA)** — 마커 없는 RGB-D 8대 + HoloLens
- RGB-D 8대 + HoloLens, 영상 70개 약 699K 프레임, 피험자 9명, 물체 64개, MediaPipe 손 + BundleSDF 물체 재구성 + FoundationPose 추적 + SDF 기반 손·물체 자세 공동 정제 — [arXiv 2406.06843](https://arxiv.org/html/2406.06843v4)
- 공식 페이지: RealSense 8대 + HoloLens, 물체 메시는 Azure Kinect 1대 녹화로 재구성, CC BY 4.0, 총 약 102 GB — [irvlutd.github.io/HOCap](https://irvlutd.github.io/HOCap/)

**HUMOTO (ICCV 2025, Adobe + UT Austin)** — 전신 모션 캡처 + 관절 부품, 영상 중심 아님
- 736 시퀀스(7,875초, 30 fps), 정밀 모델링 물체 63개, 관절 부품 72개, 전자기 추적 장갑·슈트 + Kinect 2대, 아티스트가 발 미끄러짐·물체 관통을 정리, 관통·접촉 엔트로피 등 품질 지표 제안 — [arXiv 2504.10414](https://arxiv.org/html/2504.10414v1)

**EgoPressure (CVPR 2025 Highlight, ETH + Microsoft)** — 압력은 있으나 물체는 평판
- 5시간, 21명, egocentric 1대 + Azure Kinect 7대(30 Hz), Sensel Morph 터치패드 압력, 손당 32 제스처, 손 메시는 HaMeR 초기화 후 다중뷰 최적화; 실제 물체 없음(터치패드만); 데이터는 "Pre-Release"(Google Drive) — [EgoPressure 페이지](https://yiming-zhao.github.io/EgoPressure/), [arXiv 2409.02224](https://arxiv.org/html/2409.02224v2)

**2025~2026 신규 egocentric 3D 손(+물체) 데이터**
- EgoExo-Hands(arXiv 2510.02601, Meta Reality Labs): 등 뒤 8대 외부 카메라 + Quest 3 egocentric 2뷰 이동 리그, 약 30,000 주석 프레임, 피험자 3명, 손당 21 키포인트(42개), UmeTrack/HOT3D 호환 개인화 메시; 물체 메시·6자유도 자세·접촉은 "향후 대규모 데이터에서 계획". 정답 품질(30대 돔 대비 MPJPE 중앙값/P90): 상호작용 없음 6.28/7.31 mm, 손-손 7.46/8.95 mm, 손-물체 7.90/12.53 mm. 공개 여부·라이선스 미기재 — [arXiv 2510.02601](https://arxiv.org/html/2510.02601v1)
- SHOW3D(arXiv 2603.28760, CVPR 2026): 같은 종류의 리그로 야외 포함 "3D annotations of hands and objects"; 시간·프레임·형식·라이선스는 초록에 없음, 프로젝트 사이트 show3d-dataset.github.io — [arXiv 2603.28760](https://arxiv.org/abs/2603.28760)
- OpenEgo(arXiv 2509.05513): 공개 데이터 6종을 통합한 1,107시간, 290 작업, 600+ 환경, 표준화 손 자세 + 의도 정렬 동작 프리미티브, CC BY 4.0, "will be released" — [arXiv 2509.05513](https://arxiv.org/abs/2509.05513)
- Aria Everyday Objects: 26 시퀀스, 20 장면, 3D 경계상자 584개(17 클래스), 비상업 — [projectaria_tools AEO](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_objects), [EFM3D](https://arxiv.org/pdf/2406.10224)
- Digital Twin Catalog(DTC, Meta 2025): 스캔 물체 2,000개(밀리미터 기하 정확도, PBR), Aria 시퀀스 200개(360도 100 + 자연 보행 100) — [arXiv 2504.08541](https://arxiv.org/abs/2504.08541), [projectaria_tools DTC](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/digital_twin_catalog)

### Inferences
- 사용자의 5가지 입력(물체 표면 기하, MANO 손, 접촉, 6자유도 궤적, 질량)에 대해 공식 주석만으로 1~4를 채울 수 있는 egocentric 데이터는 HOT3D 뿐입니다. 접촉은 공식 라벨이 없으므로 DexMachina/CHORD 방식(거리 임계값)으로 메시에서 계산해야 합니다(아래 질문 2). 질량(5)은 어떤 데이터셋도 제공하지 않습니다.
- HOT3D 의 정답은 마커 기반이지만 mm 수치가 공개되어 있지 않고, 저자 스스로 일부 프레임 품질 저하를 인정하므로 "밀리미터 단위로 접촉 성패가 갈린다"는 사용자 실험 기준에서는 HOT3D 접촉도 재검증이 필요합니다. 특히 MANO 형식은 UmeTrack 피팅 결과를 다시 변환한 것이라 UmeTrack 보다 덜 정확하다고 논문이 밝히고 있으므로, 접촉 계산은 가능하면 UmeTrack 메시로 하고 MANO 는 16마디 분할용으로만 쓰는 것이 안전합니다(추론).
- EgoDex 는 규모(829 h)와 다양성은 압도적이나 물체 3D 가 전혀 없어, 쓰려면 질문 3의 단안 재구성 경로(SAM3D 등 생성 메시 + FoundationPose)가 반드시 필요하고 그 결과의 접촉 정확도는 cm 수준입니다(질문 4). "초기 파지 앵커" 학습용 대규모 손 모션 프라이어로는 유용하나, 접촉 정답 공급원으로는 부적합합니다.
- 관절 물체가 필요한 작업은 ARCTIC 이 여전히 유일하게 "MANO + 관절 물체 메시 + egocentric 뷰"를 동시에 만족합니다. HOI4D 는 카테고리 수준 자세라 특정 물체 메시와 궤적의 정합이 느슨하고 손 주석도 반자동이므로 접촉 품질이 낮을 가능성이 큽니다.
- EPIC-Contact 는 "야외 egocentric + 접촉 대응"이라는 점에서 사용자 과제와 가장 가깝지만, 물체 스케일이 VLM 추정(MAE 0.94 cm)이고 자세는 한 프레임을 클립 전체에 복사한 것이라 궤적(4) 용도로는 쓸 수 없고, 접촉 위치 분포 학습·평가용으로만 적합합니다.

### Gaps
- HOT3D 데이터 라이선스의 구체 조항(상업 이용 가능 여부)과 전체 다운로드 용량은 공식 페이지에 없어 확인하지 못했습니다.
- HOT3D 물체 메시의 단위(미터)와 좌표계는 공식 문서에서 명시적으로 확인하지 못했습니다(마커 기반 모션 캡처와 정합되므로 미터 단위일 것으로 추론되지만 검증 필요).
- OakInk2 의 시퀀스·물체 수·카메라 구성(egocentric 포함 여부)·라이선스, TACO 의 라이선스·용량·카메라 대수, HOGraspNet 의 접촉 맵 정의는 원문에서 확인하지 못했습니다.
- Ego4D(2D 전용)와 DexYCB·GRAB(비 egocentric, CHORD 입력) 은 이번 조사에서 직접 수치를 확인하지 않았습니다.
- EgoDex 가 ICLR 2026 발표 시점에 물체 주석을 추가했는지는 저장소 README 기준으로 "언급 없음"까지만 확인했습니다.

---

## 핵심 질문 2. 로봇 논문에서 이미 쓰인 접촉 추출 — DexMachina 와 CHORD(video_to_data)가 데이터셋별로 하는 일

### Takeaway
CHORD 의 공개 저장소 video_to_data 는 ARCTIC·TACO·HOT3D·OakInk2·DexYCB·GRAB·H2O 7종을 "손목 자세 + MANO 21관절 + 링크별 손-물체 접촉 + 물체 6자유도"라는 통일 표현으로 변환하며, 접촉은 메시 거리 임계값(코드 1 cm/16마디, HOT3D 클립 분할 시 손끝 2 cm)으로 기하학적으로 계산합니다. 단안 RGB 영상 경로(MoGe + Grounding DINO + SAM2 + SAM3D + FoundationPose + ViPE/Dyn-HaMR + Gaussian splat 정제)는 "roughly-metric" 결과를 내며, 논문 스스로 RGB 재구성에서는 접촉 추정이 잡음이 많아 렌치 직접 대조 대신 완화된 목적함수로 전환한다고 밝힙니다.

### Cited Findings
- DexMachina 의 ARCTIC 처리 코드: `--contact_threshold` 기본값 0.01(미터, 즉 1 cm), `--max_contact_per_step` 기본값 50, 최근접 꼭짓점 거리 < 임계값으로 접촉 마스크 생성, 물체 부품 ID(1 또는 2)를 접촉점에 부여, 최원점 샘플링 옵션 — [로컬 파일 process_arctic.py](file:///home/robin/a_workspace/condex/extlib/dexmachina/dexmachina/retargeting/process_arctic.py)
- CHORD 논문 부록 D: "It ingests per-frame MANO hand fits and rigid-object 6-DoF trajectories from seven motion-capture datasets: ARCTIC, TACO, HOT3D, OakInk2, DexYCB, GRAB, and H2O. Each dataset-specific loader parses the source's native format and exports every sequence into a unified representation containing the wrist pose, 21 MANO joint positions, per-link hand-object contacts, and the object's full 6-DoF pose." — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf) (arXiv 2607.00033)
- HOT3D 전처리: 긴 egocentric 녹화를 원자 클립으로 분할. "Contact is detected when the minimum fingertip-to-object distance falls below 2 cm. Brief contact breaks of up to 15 frames are bridged… per-frame majority vote. This process expands roughly 294 raw recordings into approximately 4,045 atomic clips." — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 품질 검사: 로봇 손을 캡슐로 모델링해 손-물체(물체 볼록 껍질 기준)·손-손 관통이 어느 프레임에서든 2 cm 를 넘으면 시퀀스 기각; 볼록껍질/메시 부피비 > 3 인 오목 물체(머그, AR 안경)는 손-물체 검사에서 제외; Isaac Lab 에서 300 스텝 재생 검사 — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 코드 수준 접촉 정의(2026-10-05 커밋 cbee067a 에서 확인해 메모리에 기록): 부품당 물체 표면점 4,096개 ↔ MANO 손 메시 꼭짓점 거리 < 1 cm, 손 마디 16개(손바닥 + 손가락당 3)에 배정해 마디당 대표점 1개(거리 가중 평균), 위치는 물체 표면 최근접점, 법선은 손 메시 쪽 법선, 7종 데이터셋 공통; 렌치 공간은 마찰 뿔 모서리 8 + 순수 법선 1, 무작위 방향 512, 마찰계수 기본 0.1; 전신 로봇 경로는 손끝 기준점 11개/손, 거리 1.8 cm — [로컬 메모리 reference-chord-paper-code.md](file:///home/robin/.claude/projects/-home-robin-a-workspace-condex/memory/reference-chord-paper-code.md), 코드 위치 [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data)
- 단안 RGB 영상 경로(논문 부록 C, 붓 조작 과제 1건으로 검증): 입력은 단일 단안 RGB 영상 + 물체 이름 텍스트, 출력은 프레임별 MANO + 텍스처 물체 메시 + 6자유도 자세를 "a shared, roughly-metric space"에서 생성. MoGe 단안 깊이가 미터 기준점. 물체: Grounding DINO → SAM2 → SAM3D 로 메시(스케일 미정) → MoGe 깊이에 등록해 미터 크기 추정 → FoundationPose 로 추적. 손: ViPE + Dyn-HaMR 로 MANO 추정 후 렌더 깊이를 MoGe 에 맞춰 시선 방향 이동(프레임별 평행이동 + 전역 손 스케일). 정제: DROID-SLAM 으로 카메라 초기화한 Gaussian splat 장면을 광도·분할·깊이 손실 + 시간 평활로 공동 최적화 — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 논문 본문: "When contact estimates are noisy due to hand-object misalignment, as in demonstrations reconstructed from RGB videos, directly matching human contact wrenches can be unreliable. In such scenarios we pivot from direct matching to a reduced contact wrench guidance objective" — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 저장소 reconstruction README 의 모듈: `v2d_ego_hand_reconstruction`("4D hand reconstruction from egocentric video (ViPE + Dyn-HaMR)"), `v2d_foundation_pose`(6자유도 추적 + 메시 스케일 추정 `run_estimate_scale`/깊이 정렬 `run_align_mesh_scale`), `v2d_sam3d`(단일 영상+마스크 → GLB), `v2d_bundlesdf`(기존 자세에서 SDF 학습·텍스처, BundleTrack 추적은 미지원), `v2d_hoi_object_reconstruction`(CuSFM → 깊이 → 마스크 → 1단계 NeRF → FoundationPose → 2단계 NeRF), 깊이는 FoundationStereo/UniDepth/MoGe, 사람은 SAM3D-Body; 검증 GPU RTX A6000·L40S, Blackwell(sm_120) 미지원; 정확도·시간 수치 없음 — [video_to_data reconstruction README](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/README.md)
- 최상위 README: 3단계(Video Ingestion Agent, Reconstruction, Robotic Grounding), 로봇 학습은 Isaac Lab 2.3.1 + RSL-RL PPO, 재현 명령 예시가 "(arctic / hot3d / taco)" 세 가지만 노출 — [video_to_data](https://github.com/nvidia-isaac/video_to_data)

### Inferences
- 사용자의 현재 방식(MANO 16마디, 1 cm)은 CHORD 코드의 정의와 동일하므로, HOT3D·TACO·OakInk2 로 확장할 때 video_to_data 의 데이터셋별 로더를 그대로 가져와 쓰는 것이 가장 빠릅니다. 다만 HOT3D 손은 UmeTrack 원본이 더 정확하다고 논문이 밝히므로 로더가 MANO 변환본을 쓰는지 확인할 가치가 있습니다.
- CHORD 자체가 (i) 로봇 관통 2 cm 게이트, (ii) RGB 경로에서는 접촉 직접 대조 포기라는 두 가지 완화를 두고 있다는 점은, 그들의 파이프라인도 mm 수준 접촉 정확도를 보장하지 못함을 시사합니다. 사용자 실험이 mm 민감하다면 RGB 재구성 결과를 그대로 접촉 앵커로 쓰는 것은 위험하고, 시뮬레이션 내 재최적화(질문 4의 Grasp-and-Lift 류)가 필요합니다.

### Gaps
- video_to_data 의 데이터셋별 로더(`v2d_task_library_loader`)가 각 데이터셋에서 어떤 좌표계·단위 변환을 하는지는 오늘 README fetch 에는 나오지 않았고, 2026-10-05 로컬 확인 기록에만 의존했습니다. 최신 커밋에서 바뀌었을 수 있습니다.
- CHORD 가 HOT3D 클립을 몇 개나 최종 학습에 채택했는지(관통 게이트 통과율)는 본문에서 찾지 못했습니다.

---

## 핵심 질문 3. 단안·egocentric RGB 재구성 도구 — 손, 손-물체, 물체 자세, 접촉

### Takeaway
손 단독 재구성은 카메라 좌표계 PA-MPJPE 5~8 mm 수준까지 왔고 세계 좌표계도 HaWoR(HOT3D W-MPJPE 33 mm)로 크게 개선됐지만, 물체는 "생성 메시(SAM3D 등) + RGB-D 추적기(FoundationPose)"에 의존해 egocentric RGB 단독에서는 BOP 2024 기준 미지 물체 6자유도 검출 AP 26.8 에 불과하며, 손-물체 접촉 위치 오차는 최신 HOPformer 도 20~32 mm 입니다.

### Cited Findings

**손 메시(단안)**
- 카메라 좌표계 PA-MPJPE(여러 논문 표에서 취합, 프로토콜 상이): FreiHAND WiLoR 5.5 / Hamba 5.8 / HaMeR 6.0 mm; HO-3D WiLoR 7.5 / HaMeR 7.7 mm; ARCTIC 양손 공동 예측에서 WiLoR PA-MPJPE 7.35, Hamba 9.41, Dyn-HaMR 9.61 mm(MPJPE 18.05 / 25.74 / 22.10) — [검색 취합(UniHand, 비디오 확산 손 재구성, StableHand, GeoHand 등)](https://arxiv.org/pdf/2606.30308), [arXiv 2602.21631](https://arxiv.org/pdf/2602.21631), [arXiv 2605.18553](https://arxiv.org/pdf/2605.18553) (어느 표가 어느 수치인지 1:1 매핑은 미확인)
- HaWoR(CVPR 2025, egocentric 세계 좌표계): HOT3D 에서 PA-MPJPE 4.79, W-MPJPE 33.20, WA-MPJPE 11.27 mm vs WiLoR-SLAM 6.00 / 151.67 / 39.49, HaMeR-SLAM 9.39 / 156.03 / 43.37; DexYCB PA-MPJPE 4.76 vs WiLoR 5.01; 프레임당 순전파 약 40 ms(최적화형 HMP-SLAM 160 ms), "still far from real-time"; 손이 화면 밖으로 나간 구간은 motion infiller 로 보간; 코드 공개 — [arXiv 2501.02973](https://arxiv.org/html/2501.02973v1)
- Dyn-HaMR(CVPR 2025, 움직이는 카메라에서 4D 양손): H2O G-MPJPE 45.6 vs HaMeR 96.9, MPJPE 22.5; HOI4D G-MPJPE 58.5 vs 201.6, MPJPE 19.5; 양손 공동 최적화(관통·생체역학 제약); A100 에서 128프레임 클립당 초기화 3~6분 + 최적화 약 4.19분 + 렌더 약 1분; 코드는 "will be released" — [arXiv 2412.12861](https://arxiv.org/html/2412.12861v2). CHORD 저장소가 이를 모듈로 포함 — [video_to_data reconstruction README](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/README.md)
- Hand3R(2026-02): 단일 순전파 온라인 4D 손-장면, DexYCB(국소)·HOI4D(전역) 평가, HaMeR-SLAM/WiLoR-SLAM 보다 우수하나 전역 궤적에서는 HaWoR 에 뒤짐 — [arXiv 2602.03200](https://arxiv.org/abs/2602.03200v1)
- EgoGrasp(2026-01, v2 2026-03): egocentric 영상에서 세계 좌표계 손-물체 상호작용, 몸 사전지식 유도 확산 모델로 손 자세, HOI 사전지식 확산 모델로 6자유도 물체 자세 보간; 초록에 수치·코드 없음 — [arXiv 2601.01050](https://arxiv.org/abs/2601.01050)
- 2026년 추가 목격: StableHand(egocentric 세계 좌표계 양손, 흐름 매칭) — [arXiv 2605.18553](https://arxiv.org/pdf/2605.18553); UniHand — [arXiv 2602.21631](https://arxiv.org/pdf/2602.21631); 비디오 확산 모델 기반 손 재구성 — [arXiv 2606.30308](https://arxiv.org/pdf/2606.30308) (세 편 모두 상세 미확인)

**손-물체 공동 재구성**
- HOLD(CVPR 2024): 단안 상호작용 영상에서 카테고리 무관 손+물체 공동 재구성(관절 암시 모델), 3D 주석 불필요 — [arXiv 2311.18448](https://arxiv.org/abs/2311.18448v1)
- GHOST(2026-03): 2D Gaussian splatting 으로 단안 영상에서 동적 손·물체, 기존 카테고리 무관 방법보다 "an order of magnitude faster"(검색 요약은 13배 이상), ARCTIC·HO3D·야외 평가, 코드 github.com/ATAboukhadra/GHOST — [arXiv 2603.18912](https://arxiv.org/abs/2603.18912)
- HORT(ICCV 2025): 단일 RGB 에서 손 기하를 사전지식으로 희소→조밀 점군 물체 재구성, ObMan·HO3D·DexYCB·MOW 평가, 피드포워드라 확산 기반보다 빠름 — [arXiv 2503.21313](https://arxiv.org/html/2503.21313)
- MCC-HO + RAR(2024): 단일 RGB + 추정 손 → 손·물체 기하; GPT-4V 로 물체 인식 → Genie 텍스트→3D 생성 → ICP 강체 정렬, 영상 전체에 일관된 템플릿 제공 — [arXiv 2404.06507](https://arxiv.org/abs/2404.06507)
- EasyHOI(CVPR 2025): 단일 영상, LISA 분할 + 확산 기반 인페인팅·3D 재구성 + 물리 제약 사전지식 유도 최적화 — [arXiv 2411.14280](https://arxiv.org/html/2411.14280v4)
- HOPformer(ECCV 2026): 손-물체 공동 자세 추정 트랜스포머, ARCTIC(ego) CDev 31.9 mm / MPJPE 16.1 / SR@0.05 82.4%, EPIC-Contact CDev 20.7 mm / MPJPE 19.9 / SR@0.05 29.8%(JointTransformer 30.1 / 22.9 / 17.6, ArcticNet-SF 94.2 / 44.8 / 16.5) — [arXiv 2606.30598](https://arxiv.org/html/2606.30598)
- 장면 내 조작 영상 제로샷 재구성(2025-12): 단안 영상 → 물체 메시·자세 + 장면 점군 + 손 자세, 2단계 최적화로 물리적 타당성 — [arXiv 2512.19684](https://arxiv.org/abs/2512.19684)

**물체 자세 추적·재구성**
- FoundationPose(CVPR 2024): RGB-D 입력, 모델 기반·모델 프리 모두 지원, 추적 약 32 Hz, 추정 물체당 약 1.3 s(RTX 3090); YCBInEOAT 추적 ADD-S 96.42 / ADD 93.09 vs BundleSDF 93.77 / 86.95, BundleTrack 92.53 / 87.34; YCB-Video 모델 프리 추정 ADD-S 97.4 / ADD 91.5 — [arXiv 2312.08344](https://arxiv.org/html/2312.08344v2)
- 손 가림 취약성 보고: BundleSDF 는 부분·완전 가림 후 추적 실패 사례 — [arXiv 2505.02335](https://arxiv.org/pdf/2505.02335); FoundationPose 는 HO3D 에서 빈번하고 강한 회전 시 부정확 — [arXiv 2506.07996](https://arxiv.org/pdf/2506.07996)
- Any6D(CVPR 2025): RGB-D 참조 1장만으로 미지 물체 자세 + 미터 크기, 영상→3D 생성 모델로 형상 후 2D-3D 정렬·렌더 비교, HO3D 포함 평가, 코드 공개 — [arXiv 2503.18673](https://arxiv.org/html/2503.18673v2)
- BOP 2024 의 HOT3D(RGB/흑백만, 깊이 없음) 결과: 미지 물체 모델 기반 6자유도 검출 AP GigaPose+GenFlow 26.8(5.3 s), GigaPose 7.2(0.9 s); BOP-H3 평균 31.2 vs BOP-Classic-Core 50.4; 2D 검출 AP MUSE 42.6, CNOS(FastSAM) 35.0; 모델 프리 2D 검출 GFreeDet 33.8; 지표는 MSSD·MSPD; "2D detection is the main bottleneck" — [arXiv 2504.02812](https://arxiv.org/html/2504.02812v2)

**접촉 추정(영상→접촉)**
- DECO(ICCV 2023): 단일 영상에서 SMPL 몸 표면 전체의 조밀 접촉, DAMON 테스트 F1 0.55(BSTRO 0.46), 코드·데이터 공개 — [arXiv 2309.15273](https://arxiv.org/abs/2309.15273v1)
- HACO(NeurIPS 2025): 조밀 손 접촉 추정, 클래스 불균형(대부분 꼭짓점 비접촉, 접촉은 손끝 집중) 대응 손실, 손-물체·손-손·손-장면·손-몸 포함, 코드 github.com/dqj5182/HACO_RELEASE — [arXiv 2505.11152](https://arxiv.org/abs/2505.11152v2)
- PressureFormer(EgoPressure): 단일 egocentric RGB 에서 손 메시 UV 맵 형태의 압력 예측 — [arXiv 2409.02224](https://arxiv.org/html/2409.02224v2)

### Inferences
- "손은 충분히 좋고 물체가 병목"이 2026년의 상태입니다. 손 PA-MPJPE 5 mm 는 사용자 요구에 근접하지만, 이는 손목 정렬 후 수치이고 접촉에 중요한 것은 물체 좌표계 안의 절대 위치입니다. 세계 좌표계 오차(W-MPJPE 33 mm)와 물체 자세 오차가 그대로 접촉 오차로 전이됩니다.
- EgoDex·Ego4D 같은 RGB 전용 데이터에 적용 가능한 "현실적 최선"은 CHORD 부록 C 경로(MoGe 깊이 기준점 + SAM3D 메시 + FoundationPose + Dyn-HaMR/HaWoR + splat 정제)이며, 여기에 HACO 류 접촉 추정기를 손 메시 위 사전확률로 결합해 EPIC-Contact 식 접촉 손실로 재최적화하면 cm 이하로 내릴 여지가 있다고 봅니다(추론, 수치 근거 없음).
- FoundationPose 는 깊이를 요구하므로 Aria/Quest/Vision Pro 영상에는 MoGe·UniDepth 같은 단안 깊이로 대체해야 하고, 그 경우 스케일 오차(EPIC-Contact 의 VLM 스케일 5.9%, 10 cm 물체면 약 6 mm)가 접촉 오차에 직접 더해집니다.

### Gaps
- FoundationPose 의 HO3D(손 가림) 수치는 논문 HTML 에서 찾지 못했습니다.
- HORT·EasyHOI·MCC-HO·GHOST 의 접촉/관통 지표 수치는 초록 수준에서만 확인했고 표를 읽지 못했습니다.
- Dyn-HaMR 공식 코드 공개 여부는 논문 문구("will be released")만 확인했고 저장소 존재는 확인하지 못했습니다(단, video_to_data 가 래핑하고 있어 실행 가능한 구현은 존재).
- 손 PA-MPJPE 취합 수치의 논문별 출처 매핑이 불확실합니다.

---

## 핵심 질문 4. 손-물체 접촉 위치 정확도: 실험실 다중 카메라 vs 단안 야외 egocentric, 그리고 물리 검증

### Takeaway
실험실 정답조차 손-물체 상호작용 구간에서는 중앙값 약 8 mm, 90퍼센타일 12.5 mm 의 키포인트 오차(EgoExo-Hands 대 30대 돔)를 보이고 DexYCB·HO3D·HOI4D 정답은 시뮬레이터에서 관통·미접촉이 발생한다고 보고되며, 단안 야외 egocentric 에서는 최고 방법도 접촉 편차 20~32 mm, 제약 없는 재구성은 평균 관통 9~19 mm 입니다. 따라서 "mm·수 도 단위로 성패가 갈리는" 접촉을 영상에서 바로 얻는 것은 현재 불가능하고, 시뮬레이션 내 물리 재최적화(DeepSimHO·Grasp-and-Lift·CHORD 재생 검사)가 필수 단계로 자리잡고 있습니다.

### Cited Findings

**(a) 실험실·다중 카메라 데이터의 정답 품질**
- EgoExo-Hands 이동 리그 정답 vs 30대 카메라 돔: MPJPE 중앙값/P90 — 상호작용 없음 6.28/7.31 mm, 손-손 7.46/8.95 mm, 손-물체 7.90/12.53 mm — [arXiv 2510.02601](https://arxiv.org/html/2510.02601v1)
- AssemblyHands 자동 주석: 수작업 대비 평균 4.20 mm — [arXiv 2304.12301](https://arxiv.org/abs/2304.12301v1)
- HOT3D: 마커 기반이지만 정답 mm 오차 미공개, 1.5M 중 1.16M 프레임만 검수 통과, "lower quality" 프레임 존재 인정 — [arXiv 2411.19167](https://arxiv.org/html/2411.19167)
- HOI4D: 20% 프레임만 수작업 2D 키포인트 후 MANO 최적화 — [arXiv 2203.01577](https://arxiv.org/pdf/2203.01577); 정렬 불일치 증거 보고 — [arXiv 2409.07337](https://arxiv.org/html/2409.07337v1)
- Grasp-and-Lift(2026-01): "Existing resources such as DexYCB and HO3D are primarily optimized for visual alignment but often yield physically implausible interactions when replayed in physics simulators"(관통·미접촉·불안정 파지), 스플라인 키프레임 + CMA-ES 로 물리 엔진을 블랙박스 목적함수 삼아 실행 가능한 궤적으로 정제, ManipTrans 대비 재생 시 손·물체 자세 오차 감소 — [arXiv 2601.18121](https://arxiv.org/abs/2601.18121)
- HUMOTO 는 아티스트가 관통을 수작업 정리하고 관통·접촉 엔트로피를 데이터 품질 지표로 제안 — [arXiv 2504.10414](https://arxiv.org/html/2504.10414v1)
- CHORD 는 리타게팅 후 로봇 캡슐 관통 2 cm 초과 시 기각, Isaac Lab 300스텝 재생 검사 — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)

**(b) 단안·야외 egocentric 재구성의 접촉 정확도**
- EPIC-Contact 테스트에서 HOPformer CDev 20.7 mm(SR@0.05 29.8%, SR@0.1 69.7%), JointTransformer 30.1 mm, ArcticNet-SF 94.2 mm; ARCTIC egocentric 뷰에서는 HOPformer 31.9 mm — [arXiv 2606.30598](https://arxiv.org/html/2606.30598)
- 같은 논문: 야외 물체 스케일을 VLM 으로 추정하면 MAE 0.94 cm(5.9%), 정답 자체가 모션 캡처가 아니라 주석 기반이며 떨림 프레임은 낮은 신뢰도로 표시 — [arXiv 2606.30598](https://arxiv.org/html/2606.30598)
- 제약 없는 학습 재구성의 평균 관통 깊이: ObMan 9 mm, FHB 19 mm; 관통을 줄이면 시뮬레이션 파지 안정성이 떨어지는 상충 관계, 두 항 균형 시 둘 다 개선 — [Hasson et al. 2019](https://arxiv.org/pdf/1904.05767)
- DeepSimHO: 물리 시뮬레이터로 자세 안정성을 평가·학습, 지표는 200 ms 시뮬레이션 후 물체 중심 변위, 접촉 비율, 관통 깊이 — [arXiv 2310.07206](https://arxiv.org/pdf/2310.07206)
- 세계 좌표계 손 오차: HaWoR 33.2 mm vs SLAM 결합 단일 프레임 방법 150 mm 대(HOT3D) — [arXiv 2501.02973](https://arxiv.org/html/2501.02973v1)
- egocentric RGB 단독 미지 물체 6자유도 검출 AP 26.8(HOT3D, BOP 2024 최고) — [arXiv 2504.02812](https://arxiv.org/html/2504.02812v2)
- CHORD 의 RGB 경로 결과는 "roughly-metric"이며 접촉 직접 대조를 포기 — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 2024 HANDS 벤치마크 결론: egocentric 에서는 빠른 손 동작, 좁은 시야에서의 물체 재구성, 양손-물체 근접 접촉이 현재 방법들의 난점 — [arXiv 2403.16428](https://arxiv.org/html/2403.16428v2)

### Inferences
- 접촉 위치 오차의 현실적 하한을 정리하면: 실험실 마커/다중뷰 정답 ≈ 5~13 mm(상호작용 구간), 단안 야외 최신 ≈ 20~30 mm, 제약 없는 단안 ≈ 관통만 10~20 mm. 사용자 실험이 요구하는 "mm·수 도" 정밀도는 어떤 영상 기반 출처도 직접 만족하지 못하므로, 데이터에서 얻는 것은 "접촉 영역과 의도(어느 마디가 어느 면을 누르는가)"로 한정하고 정확한 접촉점은 시뮬레이터에서 물리적으로 재확정하는 2단계 설계가 문헌 흐름과 일치합니다.
- 사용자의 렌치 공간 점수(CHORD 방식)는 위치보다 법선·마찰 뿔에 민감하므로, 접촉 "위치" 오차 1~2 cm 보다 재구성된 물체 메시의 "표면 법선" 품질(생성 메시의 과도한 평활화 — HORT 논문이 지적)이 더 큰 위험 요인일 수 있습니다(추론).

### Gaps
- 접촉 위치 오차를 mm 로 직접 보고하는 논문은 EPIC-Contact/HOPformer 한 건만 찾았습니다. HOT3D·ARCTIC 에서 "정답 접촉 vs 재구성 접촉"을 체계적으로 수치화한 연구는 발견하지 못했습니다.
- 시뮬레이터에서 데이터셋 정답을 재생했을 때의 성공률(예: DexYCB 몇 %가 파지 유지 실패)을 수치로 제시한 출처는 초록 수준에서 확인하지 못했습니다(Grasp-and-Lift 본문 표 미열람).
- 2025~2026 egocentric 손-물체 재구성 전용 벤치마크는 찾지 못했습니다(2024 HANDS 벤치마크가 최신).

---

## 핵심 질문 5. 실용 정보 — 미터 스케일 메시·질량·재질을 주는 데이터셋, 영상에서 물리량 추정 도구

### Takeaway
미터 스케일 물체 메시는 HOT3D·ARCTIC·TACO·GigaHands·HO-Cap·H2O·HOI4D·ADT·DTC 가 제공(스캔 또는 RGB-D 재구성)하지만, 조사한 어떤 손-물체 데이터셋도 질량·마찰·재질을 공식 주석으로 제공하지 않습니다. 영상에서 질량·마찰을 추정하는 2025~2026 연구는 존재하나 실물에서는 정확도가 낮고 "한 번 밀기" 같은 능동 상호작용이나 밀도 가정에 의존합니다.

### Cited Findings
- 메시 출처별: HOT3D 사내 스캐너 PBR 메시(단위·질량 미기재) — [arXiv 2411.19167](https://arxiv.org/html/2411.19167); ADT .glb 물체 모델(2024-09) — [projectaria_tools ADT](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset); DTC 스캔 2,000개, 밀리미터 정확도, PBR, 질량 여부는 확인 안 됨 — [arXiv 2504.08541](https://arxiv.org/abs/2504.08541); GigaHands "scanned and generated meshes" — [brown-ivl/GigaHands](https://github.com/brown-ivl/GigaHands); HO-Cap Azure Kinect 1대로 재구성 — [irvlutd.github.io/HOCap](https://irvlutd.github.io/HOCap/); H2O BADSLAM 재구성 — [H2O 페이지](https://taeinkwon.com/projects/h2o/); HOI4D 재구성 메시 — [arXiv 2203.01577](https://arxiv.org/abs/2203.01577v4); ARCTIC 물체는 제3자 URDF 파일 링크 존재 — [awesome 목록](https://awesome.ecosyste.ms/projects/github.com%2Fzc-alexfan%2Farctic)
- HOGraspNet(22개)·DexYCB 는 YCB 물체 세트를 사용 — [arXiv 2409.04033](https://arxiv.org/abs/2409.04033); CHORD 는 DexYCB 를 7종 중 하나로 사용 — [로컬 PDF 2607.00033v1](file:///home/robin/a_workspace/condex/harness/research/references/papers/2607.00033v1.pdf)
- 영상 기반 물리량 추정: 비디오 파운데이션 모델로 동마찰·탄성·점성 추론, 합성에서는 강하나 실물(특히 마찰)에서 정확도 하락, 실데이터 미세조정이 마찰에 가장 효과 — [arXiv 2510.02311](https://arxiv.org/pdf/2510.02311v1); PhyPush(2026): 한 번 밀기 영상으로 질량·마찰 추정, 질량은 과소·마찰은 과대 추정 경향 — [arXiv 2605.26284](https://arxiv.org/pdf/2605.26284); RGB-D 강체 추적 + 마찰 계수 필터링(불확실성 포함) — [arXiv 2309.15703](https://arxiv.org/html/2309.15703v3); 용기 질량은 내용물 높이·부피·밀도 가정 + 용기 질량으로 계산 — [arXiv 2107.01309](https://arxiv.org/pdf/2107.01309); 사람이 드는 동작의 운동학만으로는 가벼운/무거운 물체 구분이 "inadequate" — [arXiv 2003.12045](https://ar5iv.arxiv.org/html/2003.12045); 인터넷 영상에서 사람-물체 힘 추정은 물체 유형별 공통 질량 분포 가정 — [arXiv 2111.01591](https://arxiv.org/pdf/2111.01591)

### Inferences
- 질량·마찰은 데이터에서 읽는 대신 (1) YCB 계열 물체는 YCB 공식 물성표(질량 공개로 알려져 있으나 이번 세션에서 미확인)를 참조하고, (2) 나머지는 스캔 메시 부피 × 재질별 밀도 가정 + 시뮬레이션 영역 무작위화로 처리하는 것이 현실적입니다. 사용자가 스케일업 물체(2~3대 로봇 팔)를 다루므로 어차피 사람 쪽 질량은 그대로 쓰이지 않고 스케일 법칙으로 재설정되어야 하며, 이 경우 "정확한 질량"보다 "질량중심 위치와 마찰 범위"가 렌치 점수에 더 중요합니다(추론).
- PhyPush 류는 로봇이 실제로 밀어볼 수 있을 때만 유효하므로 오프라인 데이터셋 전처리에는 적용 불가합니다.

### Gaps
- DTC 와 HOT3D 물체의 질량·치수 메타데이터 유무는 공식 문서 본문을 끝까지 읽지 못해 확정하지 못했습니다(공식 "object models" 데이터 포맷 페이지 확인 필요).
- YCB 물체 질량표의 존재와 수치는 이번 조사에서 직접 확인하지 않았습니다.
- 손 조작 영상만으로 질량을 추정해 로봇 전이에 쓴 2025~2026 논문은 찾지 못했습니다.

---

### 우선순위 제안 (질문 5 Inferences 보충 — 사용자 과제 기준 추론)
1. HOT3D(Aria 분할, UmeTrack 손 + 스캔 메시 + 마커 기반 6자유도) → video_to_data 로더로 16마디·1 cm 접촉 추출 → 강체 과제 egocentric 데이터 1순위. 전처리: 클립 분할(CHORD 방식 손끝 2 cm/15프레임 브리지), 검수 통과 마스크(`mask_qa_pass.csv`) 적용, 테스트 피험자 제외.
2. ARCTIC egocentric 뷰 + TACO egocentric 뷰 → 관절 물체·양손 도구 과제. 이미 보유한 ARCTIC 파이프라인을 TACO 로더로 확장.
3. EPIC-Contact → 야외 접촉 분포의 평가·사전확률(궤적은 없음).
4. EgoDex/Ego4D/HD-EPIC → 물체 3D 가 없으므로 CHORD 부록 C 경로로 재구성하되, 접촉은 앵커가 아니라 "접촉 영역 힌트"로만 쓰고 시뮬레이터 재최적화(Grasp-and-Lift/DeepSimHO 류)를 거칠 것.
