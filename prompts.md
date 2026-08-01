일단 다시 지금 최적화 수식을 어떤 걸 풀고있는건지 다시 정리해서 보여줘. 읽기 쉽게 html로 작성해줘봐. theta 가 왜 117차원인지도 설명해줘 

지금 힘이 누르는 힘만 작용하짆아. 하지만 friction cone내에서는 수직항력을 바탕으로 수평 힘도 가할 수 있는거잖아. 내가 이해하기론 지금 이 수평방향 힘은 passive하게만 작용하고 있는 것 같은데, 이걸 active하게 줄 수 도 있는 거 아닌가? 그래서 지금 뚜껑 여는 것이 힘든 것 아냐? 내가 이해한게 맞나?

실험해보는 건 찬성인데 그전에 확실히 하고 가야 할 게 있어. 경로만 나오는 경우는 sim2real을 위해 다양한 물성치에 대해서 강건하게 최적화를 돌린건데, 액티브하게 접선력을 추가하는 경우는 sim2real에서 문제가 생기지 않을까?





---
# Contraction Net — Stage 1: Topology & Grouping Network Implementation Specification

**문서 작성일:** 2026-07-27  
**목적:** 사람 시연 파지(Human Grasp) 조건을 입력받아, 로봇 팔 $N \in \{2, 3, 4\}$개의 앵커 및 손가락 유닛 그룹 파티션 조합(총 26가지 유효 후보)을 예측하는 **1단계 그래프 트랜스포머 네트워크(Stage 1 Network)** 구현 명세서.

---

## 1. 개요 및 설계 목표

본 네트워크는 **1단계 Topology & Grouping Planner**로서, 사람의 파지 데이터로부터 이산적인(Discrete) 로봇 팔-손가락 대응 관계(`unit_group`) 및 앵커 개수 $N$을 결정한다.

* **입력:** 물체 표면 128개 패치 정보 + 6개 손가락 유닛 접촉 정보 + 0/1 바이너리 인접 행렬 + 전역 손 관절각(`qpos`)
* **출력:** 총 26가지 유효 손가락 파티션 조합 후보에 대한 확률 분포 Logits ($\hat{\mathbf{y}} \in \mathbb{R}^{26}$)
* **핵심 특성:**
  1. **다중 모달리티 보존 (D12 원칙):** One-Hot 원핫 라벨 대신, 데이터셋의 내하중 물리 품질($\min \alpha$) 기반 **Quality-Weighted Soft Target** 확률 분포를 정답(GT)으로 학습 (KL-Divergence Loss 사용).
  2. **[CLS] Token 전역 요약:** 134개 노드(물체 128 + 손가락 6) 간 전역 Attention 후 `[CLS]` 토큰으로 알짜배기 문맥만 압축.
  3. **Two-Stream Late Fusion:** 국소적 기하·접촉 그래프 문맥과 전역 손 자세(`qpos`, Taxonomy Prior)를 분류 직전 융합.

---

## 2. 모듈 입력 명세 (Data Input Specs)

### 2.1 그래프 노드 텐서 ($X \in \mathbb{R}^{B \times 135 \times D_{\text{node}}}$)
총 135개 노드로 구성되며, 각 노드는 공통 임베딩 차원 $D_{\text{node}}$ (기본값: `256`)로 투영된다.

1. **물체 패치 노드 (Index `0 ~ 127`, 총 128개):**
   * **입력 구성:**
     * `center_pos` $[B, 128, 3]$ : 패치 중심 좌표 (물체 외접반경 `r_obj`로 정규화)
     * `normal` $[B, 128, 3]$ : 패치 안쪽 방향 법선 벡터
     * `curvature` $[B, 128, 1]$ : 표면 곡률
     * `pointnet_feat` $[B, 128, D_{\text{enc}}]$ : 패치 내부 로컬 포인트 클라우드를 Mini-PointNet으로 인코딩한 특징 벡터
   * **인코더:** `ObjectNodeEncoder` (Concat 후 `nn.Linear` $\rightarrow D_{\text{node}}$)

2. **손가락 유닛 노드 (Index `128 ~ 133`, 총 6개):**
   * **유닛 순서:** 0=엄지(Thumb), 1=검지(Index), 2=중지(Middle), 3=약지(Ring), 4=소지(Little), 5=손바닥(Palm)
   * **입력 구성:**
     * `unit_id` $[B, 6]$ : 유닛 인덱스 (0~5) $\rightarrow$ `nn.Embedding(6, D_id)`
     * `joint_ratios` $[B, 6, 2]$ : 마디 비율 인코딩 (`[distal_ratio, mid_ratio]`)
   * **인코더:** `FingerNodeEncoder` (Concat 후 `nn.Linear` $\rightarrow D_{\text{node}}$)

3. **`[CLS]` 전역 요약 토큰 (Index `134`, 1개):**
   * 학습 가능한 벡터 `nn.Parameter(torch.randn(1, 1, D_node))`

### 2.2 인접 행렬 ($A \in \{0, 1\}^{B \times 135 \times 135}$) — 바이너리 엣지
복잡한 3D 변환 엣지 대신 0과 1의 바이너리 인접 행렬로 연결 통로를 제공한다.
* 패치 $\leftrightarrow$ FPS 상의 인접 패치 (1-hop 이웃) = `1`
* 손가락 유닛 $\leftrightarrow$ 실제 닿아 있는 물체 패치 (`contact_mask == 1`) = `1`
* `[CLS]` 토큰 $\leftrightarrow$ 모든 노드 (135개 전체) = `1`
* 모든 노드의 Self-loop (자기 자신) = `1`
* 그 외 비연결 노드 쌍 = `0`

### 2.3 전역 손 Joint Pose ($Q_{\text{pose}} \in \mathbb{R}^{B \times 29}$)
* 원본 손 자세 `qpos` 29차원 (손목 위치 3 + 회전 4 + 관절각 22)
* `PoseEncoder`: `MLP([29, 64, D_pose])` (기본값: $D_{\text{pose}} = 64$)

---

## 3. 네트워크 아키텍처 (Model Architecture)

```text
  [128 Object Patches]     [6 Finger Units]     [1 Learnable CLS Token]
           │                      │                        │
           ▼                      ▼                        ▼
  [ObjectNodeEncoder]    [FingerNodeEncoder]               │
           │                      │                        │
           └──────────────────────┼────────────────────────┘
                                  ▼
                    Node Matrix X [B, 135, D_node]
                                  │
                                  ├────────────────┐ (Attention Bias 주입)
                                  ▼                ▼
                     [ Graph Transformer Encoder ] ◄── Adjacency Matrix A [B, 135, 135]
                                  │
                                  ▼
                   Updated CLS Token h_CLS [B, D_node]
                                  │
    [qpos 29차원] ──► [PoseEncoder] ──► Pose Vector h_pose [B, D_pose]
                                  │
                                  ▼
                       [ Concat (Late Fusion) ]
                      h_fused [B, D_node + D_pose]
                                  │
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
   [ Main Classification Head ]             [ Auxiliary Anchor Head ]
             │                                         │
             ▼                                         ▼
   26-dim Partition Logits y_hat               3-dim Anchor Logits (N=2,3,4)
    

---
데이터 생성 쪽은 사실상 끝났고, 남은 건 "확장 결정" 하나입니다:

1. 전량 158만 확장 — 유일한 실질 작업. 지금 상태로는:

- 준비물은 전부 완비 (물체 캐시 10,690개 전체, 배칭 러너 + 체크포인트, 시뮬 판정 러너 2종)
- 비용: 탐색 2.5~3일 + 시뮬 판정 ~4일 = 총 6~7일 컴퓨터 점유
- 시작 조건(제 권고): Stage 1 학습이 10만(학습 가용 68.8만 행)으로 돌아본 뒤 데이터 부족이 병목으로 확인되면 — 그 신호 없이 미리 걸 이유가 없습니다
- 걸기 전에 할 일 하나: 배칭 러너 2~4프로세스 경합 실측으로 정확한 일정 확정 (30분짜리)

2. 950만 확장 예비 카드 — 어제 실측으로 "같은 파일의 나머지 5/6은 준중복 0%, 진짜 다른 파지"임이 확인돼서, 158만으로도 모자라면 6배까지 늘릴 실물 자원이 있다는 것만 기록된 상태. 실행은 필요성 입증 후.

3. 실기 하드웨어가 정해질 때의 캘리브레이션 항목 (지금 할 수 없는 것들): 팁의 접촉면 비율(hertz_frac=0.5 자리표시자), 실제 마찰계수, 곡률 보정 필요성 재확인. 전부 계획서에 "실기 캘리브레이션 항목"으로 표시돼 있습니다.

즉 지금 당장 이 세션이 손 놀릴 일은 없고, Stage 1 첫 학습 결과가 나오면 그 신호로 전량 확장 여부를 결정하는 대기 상태입니다.
