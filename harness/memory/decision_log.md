# Research decision log

Append, never silently rewrite historical decisions.

Template:
- Date:
- Idea / Experiment ID:
- Researcher decision:
- Rationale and rejected alternatives:
- Evidence (paths / run IDs):
- What would change our mind:
- Next action:

---

- Date: 2026-11-13
- Idea / Experiment ID: IDEA-001-imitate-verified-gt / EXP-001-gt-seed-consistency
- Researcher decision: net2는 정답(GT) 위치에 가깝게 내는 것을 목표로 하고, 성공은 정답 품질로 해결하는 방향을 추진한다. 첫 단계로 EXP-001(씨앗만 바꾼 정답 탐색의 자기 일치율)을 승인한다. "합격한 정답만으로 net2 재학습"은 EXP-001 결과를 본 뒤로 보류한다. 아이디어 번호는 양식 예시였던 기존 IDEA-001을 지우고 이 아이디어에 준다.
- Rationale and rejected alternatives: 합격한 정답 근처에 간 후보는 150개 중 140개가 합격해 방향의 논리가 성립한다. 그러나 세 팔 모두 정답 한 칸 이내인 출력이 1.5%뿐이고, 그 원인이 네트워크인지 정답의 뽑기 운인지 갈리지 않았다. "합격한 정답만으로 재학습"을 먼저 하는 안은 한 칸 이내 비율을 올리는 실험이 아니어서 뒤로 미뤘다.
- Evidence (paths / run IDs): harness/research/IDEA-001-imitate-verified-gt/ (idea.md, discussion.md), 근거 수치의 원자료 scratchpad/net2_kshot_cases.pkl, scratchpad/net2_rule_eval_test.pkl, scratchpad/net2_viz_cases.pkl, scratchpad/net2_viz_cases_test.pkl
- What would change our mind: EXP-001에서 세 팔 모두 한 칸 이내 일치율이 80% 이상이면 정답은 일관된 것이므로 네트워크 쪽을 먼저 고친다. 50% 미만이면 정답 생성을 먼저 고친다.
- Next action: EXP-001 실행 후 analysis.md 작성, 다음 단계는 결과 보고 뒤 연구자가 정한다.

- Date: 2026-11-13
- Idea / Experiment ID: IDEA-001-imitate-verified-gt / EXP-002-gt-seed-site-attribution
- Researcher decision: EXP-002(정답 탐색에서 난수 씨앗이 들어가는 세 자리를 한 자리씩만 바꿔 다시 돌리는 측정)의 실행을 승인한다. EXP-001 결과에 따른 다음 방향(정답 생성을 고칠지, 네트워크를 고칠지)은 아직 정하지 않았다.
- Rationale and rejected alternatives: EXP-001에서 씨앗만 바꿔도 세 팔 모두 0.078배 이내로 일치하는 정답이 29.0%뿐이었다. 어느 자리가 원인인지 알아야 정답 생성을 어떻게 고칠지 정할 수 있다는 Claude의 제안을 연구자가 "일단 해봐바"로 받아들였다. 다른 후보(조건부 net2 집계, 촘촘한 탐색의 수렴 확인, 흔들어도 남는 α의 사전 확인)는 이번에 승인되지 않았다.
- Evidence (paths / run IDs): harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-001-gt-seed-consistency/ (analysis.md, runs/run-001), harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-002-gt-seed-site-attribution/plan.md
- What would change our mind: 한 자리의 불일치율이 전체 불일치율(71.0%)의 절반 이상이면 그 자리를 주된 원인으로 본다. 어느 자리도 절반에 못 미치면 한 자리만 고쳐서는 해결되지 않는다고 본다.
- Next action: EXP-002 실행 후 analysis.md 작성, 다음 단계는 결과 보고 뒤 연구자가 정한다.

- Date: 2026-11-13
- Idea / Experiment ID: IDEA-001-imitate-verified-gt / EXP-003-coarse-candidate-precheck
- Researcher decision: 두 단계로 간다. 1단계는 물체 표면의 후보 점을 128개로 줄여 α 기준 정답을 맞히는 모델을 학습하는 것이고, 잘 되면 2단계로 시뮬레이션 성공을 맞히는 모델을 학습한다. 1단계에 앞서 EXP-003(후보 128개로 조합을 전부 채점하는 사전 확인)의 실행을 승인한다.
- Rationale and rejected alternatives: EXP-001과 EXP-002에서 정답이 탐색의 뽑기(팔마다 후보를 32개로 줄이는 자리)에 좌우된다는 것이 확인되었다. 후보를 줄이면 조합을 전부 채점할 수 있어 정답이 씨앗과 무관해진다는 것이 연구자의 제안이다. Claude가 제안한 "합격 예측 모델을 먼저" 안은 이번에 채택되지 않았고, 필요할 때 나란히 할 수 있는 것으로 남겼다. 후보가 모자란 팔의 처리 규칙과 128개 대 256개는 EXP-003 결과를 보고 정한다.
- Evidence (paths / run IDs): harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-003-coarse-candidate-precheck/plan.md, harness/research/IDEA-001-imitate-verified-gt/discussion.md (2026-11-13의 후보 점 논의 세션들)
- What would change our mind: 네 후보 규칙 모두에서 새 정답의 α가 저장된 정답 α의 0.8배 이상인 레코드가 75% 미만이면 128개는 부적합하다고 본다. α가 최선의 0.95배 이상인 다른 조합이 있는 레코드가 50%를 넘으면 정답 하나 대신 정답 묶음을 쓴다.
- Next action: EXP-003 연기 시험과 본 실행, analysis.md 작성. 1단계의 정답 재생성과 학습은 결과 보고 뒤 따로 계획하고 승인받는다.

- Date: 2026-11-14
- Idea / Experiment ID: IDEA-001-imitate-verified-gt / EXP-004-c128-learnability
- Researcher decision: EXP-003에서 후보 128개는 α 기준으로 부적합하다고 나왔지만, 그와 별개로 "후보 128개로 만든 정답으로 학습한 네트워크가 그 정답을 그대로 내는가"를 먼저 본다. 정답 생성(약 4시간) 뒤 모델 A를 먼저 학습하고, A가 끝나면 모델 B 학습을 시작하면서 동시에 A의 결과를 분석한다.
- Rationale and rejected alternatives: α가 깎이는 문제와 "정답이 일관되면 네트워크가 맞히는가"는 별개의 질문이고, 뒤쪽 답은 후보를 256개로 하든 다른 방식으로 하든 그대로 쓰인다는 것이 연구자의 판단이다. Claude가 제안한 "256개 일부 탐색을 먼저" 안과 "128개로 고른 뒤 촘촘한 점으로 다듬기" 안은 이번에 채택되지 않았다. 대조군(모델 B: 지금 정답을 가장 가까운 후보로 옮겨 학습)은 Claude의 제안이고 연구자가 실행 순서에 포함했다.
- Evidence (paths / run IDs): harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-003-coarse-candidate-precheck/analysis.md, harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-004-c128-learnability/plan.md
- What would change our mind: 모델 A가 세 팔 모두 정답과 같은 점을 고르는 비율이 25% 미만이면 정답이 일관되어도 네트워크가 못 맞히는 것으로 본다. A와 B의 차이가 5%포인트 미만이면 정답의 일관성은 원인이 아니라고 본다.
- Next action: 구현과 연기 시험, 정확한 실행 명령을 보여 주고 긴 실행의 승인을 받는다.

- Date: 2026-11-20
- Idea / Experiment ID: IDEA-001-imitate-verified-gt / EXP-005-net1-data-scaling, EXP-006-gt-v2-pilot
- Researcher decision: GT 를 새로 만든다. 게이트는 법선 힘 제어(하이브리드) + 외란별 최적 배분, 채점은 논문 III-B 의 ℓ∞ 여유 ρ, 파티션마다 3시드 최댓값. 방식과 규모를 정하기 전에 EXP-005(데이터 규모 시험)와 EXP-006(1,000 파지 시범 생성)을 먼저 한다.
- Rationale and rejected alternatives: 2026-11-15~19 진단에서 게이트 불합격의 약 절반이 위치 서보의 접촉 놓침이었고 힘 제어로 회복되었다. 3시드 최댓값은 EXP-001/002 의 탐색 노이즈 대책(연구자 제안). 전량 생성 추정(ρ64 + 3시드 약 310시간)이 커서 실측과 데이터 필요량 확인을 먼저 한다. 8꼭짓점(힘만) 축약형은 토크가 최약 방향인 경우가 많아 Claude 가 비권장.
- Evidence (paths / run IDs): scratchpad/2026-11-gate-diagnosis/, EXP-005/006 plan.md
- What would change our mind: EXP-005 에서 데이터가 평평하면 생성 규모를 줄인다. EXP-006 에서 two_stage 의 ρ0 가 rho64 의 0.95배 이상이면 two_stage 로 시간을 줄인다.
- Next action: 구현·연기 시험 → EXP-005, EXP-006 실행 → 결과 보고 후 연구자가 전량 재생성 방식을 정한다.
