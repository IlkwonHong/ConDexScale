"""조건부 net2 전체 구조도(그림 2) SVG 생성 → net12_figure HTML 의 fig2 교체."""
import re
from html import escape

G = "#2E8B57"  # 새로 추가 · 바뀜 표시 색
K = {"input": ("#E8F1FC", "#2F6FC2", None), "reuse": ("#EFEAFB", "#6E56C2", None),
     "learned": ("#FFFFFF", "#1F2937", None), "fixed": ("#FFFFFF", "#6B7280", "4 3"),
     "output": ("#FDEBDD", "#C8621B", None)}
out = []


def t(x, y, ko, en=None, size=12, weight=400, fill="#374151", anchor="middle"):
    a = f' data-en="{escape(en)}"' if en and en != ko else ""
    out.append(f'<text x="{x}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
               f'fill="{fill}"{a}>{escape(ko)}</text>')


def box(x, y, w, h, kind, title, lines=(), new=False):
    fill, stroke, dash = K[kind]
    sw = 1.3
    if new:
        stroke, sw = G, 2.2
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    rows = [(title[0], title[1], 12.5, 600, "#111827")] + [
        (l[0], l[1], l[2] if len(l) > 2 else 11.5, 400, l[3] if len(l) > 3 else "#4B5563") for l in lines]
    hs = [15.5 if r[2] >= 12 else 14.5 for r in rows]
    cy = y + (h - sum(hs)) / 2 + 11.5
    for (ko, en, sz, wt, col), hh in zip(rows, hs):
        t(x + w / 2, cy, ko, en, sz, wt, col)
        cy += hh


def arrow(pts, label=None, lx=None, ly=None, size=10.5, color="#4B5563"):
    p = " ".join(f"{a},{b}" for a, b in pts)
    out.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="1.3" marker-end="url(#ah2)"/>')
    if label:
        t(lx, ly, label[0], label[1], size, 400, "#374151")


def panel(y, h):
    out.append(f'<rect x="8" y="{y}" width="1104" height="{h}" rx="12" fill="#F6F6F3"/>')


W, H = 1120, 1022
# ───────── (a) 조건부 net2 전체 ─────────
panel(8, 534)
t(24, 32, "(a) 조건부 net2 — 그룹 k의 점수", "(a) Conditional net2 — score of group k", 14, 700, "#111827", "start")
t(1096, 32, "초록 = 원래 net2(그림 1 (c))에서 새로 추가되거나 바뀐 부분",
  "green = added or changed vs. the original net2 (Fig. 1 (c))", 11.5, 600, G, "end")
# 입력 열
box(24, 50, 176, 40, "reuse", ("패치 노드 출력 P", "Patch tokens P"), [("128 × 256 ← 그림 1 (a)", "128 × 256 ← Fig. 1 (a)", 11.5, "#374151")])
box(24, 98, 176, 40, "input", ("가까운 패치 3개", "3 nearest patches"), [("2048 × 3 (번호·비율)", "2048 × 3 (id, weight)", 11.5, "#374151")])
box(24, 146, 176, 40, "input", ("표면점 특징", "Surface-point features"), [("2048 × 10", None, 11.5, "#374151")])
box(24, 200, 176, 40, "reuse", ("부위 MLP 출력", "Unit MLP output"), [("6 × 128 ← 그림 1 (b)", "6 × 128 ← Fig. 1 (b)", 11.5, "#374151")])
box(24, 248, 176, 52, "input", ("선택된 파티션", "Selected partition"),
    [("6 (부위 → 그룹 번호)", "6 (unit → group id)", 11.5, "#374151"),
     ("바뀜: 엄지 그룹 먼저 순서", "changed: thumb group first", 10.5, G)])
box(24, 316, 176, 40, "reuse", ("이미 놓인 앵커 g_j", "Placed anchors g_j"), [("j = 1 … k−1 (표면점 번호)", "j = 1 … k−1 (point ids)", 11, "#374151")], new=True)
box(24, 366, 176, 40, "input", ("표면점 위치 · 법선", "Point position · normal"), [("2048 × 6", None, 11.5, "#374151")], new=True)
# 둘째 열
box(240, 50, 136, 88, "fixed", ("거리 비율로 섞기", "Weighted mix"), [("점마다 가까운", "blend the 3 nearest"), ("패치 3개 출력을 더함", "patch tokens per point")])
box(240, 200, 136, 80, "fixed", ("그룹별 합", "Group sum"), [("파티션대로", "pool units"), ("부위 벡터 묶기", "by partition")])
box(240, 308, 136, 40, "fixed", ("앞 자리 벡터 꺼내기", "Gather anchor vector"), [("E[g_j] → 128", None, 11)], new=True)
box(240, 362, 136, 40, "fixed", ("상대 기하", "Relative geometry"), [("점마다 5 (p와 g_j)", "5 per point (p vs g_j)", 11)], new=True)
# 셋째 열
box(430, 66, 140, 108, "learned", ("점 MLP", "Point MLP"), [("(256+10)", None, 12, "#374151"), ("→256→128", None, 12, "#374151"), ("→ 점 벡터 E", "→ point embedding E")])
box(430, 218, 140, 44, "learned", ("그룹 선형층", "Group linear"), [("128→128", None, 12, "#374151")])
box(430, 316, 140, 86, "learned", ("조건 MLP", "Condition MLP"), [("(128+5)→128→128", None, 12, "#374151"), ("놓인 앵커 1개마다", "once per placed anchor", 11)], new=True)
# 넷째 열
box(620, 316, 120, 86, "fixed", ("앞 자리끼리 합", "Sum over j < k"), [("C_k: 2048 × 128", None, 11.5, "#374151"), ("k = 1이면 0", "zero when k = 1", 11)], new=True)
# 다섯째 열
box(786, 200, 110, 80, "fixed", ("더하기", "Add"), [("E + T_k + C_k", None, 11.5, "#374151"), ("(+ S_k 실험)", "(+ S_k, experimental)", 10, "#B45309")])
t(841, 297, "바뀜: C_k 추가", "changed: C_k added", 10.5, 600, G)
# 여섯째 열
box(936, 200, 160, 56, "learned", ("점 점수 MLP", "Point-score MLP"), [("128→128→1", None, 12, "#374151"), ("그림 1 (c)와 같은 층", "same layer as Fig. 1 (c)", 11)])
box(936, 290, 160, 66, "output", ("그룹 k 확률 P_k", "Group-k probability P_k"),
    [("2048", None, 12, "#374151"), ("후보 범위 밖 제외 → 확률", "mask → softmax", 11), ("바뀜: 그룹마다 차례로", "changed: one group at a time", 10.5, G)])
box(936, 372, 160, 40, "input", ("후보 범위 (그룹 k)", "Candidate mask (group k)"), [("2048 (참/거짓)", "2048 (bool)", 11.5, "#374151")])
# ── 지지 함수 경로 (2026-10-24 실험, 옵션 --wrench-feat support, 채택 안 함) ──
V = "#B45309"   # 실험 경로 색 (주황)
def vbox(x, y, w, h, kind, title, lines=()):
    fill, _, dash = K[kind]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{V}" stroke-width="2"{d}/>')
    rows = [(title[0], title[1], 12.5, 600, "#111827")] + [(l[0], l[1], l[2] if len(l) > 2 else 11.5, 400, l[3] if len(l) > 3 else "#4B5563") for l in lines]
    hs = [15.5 if r[2] >= 12 else 14.5 for r in rows]; cy = y + (h - sum(hs)) / 2 + 11.5
    for (ko, en, sz, wt, col), hh in zip(rows, hs):
        t(x + w / 2, cy, ko, en, sz, wt, col); cy += hh
t(240, 438, "실험 경로 (주황): 렌치 지지 함수 — 2026-10-24 시험, 효과 없어 채택하지 않음. 옵션 --wrench-feat support",
  "experimental path (orange): wrench support function — tested 2026-10-24, no gain, not adopted. flag --wrench-feat support", 11.5, 600, V, "start")
vbox(24, 452, 176, 40, "input", ("표면점 위치 · 법선", "Point position · normal"), [("2048 × 6 (위와 같음)", "2048 × 6 (same as above)", 11, "#374151")])
vbox(240, 452, 136, 54, "fixed", ("법선 렌치 → 방향값", "Normal wrench → projections"), [("w_p = [n ; pos × n]", None, 10.5), ("proj_p(d) = d·w_p, 28방향", "proj_p(d) = d·w_p, 28 dirs", 10.5)])
vbox(416, 452, 168, 54, "fixed", ("지지 함수 h_k", "Support function h_k"), [("h_k(d) = max_j proj_(g_j)(d)", None, 10.5), ("앞 자리마다 갱신 · k=1이면 0", "updated per placed anchor; 0 at k=1", 10)])
vbox(620, 452, 120, 54, "learned", ("렌치 MLP", "Wrench MLP"), [("[h_k ; proj_p]", None, 10.5), ("56→128→128", None, 10.5)])
arrow([(200, 472), (240, 472)], color=V)
arrow([(376, 472), (416, 472)], ("2048×28", None), 396, 466, 9, color=V)
arrow([(308, 506), (308, 522), (680, 522), (680, 506)], ("proj_p: 2048×28", None), 494, 517, 9.5, color=V)   # 후보점 방향값도 렌치 MLP 로
arrow([(584, 479), (620, 479)], ("28", None), 602, 473, 9.5, color=V)
arrow([(740, 479), (841, 479), (841, 280)], ("S_k: 2048×128", None), 790, 473, 9.5, color=V)
arrow([(60, 406), (60, 452)], color=V)                                        # 놓인 앵커 g_j → 아래 경로 (h_k 갱신에 쓰임)
t(66, 433, "g_j → h_k", None, 10, 400, V, "start")
# 화살표
arrow([(200, 70), (240, 70)])
arrow([(200, 118), (240, 118)])
arrow([(376, 94), (430, 94)], ("2048×256", None), 403, 88, 10)
arrow([(200, 166), (430, 166)])
arrow([(570, 110), (762, 110), (762, 218), (786, 218)], ("E: 2048 × 128", None), 666, 104)
arrow([(600, 110), (600, 292), (308, 292), (308, 308)])
arrow([(200, 220), (240, 220)])
arrow([(200, 268), (240, 268)])
arrow([(376, 240), (430, 240)], ("4×128", None), 403, 234, 10)
arrow([(570, 240), (786, 240)], ("T: 4 × 128", None), 690, 234)
arrow([(200, 328), (240, 328)], color=G)
arrow([(220, 328), (220, 380), (240, 380)], color=G)
arrow([(200, 392), (240, 392)], color=G)
arrow([(376, 328), (430, 328)], ("128", None), 403, 322, 10)
arrow([(376, 382), (430, 382)], ("2048×5", None), 403, 398, 10)
arrow([(570, 358), (620, 358)], ("2048×128", None), 595, 352, 9.5)
arrow([(740, 358), (770, 358), (770, 262), (786, 262)], color=G)
arrow([(896, 228), (936, 228)])
arrow([(1016, 256), (1016, 290)])
arrow([(1016, 372), (1016, 356)])

# ───────── (b) 차례로 고르기 ─────────
o = 554
panel(o, 206)
t(24, o + 24, "(b) 그룹 순서대로 한 점씩 고르기 (팔 3대 예)", "(b) Pick one anchor per group in a fixed order (3-arm example)", 14, 700, "#111827", "start")
t(1096, o + 24, "새 방식 · 원래는 모든 그룹의 점수를 서로 모른 채 한 번에 냄",
  "new · the original scored all groups at once, independently", 11.5, 600, G, "end")
steps = [(24, "단계 1 · 엄지 그룹", "Step 1 · thumb group", "조건 없음 (C_1 = 0)", "no condition (C_1 = 0)", "P_1(p) → 앵커 g_1", "P_1(p) → anchor g_1"),
         (410, "단계 2 · 둘째 그룹", "Step 2 · second group", "조건: g_1", "condition: g_1", "P_2(p | g_1) → 앵커 g_2", "P_2(p | g_1) → anchor g_2"),
         (796, "단계 3 · 셋째 그룹", "Step 3 · third group", "조건: g_1, g_2", "condition: g_1, g_2", "P_3(p | g_1, g_2) → 앵커 g_3", "P_3(p | g_1, g_2) → anchor g_3")]
for i, (x, a, ae, b, be, c, ce) in enumerate(steps):
    lines = [(b, be, 12, "#374151"), (c, ce, 12, "#374151")]
    if i == 0:
        lines.append(("(a)에서 C = 0 → 원래 net2와 같은 식", "(a) with C = 0 → same as original", 10.5))
    box(x, o + 42, 300, 84, "learned", (a, ae), lines, new=(i > 0))
arrow([(324, o + 84), (410, o + 84)], ("g_1", None), 367, o + 78, 11.5)
arrow([(710, o + 84), (796, o + 84)], ("g_2", None), 753, o + 78, 11.5)
arrow([(174, o + 126), (174, o + 148), (946, o + 148), (946, o + 126)], ("g_1", None), 560, o + 142, 11.5)
t(560, o + 174, "조합 확률 = P_1(g_1) × P_2(g_2 | g_1) × P_3(g_3 | g_1, g_2)",
  "Joint probability = P_1(g_1) · P_2(g_2 | g_1) · P_3(g_3 | g_1, g_2)", 12.5, 600, "#111827")
t(560, o + 194, "그룹 순서: 엄지 든 그룹 먼저, 나머지는 검지 · 중지 · 약지 · 소지 · 손바닥 순서에서 가장 앞 손가락 기준 (학습 · 추론 공통)",
  "Group order: thumb group first, then by the earliest of index · middle · ring · little · palm (training and inference)", 11, 400, "#4B5563")

# ───────── (c) 실제로 쓸 때 ─────────
o = 772
panel(o, 190)
t(24, o + 24, "(c) 실제로 쓸 때 — 빔 탐색 · α 채점 · 다듬기 · 시뮬레이션 점검", "(c) Inference — beam search · α scoring · refinement · simulation check", 14, 700, "#111827", "start")
t(1096, o + 24, "바뀜 · 원래는 그룹마다 상위 8점 → 8×8×8 = 512조합 전부 채점",
  "changed · the original took top-8 per group → scored all 512 combos", 11.5, 600, G, "end")
y0 = o + 42
box(24, y0, 150, 92, "reuse", ("파티션 상위 3개", "Top-3 partitions"), [("net1 ← 그림 1 (b)", "net1 ← Fig. 1 (b)", 11.5, "#374151")])
box(200, y0, 170, 92, "fixed", ("빔 탐색", "Beam search"), [("첫 그룹 상위 8개", "group 1: top 8", 11), ("둘째 8개씩 → 64개", "group 2: 8 each → 64", 11), ("조합 확률 상위 8개 유지", "keep top 8 by joint prob.", 11)], new=True)
box(396, y0, 176, 92, "fixed", ("마지막 그룹", "Last group"), [("네트워크 상위 8개씩 (64조합)", "network top 8 each (64)", 11), ("또는 후보 범위 전체", "or every point in the mask", 11), ("(조합 약 2천~1만)", "(≈2k–10k combos)", 11)], new=True)
box(598, y0, 140, 92, "fixed", ("α 채점", "α scoring"), [("조합마다 선형계획", "LP per combination", 11)])
box(764, y0, 156, 92, "fixed", ("다듬기", "Refinement"), [("파티션당 상위 2개", "top 2 per partition", 11), ("앵커마다 가까운 7점", "7 nearest points per anchor", 11), ("(343조합)", "(343 combos)", 11)])
box(946, y0, 150, 92, "output", ("후보 6개 · α 순서", "6 candidates, α order"), [("시뮬레이션 점검", "simulation check", 11), ("처음 통과한 후보", "first one to pass", 11), ("→ 로봇 실행", "→ execute on robot", 11)])
for a, b in ((174, 200), (370, 396), (572, 598), (738, 764), (920, 946)):
    arrow([(a, y0 + 46), (b, y0 + 46)])
out.append(f'<polyline points="200,{y0+100} 200,{y0+108} 920,{y0+108} 920,{y0+100}" fill="none" stroke="#6B7280" stroke-width="1.1"/>')
t(560, y0 + 126, "파티션 3개마다 따로", "repeated for each of the 3 partitions", 11.5)

# ───────── 범례 ─────────
ly = 976
items = [("input", "외부 입력", "input", False), ("reuse", "다른 블록 출력", "from another block", False),
         ("learned", "학습되는 층", "learned layer", False), ("fixed", "고정 연산 (학습 안 함)", "fixed op (no parameters)", False),
         ("output", "최종 출력", "output", False), ("learned", "새로 추가 (초록 테두리)", "added (green border)", True),
         ("learned", "실험 경로 (주황, 채택 안 함)", "experimental (orange, not adopted)", "orange")]
for i, (kind, ko, en, new) in enumerate(items):
    x = 24 + i * 152
    fill, stroke, dash = K[kind]
    if new == "orange":
        stroke = "#B45309"
    elif new:
        stroke = G
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{ly}" width="22" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if new else 1.3}"{d}/>')
    t(x + 30, ly + 11.5, ko, en, 11.5, 400, "#374151", "start")
t(24, ly + 36, "⊕ = 이어 붙이기 · 모든 차원은 파지 1개 기준 (배치 차원 생략) · k = 정해진 순서에서 몇 번째 그룹인지 · 초록 글씨 = 바뀐 점",
  "⊕ = concatenation · shapes per grasp (batch omitted) · k = group index in the fixed order · green text = what changed", 11.5, 400, "#374151", "start")

aria = ("조건부 net2 전체 구조도. (a) 패치 노드 출력을 가까운 패치 3개 비율로 섞고 표면점 특징과 함께 점 MLP에 넣어 점 벡터 E를 만들고, "
        "부위 MLP 출력을 파티션대로 묶어 그룹 선형층으로 그룹 벡터 T를 만든다. 새로 추가된 조건 MLP가 이미 놓인 앵커마다 조건 항을 만들어 더하고, "
        "점 점수 MLP가 그룹 k의 확률을 낸다. 주황 경로는 렌치 지지 함수 실험(채택 안 함). (b) 엄지 그룹부터 차례로 한 점씩 고른다. (c) 빔 탐색, α 채점, 다듬기, 시뮬레이션 점검.")
svg = (f'<svg id="fig2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(aria)}" '
       f'font-family="\'Noto Sans KR\',\'Apple SD Gothic Neo\',\'Malgun Gothic\',NanumGothic,sans-serif">'
       '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#4B5563"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n' + "\n".join(out) + "</svg>")
p = "md_claude/plans/net12_figure_2026-10-17.html"
h = open(p, encoding="utf-8").read()
h2, n = re.subn(r'<svg id="fig2".*?</svg>', lambda m: svg, h, flags=re.S)
assert n == 1
open(p, "w", encoding="utf-8").write(h2)
open("scratchpad/fig2_preview.svg", "w", encoding="utf-8").write(svg)
print("교체 완료, 요소", len(out))
