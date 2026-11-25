"""키프레임 추출 타임라인 시각화 — 데모 시퀀스 2개.

행 구성 (시퀀스당 3행): ① 손-물체 최소 거리 (오른손/왼손, 문턱 1.5/2.2cm)
② 손 접촉 상태 띠 + hold 구간 라벨 ③ 뚜껑 각도 + 모션 구간 음영.
키프레임 = 세로 점선 (전 행 관통).
출력: md_claude/keyframes/keyframe_timeline.html
실행: ./venv/bin/python scratchpad/keyframe_viz.py
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "md_claude/keyframes/keyframe_timeline.html"
SEQS = ["s01__box_grab_01", "s01__box_use_02"]
STATE_COLOR = {"none": "#d9d9d9", "right": "#5b9bd5", "left": "#ed7d31", "both": "#70ad47"}
STATE_KO = {"none": "무접촉", "right": "오른손", "left": "왼손", "both": "양손"}

fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.035,
    row_heights=[0.22, 0.10, 0.16, 0.22, 0.10, 0.16],
    subplot_titles=[
        f"{SEQS[0]} — 손-물체 거리", "접촉 상태 / hold 구간", "뚜껑 각도",
        f"{SEQS[1]} — 손-물체 거리", "접촉 상태 / hold 구간", "뚜껑 각도"])

for si, seq in enumerate(SEQS):
    base = si * 3
    with np.load(ROOT / "data/arctic_cache/contact_dist" / f"{seq}.npz") as z:
        dist, theta = z["dist"], z["theta"]
    kf = json.load(open(ROOT / "data/contraction_dyn/keyframes" / f"{seq}.json"))
    T = kf["n_frames"]
    t = np.arange(T)

    # ① 손별 최소 거리
    for cols, name, color in [(slice(0, 5), "오른손", "#5b9bd5"),
                              (slice(5, 10), "왼손", "#ed7d31")]:
        fig.add_trace(go.Scatter(
            x=t, y=dist[:, cols].min(1) * 100, name=name, legendgroup=name,
            showlegend=(si == 0), line=dict(color=color, width=1.5)),
            row=base + 1, col=1)
    for y, dash in [(1.5, "dot"), (2.2, "dash")]:
        fig.add_hline(y=y, line=dict(color="#999", width=1, dash=dash),
                      row=base + 1, col=1)
    fig.update_yaxes(title_text="cm", range=[0, 12], row=base + 1, col=1)

    # ② 상태 띠 (hold 구간 사각형)
    for s in kf["segments"]:
        fig.add_shape(type="rect", x0=s["start"], x1=s["end"], y0=0, y1=1,
                      fillcolor=STATE_COLOR[s["hand_state"]], line_width=0,
                      row=base + 2, col=1)
        if s["n_frames"] > 25:
            fig.add_annotation(x=(s["start"] + s["end"]) / 2, y=0.5,
                               text=STATE_KO[s["hand_state"]], showarrow=False,
                               font=dict(size=9), row=base + 2, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=base + 2, col=1)

    # ③ 뚜껑 각도 + 모션 구간 음영
    fig.add_trace(go.Scatter(x=t, y=theta, name="뚜껑 각도", showlegend=(si == 0),
                             legendgroup="arti", line=dict(color="#7030a0", width=1.5)),
                  row=base + 3, col=1)
    for a, b in kf["arti_segments"]:
        fig.add_shape(type="rect", x0=a, x1=b, y0=0, y1=max(theta.max(), 0.1),
                      fillcolor="rgba(112,48,160,0.12)", line_width=0,
                      row=base + 3, col=1)
    fig.update_yaxes(title_text="rad", row=base + 3, col=1)

    # 키프레임 세로선 (3행 관통)
    for f in kf["keyframes"]:
        for r in (1, 2, 3):
            fig.add_vline(x=f, line=dict(color="#c00000", width=1, dash="dot"),
                          row=base + r, col=1)

    fig.layout.annotations[base].text = (
        f"{seq} — 손-물체 거리 (전체 {T}프레임 = {T/30:.1f}초, "
        f"키프레임 {len(kf['keyframes'])}개 = 빨간 점선)")

fig.update_layout(height=950, width=1250, template="plotly_white",
                  title="키프레임 추출 타임라인 — 데모 시퀀스 2개 (2026-08-08)",
                  legend=dict(orientation="h", y=1.04))
fig.update_xaxes(title_text="프레임 (30fps)", row=6, col=1)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"저장: {OUT}")
