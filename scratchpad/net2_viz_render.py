"""net2 사례 카드(뷰 A) + 집계 곡선(뷰 C) 렌더.

입력: scratchpad/net2_viz_cases.pkl (net2_viz_mine.py 출력)
출력: md_claude/plans/stage2_net2_cases.html

색 규약 (dataviz 검증 통과: #2a78d6/#eb6834/#1baf7a/#4a3aa7 all-pairs PASS):
  히트맵 = 파랑 순차 램프(#cde2fb→#0d366b, 마스크 안만) · 마스크 밖 = 회색
  GT 앵커 = 주황 다이아 · top-8 = 아쿠아 원(테두리) · 최종 선택 = 보라 사각
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "md_claude/plans/stage2_net2_cases.html"

SEQ = [(0.0, "#cde2fb"), (0.25, "#86b6ef"), (0.5, "#3987e5"),
       (0.75, "#1c5cab"), (1.0, "#0d366b")]
C_GT, C_TOP8, C_SEL = "#eb6834", "#1baf7a", "#4a3aa7"
C_MUT, C_GRID, INK, INK2 = "#e1e0d9", "#e1e0d9", "#0b0b0b", "#52514e"
SURF = "#fcfcfb"
TYPE_META = {
    1: ("유형 1 — 성공 · GT 근처", "#0ca30c",
        "최종 앵커 전부가 GT 분지(0.15r) 안. 교과서 경로."),
    2: ("유형 2 — 성공 · GT와 다른 자리", "#0ca30c",
        "GT 분지 밖 앵커가 있는데 시뮬 합격 — 다중 모달(동급 해 여럿)의 눈 증거."),
    3: ("유형 3 — 실패 · 분지 미달", "#d03b3b",
        "선택 파티션에서 top-8이 GT 분지에 못 닿음 — net2 순위 실패 (일부는 다른 동급 분지일 수 있음)."),
    4: ("유형 4 — 실패 · 선별 오류", "#d03b3b",
        "top-3에 합격 파티션이 있고 top-8도 분지에 닿았는데 최종 선택이 실패 — LP 과대평가(1발↔상한 격차의 실체)."),
}


def camera_for(pts):
    """물체 주성분축 기반 카메라: 최소 분산축(면 법선) 쪽에서 내려다보기.

    납작한 물체를 기본 카메라(대각선)로 보면 마스크 영역이 프레임 밖으로
    잘리는 문제가 있어, 면을 정면으로 보는 시점을 케이스별로 계산한다.
    """
    c = pts.mean(0)
    _ev, evec = np.linalg.eigh(np.cov((pts - c).T))
    d = evec[:, 0] + 0.35 * evec[:, 1]        # 법선 + 약간 기울임
    d /= np.linalg.norm(d)
    if d[2] < 0:
        d = -d
    return dict(eye=dict(x=float(d[0] * 2.6), y=float(d[1] * 2.6),
                         z=float(d[2] * 2.6)))


def case_fig(m):
    p = m["parts"][m["sel"]]
    n_g, r = p["n_g"], m["r_obj"]
    pts = m["pts"]
    fig = make_subplots(rows=1, cols=n_g, specs=[[{"type": "scene"}] * n_g],
                        subplot_titles=[f"그룹 {g} · 담당 부위: " +
                                        "·".join(u for u in
                                                 ["index", "middle", "ring",
                                                  "little", "thumb", "palm"]
                                                 if p["ug"][["index", "middle",
                                                             "ring", "little",
                                                             "thumb", "palm"
                                                             ].index(u)] == g)
                                        for g in range(n_g)])
    for g in range(n_g):
        mask = p["masks"][g]
        heat = p["heat"][g].astype(np.float32)
        hin = heat[mask]
        prob = np.exp(hin - hin.max())
        prob /= prob.sum()
        rank = np.empty(len(hin), np.int32)
        rank[np.argsort(hin)[::-1]] = np.arange(1, len(hin) + 1)
        lo, hi = float(hin.min()), float(hin.max())
        cval = (hin - lo) / max(hi - lo, 1e-9)
        out_i = np.nonzero(~mask)[0]
        if len(out_i) > 1200:
            out_i = out_i[:: len(out_i) // 1200 + 1]
        sc = dict(row=1, col=g + 1)
        fig.add_trace(go.Scatter3d(
            x=pts[out_i, 0], y=pts[out_i, 1], z=pts[out_i, 2],
            mode="markers", marker=dict(size=2, color=C_MUT),
            hoverinfo="skip", showlegend=False), **sc)
        mi = np.nonzero(mask)[0]
        gt_d = np.linalg.norm(pts[mi] - p["gt_anchor"][g], axis=1) / r
        fig.add_trace(go.Scatter3d(
            x=pts[mi, 0], y=pts[mi, 1], z=pts[mi, 2], mode="markers",
            marker=dict(size=3, color=cval, colorscale=SEQ, cmin=0, cmax=1),
            customdata=np.stack([prob * 100, rank, gt_d], 1),
            hovertemplate=("확률 %{customdata[0]:.2f}%% · 순위 %{customdata[1]:.0f}"
                           "위 · GT거리 %{customdata[2]:.3f}r<extra>후보</extra>"),
            name="히트맵(마스크 안)", showlegend=(g == 0),
            legendgroup="hm"), **sc)
        t8 = p["top8"][g]
        d8 = np.linalg.norm(pts[t8] - p["gt_anchor"][g], axis=1) / r
        fig.add_trace(go.Scatter3d(
            x=pts[t8, 0], y=pts[t8, 1], z=pts[t8, 2], mode="markers",
            marker=dict(size=6, color=C_TOP8, symbol="circle-open",
                        line=dict(color="#0b0b0b", width=2)),
            customdata=d8[:, None],
            hovertemplate="top-8 · GT거리 %{customdata[0]:.3f}r<extra></extra>",
            name="top-8 후보", showlegend=(g == 0), legendgroup="t8"), **sc)
        a = p["gt_anchor"][g]
        fig.add_trace(go.Scatter3d(
            x=[a[0]], y=[a[1]], z=[a[2]], mode="markers",
            marker=dict(size=8, color=C_GT, symbol="diamond",
                        line=dict(color="#ffffff", width=2)),
            hovertemplate="GT 앵커<extra></extra>",
            name="GT 앵커", showlegend=(g == 0), legendgroup="gt"), **sc)
        s = pts[p["combo"][g]]
        fig.add_trace(go.Scatter3d(
            x=[s[0]], y=[s[1]], z=[s[2]], mode="markers",
            marker=dict(size=7, color=C_SEL, symbol="square",
                        line=dict(color="#ffffff", width=2)),
            customdata=[[np.linalg.norm(s - a) / r]],
            hovertemplate="최종 선택 · GT거리 %{customdata[0]:.3f}r<extra></extra>",
            name="최종 선택(다듬기 후)", showlegend=(g == 0),
            legendgroup="sel"), **sc)
    ax = dict(visible=False)
    cam = camera_for(pts)
    for g in range(n_g):
        fig.update_scenes(dict(xaxis=ax, yaxis=ax, zaxis=ax,
                               aspectmode="data", bgcolor=SURF, camera=cam),
                          row=1, col=g + 1)
    fig.update_layout(
        height=380, margin=dict(l=4, r=4, t=42, b=4), paper_bgcolor=SURF,
        font=dict(family="'Noto Sans KR', sans-serif", color=INK, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, x=0.0,
                    font=dict(size=11)))
    return fig


def aggregates(mined):
    """뷰 C: top-8→GT 거리 CDF + GT 정확점 순위 히스토그램 (선택 파티션 기준)."""
    dists, ranks = [], []
    for m in mined:
        p = m["parts"][m["sel"]]
        for g in range(p["n_g"]):
            t8 = p["top8"][g]
            d = np.linalg.norm(m["pts"][t8] - p["gt_anchor"][g], axis=1).min()
            dists.append(d / m["r_obj"])
            mask = p["masks"][g]
            heat = np.where(mask, p["heat"][g].astype(np.float32), -np.inf)
            gt_i = int(np.linalg.norm(m["pts"] - p["gt_anchor"][g], axis=1).argmin())
            order = np.argsort(heat)[::-1]
            ranks.append(int(np.nonzero(order == gt_i)[0][0]) + 1)
    dists, ranks = np.sort(dists), np.array(ranks)

    cdf = go.Figure()
    y = np.arange(1, len(dists) + 1) / len(dists) * 100
    cdf.add_trace(go.Scatter(x=dists, y=y, mode="lines",
                             line=dict(color="#2a78d6", width=2),
                             hovertemplate="거리 %{x:.3f}r 이내 %{y:.1f}%<extra></extra>",
                             name="top-8 최근접→GT 거리"))
    for thr, lab in [(0.078, "한 점 간격 0.078r"), (0.15, "분지 0.15r")]:
        frac = (dists <= thr).mean() * 100
        cdf.add_vline(x=thr, line=dict(color="#898781", width=1, dash="dot"))
        cdf.add_annotation(x=thr, y=frac, text=f"{lab}<br>{frac:.0f}%",
                           showarrow=True, arrowhead=0, ax=42, ay=18,
                           font=dict(size=11, color=INK2))
    cdf.update_layout(
        height=330, paper_bgcolor=SURF, plot_bgcolor=SURF,
        font=dict(family="'Noto Sans KR', sans-serif", color=INK, size=12),
        margin=dict(l=52, r=16, t=8, b=42), showlegend=False,
        xaxis=dict(title="top-8 중 최근접 후보 → GT 앵커 거리 (r_obj 배)",
                   gridcolor=C_GRID, zeroline=False, range=[0, 0.5]),
        yaxis=dict(title="누적 비율 (%)", gridcolor=C_GRID, zeroline=False,
                   range=[0, 100]))

    hist = go.Figure()
    bins = list(range(1, 41))
    cnt = np.array([(ranks == b).sum() for b in bins])
    cnt_over = (ranks > 40).sum()
    hist.add_trace(go.Bar(x=bins + ["40+"], y=list(cnt) + [cnt_over],
                          marker=dict(color=["#2a78d6" if b <= 8 else "#9ec5f4"
                                             for b in bins] + ["#9ec5f4"],
                                      line=dict(color=SURF, width=1)),
                          hovertemplate="순위 %{x} · 그룹 %{y}개<extra></extra>"))
    hist.add_vline(x=8.5, line=dict(color="#898781", width=1, dash="dot"))
    hist.add_annotation(x=8.5, y=float(cnt.max()), text="top-8 경계",
                        showarrow=False, xshift=34, font=dict(size=11, color=INK2))
    hist.update_layout(
        height=330, paper_bgcolor=SURF, plot_bgcolor=SURF, showlegend=False,
        font=dict(family="'Noto Sans KR', sans-serif", color=INK, size=12),
        margin=dict(l=52, r=16, t=8, b=42),
        xaxis=dict(title="GT 정확점의 히트맵 순위 (마스크 안)", gridcolor=C_GRID),
        yaxis=dict(title="그룹 수", gridcolor=C_GRID, zeroline=False))
    return cdf, hist, len(dists)


def main():
    with open(ROOT / "scratchpad/net2_viz_cases.pkl", "rb") as f:
        mined = pickle.load(f)
    from collections import Counter
    print("유형 분포:", Counter(m["case_type"] for m in mined))

    picks = {}
    for t in (1, 2, 3, 4):
        cand = [m for m in mined if m["case_type"] == t]
        seen, sel = set(), []
        for m in cand:                       # 물체 다양성 우선
            if m["ds_oid"] not in seen:
                sel.append(m)
                seen.add(m["ds_oid"])
            if len(sel) == 3:
                break
        picks[t] = sel or cand[:3]

    # 3D 사례는 지연 마운트 (WebGL 컨텍스트 상한 ~16개 — 36장면을 다 띄우면
    # 브라우저가 오래된 컨텍스트를 강제 해제해 앞 카드가 사라짐)
    parts_html = []
    fig_k = 0
    for t in (1, 2, 3, 4):
        title, color, desc = TYPE_META[t]
        n_all = sum(1 for m in mined if m["case_type"] == t)
        parts_html.append(
            f'<h2><span class="badge" style="background:{color}">'
            f'{"성공" if t in (1, 2) else "실패"}</span> {title}'
            f' <span class="cnt">— 채굴 {len(mined)}건 중 {n_all}건</span></h2>'
            f'<p class="desc">{desc}</p>')
        for m in picks[t]:
            p = m["parts"][m["sel"]]
            d = np.linalg.norm(m["pts"][p["combo"]] - p["gt_anchor"],
                               axis=1) / m["r_obj"]
            cap = (f'<b>{m["ds_oid"][0]}/{m["ds_oid"][1]}</b> · {m["gid"]} · '
                   f'시뮬 <b>{m["verdict"]}</b> · 달성 α {p["min_alpha"]:.3f} / '
                   f'GT최선 {m["gt_best_pass"]:.3f} '
                   f'(비 {p["min_alpha"]/m["gt_best_pass"]:.2f}) · '
                   f'앵커→GT 거리 [{", ".join(f"{x:.2f}r" for x in d)}]')
            parts_html.append(f'<p class="cap">{cap}</p>')
            spec = case_fig(m).to_json().replace("</", "<\\/")
            parts_html.append(
                f'<script type="application/json" id="figspec-{fig_k}">{spec}'
                f'</script><div class="lazy-plot" data-spec="figspec-{fig_k}" '
                f'style="height:380px"></div>')
            fig_k += 1

    cdf, hist, n_grp = aggregates(mined)
    agg_html = (
        f'<h2>집계 — 점 3개짜리 사다리를 곡선 전체로 (선택 파티션 기준, 그룹 {n_grp}개)</h2>'
        '<p class="desc">왼쪽: top-8 중 최근접 후보가 GT 앵커에서 얼마나 떨어졌는지의 누적분포 — '
        '성적표의 이웃/분지 지표는 이 곡선 위의 두 점이다. 오른쪽: GT 정확점이 히트맵에서 몇 위였는지 '
        '— 진한 막대(8위 이내)가 top-8 적중분, 9~20위 꼬리가 두꺼우면 top-K 확대 카드의 근거가 된다.</p>'
        '<div class="row"><div class="half">'
        + pio.to_html(cdf, include_plotlyjs=False, full_html=False,
                      config=dict(displayModeBar=False))
        + '</div><div class="half">'
        + pio.to_html(hist, include_plotlyjs=False, full_html=False,
                      config=dict(displayModeBar=False))
        + '</div></div>')

    from plotly.offline import get_plotlyjs
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>net2 출력 시각화 — 사례 카드 + 집계</title>
<script>{get_plotlyjs()}</script>
<style>
 body {{ font-family:'Noto Sans KR',sans-serif; background:#f7f8fa; color:#1c2733;
        margin:0; padding:32px; }}
 .col {{ max-width:1200px; margin:0 auto; }}
 h1 {{ font-size:21px; margin:0 0 4px; }}
 h2 {{ font-size:15px; margin:30px 0 4px; border-top:2px solid #dfe5ec; padding-top:16px; }}
 .sub,.desc {{ color:#52514e; font-size:12.5px; margin:4px 0 10px; }}
 .cap {{ font-size:12px; color:#52514e; margin:14px 0 2px; }}
 .badge {{ color:#fff; font-size:11px; border-radius:20px; padding:2px 10px;
          vertical-align:2px; }}
 .cnt {{ color:#898781; font-weight:400; font-size:12px; }}
 .row {{ display:flex; gap:12px; flex-wrap:wrap; }}
 .half {{ flex:1 1 480px; min-width:420px; }}
 .lazy-plot, .half .plotly-graph-div {{ border:1.5px solid #c9d2dd;
                     border-radius:10px; background:#fcfcfb; }}
</style></head><body><div class="col">
<h1>net2 출력 시각화 — 1발 파이프라인 사례 카드 (N=3, 표준화 모델)</h1>
<p class="sub">파지당: net1 top-3 파티션 → 파티션별 히트맵 → 그룹당 top-8 → 8³ 조합 LP →
이웃 다듬기 → 전체 1등(1발) → 짧은 시뮬 판정. 마스크에 GT 정답점 강제 포함 없음(추론 공정성).
색: 파랑 진할수록 히트맵 점수 높음(마스크 안 min-max) · 회색 = 채점 대상 아님 ·
<b style="color:#eb6834">◆ GT 앵커</b> · <b style="color:#1baf7a">○ top-8</b> ·
<b style="color:#4a3aa7">■ 최종 선택</b>. 마우스로 회전·확대, 호버로 수치.
생성: scratchpad/net2_viz_mine.py → net2_viz_render.py · {len(mined)}파지 채굴.</p>
{"".join(parts_html)}
{agg_html}
<script>
const obs = new IntersectionObserver(entries => {{
  for (const e of entries) {{
    const div = e.target;
    if (e.isIntersecting && !div._mounted) {{
      const spec = JSON.parse(document.getElementById(div.dataset.spec).textContent);
      Plotly.newPlot(div, spec.data, spec.layout,
                     {{displayModeBar: false, responsive: true}});
      div._mounted = true;
    }} else if (!e.isIntersecting && div._mounted) {{
      Plotly.purge(div);
      div._mounted = false;
    }}
  }}
}}, {{rootMargin: "500px 0px 500px 0px"}});
document.querySelectorAll(".lazy-plot").forEach(d => obs.observe(d));
</script>
</div></body></html>"""
    OUT.write_text(html)
    print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
