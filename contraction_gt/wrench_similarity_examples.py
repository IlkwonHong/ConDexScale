"""렌치 공간 닮음 — 파지별 예시 HTML (계획서 §5-5, 질문 4 결정: 예시는 돌려 볼 수 있는 HTML).

한 파지당 한 화면:
  왼쪽  3차원 장면 — 물체 표면점(회색), 사람 손 마디 접촉(A, 파랑), 계획 앵커(B, 주황), 실제 접촉(C, 청록)
  가운데 힘 방향 구 위의 그림자 덩어리 (반지름 = 그 방향의 그림자 길이), A·B·C 겹침
  오른쪽 토크 방향 구 위의 그림자 덩어리
  아래  방향 512개를 사람 값 순으로 늘어놓고 비율 B/A, C/A 띠
표본은 realized_contacts.pkl (C 가 있는 기록) 에서 합격 10·불합격 10·정착 실패 5.

실행: ./venv/bin/python -m contraction_gt.wrench_similarity_examples [--n-pass 10 --n-fail 10 --n-settle 5]
출력: md_claude/gt_viz/wrench_similarity/examples/{index.html, ex_<idx>_<verdict>.html}
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .wrench_similarity_eval import (ROOT, OUT, CACHE_ROOT, CONTACT_ROOT, MU, N_EDGES, N_DIRS, SEED,
                                     manifest_map)
from .wrench_support import sample_directions, cone_edge_wrenches, support_max, similarity
from .wrench_similarity_report import C_SPACE, K_VERDICT

EX_OUT = Path("md_claude/gt_viz/wrench_similarity/examples")
NAME = {"A": "사람 손 (마디 접촉)", "B": "계획 앵커", "C": "실제 접촉 (시뮬)"}


def sphere_grid(n_lat=36, n_lon=72):
    th = np.linspace(0, np.pi, n_lat); ph = np.linspace(0, 2 * np.pi, n_lon)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)   # [lat, lon, 3]
    return d


def sig_of(pos, nrm, com, r_obj, dirs6):
    if len(pos) == 0:
        return np.zeros(len(dirs6))
    W = cone_edge_wrenches(torch.as_tensor(np.asarray(pos), dtype=torch.float64)[None],
                           torch.as_tensor(np.asarray(nrm), dtype=torch.float64)[None],
                           torch.as_tensor(com, dtype=torch.float64)[None], float(r_obj), mu=MU, n_edges=N_EDGES)
    return support_max(W, torch.as_tensor(dirs6, dtype=torch.float64))[0].numpy()


def blob(ax_dirs, sig, color, name, opacity=0.45):
    """구 격자 방향 [lat,lon,3] 과 값 [lat*lon] → 반지름 = 값인 표면."""
    r = sig.reshape(ax_dirs.shape[:2])
    xyz = ax_dirs * r[..., None]
    return go.Surface(x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2], surfacecolor=np.zeros_like(r),
                      colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity, name=name,
                      showlegend=True, hovertemplate=f"{name}<br>그림자 길이 %{{customdata:.2f}}<extra></extra>",
                      customdata=r)


def make_example(r, rec_row, gr, mmap, dirs512, grid, pts_cache):
    gidx = int(rec_row["grasp_idx"]); gid = str(gr["grasp_id"][gidx])
    com = gr["com"][gidx].astype(np.float64); r_obj = float(gr["r_obj"][gidx]); scale = float(gr["total_scale"][gidx])
    rel, row = gid.split("#"); row = int(row)
    ds, oid = mmap[rel]
    key = (ds, oid)
    if key not in pts_cache:
        pts_cache[key] = np.load(CACHE_ROOT / ds / f"{oid}.npz")["points"]
    P = pts_cache[key] * scale
    z = np.load(CONTACT_ROOT / rel.replace(".npy", ".npz"))
    act = z["link_contact"][row]
    hp, hn = z["link_contact_point"][row][act], -z["link_contact_normal"][row][act]
    bp, bn = r["planned_pos"], r["planned_nrm"]
    cp = np.array([c["pos_obj"] for c in r["contacts"]]).reshape(-1, 3)
    cn = np.array([c["normal_obj"] for c in r["contacts"]]).reshape(-1, 3)
    spaces = {"A": (hp, hn), "B": (bp, bn), "C": (cp, cn)}

    # 그림자: 512 방향(비율 띠), 힘 구, 토크 구
    gd = grid.reshape(-1, 3)
    d_force = np.concatenate([gd, np.zeros_like(gd)], -1)
    d_torque = np.concatenate([np.zeros_like(gd), gd], -1)
    s512 = {k: sig_of(p, n, com, r_obj, dirs512) for k, (p, n) in spaces.items()}
    sF = {k: sig_of(p, n, com, r_obj, d_force) for k, (p, n) in spaces.items()}
    sT = {k: sig_of(p, n, com, r_obj, d_torque) for k, (p, n) in spaces.items()}
    sm_AB = similarity(torch.as_tensor(s512["A"])[None], torch.as_tensor(s512["B"])[None], beta=0.1)
    sm_BC = similarity(torch.as_tensor(s512["B"])[None], torch.as_tensor(s512["C"])[None], beta=0.1)

    fig = make_subplots(rows=2, cols=3, specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                                               [{"type": "xy", "colspan": 3}, None, None]],
                        row_heights=[0.72, 0.28], horizontal_spacing=0.02, vertical_spacing=0.08,
                        subplot_titles=("3차원 장면 (물체 좌표, m)", "힘 방향 구: 반지름 = 그림자 길이",
                                        "토크 방향 구: 반지름 = 그림자 길이",
                                        "방향 512개 (사람 값이 큰 순) — 비율 로봇 ÷ 사람"))
    # 장면
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                               marker=dict(size=1.6, color="#b8b7b2"), name="물체 표면점", hoverinfo="skip"), 1, 1)
    L = 0.12 * r_obj
    for k, (p, n) in spaces.items():
        if len(p) == 0:
            continue
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                   marker=dict(size=6 if k != "A" else 4.5, color=C_SPACE[k],
                                               symbol="circle" if k != "C" else "diamond"),
                                   name=NAME[k], legendgroup=k), 1, 1)
        segs = np.stack([p, p + L * n, np.full_like(p, np.nan)], 1).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=segs[:, 0], y=segs[:, 1], z=segs[:, 2], mode="lines",
                                   line=dict(color=C_SPACE[k], width=4), showlegend=False, hoverinfo="skip",
                                   legendgroup=k), 1, 1)
    fig.add_trace(go.Scatter3d(x=[com[0]], y=[com[1]], z=[com[2]], mode="markers",
                               marker=dict(size=5, color="#0b0b0b", symbol="x"), name="질량중심"), 1, 1)
    # 구 두 개
    for k in ("A", "B", "C"):
        fig.add_trace(blob(grid, sF[k], C_SPACE[k], NAME[k] + " — 힘"), 1, 2)
        fig.add_trace(blob(grid, sT[k], C_SPACE[k], NAME[k] + " — 토크"), 1, 3)
    # 비율 띠
    order = np.argsort(-s512["A"])
    a = s512["A"][order]
    for k in ("B", "C"):
        ratio = np.where(a > 1e-3, s512[k][order] / np.maximum(a, 1e-3), np.nan)
        fig.add_trace(go.Scatter(x=np.arange(N_DIRS), y=ratio, mode="lines", line=dict(color=C_SPACE[k], width=1.5),
                                 name=f"비율 {k}/A", hovertemplate="방향 %{x}<br>비율 %{y:.2f}<extra></extra>"), 2, 1)
    # (plotly 의 add_hline 은 3차원 trace 가 있는 그림에서 오류 → 선을 직접 그린다)
    for yv, dash, label in ((1.0, "solid", "사람과 같음 (비율 1)"), (0.9, "dot", "허용 오차 β=0.1 (비율 0.9)")):
        fig.add_trace(go.Scatter(x=[0, N_DIRS - 1], y=[yv, yv], mode="lines",
                                 line=dict(color="#52514e", width=1, dash=dash), name=label,
                                 hoverinfo="skip"), 2, 1)
    fig.update_yaxes(title_text="로봇 ÷ 사람", range=[0, 2.5], row=2, col=1)
    fig.update_xaxes(title_text="방향 번호 (사람 그림자 길이 내림차순)", row=2, col=1)

    rng = float(max(np.abs(P - com).max(), 1e-3)) * 1.1
    fig.update_scenes(aspectmode="cube", row=1, col=1)
    fig.update_scenes(xaxis=dict(range=[com[0]-rng, com[0]+rng]), yaxis=dict(range=[com[1]-rng, com[1]+rng]),
                      zaxis=dict(range=[com[2]-rng, com[2]+rng]), row=1, col=1)
    smax = max(max(sF[k].max() for k in sF), max(sT[k].max() for k in sT), 1e-3) * 1.05
    for c in (2, 3):
        fig.update_scenes(aspectmode="cube", xaxis=dict(range=[-smax, smax], title="x"),
                          yaxis=dict(range=[-smax, smax], title="y"), zaxis=dict(range=[-smax, smax], title="z"), row=1, col=c)
    title = (f"기록 {int(r['idx'])} · {K_VERDICT[r['verdict']]} · 팔 {len(bp)}대 · α={float(rec_row['min_alpha']):.3f} · "
             f"사람 대비 잃음(A→B) {float(sm_AB['lost_frac'][0]):.2f} · 계획 대비 실제 잃음(B→C) {float(sm_BC['lost_frac'][0]):.2f} · "
             f"계획↔실제 거리/팁반지름 최대 {np.nanmax(r['tip_dist']/r['tip_radius']) if np.isfinite(r['tip_dist']).any() else float('nan'):.2f} · "
             f"접촉 없는 팁 {int((r['tip_ncontact']==0).sum())}")
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=14)), height=900, width=1500,
                      paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=dict(color="#0b0b0b", size=12),
                      legend=dict(orientation="h", y=-0.12, x=0.0, font=dict(size=11)), margin=dict(l=10, r=10, t=60, b=90))
    return fig, dict(idx=int(r["idx"]), verdict=r["verdict"], alpha=float(rec_row["min_alpha"]),
                     lost_AB=float(sm_AB["lost_frac"][0]), lost_BC=float(sm_BC["lost_frac"][0]), n_arms=len(bp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pass", type=int, default=10)
    ap.add_argument("--n-fail", type=int, default=10)
    ap.add_argument("--n-settle", type=int, default=5)
    args = ap.parse_args()
    EX_OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(np.load(OUT / "records.npz", allow_pickle=True))
    gr = dict(np.load(OUT / "grasps.npz", allow_pickle=True))
    res = pickle.load(open(OUT / "realized_contacts.pkl", "rb"))
    idx_to_row = {int(i): k for k, i in enumerate(rec["idx"])}
    rng = np.random.default_rng(0)
    picks = []
    for name, cnt in (("pass", args.n_pass), ("fail", args.n_fail), ("fail_settle", args.n_settle)):
        pool = [r for r in res if r["verdict"] == name]
        picks += [pool[i] for i in rng.choice(len(pool), min(cnt, len(pool)), replace=False)]
    mmap = manifest_map()
    dirs512 = sample_directions(N_DIRS, seed=SEED).numpy()
    grid = sphere_grid()
    pts_cache = {}
    rows = []
    for r in picks:
        k = idx_to_row[int(r["idx"])]
        rec_row = {name: rec[name][k] for name in ("grasp_idx", "min_alpha", "n_anchors")}
        fig, info = make_example(r, rec_row, gr, mmap, dirs512, grid, pts_cache)
        fname = f"ex_{info['idx']}_{info['verdict']}.html"
        fig.write_html(EX_OUT / fname, include_plotlyjs="cdn", full_html=True)
        info["file"] = fname
        rows.append(info)
        print("저장", fname, flush=True)
    rows.sort(key=lambda d: (d["verdict"], d["idx"]))
    html = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>렌치 공간 닮음 — 예시 목록</title>",
            "<style>body{font-family:NanumGothic,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:24px}"
            "table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #e6e5e1;text-align:left}"
            "a{color:#2a78d6}</style></head><body>",
            "<h2>렌치 공간 닮음 — 파지별 예시</h2>",
            "<p>A = 사람 손 마디 접촉, B = 계획 앵커, C = 시뮬에서 실제로 닿은 접촉. 잃음 = 앞 공간이 밀 수 있는 방향 중 뒤 공간이 90% 미만으로 미는 방향의 비율 (β=0.1).</p>",
            "<table><tr><th>판정</th><th>기록</th><th>팔</th><th>α</th><th>잃음 A→B</th><th>잃음 B→C</th><th>열기</th></tr>"]
    for d in rows:
        html.append(f"<tr><td>{K_VERDICT[d['verdict']]}</td><td>{d['idx']}</td><td>{d['n_arms']}</td><td>{d['alpha']:.3f}</td>"
                    f"<td>{d['lost_AB']:.2f}</td><td>{d['lost_BC']:.2f}</td><td><a href='{d['file']}'>{d['file']}</a></td></tr>")
    html.append("</table></body></html>")
    (EX_OUT / "index.html").write_text("\n".join(html))
    print("목록", EX_OUT / "index.html")


if __name__ == "__main__":
    main()
