"""정적 파지 용량 연구 시각화 — 구간별: GT 실사진 | 3D(사람 접촉 + 기준선/튜닝 앵커).

static_capacity.py(배치·접촉) + static_hold_test.py(시뮬 판정) 산출을 합쳐,
성공/실패 모두 사람 접촉과 contraction 결과를 나란히 보여준다 (그릴링 확정).

표기: 점 = 사람 접촉 (주황 왼손/파랑 오른손) · 다이아몬드(테두리) = 기준선(α 시간표)
· 다이아몬드(채움) = ρ 튜닝 · 초록 = 본체 / 빨강 = 뚜껑 부품.
실행: ./venv/bin/python scratchpad/static_capacity_viz.py box_grab_01
출력: md_claude/keyframes/static_capacity_viz_{seq}.html
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.hand_contacts import SeqData

IMG = Path("/home/robin/a_workspace/ext_proj/arctic/unpack/arctic_data/data/images/s01")
CROP = (300, 500, 1900, 2500)

seq = sys.argv[1] if len(sys.argv) > 1 else "box_grab_01"
cap = json.load(open(ROOT / f"data/contraction_dyn/static_capacity/s01__{seq}.json"))
hold = json.load(open(ROOT / f"data/contraction_dyn/static_hold/s01__{seq}.json"))
hres = {(r["seg"], r["placement"]): r for r in hold["rows"]}
sd = SeqData("s01", seq)
top = sd.parts == 0
Fb = sd.tfaces[~top[sd.tfaces].any(1)]
Ft = sd.tfaces[top[sd.tfaces].all(1)]


def verdict(si, name):
    r = hres.get((si, name))
    if r is None:
        return "—"
    v = "<b style='color:#1a7a1a'>유지 통과</b>" if r["hold_pass"] \
        else "<b style='color:#c00000'>유지 실패</b>"
    s = (f"ρ={r['rho_worst']:+.3f} · {v} "
         f"(드리프트 {r['drift_pos_cm']:.2f}cm/{r['drift_rot_deg']:.1f}°)")
    if r.get("weak_meas_N") is not None:
        s += f" · 최약 외란 실측 {r['weak_meas_N']}N (예측 {r['weak_pred_N']}N)"
    elif not r["hold_pass"]:
        s += " · 낙하"
    return s


blocks = []
for row in cap["rows"]:
    if not (row.get("baseline") or row.get("tuned")):
        continue
    si, s, e, th = row["seg"], row["start"], row["end"], row["theta"]
    rep = (s + e) // 2
    im = Image.open(IMG / seq / "1" / f"{rep+1:05d}.jpg").crop(CROP)
    im = im.resize((280, 350))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()

    v = sd.tverts.copy()
    if abs(th) > 1e-8:
        v[top] = v[top] @ Rotation.from_rotvec([0, 0, -th]).as_matrix().T
    cen = v.mean(0)
    tr = []
    for F_, col, nm in [(Fb, "#c9c9c9", "본체"), (Ft, "#e8862e", "뚜껑")]:
        tr.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=F_[:, 0], j=F_[:, 1], k=F_[:, 2],
                            color=col, opacity=0.4, name=nm, showlegend=True))
    for side, col in [("left", "#ed7d31"), ("right", "#2e75b6")]:
        P = np.asarray(row.get("contacts", {}).get(side, []))
        if len(P) == 0:
            continue
        P = P + cen
        tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                               marker=dict(size=2.5, color=col),
                               name=f"{side} 접촉"))
    for name, solid in [("baseline", False), ("tuned", True)]:
        plc = row.get(name)
        if not plc:
            continue
        for arm, d in plc["arms"].items():
            p = np.asarray(d["pos"]) + cen
            col = "#c00000" if d["part"] else "#1a7a1a"
            mk = (dict(size=9, symbol="diamond", color=col) if solid
                  else dict(size=8, symbol="diamond-open", color=col,
                            line=dict(width=3)))
            tr.append(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                marker=mk, name=f"{'튜닝' if solid else '기준'} {arm}",
                showlegend=False))
    fig = go.Figure(tr)
    fig.update_layout(scene=dict(aspectmode="data"), height=440, width=540,
                      margin=dict(l=0, r=0, t=5, b=0), showlegend=True)
    div = fig.to_html(full_html=False, include_plotlyjs=False,
                      config={"displayModeBar": False})
    cap_html = (f'<div class="cap">구간 {si} · 프레임 {s}-{e} · {row["hand"]}</div>'
                f'<div class="res">기준선(α): {verdict(si, "baseline")}</div>'
                f'<div class="res">ρ 튜닝: {verdict(si, "tuned")}</div>')
    blocks.append((si, b64, div, cap_html))

parts_html = []
for k, (si, b64, div, cap_html) in enumerate(blocks):
    parts_html.append(
        f'<div class="blk" id="b{k}" style="display:{"flex" if k == 0 else "none"}">'
        f'<div>{cap_html}<img src="data:image/jpeg;base64,{b64}"></div>'
        f'<div>{div}</div></div>')

html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>정적 파지 용량 — {seq}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{font-family:'Noto Sans KR',sans-serif;margin:18px}}
.blk{{display:flex;gap:16px;align-items:flex-start}}
.cap{{font-weight:700;font-size:14px;margin-bottom:4px}}
.res{{font-size:13px;margin-bottom:3px}}
button{{font-size:15px;padding:5px 16px;margin-right:6px}}</style></head><body>
<h2>키프레임 정지 파지 용량 — {seq} (공중 유지, LP 평형해 사전 인가)</h2>
<p>점 = 사람 접촉 (주황 왼손/파랑 오른손) ·
◇ 테두리 = 기준선(α 시간표) · ◆ 채움 = ρ(ℓ∞) 튜닝 ·
<b style="color:#1a7a1a">초록 = 본체</b> / <b style="color:#c00000">빨강 = 뚜껑</b>
· 3D 회전 가능</p>
<button onclick="mv(-1)">◀ 이전</button><button onclick="mv(1)">다음 ▶</button>
<span id="pos"></span>
{''.join(parts_html)}
<script>
let cur=0, n={len(blocks)};
function mv(d){{document.getElementById('b'+cur).style.display='none';
cur=(cur+d+n)%n;document.getElementById('b'+cur).style.display='flex';
document.getElementById('pos').textContent=' '+(cur+1)+' / '+n;
window.dispatchEvent(new Event('resize'));}}
mv(0);</script></body></html>"""
out = ROOT / f"md_claude/keyframes/static_capacity_viz_{seq}.html"
out.write_text(html)
print(f"저장: {out} ({out.stat().st_size/1e6:.1f}MB, 구간 {len(blocks)}개)")
