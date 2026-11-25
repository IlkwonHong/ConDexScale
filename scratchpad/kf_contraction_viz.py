"""키프레임(구간)별 contraction 시각화 — GT 실사진 | 3D(접촉+앵커) 나란히.

구간마다: 왼쪽 = 대표 프레임 실사진(카메라1), 오른쪽 = 회전 가능한 3D
(박스 부품색: 본체 회색/뚜껑 주황, 사람 접촉점: 좌손 주황·우손 파랑,
앵커: 별 — 본체 초록/뚜껑 빨강). 이전/다음 버튼으로 구간 넘기기.
실행: ./venv/bin/python scratchpad/kf_contraction_viz.py box_grab_01
출력: md_claude/keyframes/contraction_viz_{seq}.html
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
sched = json.load(open(ROOT / f"data/contraction_dyn/anchor_schedule/s01__{seq}.json"))
sd = SeqData("s01", seq)
top = sd.parts == 0
Fb = sd.tfaces[~top[sd.tfaces].any(1)]
Ft = sd.tfaces[top[sd.tfaces].all(1)]

blocks = []
for row in sched["rows"]:
    if not row.get("anchors"):
        continue
    s, e, th = row["start"], row["end"], row["theta"]
    rep = (s + e) // 2
    im = Image.open(IMG / seq / "1" / f"{rep+1:05d}.jpg").crop(CROP)
    im = im.resize((300, 375))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()

    v = sd.tverts.copy()
    Ra = Rotation.from_rotvec([0, 0, -th]).as_matrix()
    if abs(th) > 1e-8:
        v[top] = v[top] @ Ra.T
    cen = v.mean(0)
    tr = []
    for F_, col, nm in [(Fb, "#c9c9c9", "본체"), (Ft, "#e8862e", "뚜껑")]:
        tr.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=F_[:, 0], j=F_[:, 1], k=F_[:, 2],
                            color=col, opacity=0.45, name=nm, showlegend=True))
    frames = sorted(set(np.linspace(s, e - 1, min(5, e - s)).round().astype(int)))
    for side, col in [("left", "#ed7d31"), ("right", "#2e75b6")]:
        con = sd.segment_contacts(side, frames)
        if con is None:
            continue
        P = con["cpos"] + cen
        tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                               marker=dict(size=2.5, color=col),
                               name=f"{side} 접촉"))
    for arm, d in row["anchors"].items():
        q = np.asarray(d["pos_part"])
        va = (Ra @ q) if d["part"] and abs(th) > 1e-8 else q
        tr.append(go.Scatter3d(
            x=[va[0]], y=[va[1]], z=[va[2]], mode="markers+text",
            marker=dict(size=9, symbol="diamond",
                        color="#c00000" if d["part"] else "#1a7a1a"),
            text=[arm], textposition="top center",
            name=f"{arm} ({'뚜껑' if d['part'] else '본체'})"))
    fig = go.Figure(tr)
    fig.update_layout(scene=dict(aspectmode="data"), height=460, width=560,
                      margin=dict(l=0, r=0, t=25, b=0), showlegend=True,
                      title=dict(text=f"α={row.get('min_alpha')}", font=dict(size=12)))
    div = fig.to_html(full_html=False, include_plotlyjs=False,
                      config={"displayModeBar": False})
    lid_n = sum(1 for d in row["anchors"].values() if d["part"])
    blocks.append((row["seg"], s, e, row["hand"], lid_n, b64, div))

parts_html = []
for k, (si, s, e, hand, lid_n, b64, div) in enumerate(blocks):
    parts_html.append(
        f'<div class="blk" id="b{k}" style="display:{"flex" if k == 0 else "none"}">'
        f'<div><div class="cap">구간 {si} · 프레임 {s}-{e} · {hand} · '
        f'뚜껑 앵커 {lid_n}/4</div>'
        f'<img src="data:image/jpeg;base64,{b64}"></div><div>{div}</div></div>')

html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>키프레임별 contraction — {seq}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>body{{font-family:'Noto Sans KR',sans-serif;margin:18px}}
.blk{{display:flex;gap:16px;align-items:flex-start}}
.cap{{font-weight:700;font-size:14px;margin-bottom:6px}}
button{{font-size:15px;padding:5px 16px;margin-right:6px}}</style></head><body>
<h2>키프레임(구간)별 contraction — {seq} (GT 실사진 | 3D: 접촉·앵커·부품)</h2>
<p>다이아몬드 = 앵커 (<b style="color:#c00000">빨강 = 뚜껑 부품</b>,
<b style="color:#1a7a1a">초록 = 본체</b>) · 점 = 사람 접촉
(주황 왼손 / 파랑 오른손) · 3D는 마우스로 회전</p>
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
out = ROOT / f"md_claude/keyframes/contraction_viz_{seq}.html"
out.write_text(html)
print(f"저장: {out} ({out.stat().st_size/1e6:.1f}MB, 구간 {len(blocks)}개)")
