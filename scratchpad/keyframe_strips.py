"""키프레임 전후 이미지 띠 — 키프레임 f마다 f-5..f+5 의 11프레임 (카메라 1).

각 행 = 키프레임 하나: 가운데(키프레임)는 빨간 테두리, 캡션에 접촉 상태 전→후.
이미지 규약: images/{subj}/{seq}/{cam}/{프레임+1:05d}.jpg (t+1, 6월 검증).
출력: md_claude/keyframes/keyframe_strips.html (자체 포함, base64)
실행: ./venv/bin/python scratchpad/keyframe_strips.py
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = Path("/home/robin/a_workspace/ext_proj/arctic/unpack/arctic_data/data/images/s01")
OUT = ROOT / "md_claude/keyframes/keyframe_strips.html"
SEQS = ["box_grab_01", "box_use_02"]
CAM, HALF, THUMB_W = "1", 5, 160
CROP = (300, 500, 1900, 2500)          # (x0,y0,x1,y1) — 행동 영역 (카메라1, 2000x2800)
STATE_KO = {"none": "무접촉", "right": "오른손", "left": "왼손", "both": "양손"}


def thumb_b64(seq, fr):
    p = IMG / seq / CAM / f"{fr + 1:05d}.jpg"
    if not p.exists():
        return None
    im = Image.open(p).crop(CROP)
    im = im.resize((THUMB_W, int(THUMB_W * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0]))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=55)
    return base64.b64encode(buf.getvalue()).decode()


def state_around(kfdata, f):
    before = after = "?"
    for s in kfdata["segments"]:
        if s["end"] == f:
            before = STATE_KO[s["hand_state"]]
        if s["start"] == f:
            after = STATE_KO[s["hand_state"]]
    return before, after


html = ["""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>키프레임 전후 이미지 띠</title><style>
body{font-family:'Noto Sans KR',sans-serif;background:#f7f8fa;margin:20px}
h2{margin:28px 0 6px} .row{margin:14px 0;background:#fff;border:1px solid #ccd;
border-radius:8px;padding:8px;overflow-x:auto;white-space:nowrap}
.cap{font-size:13px;font-weight:700;margin-bottom:5px}
img{margin-right:3px;border:2px solid #ddd;border-radius:3px}
img.kf{border-color:#c00000;border-width:3px}
.fnum{display:inline-block;width:160px;margin-right:3px;text-align:center;
font-size:10px;color:#667}
</style></head><body>
<h1>키프레임 전후 이미지 띠 — 키프레임 ±5프레임 (카메라 1, 2026-08-08)</h1>
<p style="color:#667;font-size:13px">빨간 테두리 = 추출된 키프레임. 캡션 = 접촉 상태
전환 (키프레임에서 끝나는 구간 → 시작하는 구간). 30fps.</p>"""]

for seq in SEQS:
    kfdata = json.load(open(ROOT / "data/contraction_dyn/keyframes"
                            / f"s01__{seq}.json"))
    T = kfdata["n_frames"]
    html.append(f"<h2>{seq} (전체 {T}프레임 = {T/30:.1f}초, "
                f"키프레임 {len(kfdata['keyframes'])}개)</h2>")
    for k, f in enumerate(kfdata["keyframes"]):
        b, a = state_around(kfdata, f)
        html.append(f'<div class="row"><div class="cap">키프레임 {k+1} — 프레임 '
                    f'{f} ({f/30:.1f}초) · 상태: {b} → {a}</div>')
        frames = [x for x in range(f - HALF, f + HALF + 1) if 0 <= x < T]
        for x in frames:
            d = thumb_b64(seq, x)
            if d:
                cls = ' class="kf"' if x == f else ""
                html.append(f'<img{cls} src="data:image/jpeg;base64,{d}" '
                            f'title="프레임 {x}">')
        html.append("<br>" + "".join(
            f'<span class="fnum">{x}{" ◀" if x == f else ""}</span>'
            for x in frames))
        html.append("</div>")
        print(f"  {seq} 키프레임 {k+1}/{len(kfdata['keyframes'])}", flush=True)
html.append("</body></html>")
OUT.write_text("".join(html))
print(f"저장: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
