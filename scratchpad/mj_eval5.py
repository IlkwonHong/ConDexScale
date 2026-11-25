"""변형 비교용 5회 평가 — 물성을 살짝 흔들어 평균·최악으로 비교.

사용: ./venv/bin/python scratchpad/mj_eval5.py [--sg ..] (추가 인자는 그대로 전달)
"""
import re
import subprocess
import sys

CASES = [(1.0, 0.6), (0.9, 0.5), (1.1, 0.5), (0.9, 0.7), (1.1, 0.7)]
rows = []
for ms, fr in CASES:
    r = subprocess.run(
        ["./venv/bin/python", "-m", "contraction_dyn.play_contact_mj",
         "box_grab_01", "--dr-mass", str(ms), "--dr-fric", str(fr)]
        + sys.argv[1:], capture_output=True, text=True)
    out = r.stdout
    fell = re.search(r"낙하: step (\d+)", out)
    fell = int(fell.group(1)) if fell else 0
    pos = float(re.search(r"위치 오차: 평균 ([\d.]+)", out).group(1))
    rot = float(re.search(r"회전 오차: 평균 ([\d.]+)", out).group(1))
    lift = float(re.search(r"평균 들림: (-?[\d.]+)cm", out).group(1))
    rows.append((fell, pos, rot, lift))
    print(f"  질량×{ms} 마찰{fr}: {'낙하'+str(fell) if fell else '완주'} "
          f"위치{pos:.1f} 회전{rot:.1f} 들림{lift:+.1f}", flush=True)
n_ok = sum(1 for r in rows if r[0] == 0)
import statistics as st
print(f"[요약] 완주 {n_ok}/5 | 위치 평균 {st.mean(r[1] for r in rows):.1f} "
      f"최악 {max(r[1] for r in rows):.1f} | 회전 평균 {st.mean(r[2] for r in rows):.1f}")
