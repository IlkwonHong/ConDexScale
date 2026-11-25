"""제어 손잡이 자동 스캔 — play_contact_mj 를 조합별로 돌려 표로 정리.

축: 서보 적분 이득 sg × 되먹임 힘 축별 클립 ffc × 토크 상한 tfc × 회전 감쇠 dro.
점수: 낙하 스텝(없으면 최고), 시연 들림 구간 실물 평균 들림, 최대 들림,
위치·회전 오차 평균.
실행: ./venv/bin/python scratchpad/mj_gain_scan.py
"""
import itertools
import re
import subprocess
import sys

PY = "./venv/bin/python"
rows = []
combos = list(itertools.product([0.03, 0.05, 0.08], [4.0, 6.0],
                                [1.2, 2.0], [0.3, 1.0]))
for i, (sg, ffc, tfc, dro) in enumerate(combos):
    r = subprocess.run(
        [PY, "-m", "contraction_dyn.play_contact_mj", "box_grab_01",
         "--sg", str(sg), "--ffc", str(ffc), "--tfc", str(tfc),
         "--dro", str(dro)],
        capture_output=True, text=True)
    out = r.stdout

    def g(pat, default=float("nan")):
        m = re.search(pat, out)
        return float(m.group(1)) if m else default
    fell = re.search(r"낙하: step (\d+)", out)
    fell = int(fell.group(1)) if fell else 9999
    row = dict(sg=sg, ffc=ffc, tfc=tfc, dro=dro, fell=fell,
               max_lift=g(r"실물 최대 들림 ([\d.]+)cm"),
               avg_lift=g(r"실물 평균 들림: (-?[\d.]+)cm"),
               pos=g(r"위치 오차: 평균 ([\d.]+)cm"),
               rot=g(r"회전 오차: 평균 ([\d.]+)°"))
    rows.append(row)
    print(f"[{i+1:2d}/{len(combos)}] sg{sg} ffc{ffc} tfc{tfc} dro{dro} → "
          f"낙하 {row['fell']:4d} 들림 {row['max_lift']:.1f}/{row['avg_lift']:+.1f}cm "
          f"오차 {row['pos']:.0f}cm/{row['rot']:.0f}°", flush=True)

rows.sort(key=lambda r: (-r["fell"], -r["avg_lift"]))
print("\n=== 순위 (낙하 늦은 순 → 평균 들림) ===")
for r in rows[:10]:
    print(f"  sg{r['sg']} ffc{r['ffc']} tfc{r['tfc']} dro{r['dro']}: "
          f"낙하 {r['fell']} 들림최대 {r['max_lift']:.1f} 평균 {r['avg_lift']:+.1f} "
          f"위치 {r['pos']:.0f}cm 회전 {r['rot']:.0f}°")
