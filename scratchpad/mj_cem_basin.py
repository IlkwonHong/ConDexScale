"""CEM 2차 — 분지 채점: 물성 7케이스 × 매개변수 ±1% 흔들기 내장.

1차와 차이: ① 케이스 5→7 (질량중심 편차 ±1.5cm 추가) ② 케이스마다 조율값을
고정 패턴으로 ±1% 흔든 사본으로 롤아웃 — 바늘(정확한 값에서만 통과)은 구조적
으로 탈락, 언덕(근방 어디서든 통과)만 생존. ③ 우승자는 4자리 반올림 시험으로
분지 여부 확증. 초기 중심 = 1차 우승(웜스타트).
실행: ./venv/bin/python scratchpad/mj_cem_basin.py
"""
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

VARS = [  # (이름, 로그?, 하한, 상한)
    ("sg",     True,  0.005, 0.30),
    ("ffc",    False, 1.0,   12.0),
    ("tfc",    False, 0.3,   4.0),
    ("dro",    True,  0.05,  3.0),
    ("lever",  True,  0.03,  0.40),
    ("carrot", False, 5.0,   60.0),
    ("press",  False, 0.3,   1.5),
]
WARM = [0.036951158159670316, 2.8006712639830047, 2.717526380636423,
        0.886670732144944, 0.12375890747223402, 17.513066072600232,
        0.5677910466786238]
CASES = [(1.0, 0.6, 0.0), (0.9, 0.5, 0.0), (1.1, 0.5, 0.0),
         (0.9, 0.7, 0.0), (1.1, 0.7, 0.0),
         (1.0, 0.6, 0.015), (1.0, 0.6, -0.015)]
JIT = np.array([  # 케이스별 조율값 ±1% 고정 흔들기 패턴 (7변수)
    [0, 0, 0, 0, 0, 0, 0],
    [+1, +1, +1, +1, +1, +1, +1],
    [-1, -1, -1, -1, -1, -1, -1],
    [+1, -1, +1, -1, +1, -1, +1],
    [-1, +1, -1, +1, -1, +1, -1],
    [+1, +1, +1, -1, -1, -1, +1],
    [-1, -1, -1, +1, +1, +1, -1],
]) * 0.01
POP, ELITE, GENS, T_STEPS = 20, 5, 10, 1728
rng = np.random.default_rng(1)


def to_z(x):
    return np.array([np.log(v) if lg else v
                     for v, (_, lg, *_r) in zip(x, VARS)])


def to_x(zv):
    return [float(np.clip(np.exp(v) if lg else v, lo, hi))
            for v, (_, lg, lo, hi) in zip(zv, VARS)]


def rollout(x, k):
    ms, fr, com = CASES[k]
    xj = [v * (1 + j) for v, j in zip(x, JIT[k])]
    cmd = ["./venv/bin/python", "-m", "contraction_dyn.play_contact_mj",
           "box_grab_01", "--dr-mass", str(ms), "--dr-fric", str(fr),
           "--dr-com", str(com),
           "--sg", str(xj[0]), "--ffc", str(xj[1]), "--tfc", str(xj[2]),
           "--dro", str(xj[3]), "--lever", str(xj[4]), "--carrot", str(xj[5]),
           "--press-scale", str(xj[6])]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        pos = float(re.search(r"위치 오차: 평균 ([\d.]+)", out).group(1))
        rot = float(re.search(r"회전 오차: 평균 ([\d.]+)", out).group(1))
    except AttributeError:
        return 999.0, False
    m = re.search(r"낙하: step (\d+)", out)
    j = pos + 0.3 * rot
    if m:
        j += 200 + 200 * (1 - int(m.group(1)) / T_STEPS)
    return j, m is None


def score(x, pool):
    res = [f.result() for f in
           [pool.submit(rollout, x, k) for k in range(len(CASES))]]
    return max(r[0] for r in res), sum(1 for r in res if r[1])


def main():
    mean = to_z(WARM)
    std = np.array([0.25 if lg else (hi - lo) / 8
                    for (_, lg, lo, hi) in VARS])
    floor = np.array([0.04 if lg else (hi - lo) / 60
                      for (_, lg, lo, hi) in VARS])
    best = (float("inf"), list(WARM), 0)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for g in range(GENS):
            zs = [mean + std * rng.standard_normal(len(VARS))
                  for _ in range(POP - 1)]
            cands = [to_x(zv) for zv in zs] + [list(best[1])]
            outs = [score(x, pool) for x in cands]
            order = np.argsort([o[0] for o in outs])
            eb = order[0]
            if outs[eb][0] < best[0]:
                best = (outs[eb][0], cands[eb], outs[eb][1])
            ez = np.array([to_z(cands[i]) for i in order[:ELITE]])
            mean, std = ez.mean(0), np.maximum(ez.std(0), floor)
            bx = dict(zip([v[0] for v in VARS],
                          [round(v, 4) for v in best[1]]))
            print(f"[세대 {g+1:2d}/{GENS}] 세대최선 {outs[eb][0]:.1f} "
                  f"(완주 {outs[eb][1]}/7) | 전체최선 {best[0]:.1f} "
                  f"(완주 {best[2]}/7) {bx}", flush=True)
    json.dump(dict(score=best[0], n_ok=best[2],
                   params=dict(zip([v[0] for v in VARS], best[1]))),
              open("scratchpad/cem_basin_best.json", "w"), indent=1)
    # 분지 확증: 4자리 반올림 + 흔들기 없는 원값으로 전 케이스 재평가
    xr = [round(v, 4) for v in best[1]]
    res = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        global JIT
        JIT = np.zeros_like(JIT)               # 흔들기 끄고 반올림값만 시험
        futs = [pool.submit(rollout, xr, k) for k in range(len(CASES))]
        res = [f.result() for f in futs]
    n_ok = sum(1 for r in res if r[1])
    print(f"[분지 확증] 4자리 반올림값: 완주 {n_ok}/7, "
          f"최악 {max(r[0] for r in res):.1f}", flush=True)


if __name__ == "__main__":
    main()
