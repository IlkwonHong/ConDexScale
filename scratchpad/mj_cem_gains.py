"""CEM+DR — 제어 조율값 7개를 물성 최악 성적으로 최적화.

변수: 서보 이득 sg, 힘 한도 ffc, 회전력 한도 tfc, 회전 감쇠 dro,
     회전 가중(지렛대) lever, 회전 당근 carrot[도], 예압 배율 press.
채점: 물성 5케이스(질량 ×0.9~1.1, 마찰 0.5~0.7) 각각 롤아웃,
     케이스 점수 = 위치 평균cm + 0.3×회전 평균° + 낙하 벌점(200 + 이른 낙하 가중),
     후보 점수 = 5케이스 중 최악값 (강건성).
CEM: 후보 24 × 12세대, 상위 6개로 분포 갱신, 현 기본값을 매 세대 주입.
실행: ./venv/bin/python scratchpad/mj_cem_gains.py
"""
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

VARS = [  # (이름, 로그공간?, 하한, 상한, 초기값)
    ("sg",     True,  0.005, 0.30, 0.03),
    ("ffc",    False, 1.0,   12.0, 4.0),
    ("tfc",    False, 0.3,   4.0,  2.0),
    ("dro",    True,  0.05,  3.0,  1.0),
    ("lever",  True,  0.03,  0.40, 0.10),
    ("carrot", False, 5.0,   60.0, 20.0),
    ("press",  False, 0.3,   1.5,  1.0),
]
CASES = [(1.0, 0.6), (0.9, 0.5), (1.1, 0.5), (0.9, 0.7), (1.1, 0.7)]
POP, ELITE, GENS, T_STEPS = 24, 6, 12, 1728
rng = np.random.default_rng(0)


def to_z(x):      # 실공간 → 탐색공간 (로그 변수는 log)
    return np.array([np.log(v) if lg else v
                     for v, (_, lg, *_r) in zip(x, VARS)])


def to_x(zv):
    x = []
    for v, (_, lg, lo, hi, _i) in zip(zv, VARS):
        v = np.exp(v) if lg else v
        x.append(float(np.clip(v, lo, hi)))
    return x


def rollout(x, case):
    ms, fr = case
    cmd = ["./venv/bin/python", "-m", "contraction_dyn.play_contact_mj",
           "box_grab_01", "--dr-mass", str(ms), "--dr-fric", str(fr),
           "--sg", str(x[0]), "--ffc", str(x[1]), "--tfc", str(x[2]),
           "--dro", str(x[3]), "--lever", str(x[4]), "--carrot", str(x[5]),
           "--press-scale", str(x[6])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    try:
        pos = float(re.search(r"위치 오차: 평균 ([\d.]+)", out).group(1))
        rot = float(re.search(r"회전 오차: 평균 ([\d.]+)", out).group(1))
    except AttributeError:
        return 999.0, False                      # 실행 실패 = 최악
    m = re.search(r"낙하: step (\d+)", out)
    j = pos + 0.3 * rot
    if m:
        j += 200 + 200 * (1 - int(m.group(1)) / T_STEPS)
    return j, m is None


def score(x, pool):
    futs = [pool.submit(rollout, x, c) for c in CASES]
    res = [f.result() for f in futs]
    worst = max(r[0] for r in res)
    n_ok = sum(1 for r in res if r[1])
    return worst, n_ok


def main():
    init = [v[4] for v in VARS]
    mean = to_z(init)
    std = np.array([0.5 if lg else (hi - lo) / 4
                    for (_, lg, lo, hi, _i) in VARS])
    std_floor = np.array([0.05 if lg else (hi - lo) / 50
                          for (_, lg, lo, hi, _i) in VARS])
    best = (float("inf"), init, 0)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for g in range(GENS):
            zs = [mean + std * rng.standard_normal(len(VARS))
                  for _ in range(POP - 1)]
            cands = [to_x(zv) for zv in zs] + [list(best[1])]  # 최선 주입
            outs = [score(x, pool) for x in cands]
            order = np.argsort([o[0] for o in outs])
            eb = order[0]
            if outs[eb][0] < best[0]:
                best = (outs[eb][0], cands[eb], outs[eb][1])
            elite_z = np.array([to_z(cands[i]) for i in order[:ELITE]])
            mean = elite_z.mean(0)
            std = np.maximum(elite_z.std(0), std_floor)
            names = [v[0] for v in VARS]
            bx = dict(zip(names, [round(v, 4) for v in best[1]]))
            print(f"[세대 {g+1:2d}/{GENS}] 세대최선 {outs[eb][0]:.1f} "
                  f"(완주 {outs[eb][1]}/5) | 전체최선 {best[0]:.1f} "
                  f"(완주 {best[2]}/5) {bx}", flush=True)
    json.dump(dict(score=best[0], n_ok=best[2],
                   params=dict(zip([v[0] for v in VARS], best[1]))),
              open("scratchpad/cem_gains_best.json", "w"), indent=1)
    print("저장: scratchpad/cem_gains_best.json", flush=True)


if __name__ == "__main__":
    main()
