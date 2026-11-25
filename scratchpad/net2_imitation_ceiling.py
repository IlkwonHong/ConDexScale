"""읽기 전용 집계: net2 가 저장된 정답(GT)을 완벽히 따라 했다면 1발 성공률이 얼마였을지.
저장된 평가 자료(net2_viz_cases*.pkl)에 들어 있는 파티션별 정답의 시뮬 판정·α 만 읽는다."""
import pickle
from pathlib import Path
import numpy as np
ROOT = Path("/home/robin/a_workspace/condex")
tot = dict(n=0, cur=0, imit=0, imit_pass=0, oracle=0, top_pass=0, n_parts=[])
for f in ("scratchpad/net2_viz_cases.pkl", "scratchpad/net2_viz_cases_test.pkl"):
    sub = dict(n=0, cur=0, imit=0, oracle=0, top_pass=0)
    for m in pickle.load(open(ROOT / f, "rb")):
        ref = float(m["gt_best_pass"])
        parts = [p for p in m["parts"] if np.isfinite(p["rec_alpha"])]
        sub["n"] += 1
        sel = m["parts"][m["sel"]]
        sub["cur"] += int(m["verdict"] == "pass" and sel["min_alpha"] >= 0.8 * ref)
        if not parts:
            continue
        top = max(parts, key=lambda p: p["rec_alpha"])                 # α 가 가장 높은 파티션의 정답
        sub["top_pass"] += int(top["rec_verdict"] == "pass")
        sub["imit"] += int(top["rec_verdict"] == "pass" and top["rec_alpha"] >= 0.8 * ref)
        sub["oracle"] += int(any(p["rec_verdict"] == "pass" and p["rec_alpha"] >= 0.8 * ref for p in parts))
        tot["n_parts"].append(len(parts))
    print(f"{f}: 파지 {sub['n']}개 | 지금 파이프라인 1발 성공 {sub['cur']} | "
          f"정답을 그대로 냈다면(α 가 가장 높은 파티션의 정답) 성공 {sub['imit']} (그 정답의 시뮬 합격 {sub['top_pass']}) | "
          f"상위 3개 파티션의 정답 중 하나라도 성공 {sub['oracle']}")
    for k in ("n", "cur", "imit", "oracle", "top_pass"):
        tot[k] += sub[k]
n = tot["n"]
print(f"합계 파지 {n}개: 지금 {tot['cur']} ({100*tot['cur']/n:.1f}%) | 정답 그대로 {tot['imit']} ({100*tot['imit']/n:.1f}%) | "
      f"그 정답의 시뮬 합격 {tot['top_pass']} ({100*tot['top_pass']/n:.1f}%) | 상위 3개 파티션 정답 중 하나라도 {tot['oracle']} ({100*tot['oracle']/n:.1f}%)")
print("파지당 파티션 수 분포:", dict(zip(*np.unique(tot["n_parts"], return_counts=True))))
