"""파일럿 탐색 패스 집계 — 체크리스트 2(반경 스윕)·3(μ 감도)·6(α 분포)·8(시간).

실행: ./venv/bin/python -m contraction_gt.pilot_analysis
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("data/contraction_gt_pilot/search_records")


def load_all():
    shards = sorted(ROOT.glob("shard_*.npz"))
    merged, times, n_grasp, n_fail = {}, {}, 0, 0
    for f in shards:
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            if k in ("times", "n_grasp", "n_fail"):
                continue
            merged.setdefault(k, []).append(d[k])
        t = json.loads(str(d["times"]))
        for k, v in t.items():
            times[k] = times.get(k, 0.0) + v
        n_grasp += int(d["n_grasp"])
        n_fail += int(d["n_fail"])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    return merged, times, n_grasp, n_fail


def main():
    m, times, n_grasp, n_fail = load_all()
    R = len(m["min_alpha"])
    print(f"=== grasp {n_grasp}개 (실패 {n_fail}), 레코드 {R}개 ===\n")

    print("--- 기본 구성 ---")
    for n in (2, 3, 4):
        cnt = int((m["n_anchors"] == n).sum())
        print(f"N={n}: 레코드 {cnt}개 ({cnt / R:.1%})")
    print(f"엄지 폴백(thumbless) 비율: {m['thumbless'].mean():.1%}")
    print(f"refine 이 coarse 를 이긴 비율: {(m['stage'] == 'refine').mean():.1%}")

    print("\n--- 항목 2: 패드 반경별 min-α 분포와 N=2 생존율 ---")
    for n in (2, 3, 4):
        for pad in (0.05, 0.10, 0.20):
            sel = (m["n_anchors"] == n) & np.isclose(m["pad_rel"], pad)
            a = m["min_alpha"][sel]
            if len(a) == 0:
                continue
            qs = np.percentile(a, [50, 90])
            surv = {t: float((a > t).mean()) for t in (0.01, 0.03, 0.1)}
            print(f"N={n} 반경={pad:.2f}: 중앙값={qs[0]:.4f} p90={qs[1]:.4f} "
                  f"생존율(>0.01/0.03/0.1)={surv[0.01]:.1%}/{surv[0.03]:.1%}/{surv[0.1]:.1%}")

    print("\n--- 항목 3: μ 감도 (min-α 순위 상관, 명목 0.6 기준) ---")
    min03 = m["alpha_mu03"].min(axis=1)
    min10 = m["alpha_mu10"].min(axis=1)
    for n in (2, 3, 4):
        sel = m["n_anchors"] == n
        r03 = spearmanr(m["min_alpha"][sel], min03[sel]).statistic
        r10 = spearmanr(m["min_alpha"][sel], min10[sel]).statistic
        print(f"N={n}: ρ(0.6 vs 0.3)={r03:.3f}, ρ(0.6 vs 1.0)={r10:.3f}")

    print("\n--- 최약 방향 분포 (0-5 힘, 6-11 토크, 12-13 앵커 급소축) ---")
    for n in (2, 3, 4):
        sel = m["n_anchors"] == n
        wd = m["weak_dir"][sel]
        frac_axis = float(((wd == 12) | (wd == 13)).mean())
        frac_torque = float(((wd >= 6) & (wd <= 11)).mean())
        print(f"N={n}: 급소축 {frac_axis:.1%}, 일반 토크 {frac_torque:.1%}, 힘 {1 - frac_axis - frac_torque:.1%}")

    print("\n--- 항목 8: 시간 (4샤드 합산 CPU/GPU 초) ---")
    per_grasp = sum(times.values()) / max(n_grasp, 1)
    print({k: round(v, 1) for k, v in times.items()})
    print(f"grasp당 {per_grasp:.2f}s (샤드 합산 기준) → 158만 grasp 외삽: "
          f"{per_grasp * 1_580_000 / 3600 / 4:.0f}시간 (4병렬)")

    print("\n--- prior 이탈 (정규화 좌표, r_obj=1) ---")
    pd_ = m["prior_dist"][~np.isnan(m["prior_dist"])]
    print(f"중앙값 {np.median(pd_):.3f}, p90 {np.percentile(pd_, 90):.3f}")


if __name__ == "__main__":
    main()
