"""다중 시작 수학 탐색 vs 네트워크 — 캡슐머신 00630 (2,2).

질문: 탐색 시작점을 여러 개 뿌리면 지역 분지 문제가 풀려서
네트워크(0.658)를 수학만으로 따라잡는가.
실행: ./venv/bin/python scratchpad/multistart_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "capsulemachine_use_01"]

import box_net_eval as E
from contraction_gt.search import _batched_alpha, mu_tor_from_pad

FR = 630
UNITS = E.UNITS


def main():
    smp = E.load_smp(FR)
    mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, E.MU_T, 0.5)
    batches = torch.load(ROOT / "scratchpad/net_batches_capsulemachine_use_01.pt",
                         weights_only=False)
    cand_sets, init0 = [], []
    for side in ("left", "right"):
        b = batches[f"{FR}_{side}"]
        cpos, cunit = b["_cpos"].numpy(), b["_cunit"].numpy()
        a0 = b["_anchor0"]
        present = [UNITS[u] for u in range(6) if (cunit == u).any()]
        for blk in [[a0], [u for u in present if u != a0]]:
            mem = [UNITS.index(u) for u in blk]
            sel = np.isin(cunit, mem)
            center = cpos[sel].mean(0)
            spread = float(np.linalg.norm(cpos[sel] - center, axis=1).max())
            radius = max(spread, E.RHO_REL * smp.r_obj) + E.MARGIN_REL * smp.r_obj
            d = np.linalg.norm(smp.points - center, axis=1)
            cand_sets.append(np.where(d <= radius)[0])
            init0.append(int(np.argmin(d)))

    def ascent(cur):
        cur = list(cur)
        for _ in range(2):
            for gi, cand in enumerate(cand_sets):
                combos = np.tile(cur, (len(cand), 1))
                combos[:, gi] = cand
                a = _batched_alpha(smp, combos, E.MU_T,
                                   np.full(len(combos), mu_tor), E.DEV).min(1)
                cur[gi] = int(cand[int(np.nanargmax(a))])
        a14 = _batched_alpha(smp, np.array([cur]), E.MU_T,
                             np.array([mu_tor]), E.DEV)[0]
        return float(np.nanmin(a14)), cur

    rng = np.random.default_rng(0)
    t0 = time.time()
    results = []
    for trial in range(12):
        cur = init0 if trial == 0 else [int(rng.choice(c)) for c in cand_sets]
        s, _ = ascent(cur)
        results.append(s)
        best = max(results)
        tag = {1: "중심 시작 1회 (기존)", 4: "누적 4회", 12: "누적 12회"}.get(trial + 1)
        if tag:
            print(f"  {tag:18s}: 최고 {best:.3f}   ({time.time()-t0:.1f}s)")
    print(f"\n  네트워크 제안 (참고): 0.658 | 마스크 크기 {[len(c) for c in cand_sets]}")


if __name__ == "__main__":
    main()
