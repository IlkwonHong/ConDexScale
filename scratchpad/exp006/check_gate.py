"""검증: gate_runner 의 servo 모드가 기존 판정을 재현하고, force 모드가 진단 결과(조건 C)를 재현하는가.

servo: 기존 gate_records(_all).npz 에서 무작위 100건 → 같은 판정 비율 (기준 ≥ 98%).
force: 2026-11 진단 표본 — 원래 불합격 16 (진단 C 에서 8 합격), 원래 합격 10 (10 합격).
"""
import sys
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from contraction_gt.gate_runner import _load_records, _prepare, _worker  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
FAIL16 = [15266, 29093, 40281, 62729, 80594, 102750, 129568, 196768, 206455, 227447,
          272962, 342257, 422693, 485907, 518538, 530556]
PASS10 = [41330, 88867, 293456, 301596, 330116, 377145, 385065, 387356, 500471, 515768]


def main():
    m = _load_records(ROOT)
    old = {}
    for name in ("gate_records.npz", "gate_records_all.npz"):
        z = np.load(ROOT / name, allow_pickle=True)
        old.update({int(i): str(v) for i, v in zip(z["idx"], z["verdict"])})
    rng = np.random.default_rng(0)
    cand = np.array(sorted(i for i, v in old.items() if v in ("pass", "fail", "fail_settle")))
    sample = rng.choice(cand, 100, replace=False)
    ctx = get_context("spawn")
    with ctx.Pool(8) as pool:
        res_s = pool.map(partial(_worker, normal_mode="servo"), _prepare(sample, m, ROOT), chunksize=2)
        res_f = pool.map(partial(_worker, normal_mode="force"), _prepare(np.array(FAIL16 + PASS10), m, ROOT),
                         chunksize=1)
    same = sum(old[i] == v for i, v, _ in res_s)
    print(f"servo 재현: 같은 판정 {same}/100")
    vf = {i: v for i, v, _ in res_f}
    print(f"force: 원래 불합격 16 중 합격 {sum(vf[i] == 'pass' for i in FAIL16)} (진단 C: 8) | "
          f"원래 합격 10 중 합격 {sum(vf[i] == 'pass' for i in PASS10)} (진단 C: 10) | 판정 {vf}")


if __name__ == "__main__":
    main()
