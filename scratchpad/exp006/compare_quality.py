"""EXP-006 품질 비교: 같은 (파지, 파티션, 패드) 레코드에서 방식별로 고른 배치의 ρ0.

- rho64, two_stage 레코드: alpha 필드 = 정점 64 여유 → min = ρ0.
- alpha 레코드: 고른 배치의 ρ0 를 compute_rho(w_req=0) 로 다시 계산 (α 로 골랐을 때 ρ0 가 얼마인지).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from contraction_gt.metric import compute_rho  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
CACHE = Path("data/contraction_gt_pilot/object_cache")
rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
mmap = {f"{r[3]}#{r[4]}": (r[1], r[2]) for r in rows}


def load(sub):
    out = {}
    for f in sorted((ROOT / sub).glob("shard_*.npz")):
        z = np.load(f, allow_pickle=False)
        cols = {k: z[k] for k in ("grasp_id", "partition", "pad_rel", "alpha", "n_anchors", "anchor_pos",
                                  "anchor_normal", "r_obj", "mu_tor", "total_scale", "anchor_idx")}
        for i in range(len(cols["grasp_id"])):
            k = (str(cols["grasp_id"][i]), str(cols["partition"][i]), round(float(cols["pad_rel"][i]), 4))
            out[k] = {c: v[i] for c, v in cols.items()}
    return out


def rho_of(recs):
    """레코드 목록의 ρ0 (compute_rho, 배치)."""
    res = np.zeros(len(recs))
    by_n = {}
    for j, r in enumerate(recs):
        by_n.setdefault(int(r["n_anchors"]), []).append(j)
    for n, js in by_n.items():
        pos = np.stack([recs[j]["anchor_pos"][:n] for j in js])
        nrm = np.stack([recs[j]["anchor_normal"][:n] for j in js])
        com = np.stack([np.load(CACHE / mmap[str(recs[j]["grasp_id"])][0] / f"{mmap[str(recs[j]['grasp_id'])][1]}.npz")["com"]
                        * float(recs[j]["total_scale"]) for j in js])
        r_obj = torch.as_tensor([float(recs[j]["r_obj"]) for j in js], dtype=torch.float64)
        mt = torch.as_tensor([float(recs[j]["mu_tor"]) for j in js], dtype=torch.float64)[:, None].expand(-1, n)
        o = compute_rho(torch.as_tensor(pos), torch.as_tensor(nrm), torch.zeros(len(js), 6, dtype=torch.float64),
                        com=torch.as_tensor(com), r_obj=r_obj, mu_t=0.6, mu_tor=mt, backend="batched")
        res[js] = o["rho"].cpu().numpy()
    return res


def main():
    A, R, T = load("search_records_exp006_alpha_s0"), load("search_records_exp006_rho64_s0"), load("search_records_exp006_two_s0")
    keys = sorted(set(A) & set(R) & set(T))
    print(f"레코드: alpha {len(A)}, rho64 {len(R)}, two_stage {len(T)}, 공통 {len(keys)}")
    rr = np.array([R[k]["alpha"].min() for k in keys])
    rt = np.array([T[k]["alpha"].min() for k in keys])
    ra = rho_of([A[k] for k in keys])
    # 검산: rho64 레코드의 저장 ρ0 를 같은 함수로 재계산
    chk = rho_of([R[k] for k in keys[:300]])
    print(f"검산 rho64 저장값 vs 재계산 (300개) 최대 차이 {np.abs(chk - rr[:300]).max():.2e}")
    for name, n in (("전체", None), ("N=2", 2), ("N=3", 3), ("N=4", 4)):
        sel = np.ones(len(keys), bool) if n is None else np.array([int(R[k]["n_anchors"]) == n for k in keys])
        ok = sel & (rr > 1e-6)
        print(f"{name:4s} n={sel.sum():5d} | ρ0 중앙값: rho64 {np.median(rr[sel]):.4f}, two_stage {np.median(rt[sel]):.4f}, "
              f"alpha로 고른 배치 {np.median(ra[sel]):.4f} | two/rho64 ≥0.95 비율 {np.mean(rt[ok] >= 0.95 * rr[ok]):.3f}, "
              f"중앙값 {np.median(rt[ok] / rr[ok]):.3f} | alpha/rho64 중앙값 {np.median(ra[ok] / rr[ok]):.3f} | "
              f"같은 앵커(two=rho64) {np.mean([np.array_equal(T[k]['anchor_idx'], R[k]['anchor_idx']) for k, s in zip(keys, sel) if s]):.3f}")
    print(f"ρ0 = 0 인 레코드 비율: rho64 {np.mean(rr <= 1e-6):.3f}, two {np.mean(rt <= 1e-6):.3f}, alpha 배치 {np.mean(ra <= 1e-6):.3f}")


if __name__ == "__main__":
    main()
