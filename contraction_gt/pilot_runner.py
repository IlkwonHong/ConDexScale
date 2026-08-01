"""파일럿 러너 — 탐색 패스(체크리스트 2·3·6·8 입력 생성).

grasp 마다: 접촉 로드 → 파티션 열거(N∈{2,3,4}) → ρ 탐색(패드 반경 3종, μ=0.6)
→ 최종 앵커의 α 를 μ∈{0.3, 1.0}로 재평가(체크리스트 3) → 레코드 저장.

실행 (샤드 병렬):
  ./venv/bin/python -m contraction_gt.pilot_runner <shard_idx> <n_shards>
출력: data/contraction_gt_pilot/search_records/shard_<i>.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .contacts_io import load_grasp_contacts
from .object_cache import load_samples
from .search import SearchConfig, search_grasp, mu_tor_from_pad, _batched_alpha
from .units import enumerate_partitions, THUMB

ROOT = Path("data/contraction_gt_pilot")
CONTACT_ROOT = "data/dexonomy/contact_cache"
MU_SWEEP = (0.3, 1.0)   # 명목 0.6 은 탐색 자체가 계산


def rerun_alpha_at_mu(samples, recs, mu_t, cfg):
    """전체 레코드의 최종 앵커 α 를 다른 μ 로 일괄 재평가 (N 별 배치)."""
    out = [None] * len(recs)
    for n in (2, 3, 4):
        idxs = [i for i, r in enumerate(recs) if r.n_anchors == n]
        if not idxs:
            continue
        combos = np.stack([recs[i].anchor_idx for i in idxs])
        mu_tor = np.array([
            mu_tor_from_pad(recs[i].pad_radius_rel * samples.r_obj, mu_t, cfg.hertz_frac)
            for i in idxs])
        alpha = _batched_alpha(samples, combos, mu_t, mu_tor, cfg.device)
        for j, i in enumerate(idxs):
            out[i] = alpha[j]
    return out


def run_shard(shard_idx, n_shards, total=None, manifest_dir=None):
    cfg = SearchConfig()
    mdir = Path(manifest_dir) if manifest_dir else ROOT
    rows = [l.split("\t") for l in (mdir / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    if total:
        rows = rows[:total]
    rows = rows[shard_idx::n_shards]
    out_dir = mdir / "search_records"
    out_dir.mkdir(exist_ok=True)

    recs_out = {k: [] for k in
                ["grasp_id", "type", "n_anchors", "partition", "pad_rel", "min_alpha",
                 "weak_dir", "mu_tor", "stage", "thumbless", "n_units", "r_obj",
                 "total_scale"]}
    arr_out = {k: [] for k in ["alpha", "alpha_mu03", "alpha_mu10",
                               "anchor_pos", "anchor_normal", "prior_dist"]}
    times = {"contacts": 0.0, "samples": 0.0, "search": 0.0, "mu_sweep": 0.0}
    n_grasp = n_fail = 0

    for typ, ds, oid, rel, ridx, _sscale, _nfile in rows:
        gid = f"{rel}#{ridx}"
        try:
            t0 = time.time()
            patches, ts = load_grasp_contacts(CONTACT_ROOT, rel, int(ridx))
            times["contacts"] += time.time() - t0
            if len(patches) < 2:
                n_grasp += 1
                continue
            t0 = time.time()
            samples = load_samples(ROOT / "object_cache", ds, oid, scale=ts)
            times["samples"] += time.time() - t0
            partitions = []
            for n in (2, 3, 4):
                partitions += enumerate_partitions(patches, n)
            if not partitions:
                n_grasp += 1
                continue
            t0 = time.time()
            recs = search_grasp(samples, patches, partitions, cfg)
            times["search"] += time.time() - t0

            t0 = time.time()
            a03_all = rerun_alpha_at_mu(samples, recs, 0.3, cfg)
            a10_all = rerun_alpha_at_mu(samples, recs, 1.0, cfg)
            for r, a03, a10 in zip(recs, a03_all, a10_all):
                pad = 4 - r.n_anchors
                recs_out["grasp_id"].append(gid)
                recs_out["type"].append(typ)
                recs_out["n_anchors"].append(r.n_anchors)
                recs_out["partition"].append(json.dumps(r.partition))
                recs_out["pad_rel"].append(r.pad_radius_rel)
                recs_out["min_alpha"].append(r.min_alpha)
                recs_out["weak_dir"].append(r.weak_dir)
                recs_out["mu_tor"].append(r.mu_tor)
                recs_out["stage"].append(r.stage)
                recs_out["thumbless"].append(r.partition[0][0] != THUMB)
                recs_out["n_units"].append(len(patches))
                recs_out["r_obj"].append(samples.r_obj)
                recs_out["total_scale"].append(ts)
                arr_out["alpha"].append(r.alpha)
                arr_out["alpha_mu03"].append(a03)
                arr_out["alpha_mu10"].append(a10)
                arr_out["anchor_pos"].append(
                    np.pad(r.anchor_pos, ((0, pad), (0, 0)), constant_values=np.nan))
                arr_out["anchor_normal"].append(
                    np.pad(r.anchor_normal, ((0, pad), (0, 0)), constant_values=np.nan))
                arr_out["prior_dist"].append(
                    np.pad(r.prior_dist, (0, pad), constant_values=np.nan))
            times["mu_sweep"] += time.time() - t0
        except Exception as e:
            n_fail += 1
            print(f"실패 {gid}: {e!r}", flush=True)
        n_grasp += 1
        if n_grasp % 100 == 0:
            print(f"[shard {shard_idx}] {n_grasp}/{len(rows)} (레코드 {len(recs_out['grasp_id'])}, "
                  f"실패 {n_fail}) times={ {k: round(v,1) for k,v in times.items()} }", flush=True)

    np.savez_compressed(
        out_dir / f"shard_{shard_idx}.npz",
        **{k: np.asarray(v) for k, v in recs_out.items()},
        **{k: np.stack(v) if v else np.zeros((0,)) for k, v in arr_out.items()},
        times=json.dumps(times), n_grasp=n_grasp, n_fail=n_fail,
    )
    print(f"[shard {shard_idx}] 완료: grasp {n_grasp}, 레코드 {len(recs_out['grasp_id'])}, "
          f"실패 {n_fail}, times={times}")


if __name__ == "__main__":
    run_shard(int(sys.argv[1]), int(sys.argv[2]),
              int(sys.argv[3]) if len(sys.argv) > 3 else None,
              sys.argv[4] if len(sys.argv) > 4 else None)
