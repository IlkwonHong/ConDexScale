"""전량용 시뮬 도장 러너 (D1 확정 프로토콜).

대상: 앵커 3개 이상(N≥3)의 대표 레코드 — grasp × N 마다 min-α 최고 1개.
      (N=2 는 파일럿 검증으로 도장 불필요 — 불일치 0%)
시험: 방향 9개(LP 하위 3 + 힘 6, 중복 제거) × 각 방향 1레벨
      = 그 방향 주장값 × γ(0.5, 파일럿 보정 비율의 보수값).
판정: 전 방향 유지 → 합격. 수치 불안정으로 측정 불가한 방향이 있으면
      합격/불합격 대신 '판정 불가'로 별도 표기 (실패로 계산하지 않음).
예압: LP 평형 배분 (파일럿 확정).

실행: ./venv/bin/python -m contraction_gt.gate_runner <manifest_dir> <workers> [floor]
"""

import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .pilot_sim import _equilibrium_preload, ASSET_ROOT

GAMMA = 0.25         # 시험 레벨 = 방향별 주장값 × γ (파일럿 ROC: 검거 97%/통과 91%)
LEVEL_FLOOR = 0.01


def _load_records(manifest_dir):
    merged = {}
    for f in sorted((Path(manifest_dir) / "search_records").glob("shard_*.npz")):
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            if k in ("times", "n_grasp", "n_fail"):
                continue
            merged.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in merged.items()}


def _representatives(m, floor):
    """grasp × N(≥3) 별 min-α 최고 레코드 인덱스."""
    best = {}
    for i in range(len(m["min_alpha"])):
        n = int(m["n_anchors"][i])
        if n < 3 or m["min_alpha"][i] <= floor:
            continue
        key = (str(m["grasp_id"][i]), n)
        if key not in best or m["min_alpha"][i] > m["min_alpha"][best[key]]:
            best[key] = i
    return np.array(sorted(best.values()))


def _prepare(idxs, m, manifest_dir):
    import torch
    from .metric import direction_set
    rows = [l.split("\t") for l in
            (Path(manifest_dir) / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in rows}
    cache_root = Path("data/contraction_gt_pilot/object_cache")
    jobs = []
    for i in idxs:
        gid = str(m["grasp_id"][i])
        ds, oid = mmap[gid.split("#")[0]]
        n = int(m["n_anchors"][i])
        pos, nrm = m["anchor_pos"][i][:n], m["anchor_normal"][i][:n]
        r_obj = float(m["r_obj"][i])
        cache = np.load(cache_root / ds / f"{oid}.npz")
        com = cache["com"] * float(m["total_scale"][i])
        pos_n = torch.as_tensor((pos - com) / r_obj, dtype=torch.float64).unsqueeze(0)
        dirs_all = direction_set(pos_n)[0].numpy()
        alpha = m["alpha"][i]
        dir_set = sorted(set(np.argsort(alpha)[:3].tolist()) | set(range(6)))
        levels = np.maximum(alpha[dir_set] * GAMMA, LEVEL_FLOOR)[:, None]
        preload, _x, _W = _equilibrium_preload(
            (pos - com) / r_obj, nrm, 0.6, float(m["mu_tor"][i]) / r_obj, 5.0)
        jobs.append(dict(
            idx=int(i),
            coacd=str(ASSET_ROOT / ds / "processed_data" / oid / "urdf" / "meshes"),
            scale=float(m["total_scale"][i]), pos=pos, nrm=nrm,
            dirs=dirs_all[dir_set], levels=levels, r_obj=r_obj,
            mu_tor=float(m["mu_tor"][i]), preload=preload,
            tip_radius=max(float(m["pad_rel"][i]) * r_obj, 0.002),
        ))
    return jobs


def _worker(job):
    from .sim_eval import SimConfig, measure_capacity_multi
    try:
        cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                        timestep=1e-3, settle_steps=600, hold_steps=300)
        a_sim, settled, _pre, n_unstable = measure_capacity_multi(
            ("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"],
            job["dirs"], job["levels"], job["r_obj"], cfg,
            preload_forces=job["preload"], seed=job["idx"])
        if not settled:
            return job["idx"], "fail_settle", n_unstable
        held = a_sim >= job["levels"][:, 0] - 1e-9
        nan_mask = np.isnan(a_sim)
        if (held | nan_mask).all() and nan_mask.any():
            return job["idx"], "unstable", n_unstable
        return job["idx"], ("pass" if held[~nan_mask].all() else "fail"), n_unstable
    except Exception as e:
        return job["idx"], f"error:{e!r}", 14


def run(manifest_dir, workers, floor=0.02, limit=None):
    m = _load_records(manifest_dir)
    reps = _representatives(m, floor)
    if limit:
        reps = reps[:limit]
    print(f"레코드 {len(m['min_alpha']):,}개 → 대표(N≥3, min-α>{floor}) {len(reps):,}개", flush=True)
    jobs = _prepare(reps, m, manifest_dir)
    ctx = get_context("spawn")
    verdicts = {}
    unstable_dirs = 0
    with ctx.Pool(workers) as pool:
        for k, (idx, verdict, nu) in enumerate(
                pool.imap_unordered(_worker, jobs, chunksize=8)):
            verdicts[idx] = verdict
            unstable_dirs += nu
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} 완료", flush=True)
    idx_arr = np.array(sorted(verdicts))
    v_arr = np.array([verdicts[i] for i in idx_arr])
    np.savez_compressed(Path(manifest_dir) / "gate_records.npz",
                        idx=idx_arr, verdict=v_arr,
                        grasp_id=m["grasp_id"][idx_arr],
                        n_anchors=m["n_anchors"][idx_arr],
                        min_alpha=m["min_alpha"][idx_arr])
    from collections import Counter
    cnt = Counter(v_arr.tolist())
    print("판정 분포:", dict(cnt))
    print(f"불안정 방향 총 {unstable_dirs}개")


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]),
        float(sys.argv[3]) if len(sys.argv) > 3 else 0.02,
        int(sys.argv[4]) if len(sys.argv) > 4 else None)
