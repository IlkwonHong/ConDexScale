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


def _load_records(manifest_dir, subdir="search_records"):
    merged = {}
    for f in sorted((Path(manifest_dir) / subdir).glob("shard_*.npz")):
        d = np.load(f, allow_pickle=False)
        for k in d.files:
            if k in ("times", "n_grasp", "n_fail", "config"):
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


def _normal_alloc_lp(Wf, n, w_target):
    """ΣWx = w_target (정규화 단위), 앵커별 압력 ≤ 1 에서 총압력 최대 → 앵커별 압력 [n] (없으면 None).

    하이브리드 게이트의 외란별 법선력 배분 (압력 = 법선력 / f_cmd). 2026-11 진단 조건 C.
    """
    from scipy.optimize import linprog
    P = np.zeros((n, 6 * n))
    for j in range(n):
        P[j, 6 * j:6 * j + 6] = 1.0
    r = linprog(-np.ones(6 * n), A_ub=P, b_ub=np.ones(n), A_eq=Wf, b_eq=w_target,
                bounds=[(0, None)] * 6 * n, method="highs")
    return None if r.status != 0 else P @ r.x


def _prepare(idxs, m, manifest_dir, score="alpha"):
    """score="alpha": 기존 (14방향 α 하위 3 + 힘 6).
    score="rho64": 레코드의 alpha 필드 = 정점 64개 여유 → 시험 방향 = 최약 정점 3개(외란 −v) + 힘 6방향,
                   힘 6방향의 강도 기준은 그 방향 α 를 LP 로 새로 계산."""
    import torch
    from .metric import direction_set, vertex_direction_set, ray_wrench_matrix, _flatten_W, alpha_reference
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
        mt_n = float(m["mu_tor"][i]) / r_obj
        if score == "rho64":
            verts = vertex_direction_set(torch.float64, "cpu").numpy()
            marg = m["alpha"][i]
            weak = np.argsort(marg)[:3]
            force6 = direction_set(pos_n)[0].numpy()[:6]
            W = ray_wrench_matrix(pos_n, torch.as_tensor(nrm, dtype=torch.float64).unsqueeze(0),
                                  torch.zeros(1, 3, dtype=torch.float64), 1.0, 0.6, mt_n)
            a6 = alpha_reference(W, torch.as_tensor(force6).unsqueeze(0))[0].numpy()
            dirs = np.concatenate([-verts[weak], force6])
            claim = np.concatenate([marg[weak], a6])
        else:
            dirs_all = direction_set(pos_n)[0].numpy()
            alpha = m["alpha"][i]
            dir_set = sorted(set(np.argsort(alpha)[:3].tolist()) | set(range(6)))
            dirs, claim = dirs_all[dir_set], alpha[dir_set]
        levels = np.maximum(claim * GAMMA, LEVEL_FLOOR)[:, None]
        preload, _x, Wf = _equilibrium_preload((pos - com) / r_obj, nrm, 0.6, mt_n, 5.0)
        jobs.append(dict(
            idx=int(i),
            coacd=str(ASSET_ROOT / ds / "processed_data" / oid / "urdf" / "meshes"),
            scale=float(m["total_scale"][i]), pos=pos, nrm=nrm,
            dirs=dirs, levels=levels, r_obj=r_obj,
            mu_tor=float(m["mu_tor"][i]), preload=preload, Wf=Wf, n=n,
            tip_radius=max(float(m["pad_rel"][i]) * r_obj, 0.002),
        ))
    return jobs


def _worker(job, normal_mode="servo"):
    from .sim_eval import SimConfig, measure_capacity_multi
    try:
        cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                        timestep=1e-3, settle_steps=600, hold_steps=300, normal_mode=normal_mode)
        alloc = None
        if normal_mode == "force":
            if job.get("Wf") is None:
                return job["idx"], "error:no_W", 0
            cache = {}

            def alloc(d, a):
                key = (d, round(float(a), 9))
                if key not in cache:
                    p = _normal_alloc_lp(job["Wf"], job["n"], -a * job["dirs"][d])
                    cache[key] = (np.asarray(job["preload"], float) if p is None else p * cfg.f_cmd)
                return cache[key]
        a_sim, settled, _pre, n_unstable = measure_capacity_multi(
            ("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"],
            job["dirs"], job["levels"], job["r_obj"], cfg,
            preload_forces=job["preload"], seed=job["idx"], normal_alloc=alloc)
        if not settled:
            return job["idx"], "fail_settle", n_unstable
        held = a_sim >= job["levels"][:, 0] - 1e-9
        nan_mask = np.isnan(a_sim)
        if (held | nan_mask).all() and nan_mask.any():
            return job["idx"], "unstable", n_unstable
        return job["idx"], ("pass" if held[~nan_mask].all() else "fail"), n_unstable
    except Exception as e:
        return job["idx"], f"error:{e!r}", 14


def run(manifest_dir, workers, floor=0.02, limit=None, records_subdir="search_records",
        score="alpha", normal_mode="servo", out_name="gate_records.npz", all_records=False):
    """records_subdir/score/normal_mode/out_name 기본값 = 기존 동작.
    all_records=True 면 대표만이 아니라 N≥3·min>floor 인 전 레코드를 판정 (시범 생성용)."""
    from functools import partial
    m = _load_records(manifest_dir, records_subdir)
    if all_records:
        reps = np.array([i for i in range(len(m["min_alpha"]))
                         if int(m["n_anchors"][i]) >= 3 and m["min_alpha"][i] > floor])
    else:
        reps = _representatives(m, floor)
    if limit:
        reps = reps[:limit]
    print(f"레코드 {len(m['min_alpha']):,}개 → 판정 대상(N≥3, min>{floor}) {len(reps):,}개 "
          f"[score={score}, normal_mode={normal_mode}]", flush=True)
    jobs = _prepare(reps, m, manifest_dir, score=score)
    ctx = get_context("spawn")
    verdicts = {}
    unstable_dirs = 0
    with ctx.Pool(workers) as pool:
        for k, (idx, verdict, nu) in enumerate(
                pool.imap_unordered(partial(_worker, normal_mode=normal_mode), jobs, chunksize=8)):
            verdicts[idx] = verdict
            unstable_dirs += nu
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)} 완료", flush=True)
    idx_arr = np.array(sorted(verdicts))
    v_arr = np.array([verdicts[i] for i in idx_arr])
    out_path = Path(manifest_dir) / out_name
    if out_path.exists() and out_name == "gate_records.npz" and records_subdir != "search_records":
        raise FileExistsError("기존 gate_records.npz 를 덮어쓰지 않도록 out_name 을 지정할 것")
    np.savez_compressed(out_path,
                        idx=idx_arr, verdict=v_arr,
                        grasp_id=m["grasp_id"][idx_arr],
                        n_anchors=m["n_anchors"][idx_arr],
                        min_alpha=m["min_alpha"][idx_arr])
    from collections import Counter
    cnt = Counter(v_arr.tolist())
    print("판정 분포:", dict(cnt))
    print(f"불안정 방향 총 {unstable_dirs}개")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir"); ap.add_argument("workers", type=int)
    ap.add_argument("floor", type=float, nargs="?", default=0.02)
    ap.add_argument("limit", type=int, nargs="?", default=None)
    ap.add_argument("--records-subdir", default="search_records")
    ap.add_argument("--score", default="alpha", choices=["alpha", "rho64"])
    ap.add_argument("--normal-mode", default="servo", choices=["servo", "force"])
    ap.add_argument("--out-name", default="gate_records.npz")
    ap.add_argument("--all-records", action="store_true")
    a = ap.parse_args()
    run(a.manifest_dir, a.workers, a.floor, a.limit, a.records_subdir, a.score,
        a.normal_mode, a.out_name, a.all_records)
