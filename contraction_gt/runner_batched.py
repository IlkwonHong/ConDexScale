"""배칭 러너 v2 — 여러 grasp 의 LP 를 N 별 초대형 배치로 묶는 탐색 실행기.

프로파일 근거(2026-07-25): 탐색 시간의 97%가 LP 솔버이고, grasp 당 ~14회의
잔호출로 파이썬/커널 오버헤드가 반복마다 붙는 것이 낭비였다. 이 러너는
grasp 청크(기본 48개)의 coarse·refine·μ재평가 LP 를 각각 N 별로 이어붙여
호출 수를 청크당 십수 회로 줄인다. 결과는 기존 러너와 결정적으로 동일
(같은 후보·같은 조합 — 배치 순서만 다름).

실행: ./venv/bin/python -m contraction_gt.runner_batched <shard> <n_shards> <total|0> <manifest_dir> [chunk]
출력: <manifest_dir>/search_records/shard_<i>.npz  (pilot_runner 와 동일 스키마)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .contacts_io import load_grasp_contacts
from .object_cache import load_samples
from .metric import ray_wrench_matrix, direction_set, alpha_batched
from .search import (SearchConfig, SearchRecord, mu_tor_from_pad, _top_combos,
                     _feasible_candidates)
from .surface import candidates_in_region
from .units import enumerate_partitions, THUMB

CONTACT_ROOT = "data/dexonomy/contact_cache"
CACHE_ROOT = Path("data/contraction_gt_pilot/object_cache")
GPU_SUB = 12000            # GPU 한 호출당 조합 상한 (메모리 ~1GB 수준)


def _alpha_many(entries, cfg):
    """entries: (samples, combos[C,N], mu_tor) 목록 (같은 N) → [ΣC,14] numpy.

    서로 다른 물체의 조합을 정규화 좌표로 이어붙여 한 번에 푼다.
    """
    dtype = torch.float64
    pos_l, nrm_l, mu_l = [], [], []
    for samples, combos, mu_tor in entries:
        pos_l.append((samples.points[combos] - samples.com) / samples.r_obj)
        nrm_l.append(samples.normals[combos])
        mu_l.append(np.full(len(combos), mu_tor / samples.r_obj))
    pos = np.concatenate(pos_l)
    nrm = np.concatenate(nrm_l)
    mu = np.concatenate(mu_l)
    out = np.empty((len(pos), 14))
    for s in range(0, len(pos), GPU_SUB):
        p = torch.as_tensor(pos[s:s + GPU_SUB], dtype=dtype, device=cfg.device)
        n = torch.as_tensor(nrm[s:s + GPU_SUB], dtype=dtype, device=cfg.device)
        m = torch.as_tensor(mu[s:s + GPU_SUB], dtype=dtype, device=cfg.device)
        m = m.unsqueeze(-1).expand(-1, p.shape[1])
        com = torch.zeros(len(p), 3, dtype=dtype, device=cfg.device)
        W = ray_wrench_matrix(p, n, com, 1.0, cfg.mu_t, m)
        out[s:s + GPU_SUB] = alpha_batched(W, direction_set(p)).cpu().numpy()
    return out


def _rho_many(entries, cfg):
    """_alpha_many 와 같은 입력 → 정점 64개 방향의 여유 [ΣC,64] numpy (EXP-006).

    논문 III-B 의 ℓ∞ 여유 ρ 를 명목 렌치 0 (질량 무관 GT, D5) 으로 푼 것:
    metric.compute_rho(w_req=0) 와 같은 방향 규약 dirs = −v (v ∈ {−1,+1}^6) 이고,
    ρ0 = 행별 최솟값. 정점 수 때문에 호출당 조합 상한을 14/64 로 줄인다.
    """
    from .metric import vertex_direction_set
    dtype = torch.float64
    pos_l, nrm_l, mu_l = [], [], []
    for samples, combos, mu_tor in entries:
        pos_l.append((samples.points[combos] - samples.com) / samples.r_obj)
        nrm_l.append(samples.normals[combos])
        mu_l.append(np.full(len(combos), mu_tor / samples.r_obj))
    pos = np.concatenate(pos_l)
    nrm = np.concatenate(nrm_l)
    mu = np.concatenate(mu_l)
    out = np.empty((len(pos), 64))
    sub = max(1, GPU_SUB * 14 // 64)
    verts = vertex_direction_set(dtype, cfg.device)
    for s in range(0, len(pos), sub):
        p = torch.as_tensor(pos[s:s + sub], dtype=dtype, device=cfg.device)
        n = torch.as_tensor(nrm[s:s + sub], dtype=dtype, device=cfg.device)
        m = torch.as_tensor(mu[s:s + sub], dtype=dtype, device=cfg.device)
        m = m.unsqueeze(-1).expand(-1, p.shape[1])
        com = torch.zeros(len(p), 3, dtype=dtype, device=cfg.device)
        W = ray_wrench_matrix(p, n, com, 1.0, cfg.mu_t, m)
        dirs = (-verts).unsqueeze(0).expand(len(p), -1, -1)
        out[s:s + sub] = alpha_batched(W, dirs).cpu().numpy()
    return out


SCORE_DIMS = {"alpha": 14, "rho64": 64}


def _coarse_jobs(samples, patches, partitions, cfg):
    """search_grasp 의 coarse 준비부와 동일 — 작업 목록만 반환."""
    jobs = []
    for part in partitions:
        n_anchors = len(part)
        k_cand = cfg.k_candidates_n4 if n_anchors >= 4 else cfg.k_candidates
        cand_sets, centers = [], []
        for group in part:
            pts = np.concatenate([patches[u].points for u in group])
            areas = np.concatenate([patches[u].areas for u in group])
            w = areas / areas.sum() if areas.sum() > 0 else None
            centroid = np.average(pts, axis=0, weights=w)
            nrm = np.average(np.concatenate([patches[u].normals for u in group]),
                             axis=0, weights=w)
            patch_r = float(np.linalg.norm(pts - centroid, axis=1).max())
            rho = max(patch_r, cfg.rho_rel * samples.r_obj)
            idx = candidates_in_region(samples, centroid, rho, ref_normal=nrm,
                                       k_max=k_cand, seed=cfg.seed)
            cand_sets.append(idx)
            centers.append(samples.points[idx[np.linalg.norm(
                samples.points[idx] - centroid, axis=1).argmin()]])
        pads = cfg.pad_radius_rel if n_anchors == 2 else cfg.pad_radius_rel_multi
        width = cfg.top_combos_n2 if n_anchors == 2 else cfg.top_combos
        for pad_rel in pads:
            tip_r = max(pad_rel * samples.r_obj, 0.002)
            fsets = [_feasible_candidates(samples, idx, tip_r) for idx in cand_sets]
            if any(len(s) == 0 for s in fsets):
                continue
            combos = _top_combos(fsets, samples, width, min_sep=2.4 * tip_r,
                                 n_random=cfg.random_combos, seed=cfg.seed)
            if len(combos) == 0:
                continue
            jobs.append(dict(part=part, pad_rel=pad_rel, tip_r=tip_r,
                             centers=np.asarray(centers), combos=combos,
                             n=n_anchors))
    return jobs


def _refine_set(samples, job, best_combo, cfg):
    ref_sets = []
    for gi in range(job["n"]):
        p_best = samples.points[best_combo[gi]]
        rho_ref = max(cfg.rho_rel * samples.r_obj / cfg.refine_shrink, 1e-6)
        idx = candidates_in_region(samples, p_best, rho_ref,
                                   ref_normal=samples.normals[best_combo[gi]],
                                   k_max=cfg.refine_k, seed=cfg.seed + 1)
        idx = _feasible_candidates(samples, idx, job["tip_r"])
        if best_combo[gi] not in idx:
            idx = np.concatenate([[best_combo[gi]], idx])
        ref_sets.append(idx)
    combos = _top_combos(ref_sets, samples, cfg.top_combos,
                         min_sep=2.4 * job["tip_r"])
    if len(combos) == 0:
        combos = best_combo[None, :]
    return combos


def _score_by_n(named_jobs, cfg, combo_key, score="alpha"):
    """named_jobs: (grasp_ctx, job) 목록 → 같은 N 끼리 묶어 일괄 채점, job 에 결과 저장.

    score="alpha": 14방향 α [C,14] (기존). score="rho64": 정점 64개 여유 [C,64] (min = ρ0).
    결과 키 이름은 둘 다 combo_key + "_alpha" (호출부의 .min(axis=1) 선택 규칙을 그대로 쓰기 위함).
    """
    score_fn = {"alpha": _alpha_many, "rho64": _rho_many}[score]
    for n in (2, 3, 4):
        group = [(g, j) for g, j in named_jobs if j["n"] == n]
        if not group:
            continue
        entries = [(g["samples"], j[combo_key],
                    mu_tor_from_pad(j["pad_rel"] * g["samples"].r_obj,
                                    cfg.mu_t, cfg.hertz_frac)) for g, j in group]
        alpha = score_fn(entries, cfg)
        ofs = 0
        for (_g, j), (_s, combos, _m) in zip(group, entries):
            j[combo_key + "_alpha"] = alpha[ofs:ofs + len(combos)]
            ofs += len(combos)


def run_shard(shard_idx, n_shards, total, manifest_dir, chunk=48):
    cfg = SearchConfig()
    mdir = Path(manifest_dir)
    rows = [l.split("\t") for l in (mdir / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    if total:
        rows = rows[:total]
    rows = rows[shard_idx::n_shards]
    out_dir = mdir / "search_records"
    out_dir.mkdir(exist_ok=True)

    # 이어하기: 기존 part 파일들의 grasp_id 는 건너뜀 (중간 저장 덕에 유실 최소화)
    done_gids = set()
    part_idx = 0
    for f in sorted(out_dir.glob(f"shard_{shard_idx}_part*.npz")):
        d = np.load(f, allow_pickle=False)
        done_gids.update(str(g) for g in d["grasp_id"])
        part_idx += 1
    if done_gids:
        print(f"[shard {shard_idx}] 이어하기: 기존 part {part_idx}개, "
              f"완료 grasp {len(done_gids)}개 건너뜀", flush=True)

    recs_out = {k: [] for k in
                ["grasp_id", "type", "n_anchors", "partition", "pad_rel", "min_alpha",
                 "weak_dir", "mu_tor", "stage", "thumbless", "n_units", "r_obj",
                 "total_scale"]}
    arr_out = {k: [] for k in ["alpha", "alpha_mu03", "alpha_mu10",
                               "anchor_pos", "anchor_normal", "prior_dist"]}
    times = {"prep": 0.0, "coarse": 0.0, "refine": 0.0, "mu_sweep": 0.0}
    n_grasp = n_fail = 0
    t_wall = time.time()
    FLUSH_EVERY = 2048          # grasp 이 이만큼 쌓이면 part 파일로 저장 후 비움

    def flush(final=False):
        nonlocal part_idx, recs_out, arr_out
        if not recs_out["grasp_id"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx}_part{part_idx}.npz",
            **{k: np.asarray(v) for k, v in recs_out.items()},
            **{k: np.stack(v) if v else np.zeros((0,)) for k, v in arr_out.items()},
            times=json.dumps(times), n_grasp=n_grasp, n_fail=n_fail,
        )
        part_idx += 1
        recs_out = {k: [] for k in recs_out}
        arr_out = {k: [] for k in arr_out}

    for c0 in range(0, len(rows), chunk):
        batch_rows = [r for r in rows[c0:c0 + chunk]
                      if f"{r[3]}#{r[4]}" not in done_gids]
        if not batch_rows:
            continue
        # ---- 준비 (CPU): 접촉·표면·파티션·후보·proxy ----
        t0 = time.time()
        ctxs = []
        for typ, ds, oid, rel, ridx, _s, _n in batch_rows:
            gid = f"{rel}#{ridx}"
            try:
                patches, ts = load_grasp_contacts(CONTACT_ROOT, rel, int(ridx))
                if len(patches) < 2:
                    n_grasp += 1
                    continue
                samples = load_samples(CACHE_ROOT, ds, oid, scale=ts)
                partitions = []
                for n in (2, 3, 4):
                    partitions += enumerate_partitions(patches, n)
                jobs = _coarse_jobs(samples, patches, partitions, cfg) if partitions else []
                ctxs.append(dict(gid=gid, typ=typ, samples=samples,
                                 patches=patches, jobs=jobs, ts=ts))
            except Exception as e:
                n_fail += 1
                print(f"실패 {gid}: {e!r}", flush=True)
            n_grasp += 1
        named = [(g, j) for g in ctxs for j in g["jobs"]]
        times["prep"] += time.time() - t0

        # ---- coarse: N 별 초대형 배치 ----
        t0 = time.time()
        _score_by_n(named, cfg, "combos")
        for g, j in named:
            ma = j["combos_alpha"].min(axis=1)
            b = int(ma.argmax())
            j["best_combo"], j["best_alpha"], j["best_min"] = j["combos"][b], j["combos_alpha"][b], ma[b]
        times["coarse"] += time.time() - t0

        # ---- refine: 재샘플 후 다시 N 별 배치 ----
        t0 = time.time()
        for g, j in named:
            j["ref"] = _refine_set(g["samples"], j, j["best_combo"], cfg)
        _score_by_n(named, cfg, "ref")
        times["refine"] += time.time() - t0

        # ---- 확정 + μ 재평가 배치 ----
        t0 = time.time()
        final = []
        for g, j in named:
            rmin = j["ref_alpha"].min(axis=1)
            rb = int(rmin.argmax())
            if rmin[rb] >= j["best_min"]:
                combo, alpha, stage = j["ref"][rb], j["ref_alpha"][rb], "refine"
            else:
                combo, alpha, stage = j["best_combo"], j["best_alpha"], "coarse"
            j["final_combo"], j["final_alpha"], j["stage"] = combo, alpha, stage
            final.append((g, j))
        for mu_key, mu_val in (("mu03", 0.3), ("mu10", 1.0)):
            for n in (2, 3, 4):
                group = [(g, j) for g, j in final if j["n"] == n]
                if not group:
                    continue
                entries = [(g["samples"], j["final_combo"][None, :],
                            mu_tor_from_pad(j["pad_rel"] * g["samples"].r_obj,
                                            mu_val, cfg.hertz_frac)) for g, j in group]
                alpha = _alpha_many(entries, cfg)
                for k_, (_g, j) in enumerate(group):
                    j["alpha_" + mu_key] = alpha[k_]
        times["mu_sweep"] += time.time() - t0

        for g, j in final:
            samples = g["samples"]
            combo = j["final_combo"]
            pad = 4 - j["n"]
            mu_tor = mu_tor_from_pad(j["pad_rel"] * samples.r_obj, cfg.mu_t, cfg.hertz_frac)
            recs_out["grasp_id"].append(g["gid"])
            recs_out["type"].append(g["typ"])
            recs_out["n_anchors"].append(j["n"])
            recs_out["partition"].append(json.dumps(j["part"]))
            recs_out["pad_rel"].append(j["pad_rel"])
            recs_out["min_alpha"].append(float(j["final_alpha"].min()))
            recs_out["weak_dir"].append(int(j["final_alpha"].argmin()))
            recs_out["mu_tor"].append(mu_tor)
            recs_out["stage"].append(j["stage"])
            recs_out["thumbless"].append(j["part"][0][0] != THUMB)
            recs_out["n_units"].append(len(g["patches"]))
            recs_out["r_obj"].append(samples.r_obj)
            recs_out["total_scale"].append(g["ts"])
            arr_out["alpha"].append(j["final_alpha"])
            arr_out["alpha_mu03"].append(j["alpha_mu03"])
            arr_out["alpha_mu10"].append(j["alpha_mu10"])
            arr_out["anchor_pos"].append(
                np.pad(samples.points[combo], ((0, pad), (0, 0)), constant_values=np.nan))
            arr_out["anchor_normal"].append(
                np.pad(samples.normals[combo], ((0, pad), (0, 0)), constant_values=np.nan))
            arr_out["prior_dist"].append(np.pad(
                np.linalg.norm(samples.points[combo] - j["centers"], axis=1),
                (0, pad), constant_values=np.nan))

        if len(recs_out["grasp_id"]) and n_grasp % FLUSH_EVERY < chunk:
            flush()
        if (c0 // chunk) % 5 == 0:
            done = min(c0 + chunk, len(rows))
            rate = (time.time() - t_wall) / max(done, 1)
            print(f"[shard {shard_idx}] {done}/{len(rows)} (레코드 "
                  f"{len(recs_out['grasp_id'])}, 실패 {n_fail}) "
                  f"{rate:.3f}s/grasp times={ {k: round(v, 1) for k, v in times.items()} }",
                  flush=True)

    flush(final=True)
    print(f"[shard {shard_idx}] 완료: grasp {n_grasp}, part {part_idx}개, "
          f"실패 {n_fail}, times={times}")


if __name__ == "__main__":
    run_shard(int(sys.argv[1]), int(sys.argv[2]),
              int(sys.argv[3]) or None, sys.argv[4],
              int(sys.argv[5]) if len(sys.argv) > 5 else 48)
