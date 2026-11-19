"""다중 정답 러너 (2026-10-17) — runner_batched 에 "서로 다른 배치 여러 벌 + 거친 단계 표" 저장을 더한 판.

runner_batched 와 같은 후보·같은 거친 단계 조합을 쓰고, 그 뒤가 다르다:
  1. 거친 단계 조합을 min α 내림차순으로 훑어 서로 다른 배치(적어도 한 그룹에서 0.15·r_obj 이상
     차이)를 최대 MAX_MODES 개 고른다 (거친 min α ≥ PREFILTER × 거친 최선 인 것만).
  2. 각 배치를 runner_batched 와 같은 방식으로 다듬는다 (N 별 일괄 LP).
     배치 0 = 거친 최선 → 다듬은 결과가 runner_batched 의 레코드와 같다 (정합 검사용).
  3. 다듬은 뒤 서로 가까워진 배치는 높은 쪽만 남기고, 최선의 KEEP_FRAC 배 이상 + min α > FLOOR 만 저장.
  4. 첫 자리 조건 표: 첫 그룹(엄지 그룹) 후보 중 고르게 퍼진 N_FIRST 개를 첫 자리로 고정하고, 나머지 그룹 조합을
     대립 점검으로 TOP_COND 개 추려 LP 채점 → 첫 자리별 최선 짝 (거친 단계에서 그 점을 쓴 조합과 비교해 높은 쪽).
     (거친 단계 조합만으로 만든 표는 전수 대비 품질 중앙값 0.72·최선 일치 19% 라 폐기 — 10-17 시험)

레코드 = (파지 × 파티션 × 패드 반경) 1행. 기존 필드는 최선 배치 기준으로 runner_batched 와 같은 이름,
추가 필드:
  anchor_idx   [4]      최선 배치의 표면점 인덱스 (-1 패딩) — 물체 캐시 점 2048개 기준 (네트워크와 같은 점)
  n_modes      int
  mode_idx     [M,4]    배치별 표면점 인덱스 (-1 패딩), 0번 = 최선
  mode_alpha   [M,14]   배치별 α (NaN 패딩)
  mode_prior   [M,4]    배치별 그룹 접촉 중심 거리 (NaN 패딩)
  mode_stage   [M]      배치별 0 = coarse 채택, 1 = refine 채택, -1 = 없음
  cond_idx     [8,4]    첫 자리 조건 표 — 첫 자리 후보별 최선 조합 (-1 패딩, min α 내림차순)
  cond_min     [8]      그 조합의 min α (NaN 패딩)

실행: ./venv/bin/python -m contraction_gt.runner_multi <shard> <n_shards> <total|0> <manifest_dir> [chunk] [first_row]
출력: <manifest_dir>/search_records_multi/shard_<i>_part<k>.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .contacts_io import load_grasp_contacts
from .object_cache import load_samples
from .search import SearchConfig, mu_tor_from_pad, _top_combos
from .surface import _farthest_point_subsample
from .units import enumerate_partitions, THUMB
from .runner_batched import (CONTACT_ROOT, CACHE_ROOT, _alpha_many, _rho_many, _coarse_jobs,
                             _refine_set, _score_by_n, SCORE_DIMS)

MAX_MODES = 3          # 저장할 서로 다른 배치 수 상한
DIV_REL = 0.15         # 다른 배치 판정: 적어도 한 그룹에서 이 거리(·r_obj) 이상 차이
PREFILTER = 0.5        # 다듬을 후보 배치: 거친 min α ≥ PREFILTER × 거친 최선
KEEP_FRAC = 0.8        # 저장: 다듬은 min α ≥ KEEP_FRAC × 다듬은 최선
FLOOR = 0.02           # 저장 하한 (기존 수록 임계와 같음)
N_FIRST = 8            # 첫 자리 조건 표: 고정할 첫 자리 후보 수 (고르게 퍼진 점)
TOP_COND = 16          # 첫 자리마다 LP 로 채점할 나머지 조합 수 (대립 점검 상위)


def _diverse(pts, a, b, thr):
    """두 조합(인덱스 배열)이 서로 다른 배치인가: 적어도 한 그룹에서 thr 이상 차이."""
    return float(np.linalg.norm(pts[a] - pts[b], axis=1).max()) >= thr


def _pick_modes(samples, combos, min_a):
    """거친 조합에서 서로 다른 배치 최대 MAX_MODES 개 (첫 번째 = 거친 최선)."""
    order = np.argsort(-min_a, kind="stable")
    best = min_a[order[0]]
    thr = DIV_REL * samples.r_obj
    picked = [int(order[0])]
    for i in order[1:]:
        if len(picked) >= MAX_MODES or min_a[i] < PREFILTER * best or min_a[i] <= 0:
            break
        if all(_diverse(samples.points, combos[i], combos[p], thr) for p in picked):
            picked.append(int(i))
    return picked


def _cond_jobs(samples, job):
    """첫 자리 조건 표용 조합: 첫 그룹 후보 N_FIRST 개 × 나머지 대립 점검 상위 TOP_COND 개."""
    cand = [np.unique(job["combos"][:, g]) for g in range(job["n"])]
    first = cand[0][_farthest_point_subsample(samples.points[cand[0]], N_FIRST)]
    rows, owner = [], []
    for f in first:
        tc = _top_combos([np.array([f])] + cand[1:], samples, TOP_COND,
                         min_sep=2.4 * job["tip_r"])
        rows.append(tc)
        owner += [int(f)] * len(tc)
    combos = np.concatenate(rows) if rows else np.zeros((0, job["n"]), np.int64)
    return combos, np.asarray(owner, np.int64)


def _cond_table(job, ma_coarse, D=14):
    """첫 자리별 최선 짝 (조건 조합 + 같은 첫 자리의 거친 조합 중 최고) → (idx [8,4], min [8])."""
    best = {}
    cm = job["cond_alpha"].min(axis=1) if len(job["cond"]) else np.zeros(0)
    for c, a, f in zip(job["cond"], cm, job["cond_owner"]):
        if f not in best or a > best[f][1]:
            best[f] = (c, a)
    for c, a in zip(job["combos"], ma_coarse if ma_coarse is not None else []):
        f = int(c[0])
        if f in best and a > best[f][1]:
            best[f] = (c, a)
    idx = np.full((N_FIRST, 4), -1, np.int16)
    mins = np.full(N_FIRST, np.nan, np.float32)
    for r, (c, a) in enumerate(sorted(best.values(), key=lambda t: -t[1])[:N_FIRST]):
        idx[r, :len(c)] = c
        mins[r] = a
    return idx, mins


def run_shard(shard_idx, n_shards, total, manifest_dir, chunk=48, first_row=0,
              score="alpha", seed=0, out_subdir="search_records_multi", gid_list=None):
    """score: "alpha"(기존) | "rho64"(전 단계 ρ0) | "two_stage"(거친 단계 α 로 거르고 다듬기·확정은 ρ0).
    seed: 후보 줄이기·무작위 조합 씨앗 (EXP-002 의 주된 자리). gid_list: 처리할 grasp_id 집합(없으면 전체)."""
    cfg = SearchConfig()
    cfg.seed = int(seed)
    s_coarse = "rho64" if score == "rho64" else "alpha"
    s_fine = "alpha" if score == "alpha" else "rho64"
    D = SCORE_DIMS[s_fine]
    mdir = Path(manifest_dir)
    rows = [l.split("\t") for l in (mdir / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    if gid_list is not None:
        rows = [r for r in rows if f"{r[3]}#{r[4]}" in gid_list]
    rows = rows[first_row:]
    if total:
        rows = rows[:total]
    rows = rows[shard_idx::n_shards]
    out_dir = mdir / out_subdir
    out_dir.mkdir(exist_ok=True)

    done_gids = set()
    part_idx = 0
    for f in sorted(out_dir.glob(f"shard_{shard_idx}_part*.npz")):
        d = np.load(f, allow_pickle=False)
        done_gids.update(str(g) for g in d["grasp_id"])
        part_idx += 1
    if done_gids:
        print(f"[shard {shard_idx}] 이어하기: 기존 part {part_idx}개, "
              f"완료 grasp {len(done_gids)}개 건너뜀", flush=True)

    rec_keys = ["grasp_id", "type", "n_anchors", "partition", "pad_rel", "min_alpha",
                "weak_dir", "mu_tor", "stage", "thumbless", "n_units", "r_obj",
                "total_scale", "n_modes"]
    arr_keys = ["alpha", "alpha_mu03", "alpha_mu10", "anchor_pos", "anchor_normal",
                "prior_dist", "anchor_idx", "mode_idx", "mode_alpha", "mode_prior",
                "mode_stage", "cond_idx", "cond_min"]
    recs_out = {k: [] for k in rec_keys}
    arr_out = {k: [] for k in arr_keys}
    times = {"prep": 0.0, "coarse": 0.0, "refine": 0.0, "cond": 0.0, "mu_sweep": 0.0}
    n_grasp = n_fail = 0
    t_wall = time.time()
    FLUSH_EVERY = 2048

    def flush():
        nonlocal part_idx, recs_out, arr_out
        if not recs_out["grasp_id"]:
            return
        np.savez_compressed(
            out_dir / f"shard_{shard_idx}_part{part_idx}.npz",
            **{k: np.asarray(v) for k, v in recs_out.items()},
            **{k: np.stack(v) for k, v in arr_out.items()},
            times=json.dumps(times), n_grasp=n_grasp, n_fail=n_fail,
            config=json.dumps(dict(MAX_MODES=MAX_MODES, DIV_REL=DIV_REL, PREFILTER=PREFILTER,
                                   KEEP_FRAC=KEEP_FRAC, FLOOR=FLOOR, N_FIRST=N_FIRST,
                                   TOP_COND=TOP_COND, score=score, seed=int(seed))),
        )
        part_idx += 1
        recs_out = {k: [] for k in recs_out}
        arr_out = {k: [] for k in arr_out}

    for c0 in range(0, len(rows), chunk):
        batch_rows = [r for r in rows[c0:c0 + chunk]
                      if f"{r[3]}#{r[4]}" not in done_gids]
        if not batch_rows:
            continue
        # ---- 준비 (runner_batched 와 같음) ----
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

        # ---- 거친 단계 (같음) + 배치 고르기 + 거친 단계 표 ----
        t0 = time.time()
        _score_by_n(named, cfg, "combos", s_coarse)
        sub = []                                   # 다듬을 배치들: (g, 배치 작업)
        for g, j in named:
            ma = j["combos_alpha"].min(axis=1)
            j["picked"] = _pick_modes(g["samples"], j["combos"], ma)
            j["ma_coarse"] = ma
            j["cond"], j["cond_owner"] = _cond_jobs(g["samples"], j)
            j["modes"] = []
            for i in j["picked"]:
                m = dict(n=j["n"], pad_rel=j["pad_rel"], tip_r=j["tip_r"],
                         c_combo=j["combos"][i], c_alpha=j["combos_alpha"][i], c_min=ma[i])
                j["modes"].append(m)
                sub.append((g, m))
        if score == "two_stage" and sub:
            # 고른 거친 배치를 ρ0 로 다시 채점 (다듬기 결과와 같은 잣대로 비교하기 위함)
            for _g, m in sub:
                m["c1"] = m["c_combo"][None, :]
            _score_by_n(sub, cfg, "c1", s_fine)
            for _g, m in sub:
                m["c_alpha"] = m["c1_alpha"][0]
                m["c_min"] = float(m["c_alpha"].min())
        times["coarse"] += time.time() - t0

        # ---- 다듬기: 배치마다, N 별 일괄 ----
        t0 = time.time()
        for g, m in sub:
            m["ref"] = _refine_set(g["samples"], m, m["c_combo"], cfg)
        _score_by_n(sub, cfg, "ref", s_fine)
        times["refine"] += time.time() - t0

        # ---- 첫 자리 조건 표 ----
        t0 = time.time()
        withc = [(g, j) for g, j in named if len(j["cond"])]
        _score_by_n(withc, cfg, "cond", s_fine)
        for g, j in named:
            if not len(j["cond"]):
                j["cond_alpha"] = np.zeros((0, D))
            # two_stage 는 거친 단계 점수가 α 라 ρ0 와 섞지 않는다
            j["cond_tab"] = _cond_table(j, j["ma_coarse"] if score != "two_stage" else None)
        times["cond"] += time.time() - t0

        # ---- 확정: 배치별 최종 → 중복 제거 → 품질 필터 → μ 재평가(최선만) ----
        t0 = time.time()
        final = []
        for g, j in named:
            res = []
            for m in j["modes"]:
                rmin = m["ref_alpha"].min(axis=1)
                rb = int(rmin.argmax())
                if rmin[rb] >= m["c_min"]:
                    res.append((m["ref"][rb], m["ref_alpha"][rb], 1))
                else:
                    res.append((m["c_combo"], m["c_alpha"], 0))
            res.sort(key=lambda t: -float(t[1].min()))
            thr = DIV_REL * g["samples"].r_obj
            kept = []
            for r in res:
                if all(_diverse(g["samples"].points, r[0], k[0], thr) for k in kept):
                    kept.append(r)
            top = float(kept[0][1].min())
            kept = [kept[0]] + [r for r in kept[1:]
                                if float(r[1].min()) >= KEEP_FRAC * top and float(r[1].min()) > FLOOR]
            j["kept"] = kept
            j["final_combo"], j["final_alpha"] = kept[0][0], kept[0][1]
            j["stage"] = "refine" if kept[0][2] == 1 else "coarse"
            final.append((g, j))
        for mu_key, mu_val in (("mu03", 0.3), ("mu10", 1.0)):
            for n in (2, 3, 4):
                group = [(g, j) for g, j in final if j["n"] == n]
                if not group:
                    continue
                entries = [(g["samples"], j["final_combo"][None, :],
                            mu_tor_from_pad(j["pad_rel"] * g["samples"].r_obj,
                                            mu_val, cfg.hertz_frac)) for g, j in group]
                alpha = (_alpha_many if s_fine == "alpha" else _rho_many)(entries, cfg)
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
            recs_out["n_modes"].append(len(j["kept"]))
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
            arr_out["anchor_idx"].append(np.pad(np.asarray(combo, np.int16), (0, pad),
                                                constant_values=-1))
            mi = np.full((MAX_MODES, 4), -1, np.int16)
            ma_ = np.full((MAX_MODES, D), np.nan, np.float32)
            mp = np.full((MAX_MODES, 4), np.nan, np.float32)
            ms = np.full(MAX_MODES, -1, np.int8)
            for k, (c, a, st) in enumerate(j["kept"]):
                mi[k, :len(c)] = c
                ma_[k] = a
                mp[k, :len(c)] = np.linalg.norm(samples.points[c] - j["centers"], axis=1)
                ms[k] = st
            arr_out["mode_idx"].append(mi)
            arr_out["mode_alpha"].append(ma_)
            arr_out["mode_prior"].append(mp)
            arr_out["mode_stage"].append(ms)
            arr_out["cond_idx"].append(j["cond_tab"][0])
            arr_out["cond_min"].append(j["cond_tab"][1])

        if len(recs_out["grasp_id"]) and n_grasp % FLUSH_EVERY < chunk:
            flush()
        if (c0 // chunk) % 5 == 0:
            done = min(c0 + chunk, len(rows))
            rate = (time.time() - t_wall) / max(done, 1)
            print(f"[shard {shard_idx}] {done}/{len(rows)} (레코드 "
                  f"{len(recs_out['grasp_id'])}, 실패 {n_fail}) "
                  f"{rate:.3f}s/grasp times={ {k: round(v, 1) for k, v in times.items()} }",
                  flush=True)

    flush()
    print(f"[shard {shard_idx}] 완료: grasp {n_grasp}, part {part_idx}개, "
          f"실패 {n_fail}, times={times}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=int); ap.add_argument("n_shards", type=int)
    ap.add_argument("total", type=int); ap.add_argument("manifest_dir")
    ap.add_argument("chunk", type=int, nargs="?", default=48)
    ap.add_argument("first_row", type=int, nargs="?", default=0)
    ap.add_argument("--score", default="alpha", choices=["alpha", "rho64", "two_stage"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-subdir", default="search_records_multi")
    ap.add_argument("--gid-list", default=None, help="grasp_id 한 줄에 하나인 파일")
    a = ap.parse_args()
    gl = set(Path(a.gid_list).read_text().split()) if a.gid_list else None
    run_shard(a.shard, a.n_shards, a.total or None, a.manifest_dir, a.chunk, a.first_row,
              score=a.score, seed=a.seed, out_subdir=a.out_subdir, gid_list=gl)
