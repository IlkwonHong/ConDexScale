"""후보 점을 인코더 조각 중심 128개(비교: 256개)로 줄였을 때의 정답 사전 확인 (EXP-003, 2026-11-13).

검증 파지 480개, 팔 3대 파티션. 팔마다 후보를 조각 중심으로 제한하고, 조합을 전부 선형계획으로 채점해
min α 가 가장 높은 조합을 새 정답으로 둔다. 학습·시뮬레이션·다듬기 없음. 탐색 코드는 수정하지 않는다.

후보 규칙 (팁 간섭으로 놓을 수 없는 조각 중심은 먼저 뺀다 — search._feasible_candidates):
  가: 지금 탐색의 범위 그대로 (surface.candidates_in_region 과 같은 반경·법선 조건, 최근접 1점 폴백은 없음)
  나: 가 + (범위 반경 + 물체 반경의 0.05배 안이고 법선 조건을 만족하는 조각 중심) — 지금 net2 가림막의 반경
  다: 가의 후보가 2개 미만인 팔만 접촉 중심에서 가까운 순서로 2개까지 채움 (법선 조건 만족하는 점 먼저)
  라: 같은 방식으로 4개까지 채움 (참고용)
네 규칙의 후보는 합집합 하나로 모아 한 번만 채점하고, 규칙별 결과는 그 안에서 걸러 낸다.
앵커 사이 최소 간격(팁 반경의 2.4배)·팁 반경(물체 반경의 0.10배)·α 계산은 지금 탐색과 같은 함수.

검산 (1·2 가 통과해야 결과를 해석한다):
  1. build_patch_cache.fps 로 다시 고른 128개 = 조각 캐시의 center_idx, 캐시의 중심 좌표 = 그 번호의 표면점
  2. 저장된 정답 배치의 α 를 이 스크립트의 채점 경로로 다시 계산 = 저장된 min_alpha (0.000001 이내, 99% 이상)
  3. (연기 시험만) 채점한 조합 수 = 따로 센 조합 수, 최선 조합의 α = search._batched_alpha 로 따로 계산한 값

출력: scratchpad/gt_coarse_candidate_precheck.pkl + 요약 출력
실행: ./venv/bin/python scratchpad/gt_coarse_candidate_precheck.py [--smoke]
계획: harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-003-coarse-candidate-precheck/plan.md
"""
import itertools
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

import gt_seed_consistency as G                                                   # noqa: E402
import contraction_gt.runner_batched as RB                                        # noqa: E402
from contraction_gt.search import (SearchConfig, mu_tor_from_pad, _top_combos,    # noqa: E402
                                   _feasible_candidates, _batched_alpha)
from contraction_gt.object_cache import cache_path                                # noqa: E402
from contraction_net.build_patch_cache import fps                                 # noqa: E402

G.N_LIST = (3,)
N = 3
PATCH_ROOT = ROOT / "data/contraction_net/patch_cache"
OUT_PKL = ROOT / "scratchpad/gt_coarse_candidate_precheck.pkl"
EXP001_PKL = ROOT / "scratchpad/gt_seed_consistency.pkl"
RULES = ("가", "나", "다", "라")
RULE_TEXT = {"가": "지금 탐색의 범위 그대로", "나": "범위 반경 + 물체 반경의 0.05배",
             "다": "후보 최소 2개 보장", "라": "후보 최소 4개 보장(참고용)"}
MARGIN_REL = 0.05
KMIN = {"다": 2, "라": 4}
# ── 계획서 고정값 ──
PASS_RATIO, GO, NOGO = 0.8, 0.90, 0.75      # 주 지표: 새 α ≥ 0.8 × 저장된 α 인 레코드 비율, 90% 이상 진행 / 75% 미만 부적합
TIE, TIE_MANY, TIE_FEW = 0.95, 0.50, 0.20    # α 가 최선의 0.95배 이상인 다른 조합이 있는 레코드 비율의 읽는 기준
LP_CAP, TIME_CAP = 20_000_000, 40 * 60       # 중단 조건
VERIFY_MIN = 0.99
SUB_256 = 4                                  # 256개 비교는 파지 네 개마다 하나
CHUNK = 12
N_MANIFEST = 273_576                         # 파지 목록 전량 (pilot_manifest.tsv 의 행 수)

T0 = time.time()
LP_TOTAL = [0]


def fps_extend(points, sel, k):
    """이미 고른 점 sel 에서 이어서, 서로 가장 먼 점을 k 개가 될 때까지 더한다 (build_patch_cache.fps 와 같은 규칙)."""
    sel = [int(i) for i in sel]
    d = np.full(len(points), np.inf)
    for i in sel:
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    while len(sel) < k:
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.array(sel, dtype=np.int64)


def arm_geometry(patches, group, samples, cfg):
    """팔 하나의 접촉 중심·평균 법선·범위 반경 — runner_batched._coarse_jobs 와 같은 식."""
    pts = np.concatenate([patches[u].points for u in group])
    areas = np.concatenate([patches[u].areas for u in group])
    w = areas / areas.sum() if areas.sum() > 0 else None
    c = np.average(pts, axis=0, weights=w)
    nrm = np.average(np.concatenate([patches[u].normals for u in group]), axis=0, weights=w)
    rho = max(float(np.linalg.norm(pts - c, axis=1).max()), cfg.rho_rel * samples.r_obj)
    return c, nrm, rho


def rule_sets(samples, grid_ok, c, nrm, rho):
    """규칙별 후보 (표면점 번호 배열). grid_ok = 팁 간섭이 없는 조각 중심의 표면점 번호."""
    d_all = np.linalg.norm(samples.points - c, axis=1)
    n_all = samples.normals @ nrm > 0.1
    in_a = d_all <= rho                                   # surface.candidates_in_region 과 같은 판정 (전체 표면점 기준)
    if (in_a & n_all).any():
        in_a = in_a & n_all
    d, nm, m_a = d_all[grid_ok], n_all[grid_ok], in_a[grid_ok]
    m_b = m_a | ((d <= rho + MARGIN_REL * samples.r_obj) & nm)
    order = np.argsort(d, kind="stable")
    fill = [i for i in order if nm[i] and not m_a[i]] + [i for i in order if not nm[i] and not m_a[i]]
    masks = {"가": m_a, "나": m_b}
    for r, k in KMIN.items():
        m = m_a.copy()
        need = k - int(m.sum())
        if need > 0:
            m[fill[:need]] = True
        masks[r] = m
    return {r: grid_ok[m] for r, m in masks.items()}


def prepare_grasp(gid, ds, oid, grid_size, cfg, obj_cache, v1):
    rel, ridx = gid.rsplit("#", 1)
    patches, ts = G.load_grasp_contacts(str(ROOT / G.CONTACT_ROOT), rel, int(ridx))
    if len(patches) < 2:
        return None
    samples = G.load_samples(ROOT / G.CACHE_ROOT, ds, oid, scale=ts)
    if (ds, oid) not in obj_cache:
        pts_unit = np.load(cache_path(ROOT / G.CACHE_ROOT, ds, oid))["points"]
        z = np.load(PATCH_ROOT / ds / f"{oid}.npz")
        cidx = z["center_idx"].astype(np.int64)
        same = bool(np.array_equal(fps(pts_unit, 128), cidx)
                    and np.abs(z["centers"] - pts_unit[cidx]).max() < 1e-5)
        v1["n"] += 1
        v1["same"] += int(same)
        obj_cache[(ds, oid)] = cidx if grid_size == 128 else fps_extend(pts_unit, cidx, grid_size)
    grid = obj_cache[(ds, oid)]
    pad_rel = cfg.pad_radius_rel_multi[0]
    tip_r = max(pad_rel * samples.r_obj, 0.002)
    mu_tor = mu_tor_from_pad(pad_rel * samples.r_obj, cfg.mu_t, cfg.hertz_frac)
    grid_ok = _feasible_candidates(samples, grid, tip_r)
    parts = []
    for part in G.enumerate_partitions(patches, N):
        arms = [arm_geometry(patches, g, samples, cfg) for g in part]
        rs = [rule_sets(samples, grid_ok, c, nrm, rho) for c, nrm, rho in arms]
        S = [np.unique(np.concatenate([rs[g][r] for r in RULES])) for g in range(N)]
        if all(len(s) for s in S):
            combos = _top_combos(S, samples, int(np.prod([len(s) for s in S])), min_sep=2.4 * tip_r)
        else:
            combos = np.zeros((0, N), dtype=np.int64)
        parts.append(dict(key=(gid, json.dumps(part), round(float(pad_rel), 4)),
                          arms=arms, rs=rs, S=S, combos=combos))
    return dict(gid=gid, samples=samples, tip_r=tip_r, mu_tor=mu_tor, parts=parts)


def score_rules(p, samples):
    """파티션 하나의 규칙별 결과 — 합집합 조합의 채점 결과(p['ma'])에서 규칙의 후보만 걸러 낸다."""
    combos, ma = p["combos"], p.get("ma")
    out = {}
    for r in RULES:
        sets = [p["rs"][g][r] for g in range(N)]
        res = dict(n_cand=[len(s) for s in sets], n_combo=0, alpha=None)
        out[r] = res
        if len(combos) == 0 or any(len(s) == 0 for s in sets):
            continue
        rows = np.ones(len(combos), dtype=bool)
        for g in range(N):
            rows &= np.isin(combos[:, g], sets[g])
        idxs = np.nonzero(rows)[0]                        # 값싼 점수가 좋은 순서 (_top_combos 의 정렬)
        if len(idxs) == 0:
            continue
        b = int(idxs[int(ma[idxs].argmax())])
        best, a_best = combos[b], float(ma[b])
        near = idxs[ma[idxs] >= TIE * a_best - 1e-12]
        others = near[near != b]
        nearest = np.array([s[np.linalg.norm(samples.points[s] - p["arms"][g][0], axis=1).argmin()]
                            for g, s in enumerate(sets)])
        row = np.nonzero((combos == nearest).all(1))[0]
        pb = int(idxs[0])
        res.update(
            n_combo=int(len(idxs)), best=best.astype(np.int32), alpha=a_best,
            n_near=int(len(near)), has_other=bool(len(others)),
            other_diff2=bool(len(others) and ((combos[others] != best).sum(1) >= 2).any()),
            near_alpha=float(ma[row[0]]) if len(row) else float("nan"),
            near_same=(nearest == best), prox_alpha=float(ma[pb]), prox_same=(combos[pb] == best),
            chance=float(np.mean([1.0 / len(s) for s in sets])),
            dist_r=np.array([np.linalg.norm(samples.points[best[g]] - p["arms"][g][0]) / samples.r_obj
                             for g in range(N)]),
            dist_rho=np.array([np.linalg.norm(samples.points[best[g]] - p["arms"][g][0]) / p["arms"][g][2]
                               for g in range(N)]))
    return out


def check_v3(g, p, res, cfg, v3):
    """연기 시험용: 조합 수를 따로 세고, 최선 조합의 α 를 다른 함수로 다시 계산."""
    pts, S = g["samples"].points, p["S"]
    cnt = 0
    if all(len(s) for s in S):
        for a, b, c in itertools.product(*S):
            if min(np.linalg.norm(pts[a] - pts[b]), np.linalg.norm(pts[a] - pts[c]),
                   np.linalg.norm(pts[b] - pts[c])) >= 2.4 * g["tip_r"]:
                cnt += 1
    v3["n"] += 1
    v3["count_same"] += int(cnt == len(p["combos"]))
    r = res["라"]
    if r["alpha"] is not None:
        a = _batched_alpha(g["samples"], r["best"][None].astype(np.int64), cfg.mu_t,
                           np.full(1, g["mu_tor"]), cfg.device)[0].min()
        v3["alpha_n"] += 1
        v3["alpha_same"] += int(abs(float(a) - r["alpha"]) < 1e-9)


def run(grasps, grid_size, stored, smoke=False):
    cfg = SearchConfig()
    by_gid = defaultdict(list)
    for k, rec in stored.items():
        by_gid[k[0]].append((k, rec))
    obj_cache, recs = {}, {}
    v1 = dict(n=0, same=0)
    v2 = dict(n=0, same=0, unmatched=0, max_diff=0.0)
    v3 = dict(n=0, count_same=0, alpha_n=0, alpha_same=0)
    stat = dict(lp=0, lp_time=0.0, n_grasp=0, n_fail=0, gids=[], stopped=None)
    t_run = time.time()
    for c0 in range(0, len(grasps), CHUNK):
        ctxs = []
        for gid, ds, oid in grasps[c0:c0 + CHUNK]:
            try:
                g = prepare_grasp(gid, ds, oid, grid_size, cfg, obj_cache, v1)
            except Exception as e:
                stat["n_fail"] += 1
                print(f"  실패 {gid}: {e!r}", flush=True)
                continue
            if g is not None:
                ctxs.append(g)
        entries, slots = [], []
        for g in ctxs:
            for p in g["parts"]:
                if len(p["combos"]):
                    entries.append((g["samples"], p["combos"], g["mu_tor"]))
                    slots.append(("part", p))
            for key, rec in by_gid.get(g["gid"], []):
                dd = np.linalg.norm(g["samples"].points[None] - rec["pos"][:, None], axis=-1)   # [3, 2048]
                if dd.min(1).max() < 1e-7:
                    entries.append((g["samples"], dd.argmin(1)[None].astype(np.int64), g["mu_tor"]))
                    slots.append(("stored", rec))
                else:
                    v2["unmatched"] += 1
        n_lp = sum(len(e[1]) for e in entries)
        if LP_TOTAL[0] + n_lp > LP_CAP or time.time() - T0 > TIME_CAP:
            stat["stopped"] = (f"중단 조건 도달 (후보 {grid_size}개, 파지 {c0}/{len(grasps)}개까지 끝냄): "
                               f"선형계획 누적 {LP_TOTAL[0]:,}회 + 다음 묶음 {n_lp:,}회, 경과 {time.time() - T0:.0f}s")
            print(stat["stopped"], flush=True)
            break
        LP_TOTAL[0] += n_lp
        t = time.time()
        alpha = RB._alpha_many(entries, cfg) if entries else np.zeros((0, 14))
        stat["lp_time"] += time.time() - t
        stat["lp"] += n_lp
        ofs = 0
        for (kind, obj), (_s, combos, _m) in zip(slots, entries):
            a = alpha[ofs:ofs + len(combos)]
            ofs += len(combos)
            if kind == "part":
                obj["ma"] = a.min(1)
            else:
                diff = abs(float(a.min()) - obj["min_alpha"])
                v2["n"] += 1
                v2["same"] += int(diff < 1e-6)
                v2["max_diff"] = max(v2["max_diff"], diff)
        for g in ctxs:
            stat["n_grasp"] += 1
            stat["gids"].append(g["gid"])
            for p in g["parts"]:
                res = score_rules(p, g["samples"])
                if smoke:
                    check_v3(g, p, res, cfg, v3)
                recs[p["key"]] = res
        print(f"  후보 {grid_size}개: 파지 {min(c0 + CHUNK, len(grasps))}/{len(grasps)}, 파티션 {len(recs)}, "
              f"선형계획 누적 {stat['lp']:,}회 ({time.time() - t_run:.0f}s)", flush=True)
    stat["total_time"] = time.time() - t_run
    return dict(recs=recs, v1=v1, v2=v2, v3=v3, stat=stat, grid_size=grid_size)


def pct(x):
    return f"{100 * float(x):5.1f}%"


def q3(x):
    return "/".join(f"{v:g}" for v in np.percentile(x, [10, 50, 90]).round(2))


def summarize(out, stored, e001_best, gids, label):
    recs = out["recs"]
    keys = [k for k in stored if k[0] in gids]
    st = np.array([stored[k]["min_alpha"] for k in keys])
    sb = np.array([e001_best.get(k, stored[k]["min_alpha"]) for k in keys])
    print(f"\n===== {label}: 저장된 정답이 있는 레코드 {len(keys)}개, 나열된 파티션 {len(recs)}개 =====")
    summ = {}
    for r in RULES:
        A = [recs.get(k, {}).get(r) for k in keys]
        new = np.array([a["alpha"] if a and a["alpha"] is not None else np.nan for a in A])
        place = np.isfinite(new)
        newf = np.nan_to_num(new, nan=-1.0)
        ok = newf >= PASS_RATIO * st - 1e-12
        okb = newf >= PASS_RATIO * sb - 1e-12
        strong = st > 0.02
        v = place & (st > 1e-9)
        ratio, ratio_b = new[v] / st[v], new[v] / sb[v]
        T = [a for a in A if a and a["alpha"] is not None and a["alpha"] > 1e-9]
        allr = list(recs.values())
        ncand = np.array([x[r]["n_cand"] for x in allr]).reshape(-1)
        ncombo = np.array([x[r]["n_combo"] for x in allr], dtype=np.float64)
        near_ok = np.array([np.nan_to_num(a["near_alpha"], nan=-1.0) >= PASS_RATIO * a["alpha"] - 1e-12 for a in T])
        near_same = np.array([a["near_same"] for a in T])
        prox_same = np.array([a["prox_same"] for a in T])
        multi = np.array([[n >= 2 for n in a["n_cand"]] for a in T])
        s = dict(
            primary=float(ok.mean()), primary_strong=float(ok[strong].mean()), primary_vs_best_seed=float(okb.mean()),
            unplaceable_rec=float(1 - place.mean()), ratio_median=float(np.median(ratio)),
            ratio_ge095=float((ratio >= 0.95).mean()), ratio_gt1=float((ratio > 1 + 1e-9).mean()),
            ratio_b_median=float(np.median(ratio_b)),
            tie_other=float(np.mean([a["has_other"] for a in T])),
            tie_other_diff2=float(np.mean([a["other_diff2"] for a in T])),
            tie_n_median=float(np.median([a["n_near"] for a in T])),
            n_tie_rec=len(T), lp_per_grasp=float(ncombo.sum() / max(out["stat"]["n_grasp"], 1)))
        summ[r] = s
        print(f"[규칙 {r}: {RULE_TEXT[r]}]")
        print(f"  주 지표 — 새 정답 α 가 저장된 정답 α 의 {PASS_RATIO}배 이상: {pct(s['primary'])} "
              f"(저장된 α 가 0.02 를 넘는 레코드만 {pct(s['primary_strong'])}, 세 씨앗 중 최고 α 기준 {pct(s['primary_vs_best_seed'])}) "
              f"| 배치 불가 레코드 {pct(s['unplaceable_rec'])}")
        print(f"  α 비율(새 ÷ 저장) — 중앙값 {s['ratio_median']:.3f} | 0.95배 이상 {pct(s['ratio_ge095'])} | 1배 초과 {pct(s['ratio_gt1'])} "
              f"| 세 씨앗 중 최고 α 대비 중앙값 {s['ratio_b_median']:.3f}")
        print(f"  α 가 비슷한 조합 (레코드 {len(T)}개) — 최선의 {TIE}배 이상인 조합 수 중앙값 {s['tie_n_median']:.0f} "
              f"| 최선 말고도 있는 레코드 {pct(s['tie_other'])} | 그중 두 팔 이상에서 다른 점을 쓰는 조합이 있는 레코드 {pct(s['tie_other_diff2'])}")
        print(f"  단순 규칙 (가장 가까운 후보) — 팔 하나 일치 {pct(near_same.mean())} (후보 2개 이상인 팔만 {pct(near_same[multi].mean()) if multi.any() else '    -'}) "
              f"| 세 팔 모두 일치 {pct(near_same.all(1).mean())} | 그 조합의 α 가 최선의 {PASS_RATIO}배 이상 {pct(near_ok.mean())} "
              f"| 우연 수준(후보 수의 역수 평균) {pct(np.mean([a['chance'] for a in T]))} "
              f"| 그 조합이 간격 조건에 걸려 놓을 수 없는 레코드 {pct(np.mean([not np.isfinite(a['near_alpha']) for a in T]))}")
        print(f"  단순 규칙 (값싼 점수 1등 조합) — 팔 하나 일치 {pct(prox_same.mean())} | 세 팔 모두 일치 {pct(prox_same.all(1).mean())} "
              f"| 그 조합의 α 가 최선의 {PASS_RATIO}배 이상 {pct(np.mean([a['prox_alpha'] >= PASS_RATIO * a['alpha'] - 1e-12 for a in T]))}")
        dr = np.concatenate([a["dist_r"] for a in T])
        dh = np.concatenate([a["dist_rho"] for a in T])
        print(f"  새 정답 앵커와 접촉 중심의 거리 — 물체 반경의 배수 중앙값 {np.median(dr):.2f}, 상위 10% {np.percentile(dr, 90):.2f} "
              f"| 범위 반경의 배수 중앙값 {np.median(dh):.2f}, 상위 10% {np.percentile(dh, 90):.2f}, 범위 밖(1배 초과)인 앵커 {pct((dh > 1 + 1e-9).mean())}")
        print(f"  후보 수(팔마다, 하위10%/중앙값/상위10%) {q3(ncand)} | 조합 수(파티션마다) 중앙값 {np.median(ncombo):.0f}, 평균 {ncombo.mean():.0f} "
              f"| 조합이 없는 파티션 {pct((ncombo == 0).mean())}")
    return summ


def main():
    smoke = "--smoke" in sys.argv
    grasps = G.load_grasps()
    if smoke:
        grasps = grasps[:8]
    gids = {g for g, _, _ in grasps}
    stored = G.load_stored(gids)
    e001 = pickle.load(open(EXP001_PKL, "rb"))
    e001_best = {}
    for s in (0, 1, 2):
        for k, v in e001["seeds"][s].items():
            if v["n"] == N:
                e001_best[k] = max(e001_best.get(k, 0.0), v["min_alpha"])
    print(f"파지 {len(grasps)}개, 팔 {N}대, 저장된 정답 레코드 {len(stored)}개", flush=True)

    out128 = run(grasps, 128, stored, smoke)
    if out128["stat"]["stopped"]:
        print("후보 128개 채점이 중단 조건에 걸렸습니다 — 결과를 해석하지 않고 멈춥니다.", flush=True)
        sys.exit(3)
    v1, v2, v3 = out128["v1"], out128["v2"], out128["v3"]
    r1 = v1["same"] / max(v1["n"], 1)
    r2 = v2["same"] / max(v2["n"] + v2["unmatched"], 1)
    print(f"[검산 1] 다시 고른 128개가 조각 캐시와 같은 물체 {v1['same']}/{v1['n']} = {100 * r1:.2f}%")
    print(f"[검산 2] 저장된 정답의 α 를 다시 계산해 저장값과 같은 레코드 {v2['same']}/{v2['n'] + v2['unmatched']} = {100 * r2:.2f}% "
          f"(표면점을 못 찾은 레코드 {v2['unmatched']}, 최대 차이 {v2['max_diff']:.2e})")
    passed = r1 >= VERIFY_MIN and r2 >= VERIFY_MIN
    if smoke:
        ok3 = v3["count_same"] == v3["n"] and v3["alpha_same"] == v3["alpha_n"]
        print(f"[검산 3] 조합 수가 따로 센 값과 같은 파티션 {v3['count_same']}/{v3['n']}, "
              f"최선 α 가 따로 계산한 값과 같은 파티션 {v3['alpha_same']}/{v3['alpha_n']}")
        passed = passed and ok3
    print(f"[검산 판정] {'통과' if passed else '불통과'}", flush=True)
    if not passed:
        print("검산 불통과 — 결과를 해석하지 않고 멈춥니다 (계획서의 중단 조건).", flush=True)
        sys.exit(2)

    summ128 = summarize(out128, stored, e001_best, gids, "후보 128개, 파지 전체")
    sub = grasps[::SUB_256] if not smoke else grasps[:2]
    sub_gids = {g for g, _, _ in sub}
    out256 = run(sub, 256, {k: v for k, v in stored.items() if k[0] in sub_gids}, False)
    if out256["stat"]["stopped"]:                      # 돌린 파지까지만 요약한다
        sub_gids = set(out256["stat"]["gids"])
        sub = [g for g in sub if g[0] in sub_gids]
        print(f"후보 256개 비교는 중단 조건 때문에 파지 {len(sub)}개까지만 돌렸습니다.", flush=True)
    sub128 = dict(out128, recs={k: v for k, v in out128["recs"].items() if k[0] in sub_gids},
                  stat=dict(out128["stat"], n_grasp=len(sub)))
    summ128_sub = summarize(sub128, stored, e001_best, sub_gids, f"후보 128개, 256개와 같은 파지 {len(sub)}개")
    summ256 = summarize(out256, stored, e001_best, sub_gids, f"후보 256개, 파지 {len(sub)}개")

    # ── 생성 속도 (추정) ──
    print("\n[정답 생성 속도 — 추정]")
    for name, out, summ in (("128개", out128, summ128), ("256개", out256, summ256)):
        s = out["stat"]
        rate = s["lp"] / max(s["lp_time"], 1e-9)
        prep = (s["total_time"] - s["lp_time"]) / max(s["n_grasp"], 1)
        print(f"  후보 {name}: 파지 {s['n_grasp']}개, 선형계획 {s['lp']:,}회(네 규칙의 합집합), 채점 {s['lp_time']:.0f}s "
              f"(초당 {rate:,.0f}회), 전체 {s['total_time']:.0f}s, 채점 외 파지당 {prep:.2f}s")
        for r in RULES:
            h = (summ[r]["lp_per_grasp"] / rate + prep) * N_MANIFEST / 3600
            print(f"    규칙 {r}: 파지당 선형계획 {summ[r]['lp_per_grasp']:,.0f}회 → 팔 3대만, 파지 {N_MANIFEST:,}개 전량에 약 {h:.1f}시간")

    # ── 판정 (계획서 고정 기준) ──
    prim = {r: summ128[r]["primary"] for r in RULES}
    print(f"\n[주 지표] 후보 128개, 새 정답 α 가 저장된 정답 α 의 {PASS_RATIO}배 이상인 레코드: "
          + ", ".join(f"규칙 {r} {100 * prim[r]:.1f}%" for r in RULES))
    if prim["가"] >= GO:
        verdict = "규칙 가가 90% 이상 — 128개와 규칙 가로 1단계를 진행할 수 있습니다"
    else:
        cand = [r for r in ("나", "다", "라") if prim[r] >= GO]
        if cand:
            verdict = f"규칙 가는 90% 미만, 90% 이상인 규칙 중 범위를 가장 덜 넓히는 것은 규칙 {cand[0]} — 규칙 {cand[0]}로 진행을 제안합니다"
        elif all(prim[r] < NOGO for r in RULES):
            verdict = "네 규칙 모두 75% 미만 — 128개는 부적합합니다 (256개 결과를 보고 정합니다)"
        else:
            verdict = "어느 규칙도 90% 에 못 미치고 일부는 75% 이상 — 수치를 보고 연구자가 정합니다"
    print(f"[판정] {verdict}")
    tie = {r: summ128[r]["tie_other"] for r in RULES}
    print("[α 가 비슷한 조합] 최선 말고도 0.95배 이상인 조합이 있는 레코드: "
          + ", ".join(f"규칙 {r} {100 * tie[r]:.1f}%" for r in RULES)
          + f" (읽는 기준: {100 * TIE_MANY:.0f}% 초과면 정답 묶음을 제안, {100 * TIE_FEW:.0f}% 미만이면 정답 하나로 충분)")
    if smoke:
        print("연기 시험 끝 — 결과 파일은 쓰지 않습니다.")
        return
    slim = lambda o: dict(recs=o["recs"], v1=o["v1"], v2=o["v2"], stat=o["stat"], grid_size=o["grid_size"])
    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(grasps=grasps, sub=sub, out128=slim(out128), out256=slim(out256),
                         summ128=summ128, summ128_sub=summ128_sub, summ256=summ256, primary=prim, verdict=verdict,
                         config=dict(RULES=RULES, MARGIN_REL=MARGIN_REL, KMIN=KMIN, PASS_RATIO=PASS_RATIO, GO=GO,
                                     NOGO=NOGO, TIE=TIE, TIE_MANY=TIE_MANY, TIE_FEW=TIE_FEW, SUB_256=SUB_256)), f)
    print(f"저장 {OUT_PKL} ({OUT_PKL.stat().st_size / 1e6:.1f} MB), 총 {time.time() - T0:.0f}s, 선형계획 누적 {LP_TOTAL[0]:,}회")


if __name__ == "__main__":
    main()
