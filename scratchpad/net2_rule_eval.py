"""net2 대체 손 규칙 평가 (2026-11-07, 블로그 (가) 실험).

같은 검증 파지 240개(팔 3대), 같은 파티션(net1 상위 3개), 같은 후보 범위를 쓰고
"팔마다 상위 8점 고르기" 한 칸만 바꾼다. 뒤 과정은 net2_kshot_decompose.py 와 같다:
  파티션마다 8^3 조합 α → min α 상위 2개 조합을 이웃 7점 다듬기 → 후보 최대 6개 → α 순서로 시뮬.
  성공 = 시뮬 합격 ∧ min α ≥ 0.8 × GT 최선 합격 α.

변형:
  net2          저장된 net2 상위 8점 그대로 (재현 대조: 1발 154, 6발 190 이어야 함)
  rand0~2       후보 범위 안 무작위 8점 (바닥선)
  rule          손 규칙 ①~⑤ 전부
  rule-prox / rule-oppose / rule-cover / rule-flat   항 하나씩 뺌
  rule-diverse  ⑤ 대신 그냥 점수 상위 8점
  rule-lpcover  ③ 을 닫힌식 대신 선형계획 α 로 (다른 팔은 대표점에 고정) — 닫힌식 근사의 손실 확인용
  mix           손 규칙 4점 + net2 4점

시뮬 판정은 (파지, 조합) 별로 scratchpad/net2_rule_simcache.pkl 에 쌓아 두고 재사용한다.
실행: ./venv/bin/python scratchpad/net2_rule_eval.py [변형 ...]   (spawn + __main__ 가드 필수)
"""
import itertools
import math
import pickle
import sys
import time
import zlib
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from net2_viz_mine import refine, build_sim_job, MU_T, PAD_REL, DEV                   # noqa: E402
from net2_rule_score import (geometry, rule_terms, group_reps, combine, select_topk,   # noqa: E402
                             select_diverse, W_DEFAULT, TERMS, K_PICK)
from contraction_gt.search import _batched_alpha, mu_tor_from_pad                       # noqa: E402
from contraction_gt.surface import SurfaceSamples                                        # noqa: E402
from contraction_gt.gate_runner import _worker                                           # noqa: E402

K_PER_PART = 2
SIM_WORKERS = 12
IN_PKL = ROOT / "scratchpad/net2_viz_cases.pkl"
REF_PKL = ROOT / "scratchpad/net2_kshot_cases.pkl"
CACHE_PKL = ROOT / "scratchpad/net2_rule_simcache.pkl"
OUT_PKL = ROOT / "scratchpad/net2_rule_eval.pkl"
ALL = ["net2", "rand0", "rand1", "rand2", "rule", "rule-prox", "rule-oppose", "rule-cover",
       "rule-flat", "rule-diverse", "rule-lpcover", "mix", "net2-diverse"]


def build_part_cands(samples, mu_tor, top8):
    """net2_kshot_decompose.build_candidates 의 파티션 1개 몫 (같은 논리)."""
    combos = np.array(list(itertools.product(*top8)), np.int64)
    a = _batched_alpha(samples, combos, MU_T, np.full(len(combos), mu_tor), DEV)
    order = np.argsort(-a.min(1))
    seen, out = set(), []
    for rank, j in enumerate(order):
        if len(out) >= K_PER_PART:
            break
        combo_r, alpha_r = refine(samples, combos[j], mu_tor)
        if alpha_r.min() < a[j].min():
            combo_r, alpha_r = combos[j], a[j]
        key = tuple(int(v) for v in combo_r)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(lp_rank=int(rank), combo=np.asarray(combo_r, np.int32),
                        alpha14=np.asarray(alpha_r, np.float32), min_alpha=float(np.min(alpha_r))))
    return out


def lp_cover_raw(samples, mu_tor, masks, reps):
    """③ 의 선형계획 판: 그룹 g 후보점 p 를 넣고 다른 그룹은 대표점에 둔 조합의 min α."""
    G, P = masks.shape
    raw = np.full((G, P), np.nan)
    for g in range(G):
        cand = np.nonzero(masks[g])[0]
        combos = np.tile(reps, (len(cand), 1)).astype(np.int64)
        combos[:, g] = cand
        raw[g, cand] = _batched_alpha(samples, combos, MU_T, np.full(len(cand), mu_tor), DEV).min(1)
    return raw


class GraspCtx:
    """파지 1개의 기하와 파티션별 항 (필요할 때만 계산)."""

    def __init__(self, m):
        self.m = m
        self.samples = SurfaceSamples(points=m["pts"].astype(np.float64), normals=m["nrm"].astype(np.float64),
                                      r_obj=m["r_obj"], com=m["com"].astype(np.float64))
        self.mu_tor = mu_tor_from_pad(PAD_REL * m["r_obj"], MU_T)
        self.x, self.n, self.dist = geometry(m["pts"], m["nrm"], m["com"], m["r_obj"])
        self._terms, self._lp = {}, {}

    def terms(self, pi):
        if pi not in self._terms:
            p = self.m["parts"][pi]
            self._terms[pi] = rule_terms(self.x, self.n, self.dist, p["masks"], self.mu_tor / self.m["r_obj"])
        return self._terms[pi]

    def terms_lp(self, pi):
        if pi not in self._lp:
            p = self.m["parts"][pi]
            reps, _ = group_reps(self.x, self.n, self.dist, p["masks"])
            raw = lp_cover_raw(self.samples, self.mu_tor, p["masks"], reps)
            self._lp[pi] = rule_terms(self.x, self.n, self.dist, p["masks"], self.mu_tor / self.m["r_obj"],
                                      cover_raw_override=raw)
        return self._lp[pi]


def variant_top8(v, ctx, pi):
    m, p = ctx.m, ctx.m["parts"][pi]
    masks = p["masks"]
    G = masks.shape[0]
    if v == "net2":
        return [np.asarray(t, np.int64) for t in p["top8"]]
    if v == "net2-diverse":          # net2 점수 그대로 + ⑤ 흩어서 고르기 (2026-11-08)
        heat = np.where(masks, p["heat"].astype(np.float32), -np.inf)
        return [select_diverse(heat[g], ctx.dist) for g in range(G)]
    if v.startswith("rand"):
        s = int(v[4:])
        out = []
        for g in range(G):
            idx = np.nonzero(masks[g])[0]
            seed = zlib.crc32(f"{m['gid']}|{pi}|{g}|{s}".encode())
            out.append(np.random.default_rng(seed).choice(idx, min(K_PICK, len(idx)), replace=False))
        return out
    if v == "mix":
        T, _ = ctx.terms(pi)
        sc = combine(T, masks, W_DEFAULT)
        heat = np.where(masks, p["heat"].astype(np.float32), -np.inf)
        out = []
        for g in range(G):
            k = min(K_PICK, int(masks[g].sum()))
            chosen = list(select_diverse(sc[g], ctx.dist, k=min(4, k)))
            for i in list(p["top8"][g]) + list(np.argsort(-heat[g], kind="stable")):
                if len(chosen) >= k:
                    break
                if np.isfinite(heat[g][i]) and int(i) not in chosen:
                    chosen.append(int(i))
            out.append(np.array(chosen, np.int64))
        return out
    # 손 규칙 계열
    T, _ = ctx.terms_lp(pi) if v == "rule-lpcover" else ctx.terms(pi)
    w = dict(W_DEFAULT)
    if v in ("rule-prox", "rule-oppose", "rule-cover", "rule-flat"):
        w[v.split("-")[1]] = 0.0
    sc = combine(T, masks, w)
    pick = (lambda s: select_topk(s)) if v == "rule-diverse" else (lambda s: select_diverse(s, ctx.dist))
    return [pick(sc[g]) for g in range(G)]


def binom_p(a, b):
    """짝지은 비교: 한쪽만 성공 a, 다른 쪽만 성공 b → 양쪽 정확 이항 검정 p."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    # 인자: 변형 이름들 + 선택적으로 in=…/out=…/cache=… (다른 파지 묶음용, 2026-11-08)
    kv = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    variants = [t for t in sys.argv[1:] if "=" not in t] or ALL
    in_pkl = ROOT / kv["in"] if "in" in kv else IN_PKL
    out_pkl = ROOT / kv["out"] if "out" in kv else OUT_PKL
    cache_pkl = ROOT / kv["cache"] if "cache" in kv else CACHE_PKL
    t0 = time.time()
    mined = pickle.load(open(in_pkl, "rb"))
    cache = pickle.load(open(cache_pkl, "rb")) if cache_pkl.exists() else {}
    print(f"파지 {len(mined)}개, 변형 {variants}, 시뮬 캐시 {len(cache)}개", flush=True)

    res = {v: [] for v in variants}
    part_memo = {}
    cover_rho = []
    for gi, m in enumerate(mined):
        ctx = GraspCtx(m)
        for v in variants:
            cands = []
            for pi in range(len(m["parts"])):
                top8 = variant_top8(v, ctx, pi)
                key = (gi, pi, tuple(tuple(int(i) for i in t) for t in top8))
                if key not in part_memo:
                    part_memo[key] = build_part_cands(ctx.samples, ctx.mu_tor, top8)
                cands += [dict(c, part=pi) for c in part_memo[key]]
            cands.sort(key=lambda c: -c["min_alpha"])
            res[v].append(dict(gid=m["gid"], cands=cands))
        if "rule-lpcover" in variants and "rule" in variants:       # 닫힌식 ③ 대 선형계획 ③ 순위 상관
            for pi in range(len(m["parts"])):
                a_ = ctx.terms(pi)[1]["cover_raw"]
                b_ = ctx.terms_lp(pi)[1]["cover_raw"]
                for g in range(a_.shape[0]):
                    mk = m["parts"][pi]["masks"][g]
                    if mk.sum() >= 5:
                        rk = lambda z: np.argsort(np.argsort(z))
                        cover_rho.append(np.corrcoef(rk(a_[g, mk]), rk(b_[g, mk]))[0, 1])
        if (gi + 1) % 40 == 0:
            print(f"  후보 구성 {gi + 1}/{len(mined)} ({time.time() - t0:.0f}s)", flush=True)

    # ── 시뮬 (캐시에 없는 (파지, 조합)만) ──
    todo = {}
    for v in variants:
        for gi, r in enumerate(res[v]):
            for c in r["cands"]:
                k = (r["gid"], tuple(int(i) for i in c["combo"]))
                if k not in cache and k not in todo:
                    todo[k] = (gi, c)
    keys = list(todo)
    jobs = []
    for j, k in enumerate(keys):
        gi, c = todo[k]
        m = mined[gi]
        s = SurfaceSamples(points=m["pts"].astype(np.float64), normals=m["nrm"].astype(np.float64),
                           r_obj=m["r_obj"], com=m["com"].astype(np.float64))
        jobs.append(build_sim_job(j, m["ds_oid"][0], m["ds_oid"][1], m["ts"], m["r_obj"], s,
                                  c["combo"], c["alpha14"].astype(np.float64),
                                  mu_tor_from_pad(PAD_REL * m["r_obj"], MU_T)))
    print(f"시뮬 {len(jobs)}개 (캐시 재사용 제외) ({time.time() - t0:.0f}s)", flush=True)
    if jobs:
        done = 0
        with get_context("spawn").Pool(SIM_WORKERS) as pool:
            for j, verdict, _nu in pool.imap_unordered(_worker, jobs, chunksize=2):
                cache[keys[j]] = verdict
                done += 1
                if done % 500 == 0:
                    print(f"  시뮬 {done}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
                    pickle.dump(cache, open(cache_pkl, "wb"))
        pickle.dump(cache, open(cache_pkl, "wb"))

    for v in variants:
        for gi, r in enumerate(res[v]):
            gb = mined[gi]["gt_best_pass"]
            for c in r["cands"]:
                c["verdict"] = cache[(r["gid"], tuple(int(i) for i in c["combo"]))]
                c["success"] = c["verdict"] == "pass" and c["min_alpha"] >= 0.8 * gb

    # ── 재현 대조 ──
    if "net2" in variants and REF_PKL.exists() and in_pkl == IN_PKL:
        ref = pickle.load(open(REF_PKL, "rb"))
        mis_c = sum(any(not np.array_equal(a["combo"], b["combo"]) for a, b in zip(r["cands"], q["cands"]))
                    or len(r["cands"]) != len(q["cands"]) for r, q in zip(res["net2"], ref))
        mis_v = sum(any(a["verdict"] != b["verdict"] for a, b in zip(r["cands"], q["cands"]))
                    for r, q in zip(res["net2"], ref))
        print(f"\n[재현 대조] net2 후보 조합 불일치 파지 {mis_c}/240, 시뮬 판정 불일치 파지 {mis_v}/240")

    # ── 보고 ──
    N = len(mined)
    base = res.get("net2")
    print(f"\n== 변형별 성적 (파지 {N}개, 팔 3대, 입력 {in_pkl.name}) ==")
    for v in variants:
        R = res[v]
        ks = [sum(any(c["success"] for c in r["cands"][:K]) for r in R) for K in range(1, 7)]
        pass1 = sum(r["cands"][0]["verdict"] == "pass" for r in R)
        ratio = np.array([r["cands"][0]["min_alpha"] / mined[i]["gt_best_pass"] for i, r in enumerate(R)])
        line = (f"{v:14s} 1발 {ks[0]:3d} ({100 * ks[0] / N:4.1f}%)  3발 {100 * ks[2] / N:4.1f}%  "
                f"6발 {ks[5]:3d} ({100 * ks[5] / N:4.1f}%) | 1발 시뮬 합격만 {100 * pass1 / N:4.1f}% | "
                f"1등 후보 α÷GT최선 중앙값 {np.median(ratio):.2f}, 0.8 이상 {np.mean(ratio >= 0.8):.2f}")
        if base is not None and v != "net2":
            for K, lab in ((1, "1발"), (6, "6발")):
                s_v = [any(c["success"] for c in r["cands"][:K]) for r in R]
                s_b = [any(c["success"] for c in r["cands"][:K]) for r in base]
                a = sum(x and not y for x, y in zip(s_v, s_b))
                b = sum(y and not x for x, y in zip(s_v, s_b))
                line += f" | {lab} 이쪽만 {a}, net2만 {b}, p={binom_p(a, b):.3f}"
        print(line)
    if cover_rho:
        print(f"\n③ 닫힌식 원값 대 선형계획 α 순위 상관 (후보 범위 {len(cover_rho)}개): "
              f"중앙값 {np.median(cover_rho):.2f}, 25/75% {np.percentile(cover_rho, 25):.2f}/{np.percentile(cover_rho, 75):.2f}")
    print("판정 분포:", {v: dict(Counter(c["verdict"] for r in res[v] for c in r["cands"])) for v in variants})
    old = pickle.load(open(out_pkl, "rb")) if out_pkl.exists() else {}
    old.update(res)
    pickle.dump(old, open(out_pkl, "wb"))
    print(f"저장 {out_pkl}, 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
