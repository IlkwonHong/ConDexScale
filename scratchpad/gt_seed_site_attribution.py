"""정답 탐색에서 난수 씨앗이 들어가는 세 자리 중 어느 것이 정답을 옮기는지 가르는 측정 (EXP-002, 2026-11-13).

씨앗이 들어가는 자리 (contraction_gt/search.py · runner_batched.py · surface.py 기준):
  (가) cand — 팔마다 범위 안 표면점을 32개로 줄일 때의 첫 점      (candidates_in_region, seed=cfg.seed)
  (나) rand — 값싼 점수 상위 128개 조합에 더하는 무작위 조합 64개 (_top_combos, seed=cfg.seed)
  (다) ref  — 다듬기 후보가 10개를 넘을 때 10개로 줄이는 첫 점    (candidates_in_region, seed=cfg.seed + 1)

탐색 코드는 수정하지 않는다. EXP-001 스크립트(gt_seed_consistency.py)의 run_seed 를 cfg.seed=0 으로 그대로 부르고,
runner_batched 가 부르는 두 함수에 넘어가는 씨앗 인자만 이 스크립트가 자리별로 바꿔 넘긴다.

검산 (둘 다 통과해야 자리별 결과를 해석한다):
  1. 세 자리 모두 씨앗 0 → 저장된 정답(search_records)과 같은 점이어야 한다.
  2. 세 자리 모두 씨앗 1 → EXP-001 의 씨앗 1 결과(gt_seed_consistency.pkl)와 같은 점이어야 한다.
     (씨앗이 들어가는 자리가 위 세 곳뿐인지 확인)

출력: scratchpad/gt_seed_site_attribution.pkl + 요약 출력
실행: ./venv/bin/python scratchpad/gt_seed_site_attribution.py [--smoke]
계획: harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-002-gt-seed-site-attribution/plan.md
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

import gt_seed_consistency as G                     # noqa: E402
import contraction_gt.runner_batched as RB          # noqa: E402

OUT_PKL = ROOT / "scratchpad/gt_seed_site_attribution.pkl"
EXP001_PKL = G.OUT_PKL
N = 3                                # 팔 3대만 (계획서)
SITE_SEEDS = (1, 2)                  # 자리별로 바꿔 볼 씨앗 값 (기준 0)
SITES = (("cand", "(가) 팔마다 후보 32점을 고르는 자리만 바꿈"),
         ("rand", "(나) 무작위 조합 64개를 뽑는 자리만 바꿈"),
         ("ref", "(다) 다듬기 후보를 고르는 자리만 바꿈"))
MAJOR_FRAC, MINOR_FRAC = 0.5, 0.1    # 읽는 기준 (계획서 고정값): 전체 불일치율의 절반 이상 / 10분의 1 미만

G.N_LIST = (N,)

# ── 씨앗 인자만 자리별로 바꿔 넘기는 덮개 ──
SITE = dict(cand=0, rand=0, ref=0, phase="coarse")
CALLS = dict(coarse=0, refine=0, top_rand=0, top_plain=0)
_orig_cand, _orig_top, _orig_refine = RB.candidates_in_region, RB._top_combos, G._refine_set


def _cand(samples, center, rho, ref_normal=None, k_max=32, normal_dot_min=0.1, seed=0):
    phase = SITE["phase"]
    # run_seed 는 cfg.seed=0 으로 부르므로 거친 단계는 0, 다듬기는 0+1 이 들어와야 한다
    assert seed == (0 if phase == "coarse" else 1), f"호출 자리와 단계가 어긋남: {phase}, seed={seed}"
    CALLS[phase] += 1
    s = SITE["cand"] if phase == "coarse" else SITE["ref"] + 1
    return _orig_cand(samples, center, rho, ref_normal=ref_normal, k_max=k_max,
                      normal_dot_min=normal_dot_min, seed=s)


def _top(cand_sets, samples, top_c, min_sep=0.0, n_random=0, seed=0):
    CALLS["top_rand" if n_random > 0 else "top_plain"] += 1
    return _orig_top(cand_sets, samples, top_c, min_sep=min_sep, n_random=n_random, seed=SITE["rand"])


def _refine(samples, job, best_combo, cfg):
    SITE["phase"] = "refine"
    try:
        return _orig_refine(samples, job, best_combo, cfg)
    finally:
        SITE["phase"] = "coarse"


RB.candidates_in_region, RB._top_combos, G._refine_set = _cand, _top, _refine


def run_cond(grasps, cand, rand, ref):
    SITE.update(cand=cand, rand=rand, ref=ref, phase="coarse")
    print(f"[조건] 후보 32점 씨앗 {cand}, 무작위 조합 씨앗 {rand}, 다듬기 씨앗 {ref}(실제 값 {ref + 1})", flush=True)
    out, n_fail = G.run_seed(grasps, 0)
    return out, n_fail


def same_rate(a, b, label):
    """두 결과에서 앵커 위치가 같은 레코드의 비율 (분모 = 양쪽 키의 합집합)."""
    keys = set(a) | set(b)
    both = [k for k in keys if k in a and k in b]
    same = [k for k in both if a[k]["n"] == b[k]["n"] and np.abs(a[k]["pos"] - b[k]["pos"]).max() < G.SAME_TOL]
    da = max((abs(a[k]["min_alpha"] - b[k]["min_alpha"]) for k in same), default=0.0)
    rate = len(same) / max(len(keys), 1)
    print(f"[{label}] 앵커 위치가 같은 레코드 {len(same)}/{len(keys)} = {100 * rate:.2f}% "
          f"(한쪽에만 있는 키 {len(keys) - len(both)}, 같은 레코드의 α 최대 차이 {da:.2e})", flush=True)
    return dict(rate=rate, n_same=len(same), n_union=len(keys), max_alpha_diff=float(da))


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    grasps = G.load_grasps()
    if smoke:
        grasps = grasps[:8]
    gids = {g for g, _, _ in grasps}
    stored = G.load_stored(gids)
    print(f"파지 {len(grasps)}개, 팔 {N}대, 저장된 정답 레코드 {len(stored)}개", flush=True)
    e001 = pickle.load(open(EXP001_PKL, "rb"))
    e001_s1 = {k: v for k, v in e001["seeds"][1].items() if v["n"] == N and k[0] in gids}
    full_dis = [1.0 - e001["summary"][(N, s, "전체 레코드")]["all_cell"] for s in (1, 2)]
    full_mean = float(np.mean(full_dis))

    res, fails, checks = {}, {}, {}
    res[(0, 0, 0)], fails[(0, 0, 0)] = run_cond(grasps, 0, 0, 0)
    checks["v1"] = same_rate(stored, res[(0, 0, 0)], "검산 1: 세 자리 모두 씨앗 0 대 저장된 정답")
    res[(1, 1, 1)], fails[(1, 1, 1)] = run_cond(grasps, 1, 1, 1)
    checks["v2"] = same_rate(e001_s1, res[(1, 1, 1)], "검산 2: 세 자리 모두 씨앗 1 대 EXP-001 의 씨앗 1 결과")
    passed = checks["v1"]["rate"] >= G.VERIFY_MIN and checks["v2"]["rate"] >= G.VERIFY_MIN
    print(f"[검산 판정] {'통과' if passed else '불통과'} (기준 둘 다 {100 * G.VERIFY_MIN:.0f}% 이상), "
          f"덮개 호출 수 {CALLS}", flush=True)
    if smoke or not passed:
        if not passed:
            print("검산 불통과 — 자리별 조건은 돌리지 않고 멈춥니다 (계획서의 중단 조건).", flush=True)
        sys.exit(0 if passed else 2)

    for site, _lab in SITES:
        for s in SITE_SEEDS:
            c = dict(cand=0, rand=0, ref=0)
            c[site] = s
            key = (c["cand"], c["rand"], c["ref"])
            res[key], fails[key] = run_cond(grasps, *key)

    is_pass = lambda a: a["verdict"] == "pass"
    summary, site_dis = {}, {}
    print(f"\n===== 팔 {N}대, 기준 = 저장된 정답(세 자리 모두 씨앗 0) =====")
    for site, lab in SITES:
        dis = []
        for s in SITE_SEEDS:
            c = dict(cand=0, rand=0, ref=0)
            c[site] = s
            key = (c["cand"], c["rand"], c["ref"])
            print(f"[{lab}, 씨앗 {s}]")
            for sub, keep in (("전체 레코드", None), ("기준 정답이 시뮬레이션에 합격한 레코드", is_pass)):
                r = G.agreement(stored, res[key], N, keep)
                summary[(site, s, sub)] = r
                G.show(sub, r)
            dis.append(1.0 - summary[(site, s, "전체 레코드")]["all_cell"])
        site_dis[site] = dis
    print("[세 자리 모두 씨앗 1 (검산 2 의 조건)]")
    r = G.agreement(stored, res[(1, 1, 1)], N)
    summary[("all", 1, "전체 레코드")] = r
    G.show("전체 레코드", r)

    print(f"\n[주 지표] 세 팔 모두 0.078배 이내로 일치하지 않는 비율(불일치율), 팔 {N}대")
    print(f"  세 자리 모두 바꿈 (EXP-001): 씨앗 1 = {100 * full_dis[0]:.1f}%, 씨앗 2 = {100 * full_dis[1]:.1f}%, "
          f"평균 {100 * full_mean:.1f}% → 절반 {100 * MAJOR_FRAC * full_mean:.1f}%, 10분의 1 {100 * MINOR_FRAC * full_mean:.1f}%")
    verdicts = {}
    for site, lab in SITES:
        d = site_dis[site]
        m = float(np.mean(d))
        v = ("주된 원인" if m >= MAJOR_FRAC * full_mean else
             "거의 영향 없음" if m < MINOR_FRAC * full_mean else "부분 원인")
        verdicts[site] = v
        print(f"  {lab}: 씨앗 1 = {100 * d[0]:.1f}%, 씨앗 2 = {100 * d[1]:.1f}%, 평균 {100 * m:.1f}% → {v}")
    print(f"탐색 중 불러오기 실패 파지 수(조건별): {fails}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(grasps=grasps, stored=stored, conds=res, checks=checks, summary=summary,
                         site_dis=site_dis, full_dis=full_dis, verdicts=verdicts, fails=fails, calls=CALLS,
                         config=dict(N=N, SITE_SEEDS=SITE_SEEDS, MAJOR_FRAC=MAJOR_FRAC, MINOR_FRAC=MINOR_FRAC,
                                     CELL=G.CELL, BASIN=G.BASIN, VERIFY_MIN=G.VERIFY_MIN)), f)
    print(f"저장 {OUT_PKL} ({OUT_PKL.stat().st_size / 1e6:.1f} MB), 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
