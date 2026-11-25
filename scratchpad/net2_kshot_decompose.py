"""6발 재시도 풀 재구성 + 시뮬 판정 + 성공 파지의 GT 근접 분해 (2026-10-11).

입력: scratchpad/net2_viz_cases.pkl (net2_viz_mine.py 산출 — 검증 파지 240개, 팔 3대,
      특징 표준화 모델, net1 상위 3개 파티션별 net2 상위 8점·GT 앵커·GT 표 판정)
후보: 파티션마다 8^3 조합을 LP 로 채점 → min α 상위 2개 조합을 각각 이웃 7점 다듬기
      → 파지당 최대 6개 (여름 retry_eval 의 "파티션 3 × 정밀화 상위 2" 를 이 정의로 재구성)
순서: 6개를 LP min α 내림차순 (파티션 무관). 1발 = 1등 — net2_viz_mine 의 sel 과 일치해야 함.
판정: gate_runner._worker (방향별 주장값 × 0.25 외란 유지). 성공 = pass ∧ min α ≥ 0.8 × GT 최선 합격 α.
출력: scratchpad/net2_kshot_cases.pkl + 요약 출력.
실행: ./venv/bin/python scratchpad/net2_kshot_decompose.py   (spawn + __main__ 가드 필수)
"""
import itertools
import pickle
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from net2_viz_mine import refine, build_sim_job, MU_T, PAD_REL, DEV   # noqa: E402
from contraction_gt.search import _batched_alpha, mu_tor_from_pad      # noqa: E402
from contraction_gt.surface import SurfaceSamples                       # noqa: E402
from contraction_gt.gate_runner import _worker                          # noqa: E402

K_PER_PART = 2
SIM_WORKERS = 6
CELL, BASIN = 0.078, 0.15          # 표면점 한 칸 / GT 탐색 분지 (r_obj 배)
IN_PKL = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "scratchpad/net2_viz_cases.pkl")
OUT_PKL = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "scratchpad/net2_kshot_cases.pkl")


def build_candidates(m):
    samples = SurfaceSamples(points=m["pts"].astype(np.float64),
                             normals=m["nrm"].astype(np.float64),
                             r_obj=m["r_obj"], com=m["com"].astype(np.float64))
    mu_tor = mu_tor_from_pad(PAD_REL * m["r_obj"], MU_T)
    cands = []
    for pi, p in enumerate(m["parts"]):
        combos = np.array(list(itertools.product(*p["top8"])), np.int64)
        a = _batched_alpha(samples, combos, MU_T, np.full(len(combos), mu_tor), DEV)
        order = np.argsort(-a.min(1))
        seen, taken = set(), 0
        for rank, j in enumerate(order):
            if taken >= K_PER_PART:
                break
            combo_r, alpha_r = refine(samples, combos[j], mu_tor)
            if alpha_r.min() < a[j].min():
                combo_r, alpha_r = combos[j], a[j]
            key = tuple(int(x) for x in combo_r)
            if key in seen:
                continue
            seen.add(key)
            taken += 1
            cands.append(dict(part=pi, lp_rank=int(rank),
                              combo=np.asarray(combo_r, np.int32),
                              alpha14=np.asarray(alpha_r, np.float32),
                              min_alpha=float(np.min(alpha_r))))
    cands.sort(key=lambda c: -c["min_alpha"])
    return samples, mu_tor, cands


def closeness(m, c):
    p = m["parts"][c["part"]]
    r = m["r_obj"]
    anc = m["pts"][c["combo"]]
    dg = np.linalg.norm(anc - p["gt_anchor"], axis=1) / r
    return dict(dg=dg,
                part_is_gt_best=abs(p["rec_alpha"] - m["gt_best_pass"]) < 1e-9,
                part_passes=(p["rec_verdict"] == "pass"),
                ratio=c["min_alpha"] / m["gt_best_pass"])


def summarize(name, items):
    """items: (m, c) 쌍 — 채택된 후보들."""
    n = len(items)
    print(f"\n[{name}] {n}개")
    if n == 0:
        return
    cl = [closeness(m, c) for m, c in items]
    for lab, thr in (("한 칸(0.078r) 안", CELL), ("분지(0.15r) 안", BASIN), ("0.30r 안", 0.30)):
        k = sum(bool((x["dg"] <= thr).all()) for x in cl)
        print(f"  모든 앵커가 GT에서 {lab}: {k}/{n} ({100*k/n:.0f}%)")
    hist = Counter(int((x["dg"] <= BASIN).sum()) for x in cl)
    print("  분지 안 그룹 수 분포:", dict(sorted(hist.items())))
    mx = np.array([x["dg"].max() for x in cl])
    allg = np.concatenate([x["dg"] for x in cl])
    print("  최대 앵커→GT 거리 분위수 25/50/75/90% (r_obj 배):", np.round(np.percentile(mx, [25, 50, 75, 90]), 3))
    print("  그룹별 거리 분위수 25/50/75/90%:", np.round(np.percentile(allg, [25, 50, 75, 90]), 3))
    kb = sum(x["part_is_gt_best"] for x in cl)
    kp = sum(x["part_passes"] for x in cl)
    print(f"  채택 파티션 = GT 최선 합격 파티션: {kb}/{n} ({100*kb/n:.0f}%)   GT 표에서 합격 파티션: {kp}/{n} ({100*kp/n:.0f}%)")
    rt = np.array([x["ratio"] for x in cl])
    print("  달성 α ÷ GT 최선 α 분위수 25/50/75%:", np.round(np.percentile(rt, [25, 50, 75]), 2),
          f" 1.0 이상 비율 {float((rt >= 1.0).mean()):.2f}")
    print("  채택 후보의 net1 파티션 순위(0=1등) 분포:", dict(sorted(Counter(c["part"] for _, c in items).items())))


def main():
    t0 = time.time()
    mined = pickle.load(open(IN_PKL, "rb"))
    print(f"파지 {len(mined)}개 로드", flush=True)

    all_c, mismatch_top1 = [], 0
    for gi, m in enumerate(mined):
        samples, mu_tor, cands = build_candidates(m)
        m["samples"], m["mu_tor"], m["cands"] = samples, mu_tor, cands
        if not np.array_equal(cands[0]["combo"], m["parts"][m["sel"]]["combo"]):
            mismatch_top1 += 1
        for ci, c in enumerate(cands):
            c["rank"] = ci
            all_c.append((gi, ci))
        if (gi + 1) % 40 == 0:
            print(f"  후보 구성 {gi + 1}/{len(mined)}  ({time.time() - t0:.0f}s)", flush=True)
    n_c = Counter(len(m["cands"]) for m in mined)
    print(f"후보 수 분포 {dict(sorted(n_c.items()))}, 1발 재현 불일치 {mismatch_top1}/{len(mined)}", flush=True)

    jobs = []
    for k, (gi, ci) in enumerate(all_c):
        m, c = mined[gi], mined[gi]["cands"][ci]
        jobs.append(build_sim_job(k, m["ds_oid"][0], m["ds_oid"][1], m["ts"], m["r_obj"],
                                  m["samples"], c["combo"], c["alpha14"].astype(np.float64),
                                  m["mu_tor"]))
    print(f"시뮬 {len(jobs)}개 시작", flush=True)
    ctx = get_context("spawn")
    done = 0
    with ctx.Pool(SIM_WORKERS) as pool:
        for idx, verdict, _nu in pool.imap_unordered(_worker, jobs, chunksize=2):
            gi, ci = all_c[idx]
            mined[gi]["cands"][ci]["verdict"] = verdict
            done += 1
            if done % 200 == 0:
                print(f"  시뮬 {done}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"시뮬 완료 ({time.time() - t0:.0f}s)", flush=True)

    for m in mined:
        for c in m["cands"]:
            c["success"] = (c["verdict"] == "pass" and
                            c["min_alpha"] >= 0.8 * m["gt_best_pass"])
        succ = [c for c in m["cands"] if c["success"]]
        m["first_success"] = succ[0] if succ else None

    N = len(mined)
    print("\n== K발 곡선 (LP 순서, 처음 K개 중 하나라도 성공) ==")
    for K in range(1, 7):
        k = sum(any(c["success"] for c in m["cands"][:K]) for m in mined)
        print(f"  {K}발: {k}/{N} = {100*k/N:.1f}%")
    v_all = Counter(c["verdict"] for m in mined for c in m["cands"])
    print("후보 전체 시뮬 판정:", dict(v_all))

    adopted = [(m, m["first_success"]) for m in mined if m["first_success"] is not None]
    summarize("6발 성공 전체 (채택 = 순서상 첫 성공 후보)", adopted)
    summarize("그중 1발에서 성공", [(m, c) for m, c in adopted if c["rank"] == 0])
    summarize("그중 재시도(2~6발)로 건진 파지", [(m, c) for m, c in adopted if c["rank"] >= 1])
    print("\n채택 후보의 순위 분포(0=1발):",
          dict(sorted(Counter(c["rank"] for _, c in adopted).items())))

    # 1발 실패 파지에서 나머지 5개 후보가 어땠나
    fail1 = [m for m in mined if not m["cands"][0]["success"]]
    rescued = sum(m["first_success"] is not None for m in fail1)
    print(f"\n1발 실패 {len(fail1)}개 중 재시도로 건진 파지 {rescued}개, 끝까지 실패 {len(fail1) - rescued}개")
    v_fail_all = Counter(c["verdict"] for m in mined if m["first_success"] is None for c in m["cands"])
    print("끝까지 실패한 파지의 후보 판정:", dict(v_fail_all))

    for m in mined:                        # 저장 전 큰 객체 제거
        del m["samples"]
    with open(OUT_PKL, "wb") as f:
        pickle.dump(mined, f)
    print(f"\n저장: {OUT_PKL} ({OUT_PKL.stat().st_size/1e6:.1f} MB), 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
