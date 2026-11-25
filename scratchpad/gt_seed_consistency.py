"""정답 탐색의 자기 일치율 측정 (EXP-001, 2026-11-13).

같은 파지에 정답 탐색을 난수 씨앗만 바꿔 다시 돌려, 새 정답이 기존 정답(씨앗 0)과
같은 자리에 나오는 비율을 잰다. 선형계획만 쓰고 시뮬레이션은 없다.

탐색은 contraction_gt/runner_batched.py 의 함수를 수정 없이 run_shard 와 같은 순서로 부른다:
  load_grasp_contacts → load_samples → enumerate_partitions → _coarse_jobs
  → _score_by_n("combos") → 거친 1등 → _refine_set → _score_by_n("ref")
  → 다듬은 결과가 같거나 높으면 채택.
원래 실행과 다른 점은 (1) 팔 2대 작업을 만들지 않는다 (2) 한 묶음에 들어가는 파지가 다르다 —
둘 다 작업별 결과에는 영향이 없어야 하고, 그것을 씨앗 0 재현 검산으로 확인한다.

입력: scratchpad/net2_viz_cases.pkl, scratchpad/net2_viz_cases_test.pkl (검증 파지 240개씩)
기준: data/contraction_gt_300k/search_records (저장된 정답, 씨앗 0) + gate_records*.npz (시뮬 판정)
출력: scratchpad/gt_seed_consistency.pkl + 요약 출력
실행: ./venv/bin/python scratchpad/gt_seed_consistency.py [--smoke]
계획: harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-001-gt-seed-consistency/plan.md
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.contacts_io import load_grasp_contacts                      # noqa: E402
from contraction_gt.object_cache import load_samples                            # noqa: E402
from contraction_gt.search import SearchConfig                                  # noqa: E402
from contraction_gt.units import enumerate_partitions                           # noqa: E402
from contraction_gt.runner_batched import (CONTACT_ROOT, CACHE_ROOT,            # noqa: E402
                                           _coarse_jobs, _refine_set, _score_by_n)

IN_PKLS = ["scratchpad/net2_viz_cases.pkl", "scratchpad/net2_viz_cases_test.pkl"]
REC_DIR = ROOT / "data/contraction_gt_300k/search_records"
OUT_PKL = ROOT / "scratchpad/gt_seed_consistency.pkl"
N_LIST = (3, 4)                 # 팔 3대(주) · 4대(보조). 팔 2대는 이 측정의 대상이 아님
SEEDS = (0, 1, 2)               # 0 = 재현 검산용
CHUNK = 48                      # runner_batched 기본값
CELL, BASIN = 0.078, 0.15       # 표면점 한 칸 / 정답 탐색 반경 (물체 반경의 배수)
SAME_TOL = 1e-9                 # "같은 점" 판정 (미터)
VERIFY_MIN = 0.99               # 씨앗 0 재현 검산 통과 기준 (계획서 고정값)


def load_grasps():
    """(gid, dataset, 물체ID) 목록 — 두 평가 묶음의 순서 그대로. 파지 목록 파일과 물체가 같은지 확인."""
    obj_of = {}
    for line in (REC_DIR.parent / "pilot_manifest.tsv").read_text().splitlines()[1:]:
        _t, ds, oid, rel, ridx, _s, _n = line.split("\t")
        obj_of[f"{rel}#{ridx}"] = (ds, oid)
    out, seen = [], set()
    for f in IN_PKLS:
        for m in pickle.load(open(ROOT / f, "rb")):
            gid = m["gid"]
            assert gid not in seen, f"두 묶음에 같은 파지: {gid}"
            assert tuple(m["ds_oid"]) == obj_of[gid], f"물체 불일치: {gid}"
            seen.add(gid)
            out.append((gid, *obj_of[gid]))
    return out


def load_stored(gids):
    """저장된 정답(씨앗 0)과 시뮬 판정 — 키 (파지, 파티션, 팁 반경)."""
    cols = {k: [] for k in ["grasp_id", "n_anchors", "partition", "pad_rel",
                            "min_alpha", "r_obj", "anchor_pos"]}
    for p in sorted(REC_DIR.glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            for k in cols:
                cols[k].append(z[k])
    rec = {k: np.concatenate(v) for k, v in cols.items()}
    verdict = np.full(len(rec["grasp_id"]), "", dtype=object)
    for name in ("gate_records.npz", "gate_records_all.npz"):     # idx = 위 이어붙인 순서의 행 번호
        gp = REC_DIR.parent / name
        if gp.exists():
            with np.load(gp, allow_pickle=True) as z:
                verdict[z["idx"].astype(int)] = z["verdict"]
    sel = np.nonzero(np.isin(rec["grasp_id"], sorted(gids))
                     & np.isin(rec["n_anchors"], N_LIST))[0]
    stored = {}
    for i in sel:
        n = int(rec["n_anchors"][i])
        key = (str(rec["grasp_id"][i]), str(rec["partition"][i]), round(float(rec["pad_rel"][i]), 4))
        assert key not in stored, f"저장된 정답에 중복 키: {key}"
        stored[key] = dict(n=n, pos=rec["anchor_pos"][i][:n].copy(),
                           min_alpha=float(rec["min_alpha"][i]),
                           r_obj=float(rec["r_obj"][i]), verdict=str(verdict[i]))
    return stored


def run_seed(grasps, seed):
    """씨앗 하나로 정답 탐색 — runner_batched.run_shard 의 준비·거친 단계·다듬기·확정과 같은 순서."""
    cfg = SearchConfig(seed=seed)
    out, n_fail, t0 = {}, 0, time.time()
    for c0 in range(0, len(grasps), CHUNK):
        ctxs = []
        for gid, ds, oid in grasps[c0:c0 + CHUNK]:
            rel, ridx = gid.rsplit("#", 1)
            try:
                patches, ts = load_grasp_contacts(str(ROOT / CONTACT_ROOT), rel, int(ridx))
                if len(patches) < 2:
                    continue
                samples = load_samples(ROOT / CACHE_ROOT, ds, oid, scale=ts)
                partitions = []
                for n in N_LIST:
                    partitions += enumerate_partitions(patches, n)
                jobs = _coarse_jobs(samples, patches, partitions, cfg) if partitions else []
                ctxs.append(dict(gid=gid, samples=samples, jobs=jobs))
            except Exception as e:
                n_fail += 1
                print(f"  실패 {gid}: {e!r}", flush=True)
        named = [(g, j) for g in ctxs for j in g["jobs"]]

        _score_by_n(named, cfg, "combos")                       # 거친 단계
        for g, j in named:
            ma = j["combos_alpha"].min(axis=1)
            b = int(ma.argmax())
            j["best_combo"], j["best_alpha"], j["best_min"] = j["combos"][b], j["combos_alpha"][b], ma[b]

        for g, j in named:                                      # 다듬기
            j["ref"] = _refine_set(g["samples"], j, j["best_combo"], cfg)
        _score_by_n(named, cfg, "ref")

        for g, j in named:                                      # 확정
            rmin = j["ref_alpha"].min(axis=1)
            rb = int(rmin.argmax())
            if rmin[rb] >= j["best_min"]:
                combo, alpha = j["ref"][rb], j["ref_alpha"][rb]
            else:
                combo, alpha = j["best_combo"], j["best_alpha"]
            key = (g["gid"], json.dumps(j["part"]), round(float(j["pad_rel"]), 4))
            out[key] = dict(n=j["n"], pos=g["samples"].points[combo].copy(),
                            min_alpha=float(alpha.min()), r_obj=float(g["samples"].r_obj))
        print(f"  씨앗 {seed}: 파지 {min(c0 + CHUNK, len(grasps))}/{len(grasps)}, "
              f"레코드 {len(out)} ({time.time() - t0:.0f}s)", flush=True)
    return out, n_fail


def verify_seed0(stored, s0):
    """씨앗 0 재현 검산: 저장된 정답과 앵커 위치가 같은 레코드의 비율 (분모 = 양쪽 키의 합집합)."""
    keys = set(stored) | set(s0)
    both = [k for k in keys if k in stored and k in s0]
    same = [k for k in both if stored[k]["n"] == s0[k]["n"]
            and np.abs(stored[k]["pos"] - s0[k]["pos"]).max() < SAME_TOL]
    da = np.array([abs(stored[k]["min_alpha"] - s0[k]["min_alpha"]) for k in same]) if same else np.zeros(1)
    res = dict(n_union=len(keys), n_both=len(both), n_same=len(same),
               only_stored=len(set(stored) - set(s0)), only_new=len(set(s0) - set(stored)),
               rate=len(same) / max(len(keys), 1), max_alpha_diff=float(da.max()))
    print(f"[씨앗 0 재현 검산] 앵커 위치가 같은 레코드 {res['n_same']}/{res['n_union']} = {100 * res['rate']:.2f}% "
          f"(저장된 정답에만 있는 키 {res['only_stored']}, 새 결과에만 있는 키 {res['only_new']}, "
          f"같은 레코드의 α 최대 차이 {res['max_alpha_diff']:.2e})", flush=True)
    return res


def agreement(ref, new, n, keep=None):
    """ref 와 new 를 같은 (파지, 파티션, 팁 반경), 같은 팔끼리 비교한 일치율 (팔 n 대 레코드만)."""
    D, ratio, missing = [], [], 0
    for k, a in ref.items():
        if a["n"] != n or (keep is not None and not keep(a)):
            continue
        b = new.get(k)
        if b is None or b["n"] != n:
            missing += 1
            continue
        D.append(np.linalg.norm(a["pos"] - b["pos"], axis=1) / a["r_obj"])
        ratio.append(b["min_alpha"] / a["min_alpha"] if a["min_alpha"] > 1e-9 else np.nan)
    if not D:
        return dict(n_rec=0, missing=missing)
    D, ratio = np.array(D), np.array(ratio)
    ok = np.isfinite(ratio)
    return dict(
        n_rec=len(D), missing=missing,
        arm_same=float((D < 1e-9).mean()), arm_cell=float((D <= CELL).mean()), arm_basin=float((D <= BASIN).mean()),
        all_same=float((D < 1e-9).all(1).mean()), all_cell=float((D <= CELL).all(1).mean()),
        all_basin=float((D <= BASIN).all(1).mean()),
        ratio_median=float(np.median(ratio[ok])) if ok.any() else float("nan"),
        ratio_ge095=float((ratio[ok] >= 0.95).mean()) if ok.any() else float("nan"),
        ratio_gt1=float((ratio[ok] > 1.0 + 1e-9).mean()) if ok.any() else float("nan"))


def show(label, r):
    if r["n_rec"] == 0:
        print(f"  {label}: 비교할 레코드 없음 (빠진 레코드 {r['missing']})")
        return
    p = lambda v: f"{100 * v:5.1f}%"
    print(f"  {label}: 레코드 {r['n_rec']}개 (새 탐색에 없는 레코드 {r['missing']}개)")
    print(f"      팔 하나 기준   — 완전히 같은 점 {p(r['arm_same'])} | 0.078배 이내 {p(r['arm_cell'])} | 0.15배 이내 {p(r['arm_basin'])}")
    print(f"      모든 팔 기준   — 완전히 같은 점 {p(r['all_same'])} | 0.078배 이내 {p(r['all_cell'])} | 0.15배 이내 {p(r['all_basin'])}")
    print(f"      새 정답 α ÷ 기준 α — 중앙값 {r['ratio_median']:.3f} | 0.95배 이상 {p(r['ratio_ge095'])} | 1배 초과 {p(r['ratio_gt1'])}")


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    grasps = load_grasps()
    if smoke:
        grasps = grasps[:8]
    gids = {g for g, _, _ in grasps}
    print(f"파지 {len(grasps)}개, 팔 {N_LIST}대, 씨앗 {SEEDS if not smoke else (0,)}", flush=True)
    stored = load_stored(gids)
    print(f"저장된 정답 레코드 {len(stored)}개 (파지 {len({k[0] for k in stored})}개), "
          f"로드 {time.time() - t0:.0f}s", flush=True)

    res, fails = {}, {}
    res[0], fails[0] = run_seed(grasps, 0)
    ver = verify_seed0(stored, res[0])
    passed = ver["rate"] >= VERIFY_MIN
    print(f"[검산 판정] {'통과' if passed else '불통과'} (기준 {100 * VERIFY_MIN:.0f}% 이상)", flush=True)
    if smoke or not passed:
        if not passed:
            print("검산 불통과 — 씨앗 1·2 는 돌리지 않고 멈춥니다 (계획서의 중단 조건).", flush=True)
        sys.exit(0 if passed else 2)

    for s in SEEDS[1:]:
        res[s], fails[s] = run_seed(grasps, s)

    is_pass = lambda a: a["verdict"] == "pass"
    summary = {}
    for n in N_LIST:
        print(f"\n===== 팔 {n}대 =====")
        for s in SEEDS[1:]:
            print(f"[씨앗 {s} 대 저장된 정답(씨앗 0)]")
            for lab, keep in (("전체 레코드", None), ("기준 정답이 시뮬레이션에 합격한 레코드", is_pass),
                              ("기준 정답의 α 가 0.02 를 넘는 레코드", lambda a: a["min_alpha"] > 0.02)):
                r = agreement(stored, res[s], n, keep)
                summary[(n, s, lab)] = r
                show(lab, r)
        print(f"[씨앗 {SEEDS[1]} 대 씨앗 {SEEDS[2]}]")
        r = agreement(res[SEEDS[1]], res[SEEDS[2]], n)
        summary[(n, "1vs2", "전체 레코드")] = r
        show("전체 레코드", r)

    prim = [summary[(3, s, "전체 레코드")]["all_cell"] for s in SEEDS[1:]]
    primary = float(np.mean(prim))
    if primary < 0.5:
        verdict = "50% 미만 — 정답이 탐색의 뽑기에 크게 좌우됩니다 (다음 단계 후보: 정답 생성 수정)"
    elif primary >= 0.8:
        verdict = "80% 이상 — 정답은 일관됩니다 (다음 단계 후보: 네트워크 쪽)"
    else:
        verdict = "50~80% 사이 — 두 원인이 섞여 있습니다 (0.15배 기준 일치율과 함께 연구자가 판단)"
    print(f"\n[주 지표] 팔 3대, 세 팔 모두 0.078배 이내 일치율: 씨앗 1 = {100 * prim[0]:.1f}%, "
          f"씨앗 2 = {100 * prim[1]:.1f}%, 평균 {100 * primary:.1f}%")
    print(f"[판정] {verdict}")
    print(f"탐색 중 불러오기 실패 파지 수(씨앗별): {fails}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(dict(grasps=grasps, stored=stored, seeds=res, verify=ver, summary=summary,
                         primary=primary, verdict=verdict, fails=fails,
                         config=dict(N_LIST=N_LIST, SEEDS=SEEDS, CHUNK=CHUNK, CELL=CELL, BASIN=BASIN,
                                     VERIFY_MIN=VERIFY_MIN, inputs=IN_PKLS)), f)
    print(f"저장 {OUT_PKL} ({OUT_PKL.stat().st_size / 1e6:.1f} MB), 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
