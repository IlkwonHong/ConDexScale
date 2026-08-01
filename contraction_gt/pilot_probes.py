"""파일럿 잔여 검사 1·2.

검사 1 (탐색 반경 곡선): 반경을 크게(c=0.3) 잡아 후보를 뽑아두고, 반경 제한을
0.02→0.30 으로 좁혀가며 도달 가능한 최고 min-α 를 잰다. "어디까지 줄여도
점수가 안 깎이는가"의 곡선 → 탐색 반경 c 확정.

검사 2 (1차 필터 검사): 표본 파티션에서 '모든 조합'을 LP 로 채점한 진짜 순위를
만들고, 싸구려 점수 상위 32개가 진짜 상위권을 얼마나 놓치는지(재현율)와
그로 인한 점수 손실(후회율)을 잰다.

실행: ./venv/bin/python -m contraction_gt.pilot_probes <probe1|probe2> [표본수]
"""

import sys
from pathlib import Path

import numpy as np

from .contacts_io import load_grasp_contacts
from .object_cache import load_samples
from .units import enumerate_partitions
from .search import (SearchConfig, mu_tor_from_pad, _top_combos, _batched_alpha,
                     _feasible_candidates)
from .surface import candidates_in_region

ROOT = Path("data/contraction_gt_pilot")
CONTACT_ROOT = "data/dexonomy/contact_cache"
PAD_REL = 0.10                     # 검사 공통: 명목 패드(10%)만 사용
C_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def _grasp_rows(stride):
    rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    return rows[::stride]


def _load(row):
    typ, ds, oid, rel, ridx, _s, _n = row
    patches, ts = load_grasp_contacts(CONTACT_ROOT, rel, int(ridx))
    samples = load_samples(ROOT / "object_cache", ds, oid, scale=ts)
    parts = []
    for n in (2, 3, 4):
        parts += enumerate_partitions(patches, n)
    return patches, samples, parts


def _group_geometry(samples, patches, group):
    pts = np.concatenate([patches[u].points for u in group])
    areas = np.concatenate([patches[u].areas for u in group])
    w = areas / areas.sum() if areas.sum() > 0 else None
    centroid = np.average(pts, axis=0, weights=w)
    nrm = np.average(np.concatenate([patches[u].normals for u in group]), axis=0, weights=w)
    patch_r = float(np.linalg.norm(pts - centroid, axis=1).max())
    return centroid, nrm, patch_r


def probe1(n_grasps=300, seed=0):
    cfg = SearchConfig()
    rows = _grasp_rows(max(1, 9982 // n_grasps))
    out = {"n_anchors": [], "curve": []}          # curve: c' 별 최고 min-α
    for gi, row in enumerate(rows):
        try:
            patches, samples, parts = _load(row)
        except Exception:
            continue
        tip_r = max(PAD_REL * samples.r_obj, 0.002)
        mu_tor = mu_tor_from_pad(tip_r, cfg.mu_t, cfg.hertz_frac)
        for part in parts:
            geoms = [_group_geometry(samples, patches, g) for g in part]
            curve = []
            for c in C_GRID:
                # 반경마다 후보를 같은 밀도(32개)로 새로 샘플 — 후보 수 교란 제거
                fsets = []
                empty = False
                for centroid, nrm, patch_r in geoms:
                    rho = max(patch_r, c * samples.r_obj)
                    idx = candidates_in_region(samples, centroid, rho,
                                               ref_normal=nrm, k_max=32, seed=seed)
                    idx = _feasible_candidates(samples, idx, tip_r)
                    if len(idx) == 0:
                        empty = True
                        break
                    fsets.append(idx)
                if empty:
                    curve.append(np.nan)
                    continue
                combos = _top_combos(fsets, samples, cfg.top_combos, min_sep=2.4 * tip_r)
                if len(combos) == 0:
                    curve.append(np.nan)
                    continue
                alpha = _batched_alpha(samples, combos, cfg.mu_t,
                                       np.full(len(combos), mu_tor), cfg.device)
                curve.append(float(alpha.min(axis=1).max()))
            out["n_anchors"].append(len(part))
            out["curve"].append(curve)
        if (gi + 1) % 50 == 0:
            print(f"{gi + 1}/{len(rows)} grasp", flush=True)
    np.savez(ROOT / "probe1_rho_curve.npz",
             n_anchors=np.array(out["n_anchors"]), curve=np.array(out["curve"]))

    curve = np.array(out["curve"])
    n_anchors = np.array(out["n_anchors"])
    print("\n=== 검사 1: 반경 제한별 도달 점수 (최대 반경 0.30 대비 비율의 중앙값) ===")
    for n in (2, 3, 4):
        sel = n_anchors == n
        cv = curve[sel]
        ref = cv[:, -1]
        good = np.isfinite(ref) & (ref > 1e-6)
        ratio = cv[good] / ref[good, None]
        med = np.nanmedian(ratio, axis=0)
        frac95 = np.nanmean(ratio >= 0.95, axis=0)
        print(f"N={n} (표본 {good.sum()}):")
        for c, m_, f_ in zip(C_GRID, med, frac95):
            print(f"  c={c:.2f}: 중앙값 {m_:.3f}, 95% 도달 비율 {f_:.1%}")


def probe2(n_partitions=120, seed=0):
    cfg = SearchConfig()
    rows = _grasp_rows(9982 // 60)
    rec = {2: [], 3: [], 4: []}                    # (재현율@32, 후회율)
    done = 0
    for row in rows:
        if done >= n_partitions:
            break
        try:
            patches, samples, parts = _load(row)
        except Exception:
            continue
        tip_r = max(PAD_REL * samples.r_obj, 0.002)
        mu_tor = mu_tor_from_pad(tip_r, cfg.mu_t, cfg.hertz_frac)
        for part in parts:
            if done >= n_partitions:
                break
            n = len(part)
            k = cfg.k_candidates_n4 if n >= 4 else cfg.k_candidates
            fsets = []
            ok = True
            for group in part:
                centroid, nrm, _pr = _group_geometry(samples, patches, group)
                rho = max(_pr, cfg.rho_rel * samples.r_obj)
                idx = candidates_in_region(samples, centroid, rho, ref_normal=nrm,
                                           k_max=k, seed=seed)
                idx = _feasible_candidates(samples, idx, tip_r)
                if len(idx) == 0:
                    ok = False
                    break
                fsets.append(idx)
            if not ok:
                continue
            # 전 조합 (간격 필터 적용) — proxy 순서로 정렬된 전체를 받는다
            all_c = _top_combos(fsets, samples, 10 ** 9, min_sep=2.4 * tip_r)
            if len(all_c) < 64:
                continue
            mins = np.empty(len(all_c))
            for s in range(0, len(all_c), 20000):     # 메모리 조각내기
                a = _batched_alpha(samples, all_c[s:s + 20000], cfg.mu_t,
                                   np.full(len(all_c[s:s + 20000]), mu_tor), cfg.device)
                mins[s:s + 20000] = a.min(axis=1)
            true_best = float(mins.max())
            row = []
            for w in (32, 64, 128, 256):
                route_best = float(mins[: min(w, len(all_c))].max())
                row.append((true_best - route_best) / max(true_best, 1e-9))
            rec[n].append(row)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{n_partitions} 파티션", flush=True)

    print("\n=== 검사 2: 통과 폭별 점수 손실 (중앙값 / p90) ===")
    for n in (2, 3, 4):
        if not rec[n]:
            continue
        r = np.array(rec[n])
        for j, w in enumerate((32, 64, 128, 256)):
            print(f"N={n} 폭 {w}: {np.median(r[:,j]):.1%} / {np.percentile(r[:,j],90):.1%}"
                  + (f"  (표본 {len(r)})" if j == 0 else ""))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "probe1":
        probe1(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        probe2(int(sys.argv[2]) if len(sys.argv) > 2 else 120)
