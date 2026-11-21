"""렌치 공간 닮음 실험의 C — 시뮬에서 팁이 실제로 닿은 자리·법선 기록 (계획서 §5-4).

판정별 표본(기본 합격 300 · 불합격 300 · 정착 실패 100, 팔 3대/4대 반반)을 gate_runner 와
같은 장면·설정으로 정착시킨 뒤(외란 없음) 접촉을 기록한다.

실행: ./venv/bin/python -m contraction_gt.realized_contacts [--n-pass 300 --n-fail 300 --n-settle 100]
                                                               [--workers 8] [--idx 12 34 ...]
출력: data/contraction_gt_300k/wrench_similarity/realized_contacts.pkl
      항목마다 idx, verdict, settled, drift, planned(pos/normal), contacts(tip, pos_obj, normal_obj, dist, f_normal),
      검증량: 팁별 계획 앵커 ↔ 실제 접촉점 거리, 법선 각도.
"""

import argparse
import pickle
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .gate_runner import _load_records, _prepare

ROOT = Path("data/contraction_gt_300k")
OUT = ROOT / "wrench_similarity"


def _worker(job):
    from .sim_eval import SimConfig, measure_capacity_multi
    log = {}
    try:
        cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                        timestep=1e-3, settle_steps=600, hold_steps=300)
        _a, settled, pre, _nu = measure_capacity_multi(
            ("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"],
            np.zeros((0, 6)), np.zeros((0, 1)), job["r_obj"], cfg,
            preload_forces=job["preload"], seed=job["idx"], contact_log=log)
        err = None
    except Exception as e:                      # noqa: BLE001
        settled, pre, err = False, float("nan"), repr(e)
    # 검증량: 팁별 최근접 실제 접촉점까지 거리 / 법선 각도
    n = len(job["pos"])
    d_tip = np.full(n, np.nan); ang_tip = np.full(n, np.nan); n_c = np.zeros(n, int)
    for c in log.get("contacts", []):
        t = c["tip"]; n_c[t] += 1
        d = float(np.linalg.norm(c["pos_obj"] - job["pos"][t]))
        if not (d >= d_tip[t]):
            d_tip[t] = d
            cosang = float(np.clip(c["normal_obj"] @ (job["nrm"][t] / np.linalg.norm(job["nrm"][t])), -1, 1))
            ang_tip[t] = np.degrees(np.arccos(cosang))
    return dict(idx=job["idx"], settled=bool(settled), preload_frac=float(pre), error=err,
                planned_pos=job["pos"], planned_nrm=job["nrm"], r_obj=job["r_obj"],
                tip_radius=job["tip_radius"], contacts=log.get("contacts", []),
                obj_qpos=log.get("obj_qpos"), drift=log.get("drift", np.nan),
                actuator_force=log.get("actuator_force"),
                tip_dist=d_tip, tip_angle_deg=ang_tip, tip_ncontact=n_c)


def _sample(verdicts, m, n_pass, n_fail, n_settle, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(sorted(verdicts)); v = np.array([verdicts[i] for i in idx]); n = m["n_anchors"][idx]
    out = []
    for name, cnt in (("pass", n_pass), ("fail", n_fail), ("fail_settle", n_settle)):
        for N in (3, 4):
            pool = idx[(v == name) & (n == N)]
            k = min(cnt // 2, len(pool))
            out += rng.choice(pool, k, replace=False).tolist()
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pass", type=int, default=300)
    ap.add_argument("--n-fail", type=int, default=300)
    ap.add_argument("--n-settle", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="직접 지정 (시험용)")
    ap.add_argument("--out", default=str(OUT / "realized_contacts.pkl"))
    args = ap.parse_args()
    t0 = time.time()
    m = _load_records(ROOT)
    from .wrench_similarity_eval import load_verdicts
    verdicts = load_verdicts()
    idxs = args.idx if args.idx else _sample(verdicts, m, args.n_pass, args.n_fail, args.n_settle)
    jobs = _prepare(np.array(idxs), m, ROOT)
    print(f"표본 {len(jobs)}개 (합격 {sum(verdicts[i]=='pass' for i in idxs)}, 불합격 {sum(verdicts[i]=='fail' for i in idxs)}, "
          f"정착 실패 {sum(verdicts[i]=='fail_settle' for i in idxs)})", flush=True)
    results = []
    if args.workers <= 1:
        for j in jobs:
            results.append(_worker(j))
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=2)):
                results.append(r)
                if (k + 1) % 100 == 0:
                    print(f"{k + 1}/{len(jobs)} 완료 {time.time()-t0:.0f}s", flush=True)
    for r in results:
        r["verdict"] = verdicts[r["idx"]]
    results.sort(key=lambda r: r["idx"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    # 요약
    for name in ("pass", "fail", "fail_settle"):
        rs = [r for r in results if r["verdict"] == name]
        if not rs:
            continue
        d = np.concatenate([r["tip_dist"] / r["tip_radius"] for r in rs])
        a = np.concatenate([r["tip_angle_deg"] for r in rs])
        miss = np.concatenate([r["tip_ncontact"] == 0 for r in rs])
        print(f"{name:12s} n={len(rs):4d} 정착 {np.mean([r['settled'] for r in rs]):.2f} | "
              f"접촉 없는 팁 {miss.mean():.2f} | 계획↔실제 거리/팁반지름 중앙값 {np.nanmedian(d):.2f} (95% {np.nanpercentile(d,95):.2f}) | "
              f"법선 각도 중앙값 {np.nanmedian(a):.1f}° (95% {np.nanpercentile(a,95):.1f}°) | 오류 {sum(r['error'] is not None for r in rs)}")
    print(f"저장 {args.out}  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
