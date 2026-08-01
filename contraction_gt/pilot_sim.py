"""파일럿 항목 4: α-시뮬 상관 측정.

층화 표본(N × 패드반경 × α 분위)을 뽑아 각 레코드의 14방향 전부를 MuJoCo 로
램프 테스트하고, LP α 와의 순위 상관·최약 방향 일치율·보정 비율을 잰다.

방향별 테스트 그리드는 LP α 의 배수 [0.25 … 3.0] — "LP 가 맞다면 1.0 근처에서
깨져야 한다"를 직접 검사하는 상대 그리드다 (α_lp ≈ 0 인 방향은 절대 바닥값 0.02).

실행:
  ./venv/bin/python -m contraction_gt.pilot_sim run <표본수> <워커수>
  ./venv/bin/python -m contraction_gt.pilot_sim analyze
"""

import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path("data/contraction_gt_pilot")
ASSET_ROOT = Path("data/dexonomy/assets/object")
LEVEL_MULTS = np.arange(0.1, 2.01, 0.1)   # LP α 의 10% 해상도 상대 그리드
ALPHA_FLOOR = 0.02
F_CMD = 5.0
SENSOR_STD = 1.5   # run() 에서 모드 인자로 덮어씀


def out_path(mode):
    return ROOT / ("sim_records.npz" if mode == "passive" else "sim_records_active.npz")


def _manifest_map():
    rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    return {r[3]: (r[1], r[2]) for r in rows}   # rel_path -> (dataset, object_id)


def stratified_sample(m, per_cell=45, seed=0):
    """(N × pad × α 4분위) 셀당 per_cell 개 → 인덱스 배열."""
    rng = np.random.default_rng(seed)
    picked = []
    for n in (2, 3, 4):
        for pad in (0.05, 0.10, 0.20):
            sel = np.nonzero((m["n_anchors"] == n) & np.isclose(m["pad_rel"], pad))[0]
            if len(sel) == 0:
                continue
            a = m["min_alpha"][sel]
            qs = np.quantile(a, [0.25, 0.5, 0.75])
            bins = np.digitize(a, qs)
            for b in range(4):
                cell = sel[bins == b]
                take = min(per_cell // 4 + 1, len(cell))
                picked.append(rng.choice(cell, take, replace=False))
    return np.unique(np.concatenate(picked))


def _equilibrium_preload(pos_norm, nrm, mu_t, mu_tor_norm, f_cmd):
    """LP 평형 배분: ΣWx = 0, 앵커별 압력 ≤ 1 에서 총압력 최대화 → 팁별 예압 [N].

    균등 상한 예압은 법선이 비대칭인 다중 앵커에서 물체를 밀어버린다는
    파일럿 진단에 따라, Dexonomy contact-aware control 과 같은 원리로
    자기평형 힘 배분을 예압 목표로 쓴다.
    """
    import torch
    from scipy.optimize import linprog
    from .metric import ray_wrench_matrix, _flatten_W, _pressure_rows
    N = len(pos_norm)
    W = ray_wrench_matrix(
        torch.as_tensor(pos_norm, dtype=torch.float64).unsqueeze(0),
        torch.as_tensor(nrm, dtype=torch.float64).unsqueeze(0),
        torch.zeros(1, 3, dtype=torch.float64), 1.0, mu_t, mu_tor_norm)
    Wf = _flatten_W(W)[0].numpy()
    P = _pressure_rows(N, torch.float64, "cpu").numpy()[:, : 6 * N]
    res = linprog(-np.ones(6 * N), A_ub=P, b_ub=np.ones(N),
                  A_eq=Wf, b_eq=np.zeros(6), bounds=[(0, None)] * 6 * N,
                  method="highs")
    if res.status != 0 or res.x is None or res.x.sum() < 1e-9:
        return np.full(N, 0.5 * f_cmd), np.zeros(6 * N), None
    return (P @ res.x) * f_cmd, res.x, Wf


def _alpha_lp_with_x(Wf, w_dir, N):
    """방향 하나의 α-LP 를 풀고 (α*, x*) 반환 — 능동 제어의 힘 배분 원본."""
    from scipy.optimize import linprog
    n = 6 * N
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.concatenate([Wf, w_dir[:, None]], axis=1)
    A_ub = np.zeros((N, n + 1))
    for j in range(N):
        A_ub[j, 6 * j : 6 * j + 6] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.ones(N), A_eq=A_eq, b_eq=np.zeros(6),
                  bounds=[(0, None)] * (n + 1), method="highs")
    if res.status != 0 or res.x is None:
        return 0.0, np.zeros(n)
    return float(res.x[-1]), res.x[:n]


def _prepare_jobs(idxs, m, active=False):
    """워커에 넘길 순수 numpy 작업 목록 (torch 는 부모에서만 사용)."""
    import torch
    from .metric import direction_set
    mmap = _manifest_map()
    jobs = []
    for i in idxs:
        gid = str(m["grasp_id"][i])
        rel = gid.split("#")[0]
        ds, oid = mmap[rel]
        n = int(m["n_anchors"][i])
        pos = m["anchor_pos"][i][:n]
        nrm = m["anchor_normal"][i][:n]
        r_obj = float(m["r_obj"][i])
        cache = np.load(ROOT / "object_cache" / ds / f"{oid}.npz")
        com = cache["com"] * float(m["total_scale"][i])
        pos_n = torch.as_tensor((pos - com) / r_obj, dtype=torch.float64).unsqueeze(0)
        dirs = direction_set(pos_n)[0].numpy()
        alpha_lp = m["alpha"][i]
        levels = np.maximum(alpha_lp, ALPHA_FLOOR)[:, None] * LEVEL_MULTS[None, :]
        coacd = ASSET_ROOT / ds / "processed_data" / oid / "urdf" / "meshes"
        preload, x_eq, Wf = _equilibrium_preload(
            (pos - com) / r_obj, nrm, 0.6, float(m["mu_tor"][i]) / r_obj, F_CMD)
        job = dict(
            idx=int(i), coacd=str(coacd), scale=float(m["total_scale"][i]),
            pos=pos, nrm=nrm, dirs=dirs, levels=levels, r_obj=r_obj,
            mu_tor=float(m["mu_tor"][i]), preload=preload, active=active,
            sensor_std=SENSOR_STD,
            tip_radius=max(float(m["pad_rel"][i]) * r_obj, 0.002),
        )
        if active and Wf is not None:
            # 방향별 LP 힘 배분 (능동 제어의 명령 원본) + 평형 힘 벡터 예압
            sols = [_alpha_lp_with_x(Wf, dirs[k], n) for k in range(len(dirs))]
            ad = np.array([s[0] for s in sols])
            xs = np.stack([s[1] for s in sols])
            job["Wf"], job["x_eq"], job["xs_dir"], job["alpha_dir"] = Wf, x_eq, xs, ad
            f_eq = _tip_forces(Wf, x_eq, n) * F_CMD
            job["preload"] = f_eq                      # [N,3] 벡터 예압 (접선 포함)
        jobs.append(job)
    return jobs


def _tip_forces(Wf, x, n_tips):
    """정규화 힘 단위의 팁별 힘 벡터 [N,3] (× F_CMD 로 물리 단위)."""
    out = np.zeros((n_tips, 3))
    for i in range(n_tips):
        out[i] = Wf[:3, 6 * i : 6 * i + 6] @ x[6 * i : 6 * i + 6]
    return out


def _worker(job):
    from .sim_eval import SimConfig, measure_capacity_multi
    try:
        cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                        timestep=1e-3, settle_steps=600, hold_steps=300,
                        active=bool(job["active"]),
                        sensor_std=float(job.get("sensor_std", 1.5)))
        force_fn = None
        if job["active"] and "xs_dir" in job:
            Wf, x_eq = job["Wf"], job["x_eq"]
            xs, ad = job["xs_dir"], job["alpha_dir"]
            n_tips = len(job["pos"])
            pe = x_eq.reshape(n_tips, 6).sum(1)

            def force_fn(d, a):
                """외란 a·ŵ_d 를 상쇄하는 LP 배분(스케일) + 평형 내부힘 여유."""
                if ad[d] > 1e-9:
                    x = xs[d] * (min(a, ad[d]) / ad[d])
                else:
                    x = np.zeros_like(x_eq)
                pb = x.reshape(n_tips, 6).sum(1)
                w = pe > 1e-9
                beta = float(np.min((1.0 - pb[w]) / pe[w])) if w.any() else 0.0
                x = x + max(beta, 0.0) * x_eq
                return _tip_forces(Wf, x, n_tips) * F_CMD

        a_sim, settled, preload, n_unstable = measure_capacity_multi(
            ("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"],
            job["dirs"], job["levels"], job["r_obj"], cfg,
            preload_forces=job["preload"], force_fn=force_fn, seed=job["idx"])
        return job["idx"], a_sim, settled, preload, n_unstable, ""
    except Exception as e:
        return job["idx"], np.full(14, np.nan), False, 0.0, 14, repr(e)


def run(n_sample, n_workers, mode="passive", sensor_std=1.5):
    global SENSOR_STD
    SENSOR_STD = sensor_std
    from .pilot_analysis import load_all
    m, _, _, _ = load_all()
    idxs = stratified_sample(m, per_cell=max(4, n_sample // 33))
    print(f"표본 {len(idxs)}개 준비, 워커 {n_workers}개, 모드 {mode}")
    jobs = _prepare_jobs(idxs, m, active=(mode == "active"))
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(n_workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4)):
            results.append(res)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)} 완료", flush=True)
    results.sort(key=lambda r: r[0])
    idx_arr = np.array([r[0] for r in results])
    np.savez_compressed(
        out_path(mode), idx=idx_arr,
        alpha_sim=np.stack([r[1] for r in results]),
        settled=np.array([r[2] for r in results]),
        preload=np.array([r[3] for r in results]),
        n_unstable=np.array([r[4] for r in results]),
        errors=np.array([r[5] for r in results]),
        alpha_lp=m["alpha"][idx_arr], n_anchors=m["n_anchors"][idx_arr],
        pad_rel=m["pad_rel"][idx_arr], min_alpha=m["min_alpha"][idx_arr],
        grasp_id=m["grasp_id"][idx_arr],
    )
    n_err = sum(1 for r in results if r[5])
    print(f"저장: {out_path(mode)} (오류 {n_err}건)")


def analyze(mode="passive"):
    from scipy.stats import spearmanr
    d = np.load(out_path(mode), allow_pickle=False)
    # 방향별 NaN = 수치 불안정으로 측정 불가 — 그 방향만 제외 (전부 NaN 인 레코드는 제외)
    all_nan = np.isnan(d["alpha_sim"]).all(axis=1)
    ok = d["settled"] & ~all_nan
    n_unst = d["n_unstable"]
    print(f"=== 표본 {len(d['settled'])}개, 정착 성공 {ok.sum()}개 "
          f"({ok.mean():.1%}), 시뮬 오류 {(d['errors'] != '').sum()}건 ===")
    print(f"수치 불안정: 방향 기준 {n_unst.sum()}/{14 * len(n_unst)}개 "
          f"({n_unst.sum() / (14 * len(n_unst)):.1%}), 재시도로도 측정 불가였던 레코드 "
          f"{(n_unst > 0).sum()}개")
    print(f"정착 직후 팁 예압 달성도: 중앙값 {np.median(d['preload'][ok]):.2f}\n")

    # 상관: 정착 실패 = min-α_sim 0 으로 포함 (LP 저품질과의 합의를 반영)
    min_lp_all = np.where(d["settled"], d["alpha_lp"].min(1), d["min_alpha"])
    with np.errstate(all="ignore"):
        nanmin_sim = np.nanmin(np.where(np.isnan(d["alpha_sim"]), np.inf, d["alpha_sim"]), axis=1)
    nanmin_sim[~np.isfinite(nanmin_sim)] = np.nan
    min_sim_all = np.where(d["settled"], nanmin_sim, 0.0)
    sel = ~np.isnan(min_sim_all)
    print("--- 순위 상관 (min-α LP vs 시뮬, 정착 실패 = 0 포함) ---")
    print(f"전체: Spearman {spearmanr(min_lp_all[sel], min_sim_all[sel]).statistic:.3f}  [기준 ≥ 0.8]")
    for n in (2, 3, 4):
        s = sel & (d["n_anchors"] == n)
        if s.sum() > 10:
            print(f"N={n}: {spearmanr(min_lp_all[s], min_sim_all[s]).statistic:.3f} (n={s.sum()})")

    a_lp, a_sim = d["alpha_lp"][ok], d["alpha_sim"][ok]
    n_anchors = d["n_anchors"][ok]
    print("\n--- 최약 방향: 결정 유효성 잣대 (argmin 일치보다 중요) ---")
    full = ~np.isnan(a_sim).any(axis=1)          # 14방향 전부 측정된 레코드만
    am_lp = a_lp[full].argmin(1)
    sim_at_dstar = a_sim[full][np.arange(full.sum()), am_lp]
    min_sim = a_sim[full].min(1)
    ratio_dstar = sim_at_dstar / np.maximum(min_sim, 1e-9)
    print(f"α_sim[d*_LP] / min α_sim: 중앙값 {np.median(ratio_dstar):.2f}, "
          f"p90 {np.percentile(ratio_dstar, 90):.2f} (완전 측정 {full.sum()}건 기준)")
    print(f"  (1.0 = LP 급소만 재도 손실 없음. p90 이 기준 — 1.5 이하면 수용 가능)")
    exact = (am_lp == a_sim[full].argmin(1)).mean()
    print(f"참고: argmin 정확 일치 {exact:.1%}")

    print("\n--- 보정 비율 α_sim/α_lp (α_lp > 0.02 인 방향만) ---")
    groups = {"힘(0-5)": range(0, 6), "토크(6-11)": range(6, 12), "급소축(12-13)": range(12, 14)}
    for name, cols in groups.items():
        r = []
        for c in cols:
            v = (a_lp[:, c] > 0.02) & ~np.isnan(a_sim[:, c])
            r.append(a_sim[v, c] / a_lp[v, c])
        r = np.concatenate(r)
        print(f"{name}: 중앙값 {np.median(r):.2f}, IQR [{np.percentile(r,25):.2f}, {np.percentile(r,75):.2f}]")

    print("\n--- 정착 실패가 저품질 레코드에 몰리는지 ---")
    fail = ~d["settled"]
    if fail.any():
        print(f"정착 실패 레코드의 min-α(LP) 중앙값: {np.median(d['min_alpha'][fail]):.4f} "
              f"vs 성공: {np.median(d['min_alpha'][d['settled']]):.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]),
            sys.argv[4] if len(sys.argv) > 4 else "passive",
            float(sys.argv[5]) if len(sys.argv) > 5 else 1.5)
    else:
        analyze(sys.argv[2] if len(sys.argv) > 2 else "passive")
