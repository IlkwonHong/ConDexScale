"""기하 실현 예측 검정 (2026-10-10): 구형 팁을 계획 앵커 자리에서 계획 법선 방향으로 밀어 넣으면
충돌 형상(볼록 조각 메시)의 어디에 먼저 닿는지를 **물리 없이** 구한다 (MuJoCo 충돌 검사만 사용, 동역학 없음,
팁 하나씩, 물체 고정). 그 "예측 접촉"을 시뮬 실측 접촉(realized_contacts.pkl, C 표본 700개)과 비교하고,
예측만으로 합격·불합격을 가를 수 있는지(AUC) 본다.

실행: ./venv/bin/python -m contraction_gt.geo_realization
출력: data/contraction_gt_300k/wrench_similarity/geo_realization.npz + 표
"""

import pickle
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
from scipy.stats import spearmanr

from .gate_runner import _load_records, _prepare
from .sim_eval import SimConfig, build_scene_xml
from .wrench_similarity_eval import ROOT, OUT, load_verdicts, manifest_map, CACHE_ROOT, MU, N_EDGES, N_DIRS, SEED
from .wrench_support import sample_directions, cone_edge_wrenches, support_sum
from .metric import ray_wrench_matrix, direction_set, alpha_batched
from .wrench_similarity_report import auc


def sweep_tip(model, data, i, obj_body, t_max, iters=18):
    """팁 i 만 법선 방향으로 밀어 넣어 처음 닿는 자리를 이분탐색. 반환 (t*, [(pos, normal_in)...]) 또는 (None, [])."""
    tip_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_g{i}")
    qadr = 7 + i                                     # freejoint 7 뒤에 팁 슬라이드 관절이 순서대로

    def contacts_at(t):
        data.qpos[:] = 0.0; data.qpos[3] = 1.0; data.qpos[qadr] = t
        mujoco.mj_forward(model, data)
        out = []
        for c in range(data.ncon):
            g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
            if tip_geom in (g1, g2) and obj_body in (int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])):
                p = np.array(data.contact.pos[c]); n = p - data.geom_xpos[tip_geom]
                out.append((p, n / max(np.linalg.norm(n), 1e-12), float(data.contact.dist[c])))
        return out

    if not contacts_at(t_max):
        return None, []
    lo, hi = 0.0, t_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if contacts_at(mid):
            hi = mid
        else:
            lo = mid
    return hi, contacts_at(hi)


def main():
    t0 = time.time()
    m = _load_records(ROOT); verdicts = load_verdicts(); mmap = manifest_map()
    res = pickle.load(open(OUT / "realized_contacts.pkl", "rb"))
    idxs = np.array([r["idx"] for r in res])
    jobs = {j["idx"]: j for j in _prepare(idxs, m, ROOT)}
    rows = []
    for r in res:
        j = jobs[int(r["idx"])]
        cfg = SimConfig(mu_tor=j["mu_tor"], tip_radius=j["tip_radius"], timestep=1e-3)
        model = mujoco.MjModel.from_xml_string(build_scene_xml(("coacd", j["coacd"], j["scale"]), j["pos"], j["nrm"], cfg))
        data = mujoco.MjData(model)
        obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        n = len(j["pos"]); R = j["tip_radius"]
        pred_pos, pred_nrm, pred_d, pred_ang, pred_t = [], [], np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        miss = 0
        for i in range(n):
            t_star, cs = sweep_tip(model, data, i, obj_body, t_max=cfg.start_gap + 2.5 * R)
            if t_star is None:
                miss += 1; continue
            nrm_i = j["nrm"][i] / np.linalg.norm(j["nrm"][i])
            d = [np.linalg.norm(p - j["pos"][i]) for p, _, _ in cs]
            k = int(np.argmin(d))
            p, nn, _ = cs[k]
            pred_pos.append(p); pred_nrm.append(nn)
            pred_d[i] = d[k]; pred_t[i] = t_star
            pred_ang[i] = np.degrees(np.arccos(np.clip(nn @ nrm_i, -1, 1)))
        rows.append(dict(idx=int(r["idx"]), verdict=r["verdict"], n=n, R=R, r_obj=j["r_obj"], miss=miss,
                         pred_pos=np.array(pred_pos).reshape(-1, 3), pred_nrm=np.array(pred_nrm).reshape(-1, 3),
                         pred_d=pred_d, pred_ang=pred_ang, pred_t=pred_t, meas_d=r["tip_dist"], meas_ang=r["tip_angle_deg"],
                         planned_pos=j["pos"], planned_nrm=j["nrm"], scale=j["scale"], gid=str(m["grasp_id"][int(r["idx"])]),
                         mu_tor=j["mu_tor"]))
    print(f"기하 예측 {len(rows)}개 완료 {time.time()-t0:.1f}s", flush=True)

    v = np.array([x["verdict"] for x in rows])
    R_ = np.array([x["R"] for x in rows])
    pd_max = np.array([np.nanmax(x["pred_d"]) if np.isfinite(x["pred_d"]).any() else np.nan for x in rows]) / R_
    pa_max = np.array([np.nanmax(x["pred_ang"]) if np.isfinite(x["pred_ang"]).any() else np.nan for x in rows])
    md_max = np.array([np.nanmax(x["meas_d"]) if np.isfinite(x["meas_d"]).any() else np.nan for x in rows]) / R_
    ma_max = np.array([np.nanmax(x["meas_ang"]) if np.isfinite(x["meas_ang"]).any() else np.nan for x in rows])
    miss = np.array([x["miss"] for x in rows])

    print("\n[1] 예측 vs 실측 (정착된 기록 = 합격·불합격 600개, 팁별 최대)")
    ok = (v != "fail_settle") & np.isfinite(pd_max) & np.isfinite(md_max)
    print(f"  거리: 스피어만 {spearmanr(pd_max[ok], md_max[ok]).statistic:.3f} | 중앙값 예측 {np.median(pd_max[ok]):.2f} 실측 {np.median(md_max[ok]):.2f} (팁 반지름 배)")
    ok2 = (v != "fail_settle") & np.isfinite(pa_max) & np.isfinite(ma_max)
    print(f"  각도: 스피어만 {spearmanr(pa_max[ok2], ma_max[ok2]).statistic:.3f} | 중앙값 예측 {np.median(pa_max[ok2]):.1f}° 실측 {np.median(ma_max[ok2]):.1f}°")
    # 팁 단위
    pt_p = np.concatenate([x["pred_d"] / x["R"] for x in rows if x["verdict"] != "fail_settle"])
    pt_m = np.concatenate([x["meas_d"] / x["R"] for x in rows if x["verdict"] != "fail_settle"])
    okp = np.isfinite(pt_p) & np.isfinite(pt_m)
    print(f"  팁 단위 거리 스피어만 {spearmanr(pt_p[okp], pt_m[okp]).statistic:.3f} (팁 {okp.sum()}개)")

    print("\n[2] 판정별 중앙값")
    for k in ("pass", "fail", "fail_settle"):
        s = v == k
        print(f"  {k:12s} n={s.sum():3d} | 예측 거리 {np.nanmedian(pd_max[s]):.2f}, 예측 각도 {np.nanmedian(pa_max[s]):.1f}° | 실측 거리 {np.nanmedian(md_max[s]):.2f}, 실측 각도 {np.nanmedian(ma_max[s]):.1f}° | 닿지 못한 팁 평균 {miss[s].mean():.2f}")

    print("\n[3] 합격·불합격 구분 AUC (불합격이 더 큼)")
    f, p = v == "fail", v == "pass"
    print(f"  실측 거리 {auc(md_max[f], md_max[p]):.3f} | 실측 각도 {auc(ma_max[f], ma_max[p]):.3f}")
    print(f"  예측 거리 {auc(pd_max[f], pd_max[p]):.3f} | 예측 각도 {auc(pa_max[f], pa_max[p]):.3f}")
    fs = v == "fail_settle"
    print(f"  (정착 실패 대 합격) 예측 거리 {auc(np.nan_to_num(pd_max[fs], nan=9), pd_max[p]):.3f} | 예측 각도 {auc(np.nan_to_num(pa_max[fs], nan=180), pa_max[p]):.3f}")

    # [4] 예측 접촉으로 다시 계산한 α(선형계획 14방향)·ρ̂ 가 합격을 가르는가
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dirs = sample_directions(N_DIRS, seed=SEED, dtype=torch.float64, device=dev); l1 = dirs.abs().sum(-1)
    com_cache = {}
    a_plan, a_geo, rh_plan, rh_geo = np.full(len(rows), np.nan), np.full(len(rows), np.nan), np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for k, x in enumerate(rows):
        ds, oid = mmap[x["gid"].split("#")[0]]
        if (ds, oid) not in com_cache:
            com_cache[(ds, oid)] = np.load(CACHE_ROOT / ds / f"{oid}.npz")["com"]
        com = com_cache[(ds, oid)] * x["scale"]; ro = float(x["r_obj"])
        for tag, P, Nn in (("plan", x["planned_pos"], x["planned_nrm"]), ("geo", x["pred_pos"], x["pred_nrm"])):
            if len(P) < 2:
                continue
            Pt = torch.as_tensor((P - com) / ro, dtype=torch.float64, device=dev)[None]
            Nt = torch.as_tensor(Nn / np.linalg.norm(Nn, axis=1, keepdims=True), dtype=torch.float64, device=dev)[None]
            mt = torch.full((1, len(P)), float(x["mu_tor"]) / ro, dtype=torch.float64, device=dev)
            W6 = ray_wrench_matrix(Pt, Nt, torch.zeros(1, 3, dtype=torch.float64, device=dev), 1.0, 0.6, mt)
            a = alpha_batched(W6, direction_set(Pt)).min().item()
            W8 = cone_edge_wrenches(Pt, Nt, torch.zeros(1, 3, dtype=torch.float64, device=dev), 1.0, mu=MU, n_edges=N_EDGES)
            rh = (support_sum(W8, dirs) / l1).min().item()
            if tag == "plan": a_plan[k], rh_plan[k] = a, rh
            else: a_geo[k], rh_geo[k] = a, rh
    print("\n[4] 예측 접촉 기하로 다시 계산한 여유")
    okg = np.isfinite(a_geo) & np.isfinite(a_plan)
    print(f"  α(예측)/α(계획) 중앙값: 합격 {np.nanmedian((a_geo/np.maximum(a_plan,1e-9))[p&okg]):.2f}, 불합격 {np.nanmedian((a_geo/np.maximum(a_plan,1e-9))[f&okg]):.2f}")
    print(f"  합격·불합격 AUC(불합격이 더 작음): α(계획) {auc(-a_plan[f&okg], -a_plan[p&okg]):.3f}, α(예측) {auc(-a_geo[f&okg], -a_geo[p&okg]):.3f}, "
          f"ρ̂(계획) {auc(-rh_plan[f&okg], -rh_plan[p&okg]):.3f}, ρ̂(예측) {auc(-rh_geo[f&okg], -rh_geo[p&okg]):.3f}")
    ratio = a_geo / np.maximum(a_plan, 1e-9)
    print(f"  α(예측)/α(계획) 비율로 가르는 AUC(불합격이 더 작음): {auc(-ratio[f&okg], -ratio[p&okg]):.3f}")
    np.savez_compressed(OUT / "geo_realization.npz", idx=np.array([x["idx"] for x in rows]), verdict=v, pred_d_max=pd_max,
                        pred_ang_max=pa_max, meas_d_max=md_max, meas_ang_max=ma_max, miss=miss, alpha_plan=a_plan, alpha_geo=a_geo,
                        rho_hat_plan=rh_plan, rho_hat_geo=rh_geo)
    print(f"\n저장 {OUT/'geo_realization.npz'}  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
