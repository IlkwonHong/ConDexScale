"""C_FULL: 법선 + 접선 힘을 모두 외란별 최적 배분(LP 해)대로 명령.

장면: sim_eval active 모드(팁마다 수평 2자유도 + 임피던스 스프링 + 힘 모터) + 법선 힘 모터.
액추에이터 순서: 법선 서보 n, 수평 모터 2n, 법선 힘 모터 n.
진단: 떨어질 때 마찰 한계에 닿은 팁에서 LP 해가 요구한 접선/법선 비율.
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
from c_fail_check import W_of  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])


def lp_x(Wf, N, w_target):
    from scipy.optimize import linprog
    P = np.zeros((N, 6 * N))
    for j in range(N):
        P[j, 6 * j:6 * j + 6] = 1.0
    r = linprog(-np.ones(6 * N), A_ub=P, b_ub=np.ones(N), A_eq=Wf, b_eq=w_target,
                bounds=[(0, None)] * 6 * N, method="highs")
    return None if r.status != 0 else r.x


def run(args):
    job, var = args
    import mujoco
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _pose_deviation
    from contraction_gt.metric import tangent_frames
    import torch
    lat_k = {"FULL": 2000.0, "FULL_K0": 0.0, "NORM_ONLY_ACT": 2000.0}[var]
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"], timestep=1e-3,
                    settle_steps=600, hold_steps=300, active=True, lat_stiffness=lat_k)
    n = len(job["pos"]); r_obj = job["r_obj"]; f = cfg.f_cmd
    Wplan = W_of(job["pos_n"], job["nrm"], job["mu_tor"] / r_obj)
    xml = build_scene_xml(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg)
    xml = xml.replace("</actuator>", "".join(
        f'<motor joint="press{i}" gear="1" ctrlrange="0 {f}"/>' for i in range(n)) + "</actuator>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    gain0 = model.actuator_gainprm[:n, 0].copy(); bias0 = model.actuator_biasprm[:n, :3].copy()
    target = np.clip(np.asarray(job["preload"], float), 0.02 * f, f)
    press = cfg.start_gap + target / cfg.kp
    mujoco.mj_forward(model, data)
    axes = []
    for i in range(n):
        R = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{i}")].reshape(3, 3)
        axes.append((R[:, 0].copy(), R[:, 1].copy()))
    nn, t1, t2 = (x[0].numpy() for x in tangent_frames(torch.as_tensor(job["nrm"], dtype=torch.float64).unsqueeze(0)))
    mu = 0.6

    def servo(on):
        model.actuator_gainprm[:n, 0] = gain0 if on else 0.0
        model.actuator_biasprm[:n, :3] = bias0 if on else 0.0

    def settle():
        servo(True)
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0.0; data.ctrl[:n] = press
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    for _ in range(4):
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    out = []
    for w, lv in zip(job["dirs"], job["levels"][:, 0]):
        settle()
        ref = data.qpos[:7].copy()
        if np.linalg.norm(ref[:3]) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
            return dict(idx=job["idx"], var=var, settled=False)
        x = lp_x(Wplan, n, -lv * w)
        if x is None:
            out.append(dict(held=False, lp=False)); continue
        X = x.reshape(n, 6)
        # 팁별 힘 벡터 (물체 프레임, N): 광선 방향 n±μt, n, n
        Fv = (X[:, 0:1] * (nn + mu * t1) + X[:, 1:2] * (nn - mu * t1) + X[:, 2:3] * (nn + mu * t2)
              + X[:, 3:4] * (nn - mu * t2) + (X[:, 4:5] + X[:, 5:6]) * nn) * f
        fn = (Fv * nn).sum(1); ft = Fv - fn[:, None] * nn
        req_ratio = np.linalg.norm(ft, axis=1) / np.maximum(fn, 1e-9)
        servo(False)
        data.ctrl[:n] = 0.0
        data.ctrl[3 * n:4 * n] = np.clip(fn, 0, f)
        for i, (ax, ay) in enumerate(axes):
            if var != "NORM_ONLY_ACT":
                data.ctrl[n + 2 * i] = float(ft[i] @ ax)
                data.ctrl[n + 2 * i + 1] = float(ft[i] @ ay)
        data.xfrc_applied[ob, :3] = lv * f * w[:3]
        data.xfrc_applied[ob, 3:] = lv * f * r_obj * w[3:]
        ok = True
        for _ in range(cfg.hold_steps):
            mujoco.mj_step(model, data)
            dp, dr = _pose_deviation(data, ref)
            if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg or not np.isfinite(data.qpos).all():
                ok = False
                break
        out.append(dict(held=ok, lp=True, req_ratio_max=float(req_ratio.max())))
    return dict(idx=job["idx"], var=var, settled=True, dirs=out)


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    fail8 = [29093, 62729, 102750, 129568, 272962, 342257, 485907, 518538]
    hy = pickle.load(open(TMP / "hybrid_raw.pkl", "rb"))
    pass10 = sorted({r["idx"] for r in hy if r["cond"] == "O" and all(r.get("held", [False]))})
    m = _load_records(ROOT)
    jobs = _prepare(np.array(fail8 + pass10), m, ROOT)
    man = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in man}
    for j in jobs:
        i = j["idx"]
        ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
        com = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")["com"] * float(m["total_scale"][i])
        j["pos_n"] = (j["pos"] - com) / j["r_obj"]
    vs = ["FULL", "FULL_K0", "NORM_ONLY_ACT"]
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(run, [(j, v) for j in jobs for v in vs], chunksize=1)
    pickle.dump(res, open(TMP / "cfull_raw.pkl", "wb"))
    for grp, ids in (("남은 불합격 8", fail8), ("원래 합격 10", pass10)):
        for v in vs:
            rs = [r for r in res if r["var"] == v and r["idx"] in ids and r["settled"]]
            print(f"{grp} {v:14s}: 파지 합격 {sum(all(d['held'] for d in r['dirs']) for r in rs)}/{len(rs)} | "
                  f"방향별 {np.mean([d['held'] for r in rs for d in r['dirs']]):.2f}")
    rs = [d for r in res if r["var"] == "FULL" and r["idx"] in fail8 and r["settled"] for d in r["dirs"] if d.get("lp")]
    for name, sel in (("떨어진", [d for d in rs if not d["held"]]), ("버틴", [d for d in rs if d["held"]])):
        if sel:
            print(f"FULL {name} 방향: LP 가 요구한 최대 접선/법선 비율 중앙값 {np.median([d['req_ratio_max'] for d in sel]):.2f} (μ=0.6)")


if __name__ == "__main__":
    main()
