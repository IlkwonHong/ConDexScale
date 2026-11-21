"""게이트 조건 비교: 예압 하한 / 법선 힘 제어(하이브리드) / 외란별 최적 배분 / + 오차.

장면은 sim_eval.build_scene_xml 그대로 + 팁 법선 관절에 힘 모터를 추가.
정착은 원래대로 위치 서보(예압 피드백 보정)로 하고, 외란 시작 순간
힘 제어 조건이면 서보를 끄고(이득 0) 모터에 법선 힘을 명령한다.
접선은 원래부터 무한 강성(팔이 잡아줌) = 위치 제어 → 법선 힘/접선 위치 하이브리드.

조건: O 원래, A 예압 하한 30%(위치 서보), B 힘 제어·정착 힘 고정,
      C 힘 제어·외란별 최대 쥐기 배분, D = C + 법선 힘 오차 σ = 상한의 2%
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])
CONDS = ["O", "A", "B", "C", "D"]
PRE_LB = 0.30


def lp_alloc(Wf, N, w_target, lb=0.0):
    """ΣWx = w_target, 앵커별 압력 ∈ [lb, 1] 에서 총압력 최대 → 앵커별 압력 [N] (없으면 None)."""
    from scipy.optimize import linprog
    P = np.zeros((N, 6 * N))
    for j in range(N):
        P[j, 6 * j:6 * j + 6] = 1.0
    A_ub = np.vstack([P, -P]); b_ub = np.concatenate([np.ones(N), -lb * np.ones(N)])
    r = linprog(-np.ones(6 * N), A_ub=A_ub, b_ub=b_ub, A_eq=Wf, b_eq=w_target,
                bounds=[(0, None)] * 6 * N, method="highs")
    return None if r.status != 0 else P @ r.x


def run(args):
    job, cond = args
    import mujoco
    import torch
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _pose_deviation
    from contraction_gt.metric import ray_wrench_matrix, _flatten_W
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    n = len(job["pos"]); r_obj = job["r_obj"]; f = cfg.f_cmd
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)
    Wf = _flatten_W(ray_wrench_matrix(t(job["pos_n"]), t(job["nrm"]), torch.zeros(1, 3, dtype=torch.float64),
                                      1.0, 0.6, job["mu_tor"] / r_obj))[0].numpy()
    preload = np.asarray(job["preload"], float)
    lb_ok = True
    if cond == "A":
        p = lp_alloc(Wf, n, np.zeros(6), PRE_LB)
        if p is None:
            lb_ok = False
        else:
            preload = p * f
    xml = build_scene_xml(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg)
    motors = "".join(f'<motor joint="press{i}" gear="1" ctrlrange="0 {f}"/>' for i in range(n))
    xml = xml.replace("</actuator>", motors + "</actuator>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    gain0 = model.actuator_gainprm[:n, 0].copy(); bias0 = model.actuator_biasprm[:n, :3].copy()
    target = np.clip(preload, 0.02 * f, f)
    press = cfg.start_gap + target / cfg.kp
    rng = np.random.default_rng(job["idx"])

    def servo(on):
        model.actuator_gainprm[:n, 0] = gain0 if on else 0.0
        model.actuator_biasprm[:n, :3] = bias0 if on else 0.0

    def settle():
        servo(True)
        mujoco.mj_resetData(model, data)
        data.ctrl[:n] = press; data.ctrl[n:] = 0.0
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    for _ in range(4):
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    held_all, info = [], []
    for w, lv in zip(job["dirs"], job["levels"][:, 0]):
        settle()
        ref = data.qpos[:7].copy()
        if np.linalg.norm(ref[:3]) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
            return dict(idx=job["idx"], cond=cond, settled=False)
        F_set = data.actuator_force[:n].copy()
        if cond in ("B", "C", "D"):
            if cond == "B":
                fn = F_set
            else:
                p = lp_alloc(Wf, n, -lv * w)
                fn = F_set if p is None else p * f
                if cond == "D":
                    fn = fn + rng.normal(0.0, 0.02 * f, n)
            servo(False)
            data.ctrl[:n] = 0.0
            data.ctrl[n:] = np.clip(fn, 0.0, f)
        data.xfrc_applied[ob, :3] = lv * f * w[:3]
        data.xfrc_applied[ob, 3:] = lv * f * r_obj * w[3:]
        ok = True
        for _ in range(cfg.hold_steps):
            mujoco.mj_step(model, data)
            dp, dr = _pose_deviation(data, ref)
            if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg or not np.isfinite(data.qpos).all():
                ok = False
                break
        held_all.append(ok)
        info.append(dict(dp=dp, dr=dr))
    return dict(idx=job["idx"], cond=cond, settled=True, held=held_all, lb_ok=lb_ok, info=info)


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    rows = pickle.load(open(TMP / "during_load_rows.pkl", "rb"))
    fail_idx = sorted({r["idx"] for r in rows if r["v"] == "fail"})[:16]
    pass_idx = sorted({r["idx"] for r in rows if r["v"] == "pass"})[:10]
    m = _load_records(ROOT)
    jobs = _prepare(np.array(fail_idx + pass_idx), m, ROOT)
    man = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in man}
    for j in jobs:
        i = j["idx"]
        ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
        com = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")["com"] * float(m["total_scale"][i])
        j["pos_n"] = (j["pos"] - com) / j["r_obj"]
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(run, [(j, c) for j in jobs for c in CONDS], chunksize=1)
    pickle.dump(res, open(TMP / "hybrid_raw.pkl", "wb"))
    for grp, ids in (("불합격", fail_idx), ("합격", pass_idx)):
        print(f"--- 원래 {grp} {len(ids)}개 ---")
        for c in CONDS:
            rs = [r for r in res if r["cond"] == c and r["idx"] in ids]
            st = [r for r in rs if r["settled"]]
            gp = np.mean([all(r["held"]) for r in st]) if st else float("nan")
            dp = np.mean([np.mean(r["held"]) for r in st]) if st else float("nan")
            extra = f" | 예압 하한 불가능 {sum(not r['lb_ok'] for r in rs)}" if c == "A" else ""
            print(f"{c}: 정착 {len(st)}/{len(rs)} | 파지 합격 비율 {gp:.2f} | 방향별 버틴 비율 {dp:.2f}{extra}")


if __name__ == "__main__":
    main()
