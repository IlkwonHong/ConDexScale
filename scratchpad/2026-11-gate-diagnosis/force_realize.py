"""C 조건(법선 힘 제어 + 외란별 최적 배분, 접선 고정)에서 팁별 실제 접촉 힘 vs LP 배분 힘.

접촉 힘(물체가 받는 힘) = mj_contactForce 를 data.contact.frame 으로 월드 변환,
부호는 기하로 결정(팁 중심 → 접촉점 방향이 미는 방향). (MuJoCo 2.3.7 함정 메모 참조)
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
from c_fail_check import W_of  # noqa: E402
from c_full import lp_x  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])
MU = 0.6


def run(job):
    import mujoco
    import torch
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _pose_deviation
    from contraction_gt.metric import tangent_frames
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"], timestep=1e-3,
                    settle_steps=600, hold_steps=300)
    n = len(job["pos"]); r_obj = job["r_obj"]; f = cfg.f_cmd
    Wplan = W_of(job["pos_n"], job["nrm"], job["mu_tor"] / r_obj)
    xml = build_scene_xml(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg)
    xml = xml.replace("</actuator>", "".join(
        f'<motor joint="press{i}" gear="1" ctrlrange="0 {f}"/>' for i in range(n)) + "</actuator>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    tipg = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_g{i}") for i in range(n)]
    gain0 = model.actuator_gainprm[:n, 0].copy(); bias0 = model.actuator_biasprm[:n, :3].copy()
    target = np.clip(np.asarray(job["preload"], float), 0.02 * f, f)
    press = cfg.start_gap + target / cfg.kp
    nu = job["nrm"] / np.linalg.norm(job["nrm"], axis=1, keepdims=True)
    nn, t1, t2 = (x[0].numpy() for x in tangent_frames(torch.as_tensor(nu).unsqueeze(0)))
    f6 = np.zeros(6)

    def servo(on):
        model.actuator_gainprm[:n, 0] = gain0 if on else 0.0
        model.actuator_biasprm[:n, :3] = bias0 if on else 0.0

    def settle():
        servo(True)
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = 0.0; data.ctrl[:n] = press
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    def contact_forces():
        """팁별 (물체가 받는 힘 벡터 합 [3], 힘 가중 실제 법선 [3]) — 월드 프레임."""
        F = np.zeros((n, 3)); Nr = np.zeros((n, 3))
        for c in range(data.ncon):
            g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
            tg = g1 if g1 in tipg else (g2 if g2 in tipg else None)
            if tg is None:
                continue
            k = tipg.index(tg)
            mujoco.mj_contactForce(model, data, c, f6)
            fr = np.array(data.contact.frame[c]).reshape(3, 3)
            fw = fr.T @ f6[:3]
            push = np.array(data.contact.pos[c]) - data.geom_xpos[tg]
            push /= max(np.linalg.norm(push), 1e-12)
            if fw @ push < 0:
                fw = -fw
            F[k] += fw; Nr[k] += abs(f6[0]) * push
        for k in range(n):
            if np.linalg.norm(Nr[k]) > 0:
                Nr[k] /= np.linalg.norm(Nr[k])
        return F, Nr

    for _ in range(4):
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    out = []
    for w, lv in zip(job["dirs"], job["levels"][:, 0]):
        settle()
        ref = data.qpos[:7].copy()
        x = lp_x(Wplan, n, -lv * w)
        if x is None:
            continue
        X = x.reshape(n, 6)
        Flp = (X[:, 0:1] * (nn + MU * t1) + X[:, 1:2] * (nn - MU * t1) + X[:, 2:3] * (nn + MU * t2)
               + X[:, 3:4] * (nn - MU * t2) + (X[:, 4:5] + X[:, 5:6]) * nn) * f
        fn_lp = (Flp * nn).sum(1)
        servo(False)
        data.ctrl[:n] = 0.0
        data.ctrl[n:] = np.clip(fn_lp, 0, f)
        data.xfrc_applied[ob, :3] = lv * f * w[:3]
        data.xfrc_applied[ob, 3:] = lv * f * r_obj * w[3:]
        snaps = []; ok = True
        for s in range(1, cfg.hold_steps + 1):
            mujoco.mj_step(model, data)
            if s in (5, 20) or s % 10 == 0:
                F, Nr = contact_forces()
                snaps.append(dict(s=s, F=F, Nr=Nr))
            dp, dr = _pose_deviation(data, ref)
            if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg:
                ok = False
                F, Nr = contact_forces(); snaps.append(dict(s=s, F=F, Nr=Nr))
                break
        out.append(dict(held=ok, Flp=Flp, nn=nn, snaps=snaps, ext=np.r_[lv * f * w[:3]]))
    return dict(idx=job["idx"], dirs=out, f=f)


def analyse(res):
    rows = []
    for r in res:
        for d in r["dirs"]:
            s5 = next(s for s in d["snaps"] if s["s"] == 5) if any(s["s"] == 5 for s in d["snaps"]) else d["snaps"][0]
            F, Nr, Flp, nn = s5["F"], s5["Nr"], d["Flp"], d["nn"]
            n_act = np.einsum("ij,ij->i", F, Nr)
            t_act = np.linalg.norm(F - n_act[:, None] * Nr, axis=1)
            n_lp = np.einsum("ij,ij->i", Flp, nn)
            t_lp = np.linalg.norm(Flp - n_lp[:, None] * nn, axis=1)
            ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", Nr, nn), -1, 1)))
            touched = np.linalg.norm(Nr, axis=1) > 0
            rat_act = np.where(n_act > 1e-3, t_act / np.maximum(n_act, 1e-9), np.nan)
            rat_lp = t_lp / np.maximum(n_lp, 1e-9)
            k = int(np.nanargmax(np.where(touched, rat_act, -1)))       # 마찰을 가장 많이 쓰는 팁
            rows.append(dict(held=d["held"], err=np.linalg.norm(F - Flp, axis=1).sum() / max(np.linalg.norm(Flp, axis=1).sum(), 1e-9),
                             net=np.linalg.norm(F.sum(0) + d["ext"]) / max(np.linalg.norm(d["ext"]), 1e-9),
                             k_rat_act=rat_act[k], k_rat_lp=rat_lp[k], k_n_act=n_act[k] / r["f"], k_n_lp=n_lp[k] / r["f"],
                             k_t_act=t_act[k] / r["f"], k_t_lp=t_lp[k] / r["f"], ang_med=np.nanmedian(ang[touched]) if touched.any() else np.nan,
                             ang_k=ang[k]))
    return rows


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
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(run, jobs, chunksize=1)
    pickle.dump(res, open(TMP / "force_realize_raw.pkl", "wb"))
    rows8 = analyse([r for r in res if r["idx"] in fail8])
    rowsP = analyse([r for r in res if r["idx"] in pass10])
    med = lambda rs, k: np.nanmedian([x[k] for x in rs]) if rs else np.nan
    print("(외란 후 5스텝 시점, 힘 단위 = 상한 5N 대비)")
    for name, rs in (("못 고친 8 · 떨어진 방향", [x for x in rows8 if not x["held"]]),
                     ("못 고친 8 · 버틴 방향", [x for x in rows8 if x["held"]]),
                     ("원래 합격 10", rowsP)):
        print(f"{name:20s} n={len(rs):3d} | 팁별 힘 오차 합/LP 힘 합 {med(rs,'err'):.2f} | 힘 합 균형 오차 {med(rs,'net'):.2f} | "
              f"마찰 최다 팁: 접선/법선 실제 {med(rs,'k_rat_act'):.2f} vs LP {med(rs,'k_rat_lp'):.2f}, "
              f"법선 실제 {med(rs,'k_n_act'):.2f} vs LP {med(rs,'k_n_lp'):.2f}, 접선 실제 {med(rs,'k_t_act'):.2f} vs LP {med(rs,'k_t_lp'):.2f} | "
              f"실제 법선↔계획 법선 각도: 전체 팁 {med(rs,'ang_med'):.1f}°, 그 팁 {med(rs,'ang_k'):.1f}°")


if __name__ == "__main__":
    main()
