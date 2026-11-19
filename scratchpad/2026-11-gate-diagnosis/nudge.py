"""안정성 시험: 외란 없이 쥔 상태에서 물체를 살짝 밀었다(속도 주입) 놓고, 0.3초 뒤 되돌아오는지.

제어: 법선 힘 제어(정착 힘 고정, 접선 고정) = hybrid_check 의 B, 그리고 위치 서보 O.
밀기: 이동 6방향(±x,±y,±z) 속도 v0, 회전 6방향 각속도 w0 — 한 스텝 주입.
지표: 최대 이탈 대비 최종 이탈(되돌아오면 작음), 실패 판정(이탈 > 0.1·r_obj 또는 20°).
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])
V0 = 0.02      # m/s
W0 = 0.5       # rad/s


def run(args):
    job, mode = args
    import mujoco
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _pose_deviation
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"], timestep=1e-3,
                    settle_steps=600, hold_steps=300)
    n = len(job["pos"]); r_obj = job["r_obj"]; f = cfg.f_cmd
    xml = build_scene_xml(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg)
    xml = xml.replace("</actuator>", "".join(
        f'<motor joint="press{i}" gear="1" ctrlrange="0 {f}"/>' for i in range(n)) + "</actuator>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    gain0 = model.actuator_gainprm[:n, 0].copy(); bias0 = model.actuator_biasprm[:n, :3].copy()
    target = np.clip(np.asarray(job["preload"], float), 0.02 * f, f)
    press = cfg.start_gap + target / cfg.kp

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
    kicks = [np.r_[s * V0 * np.eye(3)[a], 0, 0, 0] for a in range(3) for s in (1, -1)] + \
            [np.r_[0, 0, 0, s * W0 * np.eye(3)[a]] for a in range(3) for s in (1, -1)]
    out = []
    for k in kicks:
        settle()
        ref = data.qpos[:7].copy()
        if np.linalg.norm(ref[:3]) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
            return dict(idx=job["idx"], mode=mode, settled=False)
        if mode == "B":
            F = data.actuator_force[:n].copy()
            servo(False); data.ctrl[:n] = 0.0; data.ctrl[n:] = F
        data.qvel[:6] += k
        peak = 0.0; fail = False
        for _ in range(cfg.hold_steps):
            mujoco.mj_step(model, data)
            dp, dr = _pose_deviation(data, ref)
            e = max(dp / (cfg.trans_tol_rel * r_obj), dr / cfg.rot_tol_deg)   # 1 = 실패선
            peak = max(peak, e)
            if e > 1 or not np.isfinite(data.qpos).all():
                fail = True
                break
        out.append(dict(fail=fail, peak=peak, final=e, ret=(e / max(peak, 1e-9))))
    return dict(idx=job["idx"], mode=mode, settled=True, kicks=out)


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    fail8 = [29093, 62729, 102750, 129568, 272962, 342257, 485907, 518538]
    hy = pickle.load(open(TMP / "hybrid_raw.pkl", "rb"))
    fixed8 = sorted({r["idx"] for r in hy if r["cond"] == "C" and all(r["held"])}
                    - {r["idx"] for r in hy if r["cond"] == "O" and all(r["held"])})
    pass10 = sorted({r["idx"] for r in hy if r["cond"] == "O" and all(r.get("held", [False]))})
    m = _load_records(ROOT)
    jobs = _prepare(np.array(fail8 + fixed8 + pass10), m, ROOT)
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(run, [(j, md) for j in jobs for md in ("B", "O")], chunksize=1)
    pickle.dump(res, open(TMP / "nudge_raw.pkl", "wb"))
    print(f"(밀기: 이동 {V0*100:.0f}cm/s 또는 회전 {W0} rad/s, 12방향 | 이탈 1.0 = 실패선)")
    for name, g in (("하이브리드로도 못 고친 8", fail8), ("하이브리드로 고쳐진 8", fixed8), ("원래 합격 10", pass10)):
        for md in ("B", "O"):
            rs = [r for r in res if r["idx"] in g and r["mode"] == md and r["settled"]]
            ks = [k for r in rs for k in r["kicks"]]
            print(f"{name:22s} {'법선 힘 제어' if md=='B' else '위치 서보':8s} | 파지 {len(rs)} | 밀었을 때 떨어진 비율 {np.mean([k['fail'] for k in ks]):.2f} | "
                  f"한 방향이라도 떨어진 파지 {sum(any(k['fail'] for k in r['kicks']) for r in rs)}/{len(rs)} | "
                  f"버틴 경우 최대 이탈 중앙값 {np.median([k['peak'] for k in ks if not k['fail']] or [np.nan]):.2f}, 최종/최대 {np.median([k['ret'] for k in ks if not k['fail']] or [np.nan]):.2f}")


if __name__ == "__main__":
    main()
