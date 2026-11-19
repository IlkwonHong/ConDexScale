"""외란 중 접촉이 끊기는 순간의 팁 상태 기록.

팁마다: 슬라이드 관절 위치 q, 서보 목표 ctrl, 서보 힘, 접촉 여부,
물체 이동의 팁 법선 성분(+ = 물체가 팁 쪽에서 멀어짐).
"""
import pickle
import sys
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])


def run_job(job):
    import mujoco
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _pose_deviation
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    model = mujoco.MjModel.from_xml_string(build_scene_xml(
        ("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg))
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    n = len(job["pos"]); r_obj = job["r_obj"]
    nrm = job["nrm"] / np.linalg.norm(job["nrm"], axis=1, keepdims=True)
    tipg = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_g{i}") for i in range(n)]
    target = np.clip(np.asarray(job["preload"], float), 0.02 * cfg.f_cmd, cfg.f_cmd)
    press = cfg.start_gap + target / cfg.kp

    def settle():
        mujoco.mj_resetData(model, data)
        data.ctrl[:n] = press
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    def touching():
        t = np.zeros(n, bool)
        for c in range(data.ncon):
            for g in (int(data.contact.geom1[c]), int(data.contact.geom2[c])):
                if g in tipg:
                    t[tipg.index(g)] = True
        return t

    for _ in range(4):
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    out = []
    for d, (w, lv) in enumerate(zip(job["dirs"], job["levels"][:, 0])):
        settle()
        ref = data.qpos[:7].copy()
        log = []

        def snap(s):
            log.append(dict(step=s, q=data.qpos[7:7 + n].copy() if model.nq >= 7 + n else None,
                            ctrl=data.ctrl[:n].copy(), f=data.actuator_force[:n].copy(),
                            touch=touching(), obj=data.qpos[:3] - ref[:3]))
        snap(0)
        data.xfrc_applied[ob, :3] = lv * cfg.f_cmd * w[:3]
        data.xfrc_applied[ob, 3:] = lv * cfg.f_cmd * r_obj * w[3:]
        fail = None
        for s in range(1, cfg.hold_steps + 1):
            mujoco.mj_step(model, data)
            snap(s)
            dp, dr = _pose_deviation(data, ref)
            if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg:
                fail = s
                break
        out.append(dict(d=d, held=fail is None, log=log))
    return dict(idx=job["idx"], n=n, nrm=nrm, f_cmd=cfg.f_cmd, kp=cfg.kp, dirs=out)


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    rows = pickle.load(open(TMP / "during_load_rows.pkl", "rb"))
    fail_idx = sorted({r["idx"] for r in rows if r["v"] == "fail"})[:16]
    m = _load_records(ROOT)
    jobs = _prepare(np.array(fail_idx), m, ROOT)
    with get_context("spawn").Pool(8) as pool:
        sims = pool.map(run_job, jobs, chunksize=1)
    pickle.dump(sims, open(TMP / "why_raw.pkl", "wb"))

    kinds = Counter()
    gap_lost, push_lost, f_before = [], [], []
    ft_first = Counter()
    for s in sims:
        for dd in s["dirs"]:
            if dd["held"]:
                continue
            L = dd["log"]
            t0 = L[0]["touch"]
            first = None
            for k in range(1, len(L)):
                lost = t0 & ~L[k]["touch"]
                if lost.any():
                    first = k; break
            if first is None:
                kinds["접촉 유지한 채 실패"] += 1; continue
            for t in np.where(t0 & ~L[first]["touch"])[0]:
                pre = L[max(first - 1, 0)]
                f_before.append(pre["f"][t] / s["f_cmd"])
                # 물체가 이 팁에서 멀어진 거리 (팁 법선 = 팁이 미는 방향)
                away = float(L[first]["obj"] @ s["nrm"][t])
                push_lost.append(away)
                # 서보 여유: 목표 - 현재 위치 (남은 전진 가능 거리)
                if pre["q"] is not None:
                    gap_lost.append(float(pre["ctrl"][t] - pre["q"][t]))
                ft_first[L[first]["step"] // 10 * 10] += 1
            kinds["팁 접촉 끊김 후 실패"] += 1
    print("떨어진 방향 분류:", dict(kinds))
    f_before = np.array(f_before); push_lost = np.array(push_lost)
    print(f"끊기기 직전 팁 힘 / 상한: 중앙값 {np.median(f_before):.2f}, 상한 포화(>0.98) 비율 {np.mean(f_before > 0.98):.2f}, "
          f"거의 0(<0.05) 비율 {np.mean(f_before < 0.05):.2f}")
    print(f"끊긴 팁에서 물체가 멀어진 거리[mm]: 중앙값 {1e3*np.median(push_lost):.3f}, 양수(멀어짐) 비율 {np.mean(push_lost > 0):.2f}")
    if gap_lost:
        g = np.array(gap_lost)
        print(f"끊기기 직전 서보 남은 전진 여유[mm]: 중앙값 {1e3*np.median(g):.3f}, p10 {1e3*np.percentile(g,10):.3f}, p90 {1e3*np.percentile(g,90):.3f}")
    print("첫 끊김 시점(스텝, 10단위):", sorted(ft_first.items())[:10])


if __name__ == "__main__":
    main()
