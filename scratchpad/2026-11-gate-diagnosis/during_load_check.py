"""외란 중 접촉 기록 → 실패 직전/실패 순간 접촉으로 알파값 재계산.

게이트와 같은 장면·설정(수동 위치 서보, 예압 = LP 평형 배분, 시행별 리셋)으로
방향마다 '계획 알파 x 0.25' 외란을 가하고, 10스텝마다 자세와 접촉을 기록한다.
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ROOT = Path("data/contraction_gt_300k")
CHECK = 10


def merge_contacts(contacts, n):
    P, N, F = [], [], []
    for t in range(n):
        cs = [c for c in contacts if c["tip"] == t]
        if not cs:
            continue
        w = np.array([max(c["f_normal"], 1e-9) for c in cs])
        P.append((w[:, None] * np.stack([c["pos_obj"] for c in cs])).sum(0) / w.sum())
        nv = (w[:, None] * np.stack([c["normal_obj"] for c in cs])).sum(0)
        N.append(nv / np.linalg.norm(nv)); F.append(w.sum())
    return np.array(P).reshape(-1, 3), np.array(N).reshape(-1, 3), np.array(F)


def run_job(job):
    import mujoco
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _record_contacts, _pose_deviation
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    spec = ("coacd", job["coacd"], job["scale"])
    model = mujoco.MjModel.from_xml_string(build_scene_xml(spec, job["pos"], job["nrm"], cfg))
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    n = len(job["pos"]); r_obj = job["r_obj"]
    target = np.clip(np.asarray(job["preload"], float), 0.02 * cfg.f_cmd, cfg.f_cmd)
    press = cfg.start_gap + target / cfg.kp

    def settle():
        mujoco.mj_resetData(model, data)
        data.ctrl[:n] = press
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    for _ in range(4):                          # sim_eval 와 같은 예압 피드백 보정
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    out = []
    for d, (w, lv) in enumerate(zip(job["dirs"], job["levels"][:, 0])):
        settle()
        ref = data.qpos[:7].copy()
        if np.linalg.norm(ref[:3]) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
            return dict(idx=job["idx"], settled=False)
        snaps = [dict(step=0, dev=(0.0, 0.0), c=_record_contacts(model, data, n, ob)["contacts"])]
        data.xfrc_applied[ob, :3] = lv * cfg.f_cmd * w[:3]
        data.xfrc_applied[ob, 3:] = lv * cfg.f_cmd * r_obj * w[3:]
        fail_step = None
        for s in range(1, cfg.hold_steps + 1):
            mujoco.mj_step(model, data)
            if s % CHECK == 0:
                dp, dr = _pose_deviation(data, ref)
                snaps.append(dict(step=s, dev=(dp, dr), c=_record_contacts(model, data, n, ob)["contacts"]))
                if fail_step is None and (dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg):
                    fail_step = s
                    break
        out.append(dict(d=d, lv=float(lv), held=fail_step is None, fail_step=fail_step, snaps=snaps))
    return dict(idx=job["idx"], settled=True, n_tip=n, dirs=out)


def main():
    import torch
    from contraction_gt.gate_runner import _load_records, _prepare
    from contraction_gt.metric import ray_wrench_matrix, alpha_reference
    res = pickle.load(open(ROOT / "wrench_similarity/realized_contacts.pkl", "rb"))
    res = [r for r in res if r["verdict"] in ("pass", "fail") and r["settled"]]
    m = _load_records(ROOT)
    jobs = {j["idx"]: j for j in _prepare(np.array([r["idx"] for r in res]), m, ROOT)}
    rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in rows}

    def com_of(i):
        ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
        return np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")["com"] * float(m["total_scale"][i])

    def alpha(pos, nrm, com, j, dirs):
        if len(pos) == 0:
            return np.zeros(len(dirs))
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)
        W = ray_wrench_matrix(t(pos), t(nrm), t(com), j["r_obj"], 0.6, j["mu_tor"])
        return alpha_reference(W, t(dirs))[0].numpy()

    # 재현 검증을 통과한 레코드만 (계획 접촉 재계산 == 저장 알파)
    good = {"pass": [], "fail": []}
    for r in res:
        i, j = r["idx"], jobs[r["idx"]]
        ds = sorted(set(np.argsort(m["alpha"][i])[:3].tolist()) | set(range(6)))
        if np.abs(alpha(j["pos"], j["nrm"], com_of(i), j, j["dirs"]) - m["alpha"][i][ds]).max() < 1e-3:
            good[r["verdict"]].append(i)
    rng = np.random.default_rng(0)
    pick = {"fail": rng.choice(good["fail"], 40, replace=False).tolist(),
            "pass": rng.choice(good["pass"], 20, replace=False).tolist()}
    verdict = {i: k for k, v in pick.items() for i in v}
    with get_context("spawn").Pool(8) as pool:
        sims = pool.map(run_job, [jobs[i] for i in verdict], chunksize=1)
    pickle.dump(sims, open(sys.argv[1] + "/during_load_raw.pkl", "wb"))

    rows = []
    for s in sims:
        i = s["idx"]; j = jobs[i]; com = com_of(i)
        if not s["settled"]:
            rows.append(dict(v=verdict[i], settled=False)); continue
        for dd in s["dirs"]:
            dirv = j["dirs"][dd["d"]][None]
            a = []
            for sn in dd["snaps"]:
                P, N, F = merge_contacts(sn["c"], s["n_tip"])
                a.append((sn["step"], alpha(P, N, com, j, dirv)[0], len(P)))
            pre = a[-2] if (not dd["held"] and len(a) >= 2) else a[-1]
            rows.append(dict(v=verdict[i], idx=i, d=dd["d"], lv=dd["lv"], held=dd["held"],
                             a_plan=alpha(j["pos"], j["nrm"], com, j, dirv)[0],
                             a0=a[0][1], a_pre=pre[1], a_min=min(x[1] for x in a),
                             touch0=a[0][2], touch_pre=pre[2], n_tip=s["n_tip"]))
    pickle.dump(rows, open(sys.argv[1] + "/during_load_rows.pkl", "wb"))

    g = [r for r in rows if r.get("settled", True)]
    print(f"표본: 불합격 {len(pick['fail'])}, 합격 {len(pick['pass'])} | 정착 실패 {sum(not r.get('settled', True) for r in rows)}")
    for v in ("pass", "fail"):
        gr = {}
        for r in g:
            if r["v"] == v:
                gr.setdefault(r["idx"], []).append(r["held"])
        print(f"[재현] {v}: 시뮬 재실행 합격 비율 {np.mean([all(h) for h in gr.values()]):.2f} ({len(gr)}개)")
    fd = [r for r in g if not r["held"]]
    hd = [r for r in g if r["held"]]
    for name, rs in (("떨어진 방향", fd), ("버틴 방향", hd)):
        if not rs:
            continue
        q = lambda k: np.array([r[k] / max(r["lv"], 1e-9) for r in rs])
        print(f"{name} n={len(rs)} | 알파/외란 중앙값: 계획 {np.median(q('a_plan')):.2f}, 쥔 직후 {np.median(q('a0')):.2f}, "
              f"실패 직전 {np.median(q('a_pre')):.2f}, 외란 중 최소 {np.median(q('a_min')):.2f} | "
              f"실패 직전 알파<외란 비율 {np.mean(q('a_pre') < 1):.2f}, 외란 중 최소<외란 비율 {np.mean(q('a_min') < 1):.2f} | "
              f"닿은 팁 쥔직후 {np.mean([r['touch0']/r['n_tip'] for r in rs]):.2f} → 직전 {np.mean([r['touch_pre']/r['n_tip'] for r in rs]):.2f}")


if __name__ == "__main__":
    main()
