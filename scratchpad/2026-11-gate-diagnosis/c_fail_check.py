"""C(법선 힘 제어 + 외란별 최적 배분)에서도 떨어지는 파지 진단 + 처방 시험.

변형:
  C    : 계획 접촉으로 배분 (hybrid_check 의 C 와 같음) — 기록 포함
  C_R  : 정착 직후 실제 접촉(위치·법선)으로 배분
  C_T  : C + 외란을 0.1초에 걸쳐 서서히 증가
  C_RT : C_R + 서서히 증가
기록(매 스텝): 팁별 접촉 여부, 법선력, 접선력/법선력, 물체 이동·회전.
"""
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_check import lp_alloc  # noqa: E402
from during_load_check import merge_contacts  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
TMP = Path(sys.argv[1])
VARS = ["C", "C_NS", "C_R_NS", "C_IMP"]
RAMP = 100


def W_of(pos_n, nrm, mu_tor_n):
    import torch
    from contraction_gt.metric import ray_wrench_matrix, _flatten_W
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)
    return _flatten_W(ray_wrench_matrix(t(pos_n), t(nrm), torch.zeros(1, 3, dtype=torch.float64),
                                        1.0, 0.6, mu_tor_n))[0].numpy()


def alpha_dir(Wf, N, w):
    from contraction_gt.pilot_sim import _alpha_lp_with_x
    return _alpha_lp_with_x(Wf, w, N)[0] if N else 0.0


def run(args):
    job, var = args
    import mujoco
    from contraction_gt.sim_eval import SimConfig, build_scene_xml, _record_contacts, _pose_deviation
    cfg = SimConfig(mu_tor=job["mu_tor"], tip_radius=job["tip_radius"],
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    n = len(job["pos"]); r_obj = job["r_obj"]; f = cfg.f_cmd; com = job["com"]
    mt = job["mu_tor"] / r_obj
    Wplan = W_of(job["pos_n"], job["nrm"], mt)
    xml = build_scene_xml(("coacd", job["coacd"], job["scale"]), job["pos"], job["nrm"], cfg)
    xml = xml.replace("</actuator>", "".join(
        f'<motor joint="press{i}" gear="1" ctrlrange="0 {f}"/>' for i in range(n)) + "</actuator>")
    if var.endswith("NS"):
        xml = xml.replace('impratio="10"', 'impratio="10" noslip_iterations="30"')
    if var == "C_IMP":
        xml = xml.replace('impratio="10"', 'impratio="100"')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    tipg = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_g{i}") for i in range(n)]
    gain0 = model.actuator_gainprm[:n, 0].copy(); bias0 = model.actuator_biasprm[:n, :3].copy()
    target = np.clip(np.asarray(job["preload"], float), 0.02 * f, f)
    press = cfg.start_gap + target / cfg.kp
    f6 = np.zeros(6)

    def servo(on):
        model.actuator_gainprm[:n, 0] = gain0 if on else 0.0
        model.actuator_biasprm[:n, :3] = bias0 if on else 0.0

    def settle():
        servo(True)
        mujoco.mj_resetData(model, data)
        data.ctrl[:n] = press; data.ctrl[n:] = 0.0
        for _ in range(cfg.settle_steps):
            mujoco.mj_step(model, data)

    def tip_forces():
        fn = np.zeros(n); ft = np.zeros(n); other = 0
        for c in range(data.ncon):
            g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
            tg = g1 if g1 in tipg else (g2 if g2 in tipg else None)
            if tg is None:
                other += 1; continue
            if model.geom_bodyid[g1 if tg == g2 else g2] != ob:
                other += 1; continue
            mujoco.mj_contactForce(model, data, c, f6)
            k = tipg.index(tg)
            fn[k] += abs(f6[0]); ft[k] += float(np.linalg.norm(f6[1:3]))
        return fn, ft, other

    for _ in range(4):
        settle()
        press = press + 0.8 * (target - data.actuator_force[:n]) / cfg.kp
    res = []
    for d, (w, lv) in enumerate(zip(job["dirs"], job["levels"][:, 0])):
        settle()
        ref = data.qpos[:7].copy()
        if np.linalg.norm(ref[:3]) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
            return dict(idx=job["idx"], var=var, settled=False)
        F_set = data.actuator_force[:n].copy()
        P, N, _ = merge_contacts(_record_contacts(model, data, n, ob)["contacts"], n)
        touch_ids = [k for k in range(n) if any(c["tip"] == k for c in _record_contacts(model, data, n, ob)["contacts"])]
        Wreal = W_of((P - com) / r_obj, N, mt) if len(P) else None
        a_plan = alpha_dir(Wplan, n, w); a_real = alpha_dir(Wreal, len(P), w) if len(P) else 0.0
        if var in ("C", "C_T", "C_NS", "C_IMP"):
            p = lp_alloc(Wplan, n, -lv * w)
            fn_cmd = F_set if p is None else p * f
        else:
            fn_cmd = np.zeros(n)
            p = lp_alloc(Wreal, len(P), -lv * w) if len(P) else None
            if p is None:
                fn_cmd = F_set
            else:
                fn_cmd[touch_ids] = p * f
        servo(False)
        data.ctrl[:n] = 0.0
        data.ctrl[n:] = np.clip(fn_cmd, 0.0, f)
        ramp = RAMP if var in ("C_T", "C_RT") else 1
        log = []; ok = True; steps = cfg.hold_steps + (ramp if ramp > 1 else 0)
        for s in range(1, steps + 1):
            k = min(s / ramp, 1.0)
            data.xfrc_applied[ob, :3] = k * lv * f * w[:3]
            data.xfrc_applied[ob, 3:] = k * lv * f * r_obj * w[3:]
            mujoco.mj_step(model, data)
            fn, ft, other = tip_forces()
            dp, dr = _pose_deviation(data, ref)
            log.append(dict(s=s, fn=fn, ft=ft, other=other, dp=dp, dr=dr))
            if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg or not np.isfinite(data.qpos).all():
                ok = False
                break
        res.append(dict(d=d, lv=float(lv), held=ok, a_plan=a_plan, a_real=a_real, cmd=fn_cmd,
                        n_touch=len(P), log=log if not ok else log[-1:],
                        fail_kind=None if ok else ("rot" if dr > cfg.rot_tol_deg else "trans")))
    return dict(idx=job["idx"], var=var, settled=True, n=n, dirs=res)


def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    hy = pickle.load(open(TMP / "hybrid_raw.pkl", "rb"))
    fail8 = sorted(r["idx"] for r in hy if r["cond"] == "C" and r["settled"] and not all(r["held"])
                   and r["idx"] in {x["idx"] for x in hy if x["cond"] == "O" and not all(x["held"])})
    m = _load_records(ROOT)
    jobs = _prepare(np.array(fail8), m, ROOT)
    man = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    mmap = {r[3]: (r[1], r[2]) for r in man}
    for j in jobs:
        i = j["idx"]
        ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
        j["com"] = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")["com"] * float(m["total_scale"][i])
        j["pos_n"] = (j["pos"] - j["com"]) / j["r_obj"]
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(run, [(j, v) for j in jobs for v in VARS], chunksize=1)
    pickle.dump(res, open(TMP / "cfail_raw2.pkl", "wb"))
    print(f"대상 {len(fail8)}개: {fail8}")
    for v in VARS:
        rs = [r for r in res if r["var"] == v and r["settled"]]
        print(f"{v:5s}: 파지 합격 {sum(all(d['held'] for d in r['dirs']) for r in rs)}/{len(rs)} | "
              f"방향별 버틴 비율 {np.mean([d['held'] for r in rs for d in r['dirs']]):.2f}")
    # 진단 (C 의 떨어진 방향)
    fd = [(r, d) for r in res if r["var"] == "C" and r["settled"] for d in r["dirs"] if not d["held"]]
    print(f"\n[C 진단] 떨어진 방향 {len(fd)}개")
    kinds = {"rot": 0, "trans": 0}
    lost, slip, early, other, ar = 0, 0, 0, 0, []
    cmd_track = []
    for r, d in fd:
        kinds[d["fail_kind"]] += 1
        L = d["log"]; t0 = L[0]["fn"] > 1e-6
        anylost = any(((L[k]["fn"] <= 1e-6) & t0).any() for k in range(len(L)))
        lost += anylost
        sl = any(((L[k]["ft"] > 0.95 * 0.6 * L[k]["fn"]) & (L[k]["fn"] > 1e-3)).any() for k in range(len(L)))
        slip += sl
        early += L[-1]["s"] <= 20
        other += any(x["other"] > 0 for x in L)
        ar.append(d["a_real"] / max(d["lv"], 1e-9))
        cmd = np.clip(d["cmd"], 0, 5)
        cmd_track.append(np.abs(L[0]["fn"].sum() - cmd.sum()) / max(cmd.sum(), 1e-9))
    nfd = max(len(fd), 1)
    print(f"실패 형태: 회전 {kinds['rot']} / 이동 {kinds['trans']}")
    print(f"팁 접촉 끊김 있음 {lost/nfd:.2f} | 마찰 한계 도달(접선/법선 > 0.95·μ) 팁 있음 {slip/nfd:.2f} | "
          f"20스텝 안에 실패 {early/nfd:.2f} | 팁 외 접촉(팁끼리 등) {other/nfd:.2f}")
    ar = np.array(ar)
    print(f"정착 직후 실제 접촉 알파/외란: 중앙값 {np.median(ar):.2f}, 1 미만 비율 {np.mean(ar < 1):.2f}")
    print(f"첫 스텝 실제 법선력 합 vs 명령 합 상대 오차 중앙값 {np.median(cmd_track):.2f}")
    fs = [len(d["log"]) for _, d in fd]
    print(f"실패까지 스텝 중앙값 {np.median(fs):.0f}")


if __name__ == "__main__":
    main()
