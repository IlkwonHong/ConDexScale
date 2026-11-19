"""배분기 물리 검증 — 정적 유지 장면에서 "배분 여유 r" 의 예측이 시뮬 실측과 맞는가 (결정적 시험).

설계 §3-6 ②. 명제: MarginAllocator 가 r = r_req 로 배분한 힘을 팁 위치 서보에 사전 인가하고
(재배분 없음), 외란 렌치 s·D·v 를 서서히 올리면, 이탈 임계 s 가 예측(방향별 margin_dir,
최약 방향에서는 r_req)에 가까워야 한다. 배치 ρ(재배분 허용)와의 램프 대조가 0.1~1.1배로 흩어졌던
것과 달리, 이 예측은 "재배분 없는 스프링 반응" 모형이라 1.0 근처에 모여야 한다.

절차 (static_hold_test 와 동일 무대): 뚜껑을 굳힌 강체 박스, 최악 자세의 중력을 기울인 벡터로
재현, 유령 홀드 정착(중력 켬) → 해제 → 2초 자유 유지 판정 → 외란 램프.

실행: ./venv/bin/python -m contraction_dyn.alloc_static_ramp box_grab_01 [--segs 1 6] [--r-req 0.15 0.10 0.05]
출력: data/contraction_dyn/eval_rho/alloc_static_ramp_{seq}.json + 표
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.alloc_margin import MarginAllocator                   # noqa: E402
from contraction_dyn.static_hold_test import (build_model, quat_angle, G,  # noqa: E402
                                              KP, TIP_R, TIP_M, DT, SETTLE1,
                                              SETTLE2, HOLD_T, RAMP_T, PASS_POS,
                                              PASS_ROT, EXIT_POS, EXIT_ROT, Q0)

OUT = ROOT / "data/contraction_dyn/eval_rho"
MU_ALLOC = 0.5
DIRS = {                       # 램프 방향 v (외란 = s·D·v; D = (f_cap, f_cap, f_cap, f_cap·r_obj ×3))
    "+x힘": [1, 0, 0, 0, 0, 0], "-z힘": [0, 0, -1, 0, 0, 0],
    "+z토크": [0, 0, 0, 0, 0, 1], "+x토크": [0, 0, 0, 1, 0, 0],
}


def one_case(row, plc, r_req, dirs):
    arms = sorted(plc["arms"])
    pos_c = np.array([plc["arms"][a]["pos"] for a in arms])
    nrm_in = np.array([plc["arms"][a]["nrm_in"] for a in arms])
    cen, com = np.array(row["cen"]), np.array(row["com_ref"])
    r_obj, mu_sim, mass, f_cap = row["r_obj"], row["mu_t"], row["mass"], row["f_cap"]
    Rw = np.array(row["R_frames"][plc["worst_frame_i"]])
    g_dir = Rw.T @ np.array([0, 0, -1.0])
    w_req = np.concatenate([-mass * G * g_dir, np.zeros(3)])

    al = MarginAllocator(mu=MU_ALLOC, f_cap=f_cap, r_req=r_req)
    al.setup(pos_c, nrm_in, com, r_obj)
    sol = al.solve(w_req)
    f = sol["f"]
    r_det = al.margin_detail(f)
    fn = (f * nrm_in).sum(1)

    # 방향 목록: 최약 꼭짓점(배분 기준) + 지정 방향
    ramp_dirs = {"최약꼭짓점": r_det[3]}
    ramp_dirs.update({k: np.array(v, float) for k, v in dirs.items()})

    results = dict(seg=row["seg"], r_req=r_req, stage=sol["stage"], r_used=sol["r"],
                   r_margin_closed=al.margin_of(f), tip_normal_N=np.round(fn, 2).tolist(),
                   weak=dict(tip=arms[r_det[1]] if r_det[1] >= 0 else None, face=r_det[2]),
                   ramps=[])

    anchors_urdf = pos_c + cen
    model = build_model(row["theta"], list(zip(anchors_urdf, nrm_in)), cen, mu_sim)
    BID_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
    BID_T = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_top")
    TIP_BIDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{a}") for a in range(len(arms))]
    MB, MT = float(model.body_mass[BID_B]), float(model.body_mass[BID_T])

    def fresh_state():
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        ray_grp = np.zeros(6, dtype=np.uint8)
        ray_grp[0] = 1
        gid = np.zeros(1, dtype=np.int32)
        touch = []
        for a in range(len(arms)):
            anchor_w = Q0 + anchors_urdf[a]
            hit = None
            for back in (0.05, 0.03, 0.015, 0.008):
                origin = anchor_w - nrm_in[a] * back
                dist = mujoco.mj_ray(model, d, origin, nrm_in[a], ray_grp, 1, -1, gid)
                if dist >= 0 and gid[0] >= 0:
                    h = origin + nrm_in[a] * dist
                    if np.linalg.norm(h - anchor_w) < 0.04:
                        hit = h
                        break
            if hit is None:
                hit = anchor_w
            touch.append(hit - nrm_in[a] * TIP_R)
        for a in range(len(arms)):
            d.qpos[7 + 3 * a: 10 + 3 * a] = touch[a]
            d.ctrl[3 * a: 3 * a + 3] = touch[a] + f[a] / KP
        model.opt.gravity[:] = G * g_dir
        for bid in TIP_BIDS:
            d.xfrc_applied[bid, :3] = -TIP_M * G * g_dir

        def hold_wrench(gain):
            from scipy.spatial.transform import Rotation
            perr = Q0 - d.qpos[:3]
            rvec = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]).as_rotvec()
            d.xfrc_applied[BID_B, :3] = gain * (600.0 * perr - 30.0 * d.qvel[:3])
            d.xfrc_applied[BID_B, 3:] = gain * (-25.0 * rvec - 1.5 * d.qvel[3:6])
        for _ in range(int(SETTLE1 / DT)):
            hold_wrench(1.0)
            mujoco.mj_step(model, d)
        for i in range(int(SETTLE2 / DT)):
            hold_wrench(max(0.0, 1.0 - i * DT / 0.75))
            mujoco.mj_step(model, d)
        d.xfrc_applied[BID_B, :] = 0
        p0, q0 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
        max_p = max_r = 0.0
        for _ in range(int(HOLD_T / DT)):
            mujoco.mj_step(model, d)
            max_p = max(max_p, float(np.linalg.norm(d.xpos[BID_B] - p0)))
            max_r = max(max_r, quat_angle(d.qpos[3:7], q0))
        return d, bool(max_p <= PASS_POS and max_r <= PASS_ROT), max_p, max_r

    d, hold_ok, drift_p, drift_r = fresh_state()
    results.update(hold_pass=hold_ok, drift_pos_cm=round(100 * drift_p, 3),
                   drift_rot_deg=round(float(np.degrees(drift_r)), 3))
    if not hold_ok:
        return results
    Dd = al.d_diag
    for name, v in ramp_dirs.items():
        d, _ok, _p, _r = fresh_state()
        pred = al.margin_dir(f, v)                          # 예측 임계 s (정규화)
        s_max = max(2.5 * pred, 0.05) if np.isfinite(pred) else 1.0
        p1, q1 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
        n_steps = int(RAMP_T / DT)
        s_fail = None
        for i in range(n_steps):
            s = s_max * (i + 1) / n_steps
            wv = s * Dd * v
            d.xfrc_applied[BID_B, :3] = wv[:3] * (MB / (MB + MT))
            d.xfrc_applied[BID_T, :3] = wv[:3] * (MT / (MB + MT))
            d.xfrc_applied[BID_B, 3:] = wv[3:]
            mujoco.mj_step(model, d)
            if (np.linalg.norm(d.xpos[BID_B] - p1) > EXIT_POS
                    or quat_angle(d.qpos[3:7], q1) > EXIT_ROT):
                s_fail = s
                break
        results["ramps"].append(dict(dir=name, v=np.round(v, 3).tolist(),
                                     pred_r=round(float(pred), 4) if np.isfinite(pred) else None,
                                     meas_r=(round(float(s_fail), 4) if s_fail is not None else None),
                                     ratio=(round(float(s_fail / pred), 3)
                                            if s_fail is not None and np.isfinite(pred) and pred > 1e-9 else None),
                                     pred_N=round(float(pred * f_cap), 2) if np.isfinite(pred) else None))
    return results


def run(seq, subj, segs, r_reqs):
    cap = json.load(open(ROOT / "data/contraction_dyn/static_capacity" / f"{subj}__{seq}.json"))
    rows = {r["seg"]: r for r in cap["rows"]}
    out = []
    print(f"--- {subj}/{seq}  배분기 정적 램프 검증 (μ_alloc {MU_ALLOC}) ---")
    for seg in segs:
        row = rows[seg]
        plc = row["tuned"]
        for rr in r_reqs:
            res = one_case(row, plc, rr, DIRS)
            out.append(res)
            print(f"[seg {seg}] r_req={rr:.2f} 단계{res['stage']} 확보 r={res['r_used']:.3f} "
                  f"팁 누르는힘 {res['tip_normal_N']} 유지 {'통과' if res['hold_pass'] else '실패'} "
                  f"(드리프트 {res['drift_pos_cm']}cm/{res['drift_rot_deg']}°)")
            for rp in res["ramps"]:
                print(f"     {rp['dir']:8s} 예측 r={rp['pred_r']}  실측 r={rp['meas_r']}  비={rp['ratio']}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"alloc_static_ramp_{seq}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"저장: {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="box_grab_01")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--segs", type=int, nargs="*", default=[1, 6])
    ap.add_argument("--r-req", type=float, nargs="*", default=[0.15, 0.10, 0.05])
    a = ap.parse_args()
    run(a.seq, a.subj, a.segs, a.r_req)
