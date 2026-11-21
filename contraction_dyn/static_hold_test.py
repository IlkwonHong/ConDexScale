"""키프레임 정지 파지의 MuJoCo 검증 — LP 평형해 사전 인가 방식 (그릴링 확정 2026-09-04).

static_capacity.py 의 산출(JSON)을 읽어, 구간마다 {baseline(α 시간표), tuned(ρ)}
배치를 각각 시험한다:

  1. 강체 박스(뚜껑을 그 구간 θ 로 굳힘, 경첩 관절 없음)를 유령 홀드로 공중 정지.
     중력은 "최악 자세"를 시뮬 좌표로 옮긴 기울인 벡터 (박스 회전 대신 중력 회전 —
     물리 등가, 앵커·팁 좌표는 canonical 그대로).
  2. ρ 와 같은 마찰·상한으로 푼 **평형 배분**(8광선 내접 각뿔, Σf = −m·g, 토크 0)
     을 팁별 치우침(힘÷강성)으로 사전 인가 — 균일 예압 손잡이 제거, 명제가
     "LP 해의 성립과 안정성"으로 순수화된다. 비가능 구간은 최소잔차 해로 강행해
     예정된 실패를 그대로 기록한다.
  3. 유령 제거 → 2초 자유 유지: 드리프트 위치 1cm·회전 5° 이내면 통과.
  4. 통과 시 최약 정점 방향 외란 램프 → 이탈 임계 실측 vs LP 예측 (외란 규약).

실행: ./venv/bin/python -m contraction_dyn.static_hold_test box_grab_01 [box_use_02]
출력: data/contraction_dyn/static_hold/{subj}__{seq}.json + 표
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.metric import (ray_wrench_matrix, alpha_batched,  # noqa: E402
                                   vertex_direction_set)

G = 9.81
KP, KV, TIP_R, TIP_M = 1500.0, 40.0, 0.015, 0.1
DT, SETTLE1, SETTLE2, HOLD_T, RAMP_T = 1 / 600, 1.0, 1.5, 2.0, 6.0
PASS_POS, PASS_ROT = 0.01, np.deg2rad(5.0)      # 유지 판정 (그릴링 확정)
EXIT_POS, EXIT_ROT = 0.03, 0.2                  # 램프 이탈 판정 (기존 테스트 동일)
N_RAY = 8
URDF = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp/box_decomp.urdf"
DECOMP = URDF.parent
OUT = ROOT / "data/contraction_dyn/static_hold"
Q0 = np.array([0.0, 0.0, 0.6])


# ---------------------------------------------------------------- 평형 배분
def tangent_frame(n):
    h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    t1 = np.cross(h, n)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def equilibrium_forces(P_rel, N_in, w_req, mu, r_obj, f_cap):
    """마찰 원뿔(8광선 내접 각뿔) 안에서 Σf=w_req[:3], Στ=w_req[3:] 인 팁 힘.

    반환 (f [A,3], 잔차 노름 [N 등가]). 잔차가 크면 평형 비가능 — 그대로
    반환해 시뮬이 예정된 실패를 보여주게 한다. 작은 티호노프 항으로 유일화
    (힘을 팁들에 분산). 상한: 광선 좌표 합 → 팁 법선력 ≤ f_cap 사후 확인.
    """
    mu_r = mu * np.cos(np.pi / N_RAY)
    e_n = 1.0 / np.sqrt(1.0 + mu_r ** 2)
    cols, A_tips = [], len(P_rel)
    for p, n in zip(P_rel, N_in):
        t1, t2 = tangent_frame(n)
        for k in range(N_RAY):
            ang = 2 * np.pi * k / N_RAY
            s = np.cos(ang) * t1 + np.sin(ang) * t2
            e = (n + mu_r * s) / np.linalg.norm(n + mu_r * s)
            cols.append(np.concatenate([e, np.cross(p, e) / r_obj]))
    A = np.array(cols).T                                   # [6, 8A]
    w = np.concatenate([w_req[:3], w_req[3:] / r_obj])
    reg = 1e-2                                             # 분산 유도 (N 단위 대비 미소)
    A_aug = np.vstack([A, reg * np.eye(A.shape[1])])
    w_aug = np.concatenate([w, np.zeros(A.shape[1])])
    res = lsq_linear(A_aug, w_aug, bounds=(0.0, np.inf), tol=1e-12)
    x = res.x
    resid = float(np.linalg.norm(A @ x - w))
    f = np.array([np.array(cols[N_RAY * i:N_RAY * (i + 1)])[:, :3].T
                  @ x[N_RAY * i:N_RAY * (i + 1)] for i in range(A_tips)])
    fn = (f * N_in).sum(1)                                 # 팁별 법선 성분
    if fn.max() > f_cap * 1.001:
        f *= f_cap / fn.max()                              # 사후 축소 (기록용)
        resid = max(resid, 1e9)
    return f, resid


# ---------------------------------------------------------------- MJCF
def inertial_str(link):
    ine = link.find("inertial")
    pos = ine.find("origin").get("xyz")
    rpy = [float(v) for v in ine.find("origin").get("rpy").split()]
    Rm = Rotation.from_euler("xyz", rpy).as_matrix()
    mass = ine.find("mass").get("value")
    g = {k: float(ine.find("inertia").get(k))
         for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    Im = np.array([[g["ixx"], g["ixy"], g["ixz"]],
                   [g["ixy"], g["iyy"], g["iyz"]],
                   [g["ixz"], g["iyz"], g["izz"]]])
    Ib = Rm @ Im @ Rm.T
    return (f'<inertial pos="{pos}" mass="{mass}" fullinertia='
            f'"{Ib[0,0]} {Ib[1,1]} {Ib[2,2]} {Ib[0,1]} {Ib[0,2]} {Ib[1,2]}"/>')


def build_model(theta, anchors_c, cen, mu, extra=""):
    """뚜껑을 θ 로 굳힌 강체 박스 + 앵커 위치의 구 팁 (canonical 좌표 배치).

    extra: worldbody 에 추가할 XML (렌더링용 고정 카메라 등).
    ⚠ 이 빌드의 MjvCamera.lookat 은 성분이 메모리를 공유해 설정 불가 —
    렌더링 카메라는 반드시 MJCF 고정 카메라로 넣을 것 (2026-09-04 실측)."""
    urdf = ET.parse(URDF).getroot()
    links = {lk.get("name"): lk for lk in urdf.findall("link")}
    meshes, geoms = [], {"bottom": [], "top": []}
    rgba = {"bottom": "0.78 0.78 0.78 1", "top": "0.91 0.53 0.18 1"}
    for nm, lk in links.items():
        for i, col in enumerate(lk.findall("collision")):
            fpath = col.find("geometry/mesh").get("filename")
            meshes.append(f'<mesh name="{nm}_{i}" file="{fpath}"/>')
            geoms[nm].append(f'<geom type="mesh" mesh="{nm}_{i}" '
                             f'rgba="{rgba[nm]}" '
                             f'friction="{mu} 0.005 0.0001"/>')
    # canonical 의 뚜껑 변환 = rotvec [0,0,−θ] (canonical_smp 와 동일)
    qw, qz = np.cos(-theta / 2), np.sin(-theta / 2)
    tips_xml = []
    for a, (p, n_in) in enumerate(anchors_c):
        tips_xml.append(f"""
    <body name="tip{a}" pos="0 0 0">
      <joint name="t{a}x" type="slide" axis="1 0 0" damping="{KV}"/>
      <joint name="t{a}y" type="slide" axis="0 1 0" damping="{KV}"/>
      <joint name="t{a}z" type="slide" axis="0 0 1" damping="{KV}"/>
      <geom type="sphere" size="{TIP_R}" mass="{TIP_M}" group="2"
            friction="{mu} 0.005 0.0001"/>
    </body>""")
    acts = "\n".join(
        f'<position joint="t{a}{ax}" kp="{KP}" forcerange="-30 30"/>'
        for a in range(len(anchors_c)) for ax in "xyz")
    xml = f"""
<mujoco>
  <compiler autolimits="true" meshdir="{DECOMP}"/>
  <option timestep="{DT}" impratio="10" cone="elliptic" noslip_iterations="5"/>
  <visual><global offwidth="800" offheight="600"/></visual>
  <default><geom solref="0.004 1"/></default>
  <asset>{''.join(meshes)}</asset>
  <worldbody>
    <light pos="0.6 -0.9 1.6" dir="-0.3 0.5 -0.8" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.7 0.6 1.4" dir="0.4 -0.35 -0.85" diffuse="0.55 0.55 0.55"/>
    {extra}
    <body name="box_bottom" pos="{Q0[0]} {Q0[1]} {Q0[2]}">
      <freejoint/>
      {inertial_str(links['bottom'])}
      {''.join(geoms['bottom'])}
      <body name="box_top" pos="0 0 0" quat="{qw} 0 0 {qz}">
        {inertial_str(links['top'])}
        {''.join(geoms['top'])}
      </body>
    </body>
    {''.join(tips_xml)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def quat_angle(q, q0):
    d = abs(float(np.dot(q, q0)))
    return 2 * np.arccos(min(d, 1.0))


# ---------------------------------------------------------------- 시험 본체
def test_placement(row, plc, device, perturb=None, do_ramp=True):
    """한 배치의 (유지 시험, 최약 정점 램프). dict 결과 반환.

    perturb: None(명목) 또는 dict(mass=배율, fric=마찰계수, com_y=본체 중심
    y 편차 [m]) — play_contact_mj 의 물성 무작위화 관례와 동일. LP 평형해와
    ρ 는 **명목 물성 기준 그대로** 둔다 ("물체를 잘못 아는" 상황의 시험이므로
    계획-실물 불일치 자체가 시험 대상)."""
    arms = sorted(plc["arms"])
    pos_c = np.array([plc["arms"][a]["pos"] for a in arms])     # canonical(중심)
    nrm_in = np.array([plc["arms"][a]["nrm_in"] for a in arms])
    cen = np.array(row["cen"])
    com = np.array(row["com_ref"])
    r_obj, mu, mu_tor = row["r_obj"], row["mu_t"], row["mu_tor"]
    mass, f_cap = row["mass"], row["f_cap"]

    # 최악 자세의 중력을 canonical 로 회전 (박스는 identity 유지 — 물리 등가)
    Rw = np.array(row["R_frames"][plc["worst_frame_i"]])
    g_dir = Rw.T @ np.array([0, 0, -1.0])                       # 단위 중력 방향
    w_req = np.concatenate([-mass * G * g_dir, np.zeros(3)])    # 접촉 몫 (com 기준)

    f_alloc, resid = equilibrium_forces(pos_c - com, nrm_in, w_req,
                                        mu, r_obj, f_cap)

    # LP 최약 정점 + 예측 (외란 규약: ΣWx = w̃ − t·ũ → dirs = +ũ, 08-30 정정).
    # 비가능 배치(w̃ ∉ W_c)는 정점 LP 를 부르지 않는다 — IPM 수렴 보장 밖.
    feasible = bool(plc["feasible_frames"][plc["worst_frame_i"]])
    pred_t, u_phys = float("nan"), None
    if feasible:
        td = dict(dtype=torch.float64, device=device)
        W = ray_wrench_matrix(
            torch.tensor((pos_c - com)[None] / r_obj, **td),
            torch.tensor(nrm_in[None], **td),
            torch.zeros(1, 3, **td), 1.0, mu, mu_tor / r_obj)
        scale = np.array([f_cap] * 3 + [f_cap * r_obj] * 3)
        wq_n = torch.tensor((w_req / scale)[None], **td)
        Vts = vertex_direction_set(torch.float64, device)
        tv = alpha_batched(W.expand(64, *W.shape[1:]), Vts.unsqueeze(1),
                           w_req=wq_n.expand(64, 6)).cpu().numpy().reshape(64)
        v_weak = Vts[int(np.nanargmin(tv))].cpu().numpy()
        pred_t = float(np.nanmin(tv))                             # 정규화 임계
        u_phys = np.concatenate([v_weak[:3], r_obj * v_weak[3:]])  # 물리 방향

    # ---- MuJoCo ----
    anchors_urdf = pos_c + cen
    model = build_model(row["theta"], list(zip(anchors_urdf, nrm_in)), cen, mu)
    BID_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
    BID_T = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_top")
    TIP_BIDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{a}")
                for a in range(len(arms))]
    if perturb:
        for bid in (BID_B, BID_T):
            model.body_mass[bid] *= perturb.get("mass", 1.0)
            model.body_inertia[bid] *= perturb.get("mass", 1.0)
        model.body_ipos[BID_B][1] += perturb.get("com_y", 0.0)
        if "fric" in perturb:
            model.geom_friction[:, 0] = perturb["fric"]
    MB, MT = float(model.body_mass[BID_B]), float(model.body_mass[BID_T])
    d = mujoco.MjData(model)

    # 표면 정렬 (레이캐스트): 앵커는 템플릿 메시 표면점, 충돌체는 볼록 분해
    # (decomp) 메시 — 뚜껑 열린 자세의 테두리 부근에서 1~2cm 불일치 (use_02
    # seg6 진단: 팁이 박스 속에 박힌 채 시작 → 힘 왜곡 → 사출). 앵커 법선
    # 선상에서 decomp 표면 교점을 mj_ray 로 직접 찾아 팁을 그 위에 놓는다.
    # (반복 이동 방식은 비스듬한 조각면에서 발산 — 폐기, 2026-09-04)
    mujoco.mj_forward(model, d)
    ray_grp = np.zeros(6, dtype=np.uint8)
    ray_grp[0] = 1                                  # 박스 geom 만 (팁은 group 2)
    gid = np.zeros(1, dtype=np.int32)
    touch, surf_shift, align_fail = [], np.zeros(len(arms)), []
    for a in range(len(arms)):
        anchor_w = Q0 + anchors_urdf[a]
        hit = None
        for back in (0.05, 0.03, 0.015, 0.008):
            origin = anchor_w - nrm_in[a] * back
            dist = mujoco.mj_ray(model, d, origin, nrm_in[a],
                                 ray_grp, 1, -1, gid)
            if dist >= 0 and gid[0] >= 0:
                h = origin + nrm_in[a] * dist
                if np.linalg.norm(h - anchor_w) < 0.04:
                    hit = h
                    break
        if hit is None:
            align_fail.append(arms[a])
            hit = anchor_w                          # 폴백: 템플릿 점 그대로
        surf_shift[a] = float(np.linalg.norm(hit - anchor_w))
        touch.append(hit - nrm_in[a] * TIP_R)
    for a in range(len(arms)):
        d.qpos[7 + 3 * a: 10 + 3 * a] = touch[a]
        d.ctrl[3 * a: 3 * a + 3] = touch[a] + f_alloc[a] / KP

    def hold_wrench(gain):
        perr = Q0 - d.qpos[:3]
        rvec = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]).as_rotvec()
        d.xfrc_applied[BID_B, :3] = gain * (600.0 * perr - 30.0 * d.qvel[:3])
        d.xfrc_applied[BID_B, 3:] = gain * (-25.0 * rvec - 1.5 * d.qvel[3:6])

    # 1) 정착 — 중력을 처음부터 켠다 (유령 홀드 + 평형해 사전 인가).
    #    평형해는 합력이 +mg 라서 무중력 정착이면 그 몫이 박스를 밀어냄
    #    (seg6 진단: 0.45cm 이탈 + 접촉 분리 후 해제 → 발산). 균일 예압
    #    (내부 합 0)과 달리 중력이 짝으로 있어야 정착이 성립한다.
    model.opt.gravity[:] = G * g_dir
    for a, bid in enumerate(TIP_BIDS):
        d.xfrc_applied[bid, :3] = -TIP_M * G * g_dir
    for _ in range(int(SETTLE1 / DT)):
        hold_wrench(1.0)
        mujoco.mj_step(model, d)
    # 2) 유령 0.75초 램프 해제
    for i in range(int(SETTLE2 / DT)):
        hold_wrench(max(0.0, 1.0 - i * DT / 0.75))
        mujoco.mj_step(model, d)
    d.xfrc_applied[BID_B, :] = 0
    # 3) 자유 유지 2초 — 최대 드리프트 판정
    p0, q0 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
    max_p, max_r = 0.0, 0.0
    for _ in range(int(HOLD_T / DT)):
        mujoco.mj_step(model, d)
        max_p = max(max_p, float(np.linalg.norm(d.xpos[BID_B] - p0)))
        max_r = max(max_r, quat_angle(d.qpos[3:7], q0))
    hold_pass = bool(max_p <= PASS_POS and max_r <= PASS_ROT)

    out = dict(arms=len(arms), feasible_lp=feasible,
               rho_worst=plc["rho_worst"], alloc_resid_N=round(resid, 4),
               surf_shift_mm=round(float(surf_shift.max()) * 1000, 1),
               align_fail=align_fail,
               hold_pass=hold_pass,
               drift_pos_cm=round(max_p * 100, 3),
               drift_rot_deg=round(np.degrees(max_r), 3))

    # 4) 최약 정점 램프 (유지 통과 시에만 의미)
    if do_ramp and hold_pass and u_phys is not None and np.isfinite(pred_t):
        pred_N = pred_t * f_cap
        s_max = max(2.0 * pred_N, 4.0)
        p1, q1 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
        n_steps = int(RAMP_T / DT)
        s_fail = None
        for i in range(n_steps):
            s = s_max * (i + 1) / n_steps
            F, T = s * u_phys[:3], s * u_phys[3:]
            d.xfrc_applied[BID_B, :3] = F * (MB / (MB + MT))
            d.xfrc_applied[BID_T, :3] = F * (MT / (MB + MT))
            d.xfrc_applied[BID_B, 3:] = T
            mujoco.mj_step(model, d)
            if (np.linalg.norm(d.xpos[BID_B] - p1) > EXIT_POS
                    or quat_angle(d.qpos[3:7], q1) > EXIT_ROT):
                s_fail = s
                break
        out.update(weak_pred_N=round(pred_N, 2),
                   weak_meas_N=(round(s_fail, 2) if s_fail is not None
                                else None),
                   weak_ratio=(round(s_fail / pred_N, 3)
                               if s_fail is not None and pred_N > 1e-6
                               else None))
    return out


def run(seq, subj, device):
    cap = json.load(open(ROOT / "data/contraction_dyn/static_capacity"
                         / f"{subj}__{seq}.json"))
    rows_out = []
    print(f"--- {subj}/{seq}  정지 유지 시험 (LP 평형해 사전 인가, "
          f"판정 {PASS_POS*100:.0f}cm/{np.degrees(PASS_ROT):.0f}°/2s) ---")
    print(f"{'seg':>3s} {'배치':10s} {'팔':>2s} {'ρ':>7s} {'잔차N':>6s} "
          f"{'유지':4s} {'드리프트':>14s} {'램프 예측/실측':>16s} {'비':>5s}")
    for row in cap["rows"]:
        for name in ("baseline", "tuned"):
            plc = row.get(name)
            if not plc:
                continue
            r = test_placement(row, plc, device)
            r.update(seg=row["seg"], placement=name)
            rows_out.append(r)
            ramp = (f"{r.get('weak_pred_N', '-'):>7} /"
                    f" {str(r.get('weak_meas_N', '-')):>6}")
            print(f"[{row['seg']:2d}] {name:10s} {r['arms']:2d} "
                  f"{r['rho_worst']:+7.3f} {r['alloc_resid_N']:6.2f} "
                  f"{'통과' if r['hold_pass'] else '실패'} "
                  f"{r['drift_pos_cm']:6.2f}cm {r['drift_rot_deg']:5.1f}° "
                  f"{ramp} {str(r.get('weak_ratio', '-')):>5s}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj, pass_pos_cm=PASS_POS * 100,
                   pass_rot_deg=float(np.degrees(PASS_ROT)),
                   hold_s=HOLD_T, rows=rows_out),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"저장: {out}")


# 물성 변주 스위프 (그릴링 합의: 명목 통과 후 확장) — 08-16 CEM 케이스 설계와
# 동일 축: 질량 ±10%, 마찰 0.5/0.7, 본체 질량중심 y ±1.5cm. 계획(평형해·ρ)은
# 명목 기준 고정 — "잘못 아는 물체"에서 ρ 여유가 오차를 흡수하는가의 시험.
PERTURB_CASES = [
    ("명목", None),
    ("질량0.9", dict(mass=0.9)),
    ("질량1.1", dict(mass=1.1)),
    ("마찰0.5", dict(fric=0.5)),
    ("마찰0.7", dict(fric=0.7)),
    ("중심+1.5", dict(com_y=+0.015)),
    ("중심-1.5", dict(com_y=-0.015)),
]


def run_sweep(seq, subj, device):
    cap = json.load(open(ROOT / "data/contraction_dyn/static_capacity"
                         / f"{subj}__{seq}.json"))
    rows_out = []
    print(f"--- {subj}/{seq}  물성 변주 스위프 (튜닝 배치 × 케이스 {len(PERTURB_CASES)}) ---")
    for row in cap["rows"]:
        plc = row.get("tuned")
        if not plc:
            continue
        res = []
        for name, pert in PERTURB_CASES:
            r = test_placement(row, plc, device, perturb=pert, do_ramp=False)
            res.append(dict(case=name, hold_pass=r["hold_pass"],
                            drift_pos_cm=r["drift_pos_cm"],
                            drift_rot_deg=r["drift_rot_deg"]))
        n_ok = sum(x["hold_pass"] for x in res)
        fails = [x["case"] for x in res if not x["hold_pass"]]
        rows_out.append(dict(seg=row["seg"], arms=len(plc["arms"]),
                             rho_worst=plc["rho_worst"], n_pass=n_ok,
                             cases=res))
        print(f"[{row['seg']:2d}] 팔{len(plc['arms'])} ρ={plc['rho_worst']:+.3f} "
              f"완주 {n_ok}/{len(PERTURB_CASES)}"
              + (f"  실패: {', '.join(fails)}" if fails else ""), flush=True)
    out_dir = ROOT / "data/contraction_dyn/static_perturb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj,
                   cases=[c for c, _ in PERTURB_CASES], rows=rows_out),
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=["box_grab_01"])
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--sweep", action="store_true",
                    help="물성 변주 스위프 (튜닝 배치, 유지 시험만)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seq in args.seqs or ["box_grab_01"]:
        if args.sweep:
            run_sweep(seq, args.subj, dev)
        else:
            run(seq, args.subj, dev)
