"""배치 간 순위 검증 — LP(새 ρ / 기존 α)가 매긴 배치 순위 vs 시뮬 실측 순위.

기존 α의 실패(08-16)는 "배치 A(α 0.339) < B(α 0.491)라는데 폐루프는 A 가 우수"
— 배치 간 순위 붕괴였다. 여기서는 그 질문의 정적 버전을 잰다: 앵커 배치 7종을
만들어 ① 새 ρ(중력 실현 + 64정점 min) ② 기존 min-α(14방향, 무하중) ③ 시뮬
실측(쥐기 성립 여부 + 최약 정점 방향 램프의 이탈 임계)을 배치마다 구해
순위 상관을 비교한다.

실행: ./venv/bin/python scratchpad/mj_rho_layout_rank.py
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.metric import (ray_wrench_matrix, alpha_batched,
                                   vertex_direction_set, direction_set)
from contraction_gt.search import mu_tor_from_pad
from contraction_dyn.anchors import canonical_smp, MU_T, PAD_REL
from contraction_dyn.hand_contacts import SeqData
from contraction_dyn.rho_eval import urdf_inertials, mass_com, URDF

G = 9.81
F_CAP, KP, KV, TIP_R, TIP_M = 30.0, 1500.0, 40.0, 0.015, 0.1
PRESS = 20.0
DT, SETTLE1, SETTLE2, RAMP_T = 1 / 600, 1.0, 1.5, 6.0
DECOMP = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp"

sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule/s01__box_grab_01.json"))
sd = SeqData("s01", "box_grab_01")
smp = canonical_smp(sd, 0.0)
mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
iner = urdf_inertials()
M_TOT = sum(m for m, _ in iner.values())
com_ref = mass_com(iner, 0.0) - smp.cen
MASS = float(sys.argv[1]) if len(sys.argv) > 1 else M_TOT   # 질량 오버라이드 (kg)
LAM = MASS * G / F_CAP

seg1 = {k: a["idx"] for k, a in sched["rows"][1]["anchors"].items()}
seg4 = {k: a["idx"] for k, a in sched["rows"][4]["anchors"].items()}
rng = np.random.default_rng(3)


def jitter(idxs, r):
    """각 앵커를 반경 r 내 무작위 표면점으로 이동."""
    out = []
    for i in idxs:
        cand = np.where(np.linalg.norm(smp.points - smp.points[i], axis=1) < r)[0]
        out.append(int(rng.choice(cand)))
    return out


LAYOUTS = [
    ("seg4 원본 6팔", list(seg4.values())),
    ("seg1 배치 6팔", list(seg1.values())),
    ("seg4 −right_C 5팔", [v for k, v in seg4.items() if k != "right_C"]),
    ("seg4 −left_A 5팔", [v for k, v in seg4.items() if k != "left_A"]),
    ("오른손 3팔만", [v for k, v in seg4.items() if k.startswith("right")]),
    ("seg4 지터 2.5cm", jitter(list(seg4.values()), 0.025)),
    ("seg4 지터 5cm", jitter(list(seg4.values()), 0.05)),
]

DT64 = torch.float64
dev = "cuda" if torch.cuda.is_available() else "cpu"
td = dict(dtype=DT64, device=dev)
Vts = vertex_direction_set(DT64, dev)


def lp_scores(idxs):
    """(새 ρ [N 환산], 최약 외란 정점 방향, 기존 min-α, 중력 실현 가능?)"""
    pos = torch.tensor(smp.points[idxs][None], **td)
    nrm = torch.tensor(smp.normals[idxs][None], **td)
    com = torch.tensor(com_ref[None], **td)
    pn = (pos - com[:, None]) / smp.r_obj
    W = ray_wrench_matrix(pn, nrm, torch.zeros(1, 3, **td), 1.0, MU_T,
                          mu_tor / smp.r_obj)
    # 명목(중력) 실현 가능성: +z 생성 용량 vs λ
    a_up = float(alpha_batched(W, torch.tensor([[[0, 0, -1., 0, 0, 0]]], **td))[0, 0])
    if a_up < LAM:
        return a_up - LAM, None, None, False
    tv = alpha_batched(W.expand(64, *W.shape[1:]), Vts.unsqueeze(1),
                       w_req=torch.full((64, 6), 0.0, **td)
                       .index_put_((torch.arange(64, device=dev),
                                    torch.full((64,), 2, dtype=torch.long, device=dev)),
                                   torch.full((64,), LAM, **td)))
    rho = float(tv.min()) * F_CAP
    v_weak = Vts[int(tv.argmin())].cpu().numpy()
    # 기존 min-α (14방향, 무하중)
    a14 = alpha_batched(W, direction_set(pn))
    return rho, v_weak, float(a14.min()), True


# ---------------- MJCF (배치마다 재구성) ----------------
def inertial_str(link):
    from scipy.spatial.transform import Rotation
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


urdf = ET.parse(URDF).getroot()
links = {lk.get("name"): lk for lk in urdf.findall("link")}
meshes, geoms = [], {"bottom": [], "top": []}
for nm, lk in links.items():
    for i, col in enumerate(lk.findall("collision")):
        f = col.find("geometry/mesh").get("filename")
        meshes.append(f'<mesh name="{nm}_{i}" file="{f}"/>')
        geoms[nm].append(f'<geom type="mesh" mesh="{nm}_{i}" '
                         f'friction="0.6 0.005 0.0001"/>')
Q0 = np.array([0.0, 0.0, 0.6])


def build(idxs):
    tips, targets, touches = [], [], []
    for a in range(len(idxs)):
        n_in = smp.normals[idxs[a]]
        touch = Q0 + smp.points[idxs[a]] + smp.cen - n_in * TIP_R
        touches.append(touch)
        targets.append(touch + n_in * (PRESS / KP))
        tips.append(f"""
    <body name="tip{a}" pos="0 0 0">
      <joint name="t{a}x" type="slide" axis="1 0 0" damping="{KV}"/>
      <joint name="t{a}y" type="slide" axis="0 1 0" damping="{KV}"/>
      <joint name="t{a}z" type="slide" axis="0 0 1" damping="{KV}"/>
      <geom type="sphere" size="{TIP_R}" mass="{TIP_M}"
            friction="0.6 0.005 0.0001"/>
    </body>""")
    acts = "\n".join(
        f'<position joint="t{a}{ax}" kp="{KP}" forcerange="-{F_CAP} {F_CAP}"/>'
        for a in range(len(idxs)) for ax in "xyz")
    xml = f"""
<mujoco>
  <compiler autolimits="true" meshdir="{DECOMP}"/>
  <option timestep="{DT}" impratio="10" cone="elliptic" noslip_iterations="5"/>
  <default><geom solref="0.004 1"/></default>
  <asset>{''.join(meshes)}</asset>
  <worldbody>
    <body name="box_bottom" pos="{Q0[0]} {Q0[1]} {Q0[2]}">
      <freejoint/>
      {inertial_str(links['bottom'])}{''.join(geoms['bottom'])}
      <body name="box_top" pos="0 0 0">
        {inertial_str(links['top'])}{''.join(geoms['top'])}
      </body>
    </body>
    {''.join(tips)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), touches, targets


def sim_measure(idxs, v_weak):
    """(쥐기 성립?, 최약 정점 방향 이탈 임계 [N])"""
    from scipy.spatial.transform import Rotation
    model, touches, targets = build(idxs)
    BID_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
    BID_T = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_top")
    k = MASS / M_TOT                       # 질량 스케일 (관성 동반, COM 비율 불변)
    for bid in (BID_B, BID_T):
        model.body_mass[bid] *= k
        model.body_inertia[bid] *= k
    MB, MT = float(model.body_mass[BID_B]), float(model.body_mass[BID_T])
    d = mujoco.MjData(model)
    for a in range(len(idxs)):
        d.qpos[7 + 3 * a: 10 + 3 * a] = touches[a]
        d.ctrl[3 * a: 3 * a + 3] = targets[a]
        d.xfrc_applied[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         f"tip{a}"), 2] = TIP_M * G

    def hold(gain):
        perr = Q0 - d.qpos[:3]
        rvec = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]).as_rotvec()
        d.xfrc_applied[BID_B, :3] = gain * (600 * perr - 30 * d.qvel[:3])
        d.xfrc_applied[BID_B, 3:] = gain * (-25 * rvec - 1.5 * d.qvel[3:6])

    model.opt.gravity[:] = 0
    for _ in range(int(SETTLE1 / DT)):
        hold(1.0)
        mujoco.mj_step(model, d)
    model.opt.gravity[:] = (0, 0, -G)
    for i in range(int(SETTLE2 / DT)):
        hold(max(0.0, 1.0 - i * DT / 0.75))
        mujoco.mj_step(model, d)
    d.xfrc_applied[BID_B, :] = 0
    if np.linalg.norm(d.qpos[:3] - Q0) > 0.03:
        return False, 0.0                      # 중력조차 못 버팀
    if v_weak is None:
        return True, 0.0
    p0, q0 = d.xpos[BID_B].copy(), d.qpos[3:7].copy()
    u = np.concatenate([v_weak[:3], smp.r_obj * v_weak[3:]])
    s_max, nsteps = 60.0, int(RAMP_T / DT)
    for i in range(nsteps):
        s = s_max * (i + 1) / nsteps
        d.xfrc_applied[BID_B, :3] = s * u[:3] * (MB / (MB + MT))
        d.xfrc_applied[BID_T, :3] = s * u[:3] * (MT / (MB + MT))
        d.xfrc_applied[BID_B, 3:] = s * u[3:]
        mujoco.mj_step(model, d)
        dq = abs(float(np.dot(d.qpos[3:7], q0)))
        if (np.linalg.norm(d.xpos[BID_B] - p0) > 0.03
                or 2 * np.arccos(min(dq, 1.0)) > 0.2):
            return True, s
    return True, s_max


print(f"질량 {MASS:.2f}kg (mg={MASS*G:.1f}N), 예압 {PRESS:.0f}N/팁")
print(f"{'배치':22s} {'새ρ[N]':>7s} {'기존minα':>8s} {'시뮬:쥐기':>7s} {'시뮬임계[N]':>9s}")
rows = []
for name, idxs in LAYOUTS:
    rho_N, v_weak, a14, feas = lp_scores(idxs)
    held, s_fail = sim_measure(idxs, v_weak if feas else None)
    rows.append((name, rho_N, a14, held, s_fail))
    a14s = f"{a14:8.3f}" if a14 is not None else "    -   "
    print(f"{name:22s} {rho_N:7.1f} {a14s} {'성립' if held else '낙하':>6s} {s_fail:9.1f}")

ok = [(r[1], r[4]) for r in rows if r[3] and r[1] > 0]
if len(ok) >= 3:
    from scipy.stats import spearmanr
    lp_v, sim_v = zip(*ok)
    sr = spearmanr(lp_v, sim_v)
    print(f"\n쥐기 성립 배치 {len(ok)}개의 순위 상관 (새ρ vs 시뮬임계): "
          f"스피어만 {sr.statistic:.2f}")
    a14_v = [r[2] for r in rows if r[3] and r[1] > 0]
    sr2 = spearmanr(a14_v, sim_v)
    print(f"기존 min-α vs 시뮬임계: 스피어만 {sr2.statistic:.2f}")
