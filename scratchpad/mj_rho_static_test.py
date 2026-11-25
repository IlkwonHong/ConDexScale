"""ρ(LP) 예측 vs 시뮬 실측 — 정적 외란 한계의 방향별 대조 + 시간 비교.

목적: LP 가 말하는 "이 배치는 방향 u 로 s*까지 버틴다"가 실제 물리(MuJoCo,
파지 조율 동일: elliptic cone, impratio 10, solref 4ms, noslip 5)에서 얼마나
맞는지. 맞으면 시뮬 검증의 '정적 용량' 부분을 ms 급 수치 계산으로 대체 가능.

프로토콜: box_grab_01 seg4 앵커 6개 위치에 구 팁(위치 서보 kp1500, 축 클립
±30N)을 놓고 12N 예압 → 중력 켠 채 유지 확인(= w_req 실현의 실물 검증) →
방향별 외란 렌치를 합성 질량중심에 0→s_max 선형 램프 → 이탈(변위 3cm 또는
회전 0.2rad) 시점의 s 를 실측 임계로 기록. LP 예측 = w̃_req 오프셋 방향 여유.

실행: ./venv/bin/python scratchpad/mj_rho_static_test.py
"""
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.metric import (ray_wrench_matrix, alpha_batched,
                                   vertex_direction_set)
from contraction_gt.search import mu_tor_from_pad
from contraction_dyn.anchors import canonical_smp, MU_T, PAD_REL
from contraction_dyn.hand_contacts import SeqData
from contraction_dyn.rho_eval import urdf_inertials, mass_com, URDF

import json

G = 9.81
F_CAP, KP, KV, TIP_R, TIP_M = 30.0, 1500.0, 40.0, 0.015, 0.1
PRESS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0   # 예압 (N/팁)
DISP_THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
ROT_THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
DT, SETTLE1, SETTLE2, RAMP_T = 1 / 600, 1.0, 1.5, 6.0
DECOMP = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp"
SEG = 4                           # box_grab_01: 게이트 1.0(완전 들림) 구간

# ---------------------------------------------------------------- 준비: 앵커/LP
sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule/s01__box_grab_01.json"))
row = sched["rows"][SEG]
sd = SeqData("s01", "box_grab_01")
smp = canonical_smp(sd, round(row["theta"], 1))
mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
idxs = np.array([a["idx"] for a in row["anchors"].values()])
iner = urdf_inertials()
M_TOT = sum(m for m, _ in iner.values())
com_ref = mass_com(iner, smp.theta) - smp.cen          # smp frame (identity 자세)
w_req = np.array([0, 0, M_TOT * G, 0, 0, 0])

DT64 = torch.float64
dev = "cuda" if torch.cuda.is_available() else "cpu"
td = dict(dtype=DT64, device=dev)
pos_t = torch.tensor(smp.points[idxs][None], **td)
nrm_t = torch.tensor(smp.normals[idxs][None], **td)
com_t = torch.tensor(com_ref[None], **td)
W = ray_wrench_matrix((pos_t - com_t[:, None]) / smp.r_obj, nrm_t,
                      torch.zeros(1, 3, **td), 1.0, MU_T, mu_tor / smp.r_obj)
wq_n = torch.tensor(w_req[None] / np.array([F_CAP] * 3 + [F_CAP * smp.r_obj] * 3), **td)

# 최약 정점 방향 — 외란 규약: 외란 s·v 가해지면 접촉은 w_req − s·v 를 실현해야
# 하므로 dirs = +v (솔버 원 규약 ΣWx = w̃ − t·d). 시뮬이 가하는 방향과 일치.
Vts = vertex_direction_set(DT64, dev)
tv = alpha_batched(W.expand(64, *W.shape[1:]), Vts.unsqueeze(1),
                   w_req=wq_n.expand(64, 6))
v_weak = Vts[int(tv.argmin())].cpu().numpy()

def phys_dir(f, t):
    return np.concatenate([f, t]).astype(np.float64)

DIRS = [
    ("힘 +x", phys_dir([1, 0, 0], [0, 0, 0])),
    ("힘 -x", phys_dir([-1, 0, 0], [0, 0, 0])),
    ("힘 +y", phys_dir([0, 1, 0], [0, 0, 0])),
    ("힘 -y", phys_dir([0, -1, 0], [0, 0, 0])),
    ("힘 -z(가중)", phys_dir([0, 0, -1], [0, 0, 0])),
    ("토크 +z(비틀)", phys_dir([0, 0, 0], [0, 0, 1])),
    ("토크 +x(굴림)", phys_dir([0, 0, 0], [1, 0, 0])),
    ("최약 정점", phys_dir(v_weak[:3], smp.r_obj * v_weak[3:])),
]

def lp_predict(dirs_phys):
    """외란 물리 방향(per unit s) -> 예측 임계 s*. 외란 규약: ΣWx = w̃ − t·ũ
    (외란 s·u 를 s* 까지 상쇄 가능) — 솔버 dirs = +ũ. 반환 [D]."""
    dn = np.stack(dirs_phys) / np.array([F_CAP] * 3 + [F_CAP * smp.r_obj] * 3)
    d_t = torch.tensor(dn[None], **td)
    return alpha_batched(W, d_t, w_req=wq_n)[0].cpu().numpy()

t0 = time.perf_counter()
pred = lp_predict([d for _, d in DIRS])
if dev == "cuda":
    torch.cuda.synchronize()
lp_ms = (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------- MJCF 구성
def inertial_str(link):
    ine = link.find("inertial")
    pos = ine.find("origin").get("xyz")
    rpy = [float(v) for v in ine.find("origin").get("rpy").split()]
    from scipy.spatial.transform import Rotation
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

Q0 = np.array([0.0, 0.0, 0.6])            # 박스(URDF 원점)의 월드 위치, 자세 identity
anchor_urdf = smp.points[idxs] + smp.cen  # canonical(URDF) 좌표
tips_xml, tip_targets, tip_touch = [], [], []
for a in range(len(idxs)):
    n_in = smp.normals[idxs[a]]                        # 안쪽(미는 방향)
    touch = Q0 + anchor_urdf[a] - n_in * TIP_R         # 표면에 접한 팁 중심
    tip_touch.append(touch)
    tip_targets.append(touch + n_in * (PRESS / KP))    # 예압 목표
    # 몸체를 원점에 둬야 슬라이드 관절 좌표 = 세계 좌표 (position ctrl 은 관절 좌표)
    tips_xml.append(f"""
    <body name="tip{a}" pos="0 0 0">
      <joint name="t{a}x" type="slide" axis="1 0 0" damping="{KV}"/>
      <joint name="t{a}y" type="slide" axis="0 1 0" damping="{KV}"/>
      <joint name="t{a}z" type="slide" axis="0 0 1" damping="{KV}"/>
      <geom type="sphere" size="{TIP_R}" mass="{TIP_M}"
            friction="0.6 0.005 0.0001"/>
    </body>""")
acts = "\n".join(f'<position joint="t{a}{ax}" kp="{KP}" forcerange="-{F_CAP} {F_CAP}"/>'
                 for a in range(len(idxs)) for ax in "xyz")
XML = f"""
<mujoco>
  <compiler autolimits="true" meshdir="{DECOMP}"/>
  <option timestep="{DT}" impratio="10" cone="elliptic" noslip_iterations="5"/>
  <default><geom solref="0.004 1"/></default>
  <asset>{''.join(meshes)}</asset>
  <worldbody>
    <body name="box_bottom" pos="{Q0[0]} {Q0[1]} {Q0[2]}">
      <freejoint/>
      {inertial_str(links['bottom'])}
      {''.join(geoms['bottom'])}
      <body name="box_top" pos="0 0 0">
        {inertial_str(links['top'])}
        {''.join(geoms['top'])}
      </body>
    </body>
    {''.join(tips_xml)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""
# 뚜껑은 관절 없이 용접(θ=0 강체 검사 — LP 와 같은 가정)

model = mujoco.MjModel.from_xml_string(XML)
BID_B = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
BID_T = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_top")
TIP_BIDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{a}")
            for a in range(len(idxs))]
MB = float(model.body_mass[BID_B]); MT = float(model.body_mass[BID_T])

def quat_angle(q, q0):
    d = abs(float(np.dot(q, q0)))
    return 2 * np.arccos(min(d, 1.0))

def hold_wrench(d, gain):
    """정착용 유령 홀드: 박스를 Q0·identity 로 끄는 PD 렌치 (gain 0~1)."""
    from scipy.spatial.transform import Rotation
    perr = Q0 - d.qpos[:3]
    rvec = Rotation.from_quat(d.qpos[3:7][[1, 2, 3, 0]]).as_rotvec()
    F = gain * (600.0 * perr - 30.0 * d.qvel[:3])
    T = gain * (-25.0 * rvec - 1.5 * d.qvel[3:6])
    d.xfrc_applied[BID_B, :3] = F
    d.xfrc_applied[BID_B, 3:] = T


def run_direction(u_phys, s_max):
    """외란 방향 u_phys 를 0→s_max 램프. 실측 임계 s (생존 시 None)와 소요 시간."""
    d = mujoco.MjData(model)
    for a, bid in enumerate(TIP_BIDS):
        d.qpos[7 + 3 * a: 10 + 3 * a] = tip_touch[a]  # 표면 접점에서 출발
        d.ctrl[3 * a: 3 * a + 3] = tip_targets[a]
        d.xfrc_applied[bid, 2] = TIP_M * G            # 팁 자중 보상
    # 1) 무중력 + 유령 홀드로 예압 정착 (팁 도달 시차의 밀어냄 방지)
    model.opt.gravity[:] = 0
    for _ in range(int(SETTLE1 / DT)):
        hold_wrench(d, 1.0)
        mujoco.mj_step(model, d)
    # 2) 중력 켜고 홀드를 0.75초에 걸쳐 해제 → 이후 자유 유지 (= w_req 실현 검증)
    model.opt.gravity[:] = (0, 0, -G)
    n2 = int(SETTLE2 / DT)
    for i in range(n2):
        gain = max(0.0, 1.0 - i * DT / 0.75)
        hold_wrench(d, gain)
        mujoco.mj_step(model, d)
    d.xfrc_applied[BID_B, :] = 0
    p0 = d.xpos[BID_B].copy(); q0 = d.qpos[3:7].copy()
    # 3) 램프
    nsteps = int(RAMP_T / DT)
    for i in range(nsteps):
        s = s_max * (i + 1) / nsteps
        F, T = s * u_phys[:3], s * u_phys[3:]
        d.xfrc_applied[BID_B, :3] = F * (MB / (MB + MT))
        d.xfrc_applied[BID_T, :3] = F * (MT / (MB + MT))
        d.xfrc_applied[BID_B, 3:] = T
        mujoco.mj_step(model, d)
        if (np.linalg.norm(d.xpos[BID_B] - p0) > DISP_THR
                or quat_angle(d.qpos[3:7], q0) > ROT_THR):
            return s, float(np.linalg.norm(p0 - Q0))
    return None, float(np.linalg.norm(p0 - Q0))

print(f"seg{SEG} 팔{len(idxs)}  mg={M_TOT*G:.1f}N  예압 {PRESS:.0f}N/팁  "
      f"이탈판정 {DISP_THR*100:.0f}cm/{ROT_THR:.1f}rad  LP 8방향 일괄 {lp_ms:.1f}ms")
print(f"{'방향':12s} {'LP예측':>8s} {'시뮬실측':>8s} {'실측/예측':>8s} {'시뮬시간':>7s}")
for (name, u), p in zip(DIRS, pred):
    smax = max(2.0 * p, 4.0)
    tw0 = time.perf_counter()
    s_fail, drift = run_direction(u, smax)
    tw = time.perf_counter() - tw0
    unit = "N·m" if abs(u[:3]).sum() < 1e-9 else "N"
    got = f"{s_fail:7.1f}" if s_fail is not None else f">{smax:5.1f}"
    ratio = f"{(s_fail / p):7.2f}" if (s_fail is not None and p > 1e-6) else "   -  "
    print(f"{name:12s} {p:7.1f}{unit[0]} {got}{unit[0]} {ratio} {tw:6.1f}s"
          f"  (정착드리프트 {drift*100:.1f}cm)")
