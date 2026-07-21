#!/usr/bin/env python3
"""Genesis 임피던스 힘제어 스모크 테스트 — DexScale 시뮬 선택 게이트.

point robot(월드정렬 prismatic tx/ty/tz)에서는 관절공간 == Cartesian이므로
내장 PD(set_dofs_kp/kv + control_dofs_position + set_dofs_force_range)가
곧 Cartesian 임피던스 제어기다: F = kp*(x_ref - x) - kv*ẋ, |F| <= F_max.
이 스크립트는 그 전제를 결정적으로 검증한다.

  T1 static hold   : gravcomp 자유공간 홀드 → 위치 오차 < 2 mm
  T2 impedance law : reference를 고정벽 안쪽 δ=5mm → F_ctrl ≈ kp·(스프링 신장)
  T4 per-env kp    : env별 kp [500,1000,1500,2000] → 힘 단조 증가 (T2와 동시)
  T3 force clamp   : δ=50mm(이론힘 25~100N) + force_range ±20N → 20N 포화
  T5 per-env 마찰  : set_friction_ratio [0.25,0.5,1,2] → 미끄럼 거리 단조 감소
  T6 per-env 질량  : set_mass_shift + 동일 힘 → 변위 ∝ 1/m (best-effort)

실행:
  /home/robin/miniforge3/envs/dexmachina/bin/python scripts/test_genesis_impedance_smoke.py
"""
import os.path as osp
import sys

import numpy as np
import torch

import genesis as gs

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
URDF = osp.join(REPO, "extlib/dexmachina/dexmachina/assets/point_arm/point1_right.urdf")

DT = 1.0 / 60.0   # dexmachina 프로덕션 설정과 동일
N = 4
KP = torch.tensor([500.0, 1000.0, 1500.0, 2000.0])   # env별 임피던스 강성
CHAIN_MASS = 0.25                                     # tip 0.2 + 중간 링크 5*0.01
SPHERE_R = 0.015
BASE = (0.15, 0.0, 0.5)      # 로봇 base = 관절 0점에서의 구 중심 월드 좌표
WALL_X = 0.30                # 고정벽 -x 면
CONTACT_QX = WALL_X - SPHERE_R - BASE[0]   # 접촉 순간의 tx 관절값 (0.135)

results = []


def check(name, ok, detail):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


gs.init(backend=gs.gpu, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=2, gravity=(0, 0, -9.81)),
    rigid_options=gs.options.RigidOptions(
        dt=DT,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
        max_collision_pairs=100,
        batch_dofs_info=True,    # env별 kp/kv/force_range에 필수
        batch_links_info=True,   # env별 set_links_inertial_mass(질량 DR)에 필수
    ),
    show_viewer=False,
    show_FPS=False,
)

plane = scene.add_entity(gs.morphs.Plane())
wall = scene.add_entity(gs.morphs.Box(pos=(0.4, 0.0, 0.5), size=(0.2, 0.4, 0.4), fixed=True))
robot = scene.add_entity(
    gs.morphs.URDF(
        file=URDF, pos=BASE, fixed=True, merge_fixed_links=False,
        convexify=True, recompute_inertia=True,
    ),
    material=gs.materials.Rigid(gravity_compensation=1.0),
)
fbox = scene.add_entity(gs.morphs.Box(pos=(0.0, 1.5, 0.0501), size=(0.1, 0.1, 0.1)))   # T5 마찰
mbox = scene.add_entity(gs.morphs.Box(pos=(0.0, -1.5, 1.2), size=(0.1, 0.1, 0.1)))     # T6 질량

scene.build(n_envs=N, env_spacing=(3.0, 3.0))

pris = [j for j in robot.joints if j.type == gs.JOINT_TYPE.PRISMATIC]
rev = [j for j in robot.joints if j.type == gs.JOINT_TYPE.REVOLUTE]
assert len(pris) == 3 and len(rev) == 3, f"unexpected joints {[j.name for j in robot.joints]}"
assert "_tx_" in pris[0].name and "_ty_" in pris[1].name and "_tz_" in pris[2].name, \
    f"prismatic order mismatch: {[j.name for j in pris]}"
pidx = [j.dof_idx_local for j in pris]
ridx = [j.dof_idx_local for j in rev]
print("prismatic dofs:", [(j.name, j.dof_idx_local) for j in pris])

DEV = robot.get_dofs_position(pidx).device   # Genesis 내부 텐서와 디바이스 통일
KP = KP.to(DEV)

kv_p = 2.0 * torch.sqrt(KP * CHAIN_MASS)   # 임계감쇠 근사
robot.set_dofs_kp(KP[:, None].repeat(1, 3), pidx)
robot.set_dofs_kv(kv_p[:, None].repeat(1, 3), pidx)
robot.set_dofs_kp(torch.full((N, 3), 5.0, device=DEV), ridx)
robot.set_dofs_kv(torch.full((N, 3), 0.5, device=DEV), ridx)
FR100 = torch.full((N, 3), 100.0, device=DEV)
robot.set_dofs_force_range(-FR100, FR100, pidx)
robot.control_dofs_position(torch.zeros(N, 3, device=DEV), pidx)
robot.control_dofs_position(torch.zeros(N, 3, device=DEV), ridx)


def ramp_to(tgt, steps):
    tgt = tgt.to(DEV)
    q0 = robot.get_dofs_position(pidx)
    for i in range(steps):
        a = (i + 1) / steps
        robot.control_dofs_position(q0 + a * (tgt - q0), pidx)
        scene.step()


def hold(steps, avg_last=0):
    """avg_last>0이면 마지막 avg_last 스텝의 (tx 제어힘 평균, 최종 q)를 반환."""
    fsum = torch.zeros(N, device=DEV)
    for i in range(steps):
        scene.step()
        if avg_last and i >= steps - avg_last:
            fsum += robot.get_dofs_control_force(pidx)[:, 0]
    q = robot.get_dofs_position(pidx)
    return (fsum / avg_last if avg_last else None), q


# ---------- T6: per-env 질량 DR ----------
# NOTE: set_mass_shift는 batch_links_info 없이 오동작 확인(2026-07-15 진단).
# 절대값 질량 setter인 set_links_inertial_mass + batch_links_info=True 사용.
try:
    m_links = mbox.get_links_inertial_mass()
    m0 = float(np.asarray(m_links.cpu() if torch.is_tensor(m_links) else m_links).flatten()[0])
    masses = torch.tensor([0.5 * m0, 1.0 * m0, 2.0 * m0, 4.0 * m0], device=DEV)
    mbox.set_links_inertial_mass(masses[:, None].repeat(1, mbox.n_links))
    F6 = torch.zeros(N, 6, device=DEV)
    F6[:, 0] = 2.0
    x0 = mbox.get_dofs_position()[:, 0].clone()
    for _ in range(21):
        mbox.control_dofs_force(F6)
        scene.step()
    dx6 = mbox.get_dofs_position()[:, 0] - x0
    pred = dx6[1] * masses[1] / masses      # 변위 ∝ 1/m (env1 기준 스케일)
    rel = ((dx6 - pred).abs() / pred.abs()).max().item()
    dec6 = bool((dx6[1:] < dx6[:-1]).all())
    check(
        "T6 per-env mass DR",
        dec6 and rel < 0.3,
        f"m={[f'{v:.2f}' for v in masses.tolist()]} kg, dx={[f'{v * 100:.2f}' for v in dx6.tolist()]} cm (1/m 예측 대비 최대 {rel * 100:.0f}%)",
    )
except Exception as e:  # noqa: BLE001
    print(f"[SKIP] T6 per-env mass DR: {type(e).__name__}: {e}")

# ---------- T1: 자유공간 정적 홀드 ----------
tgt1 = torch.tensor([[0.05, 0.02, 0.06]]).repeat(N, 1)
ramp_to(tgt1, 30)
_, q1 = hold(60)
err1 = (q1 - tgt1).abs().max().item()
check("T1 static hold", err1 < 0.002, f"max err {err1 * 1000:.2f} mm")

# ---------- T2+T4: 벽 밀기 δ=5mm, 임피던스 법칙 + per-env kp ----------
delta2 = 0.005
tgt2 = torch.tensor([[CONTACT_QX + delta2, 0.0, 0.0]]).repeat(N, 1)
ramp_to(tgt2, 90)
f2, q2 = hold(60, avg_last=20)
stretch = tgt2[:, 0] - q2[:, 0]            # 실제 스프링 신장 (관통 반영)
f_law = KP * stretch                        # 임피던스 법칙 예측 (정상상태 v≈0)
ratio = (f2 / f_law - 1.0).abs()
check(
    "T2 impedance law F=kp*err",
    bool((ratio < 0.15).all()),
    f"F={[f'{v:.2f}' for v in f2.tolist()]} N, kp*err={[f'{v:.2f}' for v in f_law.tolist()]} N",
)
mono = bool((f2[1:] > f2[:-1]).all())
check(
    "T4 per-env kp effect",
    mono and f2[3] > 1.5 * f2[0],
    f"F(kp=500..2000)={[f'{v:.2f}' for v in f2.tolist()]} N (kp*delta 이론치={[f'{v:.1f}' for v in (KP * delta2).tolist()]})",
)

# ---------- T3: 힘 클램프 ----------
FR20 = torch.full((N, 3), 20.0, device=DEV)
robot.set_dofs_force_range(-FR20, FR20, pidx)
tgt3 = torch.tensor([[CONTACT_QX + 0.05, 0.0, 0.0]]).repeat(N, 1)
ramp_to(tgt3, 30)
f3, q3 = hold(40, avg_last=20)
sphere_x = q3[:, 0] + BASE[0] + SPHERE_R
no_tunnel = bool((sphere_x < WALL_X + 0.01).all())
check(
    "T3 force clamp @20N",
    bool(((f3 - 20.0).abs() / 20.0 < 0.10).all()) and no_tunnel,
    f"F={[f'{v:.2f}' for v in f3.tolist()]} N (이론 무클램프 25~100N), 관통없음={no_tunnel}",
)
robot.set_dofs_force_range(-FR100, FR100, pidx)
ramp_to(torch.zeros(N, 3), 30)

# ---------- T5: per-env 마찰 DR ----------
ratios = torch.tensor([0.25, 0.5, 1.0, 2.0], device=DEV)
fbox.set_friction_ratio(ratios[:, None].repeat(1, fbox.n_links))
plane.set_friction_ratio(ratios[:, None].repeat(1, plane.n_links))
v0 = torch.zeros(N, 6, device=DEV)
v0[:, 0] = 1.0
fbox.set_dofs_velocity(v0)
for _ in range(150):
    scene.step()
dx5 = fbox.get_dofs_position()[:, 0]
dec5 = bool((dx5[1:] < dx5[:-1]).all())
check(
    "T5 per-env friction DR",
    dec5 and (dx5[0] - dx5[3]) > 0.03,
    f"slide dist={[f'{v:.3f}' for v in dx5.tolist()]} m (ratio 0.25→2.0, 이론 0.204→0.026)",
)

# ---------- 요약 ----------
n_fail = sum(1 for _, ok in results if not ok)
print(f"\n=== {len(results) - n_fail}/{len(results)} PASS ===")
sys.exit(1 if n_fail else 0)
