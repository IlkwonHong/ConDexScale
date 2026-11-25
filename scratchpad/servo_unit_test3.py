"""마찰 직접 측정 — 서보 없이 위치 목표만으로.

P7: 윗면을 목표 관입 6.7mm(≈10N)로 누른 채, 목표를 x방향 1.5cm/s로 끌기.
    끌리는 동안 x축 제어힘 = 운동 마찰력. μ=0.6이면 ≈6N이어야 함.
    동시에 손목(revolute) 각도 로깅 → 구가 '구르는지' 확인.
실행: /home/robin/miniforge3/envs/dexmachina/bin/python scratchpad/servo_unit_test3.py
"""
import os.path as osp
import sys

import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets")

import genesis as gs                      # noqa: E402
import torch                              # noqa: E402

KP, KV, F_MAX = 1500.0, 40.0, 20.0
REV_KP, REV_KV, REV_FMAX = 300.0, 10.0, 30.0
SPHERE_R = 0.015

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1 / 60, substeps=2,
                                      gravity=(0, 0, -9.81)),
    rigid_options=gs.options.RigidOptions(
        dt=1 / 60, constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True, enable_joint_limit=True,
        max_collision_pairs=100, batch_dofs_info=True, batch_links_info=True),
    show_viewer=False)
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(pos=(0.3, 0.0, 0.1),
                                     size=(0.2, 0.2, 0.2), fixed=True))
rob = scene.add_entity(
    gs.morphs.URDF(file=osp.join(ASSETS, "point_arm", "point2_left.urdf"),
                   pos=(0, 0, 0), fixed=True, merge_fixed_links=False,
                   convexify=True, recompute_inertia=True),
    material=gs.materials.Rigid(gravity_compensation=1.0))
scene.build(n_envs=1)

pj = sorted((j for j in rob.joints if j.type == gs.JOINT_TYPE.PRISMATIC),
            key=lambda j: j.name)
pris = [j.dof_idx_local for j in pj]
jmap = {j.name: j.dof_idx_local for j in rob.joints}
revs = [jmap[f"L{k}_forearm_{ax}_link_joint"]
        for k in (1, 2) for ax in ("roll", "pitch", "yaw")]
DEV = rob.get_dofs_position(pris).device
rob.set_dofs_kp(torch.full((1, 6), KP, device=DEV), pris)
rob.set_dofs_kv(torch.full((1, 6), KV, device=DEV), pris)
fr = torch.full((1, 6), F_MAX, device=DEV)
rob.set_dofs_force_range(-fr, fr, pris)
rob.set_dofs_kp(torch.full((1, 6), REV_KP, device=DEV), revs)
rob.set_dofs_kv(torch.full((1, 6), REV_KV, device=DEV), revs)
fr2 = torch.full((1, 6), REV_FMAX, device=DEV)
rob.set_dofs_force_range(-fr2, fr2, revs)
rob.control_dofs_position(torch.zeros(1, 6, device=DEV), revs)
rob.set_friction_ratio(torch.full((1, rob.n_links), 0.6, device=DEV))
box.set_friction_ratio(torch.full((1, box.n_links), 0.6, device=DEV))

sub, park = pris[0:3], pris[3:6]
PARK = [0.6, 0.6, 0.6]


def cmd(p):
    rob.control_dofs_position(torch.tensor(
        [list(p)], dtype=torch.float32, device=DEV), sub)
    rob.control_dofs_position(torch.tensor(
        [PARK], dtype=torch.float32, device=DEV), park)


def fmeas():
    return rob.get_dofs_control_force(sub)[0].cpu().numpy()


def revq():
    return rob.get_dofs_position(revs[:3])[0].cpu().numpy()


top = np.array([0.3, 0.0, 0.2])
touch = top + [0, 0, SPHERE_R]
start = touch + [0, 0, 0.1]
rob.set_dofs_position(torch.tensor(
    [list(start) + PARK], dtype=torch.float32, device=DEV), pris)
cmd(start)
for _ in range(60):
    scene.step()

# 누름: 목표 관입 6.7mm → 법선 ≈ 10N
press = touch + [0, 0, -0.0067]
cmd(press)
for _ in range(90):
    scene.step()
print(f"[P7 누름] 실측 = {fmeas().round(2)}  (기대 z ≈ −10N)")
print(f"          손목각(rad) = {revq().round(4)}")

# 끌기: x 목표를 0.25mm/스텝(1.5cm/s)으로 이동, z 목표 유지
for i in range(240):
    tgt = press + [0.00025 * (i + 1), 0, 0]
    cmd(tgt)
    scene.step()
    if i % 30 == 29:
        f_ = fmeas()
        print(f"[P7 끌기] i={i+1:3d} 실측(x{f_[0]:+5.2f}, z{f_[2]:+6.2f})N "
              f"마찰비 x/|z|={abs(f_[0]/f_[2]) if abs(f_[2]) > 0.5 else float('nan'):.3f} "
              f"손목각={revq().round(4)}")
