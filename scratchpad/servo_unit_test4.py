"""마찰 법칙 규명 — 속도 의존성 + friction_ratio 의존성.

P7에서 잰 운동 마찰비 0.16(설정 0.6)이 어디서 오는지:
  (a) 끌기 속도를 0.5×/1×/2×로 → 힘이 속도에 비례하면 점성(정규화) 영역
  (b) friction_ratio를 0.6→1.5→3.0으로 → 비례하면 계수 자체는 반영됨
실행: /home/robin/miniforge3/envs/dexmachina/bin/python scratchpad/servo_unit_test4.py
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


def set_mu(ratio):
    rob.set_friction_ratio(torch.full((1, rob.n_links), ratio, device=DEV))
    box.set_friction_ratio(torch.full((1, box.n_links), ratio, device=DEV))


set_mu(0.6)
sub, park = pris[0:3], pris[3:6]
PARK = [0.6, 0.6, 0.6]


def cmd(p):
    rob.control_dofs_position(torch.tensor(
        [list(p)], dtype=torch.float32, device=DEV), sub)
    rob.control_dofs_position(torch.tensor(
        [PARK], dtype=torch.float32, device=DEV), park)


def fmeas():
    return rob.get_dofs_control_force(sub)[0].cpu().numpy()


touch_z = 0.2 + SPHERE_R
press_z = touch_z - 0.0067          # 법선 ≈ 10N
start = np.array([0.3, 0.0, touch_z + 0.1])
rob.set_dofs_position(torch.tensor(
    [list(start) + PARK], dtype=torch.float32, device=DEV), pris)
cmd(start)
for _ in range(60):
    scene.step()
cmd([0.3, 0.0, press_z])
for _ in range(90):
    scene.step()
print(f"[예압] 실측 = {fmeas().round(2)} (법선 기준값)")

x = 0.3
cases = [("속도 1×  μ설정0.6", 0.00025, 0.6),
         ("속도 0.5× μ설정0.6", 0.000125, 0.6),
         ("속도 2×  μ설정0.6", 0.0005, 0.6),
         ("속도 1×  μ설정1.5", 0.00025, 1.5),
         ("속도 1×  μ설정3.0", 0.00025, 3.0)]
for name, dv, mu in cases:
    set_mu(mu)
    sgn = -1.0 if x > 0.3 else 1.0
    fx_log = []
    for i in range(120):
        x += sgn * dv
        cmd([x, 0.0, press_z])
        scene.step()
        if i >= 60:                    # 후반 60스텝 정상 상태 평균
            f_ = fmeas()
            fx_log.append((abs(f_[0]), abs(f_[2])))
    fx, fz = np.mean(fx_log, axis=0)
    print(f"[{name}] 마찰력 {fx:5.2f}N / 법선 {fz:5.2f}N → 비 {fx/fz:.3f}")
