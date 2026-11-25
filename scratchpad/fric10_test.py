"""접촉 파라미터 스캔 — (substeps, constraint_timeconst)별 마찰의 성격.

각 구성에서:
  ① 끌기 1.5cm/s → 점성 계수 B = F/v
  ② 정하중 유지: 법선 ≈10N 누른 채 접선 x에 4N 정력(force control)
     → 표류 속도 측정. 쿨롱(한계 5.8N)이면 0, 점성이면 4/B.
실행: .../python scratchpad/servo_unit_test5.py <substeps> <timeconst>
"""
import os.path as osp
import sys

import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets")

import genesis as gs                      # noqa: E402
import torch                              # noqa: E402

SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
KP, KV, F_MAX = 1500.0, 40.0, 20.0
REV_KP, REV_KV, REV_FMAX = 300.0, 10.0, 30.0
SPHERE_R = 0.015

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1 / 60, substeps=SUB,
                                      gravity=(0, 0, -9.81)),
    rigid_options=gs.options.RigidOptions(
        dt=1 / 60, constraint_solver=gs.constraint_solver.Newton,
        constraint_timeconst=TC,
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
rob.set_friction_ratio(torch.full((1, rob.n_links), 1.0, device=DEV))
box.set_friction_ratio(torch.full((1, box.n_links), 1.0, device=DEV))

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
press_z = touch_z - 0.0067
start = np.array([0.3, 0.0, touch_z + 0.1])
rob.set_dofs_position(torch.tensor(
    [list(start) + PARK], dtype=torch.float32, device=DEV), pris)
cmd(start)
for _ in range(60):
    scene.step()
cmd([0.3, 0.0, press_z])
for _ in range(90):
    scene.step()
fz0 = abs(fmeas()[2])

# ① 끌기 → 점성 계수
x, fx_log = 0.3, []
for i in range(120):
    x += 0.00025
    cmd([x, 0.0, press_z])
    scene.step()
    if i >= 60:
        fx_log.append(abs(fmeas()[0]))
fx = float(np.mean(fx_log))
B = fx / 0.015

# ② 정하중 유지: x축만 힘 제어 4N, y·z는 위치 유지
cmd([x, 0.0, press_z])
for _ in range(60):
    scene.step()
x0 = float(rob.get_dofs_position([sub[0]])[0, 0].cpu())
for i in range(120):
    rob.control_dofs_force(torch.tensor([[4.0]], device=DEV), [sub[0]])
    rob.control_dofs_position(torch.tensor(
        [[0.0, press_z]], dtype=torch.float32, device=DEV), sub[1:3])
    rob.control_dofs_position(torch.tensor(
        [PARK], dtype=torch.float32, device=DEV), park)
    scene.step()
x1 = float(rob.get_dofs_position([sub[0]])[0, 0].cpu())
drift = (x1 - x0) / 2.0 * 1000        # 2초간 → mm/s

print(f"[스캔 substeps={SUB} timeconst={TC}] 법선 {fz0:.1f}N | "
      f"끌기마찰 {fx:.2f}N (비 {fx/fz0:.3f}, B≈{B:.0f} N·s/m) | "
      f"정하중 4N 표류 {drift:+.2f} mm/s")
