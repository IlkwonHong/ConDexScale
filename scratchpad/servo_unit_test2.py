"""마찰 전달 단위시험 — 옆면을 15N으로 누르며 수직힘을 얼마나 전달할 수 있나.

본 시뮬 로그: 옆면 팁이 법선 17N을 실현하면서도 수직 실측은 +1.6N뿐
(배분 지시 +4.5N, 서보 치우침 +15mm 포화). 배분기는 마찰계수 0.5를 가정.
→ 시뮬의 실효 마찰계수를 직접 측정한다.

P6a 법선만: f_des=(+15,0,0) — 수렴 확인
P6b 수직 추가: f_des=(+15,0,+5) — μ 0.5 가정이면 여유롭게 전달돼야 함
P6c 한계 탐색: f_des=(+15,0,+10) — 미끄러지면 포화값 = μ_eff × 15
실행: /home/robin/miniforge3/envs/dexmachina/bin/python scratchpad/servo_unit_test2.py
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

for g_ent, nm in [(box, "박스"), (rob, "로봇")]:
    try:
        mus = [float(g.friction) for lk in g_ent.links for g in lk.geoms]
        print(f"[기하 마찰] {nm}: {sorted(set(round(m, 3) for m in mus))}")
    except Exception as ex:
        print(f"[기하 마찰] {nm}: 조회 실패 ({ex})")

sub, park = pris[0:3], pris[3:6]
PARK = [0.6, 0.6, 0.6]


def cmd(p):
    rob.control_dofs_position(torch.tensor(
        [list(p)], dtype=torch.float32, device=DEV), sub)
    rob.control_dofs_position(torch.tensor(
        [PARK], dtype=torch.float32, device=DEV), park)


def fmeas():
    return rob.get_dofs_control_force(sub)[0].cpu().numpy()


def tip():
    return rob.get_dofs_position(sub)[0].cpu().numpy()


# 옆면(−x쪽 면, x=0.2, 바깥 법선 −x) 중앙 접촉. 안쪽 법선 = +x
anc = np.array([0.2, 0.0, 0.1])       # 면 위의 앵커점
n_in = np.array([1.0, 0.0, 0.0])
start = anc + np.array([-0.1 - SPHERE_R, 0, 0])
rob.set_dofs_position(torch.tensor(
    [list(start) + PARK], dtype=torch.float32, device=DEV), pris)
cmd(start)
for _ in range(60):
    scene.step()

off = np.zeros(3)
for name, fd, nstep in [("P6a 법선 15N", np.array([15.0, 0, 0]), 120),
                        ("P6b +수직 5N", np.array([15.0, 0, 5.0]), 150),
                        ("P6c +수직 10N", np.array([15.0, 0, 10.0]), 150)]:
    for i in range(nstep):
        fm = fmeas()
        off = np.clip(off + 0.15 * (fd - fm) / KP, -0.015, 0.015)
        cmd(anc - SPHERE_R * n_in + off)      # 팁 중심 = 면 밖 반경 + 치우침
        scene.step()
        if i % 30 == 29:
            f_ = fmeas()
            p_ = tip()
            print(f"[{name}] i={i+1:3d} 실측(x{f_[0]:+5.1f}, z{f_[2]:+5.1f})N "
                  f"치우침mm(x{off[0]*1000:+5.1f}, z{off[2]*1000:+5.1f}) "
                  f"팁z={p_[2]:.4f}")
    f_ = fmeas()
    mu = abs(f_[2] / f_[0]) if abs(f_[0]) > 1 else float("nan")
    print(f"[{name}] 최종 법선 {f_[0]:+.1f}N / 수직 {f_[2]:+.1f}N → "
          f"전달비(실효 μ 하한 아님, 지시 대비) = {mu:.2f}")
