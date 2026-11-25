"""힘 서보 단위시험 — 팁 1개 + 고정 박스.

play_contact.py --qp 의 서보(치우침 누적 off += 0.15·(f_des − f_meas)/KP)가
전제하는 가정들을 하나씩 검증한다:
  P1  자유공간 계단 응답: get_dofs_control_force 부호 = PD 지령 방향인가
  P2  누름 시험(결정적): 목표를 표면 4mm 아래로 → 실측힘 ≈ −KP·δ = −6N 인가
      (실측힘 = 팁이 박스에 가하는 힘과 같은 부호인지 확인)
  P3  서보 수렴: f_des=(0,0,−5) → 실측 −5N, 치우침 −3.3mm 수렴?
  P4  서보 복귀: f_des=0 → 실측 0N 복귀? (right_A 배분0 상황의 정지판)
  P5  박스 상승 추종: 박스를 1.5cm/s로 올리며 f_des=0, 기준점은 실물 박스를
      탐 → 실측이 0 근처를 유지하는가? (사용자 질문 "올라갈 때 붙어있으면
      충분한 거 아냐?"의 직접 시험)

실행: /home/robin/miniforge3/envs/dexmachina/bin/python scratchpad/servo_unit_test.py
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
PARK = [0.5, 0.5, 0.5]


def cmd(p):
    rob.control_dofs_position(torch.tensor(
        [list(p)], dtype=torch.float32, device=DEV), sub)
    rob.control_dofs_position(torch.tensor(
        [PARK], dtype=torch.float32, device=DEV), park)


def fmeas():
    return rob.get_dofs_control_force(sub)[0].cpu().numpy()


def tipz():
    return float(rob.get_dofs_position(sub)[0, 2].cpu())


box_z = 0.1          # 박스 중심 z (윗면 = box_z + 0.1)


def anchor():        # 실물 박스 윗면 중심 (play_contact의 실물 앵커에 해당)
    return np.array([0.3, 0.0, box_z + 0.1])


# 시작 배치: 박스 위 10cm 허공
start = anchor() + [0, 0, 0.1 + SPHERE_R]
rob.set_dofs_position(torch.tensor(
    [list(start) + PARK], dtype=torch.float32, device=DEV), pris)
cmd(start)
for _ in range(60):
    scene.step()
print(f"[P0 정착] 자유공간 실측힘 = {fmeas().round(2)}  (기대: ≈0, 중력보상)")

# ── P1: 자유공간 계단 — 목표를 5mm 아래로. 첫 스텝 실측힘 부호 확인 ──
tgt = start + [0, 0, -0.005]
cmd(tgt)
scene.step()
f1 = fmeas()
print(f"[P1 계단] 목표 5mm 아래 직후 실측힘 z = {f1[2]:+.2f}N "
      f"(기대: 음수 ≈ −KP·5mm − 감쇠, 즉 지령 방향과 같은 부호)")
for _ in range(59):
    scene.step()

# ── P2: 누름 결정판 — 목표 = 표면 접촉점에서 4mm 더 아래 ──
touch = anchor() + [0, 0, SPHERE_R]
cmd(touch + [0, 0, -0.004])
for _ in range(120):
    scene.step()
f2 = fmeas()
print(f"[P2 누름] 목표 4mm 관입, 120스텝 후 실측힘 = {f2.round(2)} "
      f"(기대 z ≈ −6N = −KP·4mm → '실측힘 = 팁이 박스에 가하는 힘' 부호 검증)")
print(f"          팁 z = {tipz():.4f} (표면 접촉 = {touch[2]:.4f})")

# ── P3: 서보 수렴 — f_des = (0,0,−5) ──
off = np.zeros(3)
f_des = np.array([0.0, 0.0, -5.0])
for i in range(150):
    fm = fmeas()                       # play_contact와 동일: 스텝 전 읽기
    off = np.clip(off + 0.15 * (f_des - fm) / KP, -0.015, 0.015)
    cmd(anchor() + [0, 0, SPHERE_R] + off)
    scene.step()
    if i % 30 == 29:
        print(f"[P3 서보 −5N] i={i+1:3d} 실측z={fmeas()[2]:+.2f}N "
              f"치우침z={off[2]*1000:+.2f}mm (기대: −5N / −3.3mm 수렴)")

# ── P4: 서보 복귀 — f_des = 0 (right_A 배분0 상황의 정지판) ──
f_des = np.array([0.0, 0.0, 0.0])
for i in range(150):
    fm = fmeas()
    off = np.clip(off + 0.15 * (f_des - fm) / KP, -0.015, 0.015)
    cmd(anchor() + [0, 0, SPHERE_R] + off)
    scene.step()
    if i % 30 == 29:
        print(f"[P4 서보 0N] i={i+1:3d} 실측z={fmeas()[2]:+.2f}N "
              f"치우침z={off[2]*1000:+.2f}mm (기대: 0N 복귀)")

# ── P5: 박스 상승 추종 — 박스를 1.5cm/s로 올리며 f_des=0 유지되나 ──
try:
    for i in range(180):
        box_z = 0.1 + 0.015 * (i / 60.0)          # 3초간 총 4.5cm 상승
        box.set_pos(torch.tensor([[0.3, 0.0, box_z]],
                                 dtype=torch.float32, device=DEV))
        fm = fmeas()
        off = np.clip(off + 0.15 * (0.0 - fm) / KP, -0.015, 0.015)
        cmd(anchor() + [0, 0, SPHERE_R] + off)     # 기준점이 실물 박스를 탐
        scene.step()
        if i % 30 == 29:
            gap = tipz() - (box_z + 0.1 + SPHERE_R)
            print(f"[P5 상승] i={i+1:3d} 실측z={fmeas()[2]:+.2f}N "
                  f"치우침z={off[2]*1000:+.2f}mm 표면과 간격={gap*1000:+.2f}mm "
                  f"(기대: 실측 ≈0 유지)")
except Exception as ex:
    print(f"[P5 생략] 고정 박스 이동 미지원: {ex}")
