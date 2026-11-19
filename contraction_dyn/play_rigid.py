"""사다리 1단 — Genesis rigid attach 재생기 (compile_replay 산출물 실행).

씬: plane + 받침(고정) + 박스 URDF(자유 base + 힌지) + point2 로봇 ×2 (구체 4개).
구체는 위치 제어(7월 검증 임피던스)로 목표를 따라가고, 부착 = 용접 제약 추가,
분리 = 용접 삭제. 물체는 자유 낙하 가능 — 시간표에 구멍이 있으면 떨어진다.

판정: ① 낙하 없음 ② 재생 구간의 물체 자세 오차(위치/회전) ③ 부착 순간 잔차.
실행(conda dexmachina):
  /home/robin/miniforge3/envs/dexmachina/bin/python -m contraction_dyn.play_rigid \
      box_grab_01 [--video]
"""
import argparse
import os.path as osp
import sys

import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets")

import genesis as gs                      # noqa: E402
import torch                              # noqa: E402

KP, KV, F_MAX = 1500.0, 40.0, 20.0
REV_KP, REV_KV, REV_FMAX = 300.0, 10.0, 30.0    # 손목 강성 — rigid attach 는 6차원
ARMS = ["left_A", "left_B", "right_A", "right_B"]
TIPS = ["L1_tip_link", "L2_tip_link", "R1_tip_link", "R2_tip_link"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    z = np.load(osp.join(REPO, "data/contraction_dyn/replay",
                         f"{args.subj}__{args.seq}.npz"), allow_pickle=True)
    targets, events = z["targets"], z["events"]
    ref_pos, ref_quat = z["ref_pos"], z["ref_quat"]
    ref_theta, demo_f = z["ref_theta"], z["demo_frame"]
    q0, plat = z["q0"], z["platform"]
    T = len(targets)

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
    scene.add_entity(gs.morphs.Box(
        pos=(float(plat[0]), float(plat[1]), float(plat[2]) - 0.05),
        size=(0.9, 0.9, 0.1), fixed=True))
    obj = scene.add_entity(gs.morphs.URDF(
        file=osp.join(ASSETS, "arctic", "box", "decomp", "box_decomp.urdf"),
        pos=tuple(q0[:3]), quat=tuple(q0[3:7]), fixed=False,
        convexify=True, recompute_inertia=False))
    robots = []
    for side in ("left", "right"):
        robots.append(scene.add_entity(
            gs.morphs.URDF(file=osp.join(ASSETS, "point_arm", f"point2_{side}.urdf"),
                           pos=(0, 0, 0), fixed=True, merge_fixed_links=False,
                           convexify=True, recompute_inertia=True),
            material=gs.materials.Rigid(gravity_compensation=1.0)))
    cam = None
    if args.video:
        look = (float(plat[0]), float(plat[1]), float(plat[2]) + 0.15)
        cam = scene.add_camera(pos=(look[0] + 0.1, look[1] - 1.6, look[2] + 0.5),
                               lookat=look, res=(800, 600), fov=32, GUI=False)
    scene.build(n_envs=1)

    pris, revs = [], []
    for si_, r in enumerate(robots):
        pj = sorted((j for j in r.joints if j.type == gs.JOINT_TYPE.PRISMATIC),
                    key=lambda j: j.name)
        pris.append([j.dof_idx_local for j in pj])
        # 손목: 구체별 사슬 순서 [roll, pitch, yaw] (이름으로 명시 매핑)
        jmap = {j.name: j.dof_idx_local for j in r.joints}
        pre = ("L", "R")[si_]
        ridx = []
        for k in (1, 2):
            for ax in ("roll", "pitch", "yaw"):
                ridx.append(jmap[f"{pre}{k}_forearm_{ax}_link_joint"])
        revs.append(ridx)
        DEV = r.get_dofs_position(pris[-1]).device
        r.set_dofs_kp(torch.full((1, 6), KP, device=DEV), pris[-1])
        r.set_dofs_kv(torch.full((1, 6), KV, device=DEV), pris[-1])
        fr = torch.full((1, 6), F_MAX, device=DEV)
        r.set_dofs_force_range(-fr, fr, pris[-1])
        r.set_dofs_kp(torch.full((1, 6), REV_KP, device=DEV), ridx)
        r.set_dofs_kv(torch.full((1, 6), REV_KV, device=DEV), ridx)
        fr2 = torch.full((1, 6), REV_FMAX, device=DEV)
        r.set_dofs_force_range(-fr2, fr2, ridx)
        r.control_dofs_position(torch.zeros(1, 6, device=DEV), ridx)

    tip_link = {}
    for r in robots:
        for l in r.links:
            if l.name in TIPS:
                tip_link[TIPS.index(l.name)] = l.idx
    link_of_part = {}                    # 0=본체(bottom), 1=뚜껑(top)
    for l in obj.links:
        if l.name == "bottom":
            link_of_part[0] = l.idx
        elif l.name == "top":
            link_of_part[1] = l.idx
    assert 0 in link_of_part and 1 in link_of_part, [l.name for l in obj.links]
    hinge = [j for j in obj.joints
             if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
    # 마찰 경첩 가정 (08-08 결정): 힘이 가해지면 열리고, 없으면 그 각도에 고정.
    # 건마찰 0.5N·m = 뚜껑 중력 토크(~0.13N·m)의 4배 — 방치 시 플롭 방지.
    obj.set_dofs_damping(torch.full((1, 1), 0.05, device=DEV),
                         [hinge[0].dof_idx_local])
    obj.set_dofs_frictionloss(torch.full((1, 1), 0.5, device=DEV),
                              [hinge[0].dof_idx_local])
    solver = scene.sim.rigid_solver
    events_part = z["events_part"] if "events_part" in z.files \
        else np.zeros(len(events), np.int64)

    def quat_R(q):
        w, x, y, zz = q
        return np.array([
            [1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)],
            [2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)],
            [2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)]])

    cen = z["cen"]
    obj.set_qpos(torch.as_tensor(q0, dtype=torch.float32, device=DEV)[None])
    qchk = obj.get_qpos()[0].cpu().numpy()
    p_c = qchk[:3] + quat_R(qchk[3:7]) @ cen
    print(f"초기 자가검증: 중심 오차 {np.linalg.norm(p_c - ref_pos[0])*1000:.1f}mm, "
          f"쿼터니언 일치 {abs(float(np.dot(qchk[3:7], ref_quat[0]))):.4f} (1이어야 정상)")
    t0 = torch.as_tensor(targets[0], device=DEV)                  # (4,3)
    for i, r in enumerate(robots):
        st = t0[2 * i:2 * i + 2].reshape(1, 6)
        r.set_dofs_position(st, pris[i])
        r.control_dofs_position(st, pris[i])

    ev_by_step = {}
    for (s, a, k), pt in zip(events, events_part):
        ev_by_step.setdefault(int(s), []).append((int(a), int(k), int(pt)))

    tg = torch.as_tensor(targets, device=DEV)
    rv = torch.as_tensor(z["rev_targets"], device=DEV)
    pos_err, rot_err, th_err = [], [], []
    frames = []
    z_drop = float(ref_pos[:, 2].min()) - 0.15
    fell = None
    for t in range(T):
        for a, kind, pt in ev_by_step.get(t, []):
            tip_w = solver.get_links_pos([tip_link[a]])[0].cpu().numpy().ravel()
            resid = float(np.linalg.norm(tip_w - targets[t - 1, a])) if t else 0.0
            lk = link_of_part[pt]
            if kind == 1:
                if resid > 0.05:
                    ri = 0 if a < 2 else 1
                    sub = pris[ri][(a % 2) * 3:(a % 2) * 3 + 3]
                    dq = robots[ri].get_dofs_position(sub)[0].cpu().numpy()
                    wc = solver.constraint_solver.get_weld_constraints()
                    print(f"    [진단] {ARMS[a]} 목표 {targets[t-1, a].round(3)} "
                          f"관절값 {dq.round(3)} 팁 {tip_w.round(3)} "
                          f"현재 용접쌍 {list(zip(wc['link_a'].tolist(), wc['link_b'].tolist())) if len(wc['link_a']) else []}")
                solver.add_weld_constraint(tip_link[a], lk)
                print(f"  step {t:4d}: {ARMS[a]} 용접→{'뚜껑' if pt else '본체'} "
                      f"(잔차 {resid*1000:.1f}mm)")
            else:
                solver.delete_weld_constraint(tip_link[a], lk)
                print(f"  step {t:4d}: {ARMS[a]} 분리")
        for i, r in enumerate(robots):
            r.control_dofs_position(tg[t, 2 * i:2 * i + 2].reshape(1, 6), pris[i])
            r.control_dofs_position(rv[t, 2 * i:2 * i + 2].reshape(1, 6), revs[i])
        scene.step()
        q = obj.get_qpos()[0].cpu().numpy()
        if q[2] < z_drop and fell is None:
            fell = t
        if demo_f[t] >= 0:
            pos_err.append(np.linalg.norm(q[:3] + quat_R(q[3:7]) @ cen - ref_pos[t]))
            d = abs(float(np.dot(q[3:7], ref_quat[t])))
            rot_err.append(2 * np.arccos(min(1.0, d)))
            th_err.append(abs(float(q[7]) - float(ref_theta[t])))
        if cam is not None and t % 4 == 0:
            frames.append(cam.render()[0])

    pe, re_, te = map(np.asarray, (pos_err, rot_err, th_err))
    dfv = demo_f[demo_f >= 0]
    print("\n오차 시간 곡선 (데모 프레임: 위치cm / 회전°):")
    for k in range(0, len(pe), max(1, len(pe) // 14)):
        print(f"  f{dfv[k]:4d}: {pe[k]*100:5.1f}cm  {np.degrees(re_[k]):6.1f}°")
    print(f"\n=== 사다리 1단 판정 ({args.seq}) ===")
    print(f"낙하: {'없음' if fell is None else f'step {fell} 에서 낙하!!'}")
    print(f"위치 오차: 평균 {pe.mean()*100:.1f}cm / 최대 {pe.max()*100:.1f}cm "
          f"/ 마지막 {pe[-1]*100:.1f}cm")
    print(f"회전 오차: 평균 {np.degrees(re_.mean()):.1f}° / 최대 "
          f"{np.degrees(re_.max()):.1f}°")
    print(f"뚜껑 각 오차: 최대 {np.degrees(te.max()):.1f}°")
    np.savez(osp.join(REPO, "data/contraction_dyn/replay",
                      f"{args.subj}__{args.seq}_log.npz"),
             pos_err=pe, rot_err=re_, theta_err=te,
             fell=-1 if fell is None else fell)
    if cam is not None and frames:
        import imageio
        vp = osp.join(REPO, "md_claude", "dexmachina",
                      f"rigid_replay_{args.seq}.mp4")
        imageio.mimsave(vp, frames, fps=15)
        print(f"비디오: {vp}")


if __name__ == "__main__":
    main()
