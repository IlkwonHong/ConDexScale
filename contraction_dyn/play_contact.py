"""사다리 2단 (맨몸 기준선) — 실제 마찰 접촉 재생. 용접 없음.

붙은 팔의 팁 목표 (7월 E2 검증 혼합 임피던스의 이식):
  실물 앵커 = 시뮬 물체 자세로 계산한 앵커 위치 (부품 좌표 → 세계)
  팁 목표  = 실물 앵커 + (반경 − δ)·법선  ← δ = 누름 깊이 → 쥐는 힘 ≈ kp·δ
           + clip(데모 목표 − 실물 접촉점, D_MAX)  ← 옮기는 당김 (운반 성분)
δ 는 부착 구간 양끝에서 램프 (붙을 때 서서히 누르고, 떼기 전 서서히 풂).
QP 힘배분·CEM 은 이 기준선이 무너지는 전환에 다음 단계로 얹는다.

실행(conda dexmachina):
  /home/robin/miniforge3/envs/dexmachina/bin/python -m contraction_dyn.play_contact \
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
REV_KP, REV_KV, REV_FMAX = 300.0, 10.0, 30.0
SPHERE_R = 0.015
DELTA0 = 0.006                # 기본 누름 깊이 [m] → 팁당 약 9N
RAMP_UP, RAMP_DN = 18, 12     # 부착/해제 램프 스텝
D_MAX = 0.02                  # 운반 당김 클립 [m]
FRIC_RATIO = 0.6
ARMS = ["left_A", "left_B", "right_A", "right_B"]
TIPS = ["L1_tip_link", "L2_tip_link", "R1_tip_link", "R2_tip_link"]


def quat_R(q):
    w, x, y, zz = q
    return np.array([
        [1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)],
        [2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)],
        [2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--delta", type=float, default=0.006,
                    help="누름 깊이 [m] (기본 0.006)")
    ap.add_argument("--qp", action="store_true",
                    help="QP 층: 물체 자세 되먹임 → 팁 힘 배분 (마찰 원뿔 사영)")
    # 마찰 단위시험(scratchpad/servo_unit_test5.py) 결론: 기본(substeps 2,
    # timeconst 0.01)에선 마찰이 점성(F∝v, 정지 유지 불가). timeconst 하한 =
    # 2×substep dt 이므로 substeps 16 + 0.002 에서 끌기 마찰비 0.53(μ=0.6의
    # 87%), 정하중 표류 8.7mm/s 로 회복됨.
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--tc", type=float, default=0.002,
                    help="constraint_timeconst [s]")
    args = ap.parse_args()
    d0 = args.delta
    z = np.load(osp.join(REPO, "data/contraction_dyn/replay",
                         f"{args.subj}__{args.seq}.npz"), allow_pickle=True)
    targets, att = z["targets"], z["attached"]
    a_part, a_q, a_n = z["anchor_part"], z["anchor_q"], z["anchor_n"]
    ref_pos, ref_quat = z["ref_pos"], z["ref_quat"]
    ref_theta, demo_f = z["ref_theta"], z["demo_frame"]
    q0, plat, cen = z["q0"], z["platform"], z["cen"]
    T = len(targets)

    # 누름 깊이 = 평형 예압 힘(press_N, 컴파일 LP 배분) ÷ 강성, 부착 양끝 램프
    press = z["press_N"] if "press_N" in z.files else None
    appr = z["approach_u"] if "approach_u" in z.files else -np.ones((T, 4), np.float32)
    env = np.zeros((T, 4), np.float32)
    for a in range(4):
        col = att[:, a].astype(np.int8)
        d = np.diff(col)
        starts = list(np.flatnonzero(d == 1) + 1) + ([0] if col[0] else [])
        ends = list(np.flatnonzero(d == -1) + 1) + ([T] if col[-1] else [])
        HOLD = 30                 # 전원 접촉 대기: 붙은 뒤 0.5초는 힘 0 (사용자 지적)
        for s in sorted(starts):
            e = min((x for x in sorted(ends) if x > s), default=T)
            h = min(HOLD, e - s)
            up = min(RAMP_UP, e - s - h)
            env[s + h:s + h + up, a] = np.linspace(0, 1, max(up, 1), endpoint=False)
            env[s + h + up:max(s + h + up, e - RAMP_DN), a] = 1.0
            dn = min(RAMP_DN, e - s)
            env[e - dn:e, a] = np.linspace(1, 0, dn)
    if press is not None:
        delta = env * np.clip(press, 0, F_MAX * 0.8) / KP
    else:
        delta = env * d0

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 60, substeps=args.substeps,
                                          gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=1 / 60, constraint_solver=gs.constraint_solver.Newton,
            constraint_timeconst=args.tc,
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
        r.set_friction_ratio(torch.full((1, r.n_links), FRIC_RATIO, device=DEV))

    obj.set_friction_ratio(torch.full((1, obj.n_links), FRIC_RATIO, device=DEV))
    hinge = [j for j in obj.joints
             if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
    obj.set_dofs_damping(torch.full((1, 1), 0.05, device=DEV),
                         [hinge[0].dof_idx_local])
    obj.set_dofs_frictionloss(torch.full((1, 1), 1.5, device=DEV),
                              [hinge[0].dof_idx_local])   # 누름 토크(≤0.9)보다 크게

    obj.set_qpos(torch.as_tensor(q0, dtype=torch.float32, device=DEV)[None])
    t0 = torch.as_tensor(targets[0], device=DEV)
    for i, r in enumerate(robots):
        st = t0[2 * i:2 * i + 2].reshape(1, 6)
        r.set_dofs_position(st, pris[i])
        r.control_dofs_position(st, pris[i])

    rv = torch.as_tensor(z["rev_targets"], device=DEV)
    pos_err, rot_err, th_err = [], [], []
    z_act, z_ref = [], []
    frames = []

    # ── QP 층 파라미터 (물체 임피던스 + 팁 힘 배분) ──
    KP_O, DP_O = 500.0, 30.0     # 위치 오차 → 힘 [N/m], 감쇠
    KR_O, DR_O = 5.0, 0.3        # 자세 오차 → 토크 [N·m/rad], 감쇠
    F_FB_CAP, T_FB_CAP = 4.0, 1.2   # 되먹임 렌치 노름 상한 — 오차가 커져도
    # 배분기에 실현 불가능한 렌치(검증 로그: 82N/5Nm 폭주)를 먹이지 않음.
    # 12N: 중력보상 10.8 + 12 = 22.8N 요구 > 접촉 능력 ~13.4N → 배분 왜곡
    # (윗면 하향 잠식). 되먹임은 미세 조정이면 충분 (시연 가속 ~0.6N 규모)
    M_HAT, MU_QP = 1.1, 0.5      # 명목 질량 [kg], 보수 마찰
    FN_MIN, FN_MAX = 2.5, 14.0   # 팁 법선력 하한(접촉 유지)/상한 [N]
    prev_c, prev_er = None, None
    off_acc = np.zeros((4, 3))
    alloc_dbg, fm_dbg, w_dbg = {}, {}, np.zeros(6)   # 검증 로그: 배분/실측/렌치

    LEVER = 0.1                  # 회전력 행 가중용 지렛대 길이 [m] — 앵커 팔길이 규모
    # 최소제곱에서 힘(N, ~10)과 회전력(N·m, ~0.5)의 숫자 크기가 10배 달라
    # 회전력 오차가 무시되던 결함 수정: 회전력을 "지렛대 LEVER 에서의 힘"으로
    # 환산해 같은 저울에 올림 (기울어짐 복원이 진지하게 배분되게)

    N_RAY = 8                    # 원뿔 근사 모서리 수. 4개 각뿔은 대각선에서
    # 진짜 원뿔보다 √2배 넓어 접선/법선 비 0.67(>μ0.5) 배분이 나왔음 (08-12
    # fb4 장부 t=330). 8개 + 내접 반지름(cos π/8)으로 원뿔 안쪽만 쓰게 함
    MU_RAY = MU_QP * np.cos(np.pi / N_RAY)

    def qp_forces(P, Nin, w_des, f_bias):
        # 극선 좌표 배분: 원뿔 제약 = x≥0 (종이 단위시험 검증판)
        cols = []
        for p, n in zip(P, Nin):
            t1 = np.cross(n, [0, 0, 1.0])
            nt1 = np.linalg.norm(t1)
            t1 = t1 / nt1 if nt1 > 1e-6 else np.array([1.0, 0, 0])
            t2 = np.cross(n, t1)
            for k in range(N_RAY):
                a = 2 * np.pi * k / N_RAY
                s = np.cos(a) * t1 + np.sin(a) * t2
                e = (n + MU_RAY * s) / np.linalg.norm(n + MU_RAY * s)
                cols.append(np.concatenate([e, np.cross(p, e) / LEVER]))
        A = np.array(cols).T
        w = np.concatenate([w_des[:3], w_des[3:] / LEVER])
        x = np.zeros(len(cols))
        L = np.linalg.norm(A, 2) ** 2
        # 상한 15: 10이면 옆면 팁 수직 몫이 묶여 합계 < 무게 10.7N
        # (종이 시험 qp_lever_paper_test.py 로 확인된 배분 천장)
        for _ in range(300):
            x = np.clip(x - (A.T @ (A @ x - w)) / L, 0, 15.0)
        return np.array([np.array(cols[N_RAY * i:N_RAY * i + N_RAY])[:, :3].T
                         @ x[N_RAY * i:N_RAY * i + N_RAY]
                         for i in range(len(P))])

    def _qp_forces_old(P, Nin, w_des, f_bias):
        """min‖Σ(f)−w‖²+λ‖f−bias‖² 최소제곱 → 마찰 원뿔 사영 (2회 반복)."""
        k = len(P)
        G = np.zeros((6, 3 * k))
        for i in range(k):
            G[:3, 3 * i:3 * i + 3] = np.eye(3)
            r = P[i]
            G[3:, 3 * i:3 * i + 3] = np.array([
                [0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        lam = 0.05
        f = np.linalg.solve(G.T @ G + lam * np.eye(3 * k),
                            G.T @ w_des + lam * f_bias.ravel()).reshape(k, 3)
        for _ in range(2):
            for i in range(k):
                fn = float(f[i] @ Nin[i])
                ft = f[i] - fn * Nin[i]
                nt = np.linalg.norm(ft)
                fn = np.clip(fn, FN_MIN, FN_MAX)
                if nt > MU_QP * fn:
                    # 쥐기 상향: 접선 수요를 감당할 만큼 법선력을 올림 (상한 내)
                    fn = min(nt / MU_QP, FN_MAX)
                    if nt > MU_QP * fn:
                        ft *= MU_QP * fn / nt
                f[i] = fn * Nin[i] + ft
        return f
    z_drop = float(ref_pos[:, 2].min()) - 0.15
    fell = None
    for t in range(T):
        q = obj.get_qpos()[0].cpu().numpy()
        Rw, tw, thw = quat_R(q[3:7]), q[:3], float(q[7])
        step_t = targets[t].copy()
        ancs, nrms = {}, {}
        for a in range(4):
            pt, qa, na = a_part[t, a], a_q[t, a], a_n[t, a]
            if np.isnan(qa[0]):
                continue
            Rp = Rw @ Rz(-thw) if pt else Rw
            anc = Rp @ qa + tw
            nrm = Rp @ na
            ancs[a], nrms[a] = anc, nrm
            if att[t, a]:
                carry = targets[t, a] - (anc + SPHERE_R * nrm)
                cn = float(carry @ nrm)
                ct = carry - cn * nrm
                nc = np.linalg.norm(ct)
                if nc > D_MAX:
                    ct = ct / nc * D_MAX
                # 법선 성분: 누름 완화는 표면까지(δ), 강화는 3mm까지 — 떨어지진 않음
                cn = np.clip(cn, -0.003, float(delta[t, a]))
                step_t[a] = anc + (SPHERE_R - delta[t, a] + cn) * nrm + ct
            elif appr[t, a] >= 0:
                # ② 착지 실물화: 접근 마지막 구간은 실물 물체 위 앵커로 내려앉기
                u = float(appr[t, a])
                step_t[a] = anc + (SPHERE_R + (1 - u) * 0.08) * nrm

        on = [a for a in range(4) if att[t, a] and a in ancs]
        f_cmds = {}
        if args.qp and len(on) >= 2:
            c_act = q[:3] + Rw @ cen
            R_ref = quat_R(ref_quat[t])
            e_p = ref_pos[t] - c_act
            Rrel = R_ref @ Rw.T
            ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
            e_r = np.zeros(3) if ang < 1e-6 else ang / (2 * np.sin(ang)) * np.array(
                [Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0],
                 Rrel[1, 0] - Rrel[0, 1]])
            v_c = np.zeros(3) if prev_c is None else (c_act - prev_c) * 60
            v_r = np.zeros(3) if prev_er is None else (e_r - prev_er) * 60
            prev_c, prev_er = c_act, e_r
            fb_f = KP_O * e_p - DP_O * v_c
            nf_ = np.linalg.norm(fb_f)
            if nf_ > F_FB_CAP:
                fb_f *= F_FB_CAP / nf_
            fb_t = KR_O * e_r + DR_O * v_r
            nt_ = np.linalg.norm(fb_t)
            if nt_ > T_FB_CAP:
                fb_t *= T_FB_CAP / nt_
            w_des = np.concatenate(
                [fb_f + np.array([0, 0, M_HAT * 9.81]), fb_t])
            P = np.array([ancs[a] - c_act for a in on])
            Nin = np.array([-nrms[a] for a in on])
            env_t = np.array([max(delta[t, a], 1e-6) / max(delta[:, a].max(), 1e-6)
                              for a in on])       # 부착 램프 비율
            f_bias = np.array([press[t, a] * Nin[k] for k, a in enumerate(on)])
            f = qp_forces(P, Nin, w_des, f_bias)
            f *= env_t[:, None]
            alloc_dbg.clear(); fm_dbg.clear(); w_dbg = w_des.copy()
            if 200 < t < 220 and t % 6 == 0:   # 첫 램프 직전: 기준-실물 어긋남 분해
                print(f"[기준점검 t={t}] 위치오차cm="
                      f"({100*e_p[0]:+.1f},{100*e_p[1]:+.1f},{100*e_p[2]:+.1f}) "
                      f"자세오차°=({np.degrees(e_r[0]):+.1f},"
                      f"{np.degrees(e_r[1]):+.1f},{np.degrees(e_r[2]):+.1f})",
                      flush=True)
            # 실현: 스프링 목표 치우침 + 실측 힘 서보 (pilot_sim 예압 서보 방식)
            for k, a in enumerate(on):
                i, s_ = a // 2, a % 2
                sub = pris[i][s_ * 3:s_ * 3 + 3]
                fm = robots[i].get_dofs_control_force(sub)[0].cpu().numpy()
                off_acc[a] = np.clip(
                    off_acc[a] + 0.15 * (f[k] - fm) / KP, -0.015, 0.015)
                step_t[a] = ancs[a] + SPHERE_R * nrms[a] + off_acc[a]
                alloc_dbg[a], fm_dbg[a] = f[k].copy(), fm.copy()
        tg = torch.as_tensor(step_t, dtype=torch.float32, device=DEV)
        for i, r in enumerate(robots):
            for s_ in range(2):
                a = 2 * i + s_
                sub = pris[i][s_ * 3:s_ * 3 + 3]
                if a in f_cmds:
                    r.control_dofs_force(torch.as_tensor(
                        f_cmds[a], dtype=torch.float32, device=DEV).reshape(1, 3), sub)
                else:
                    r.control_dofs_position(tg[a].reshape(1, 3), sub)
            r.control_dofs_position(rv[t, 2 * i:2 * i + 2].reshape(1, 6), revs[i])
        scene.step()
        if 180 < t < 420 and t % 30 == 0:      # 첫 들기 구간: 팁별 작용력 장부
            fs = [robots[a // 2].get_dofs_control_force(
                pris[a // 2][(a % 2) * 3:(a % 2) * 3 + 3])[0].cpu().numpy()
                for a in range(4)]
            print(f"[힘 t={t}] " + " | ".join(
                f"{ARMS[a]} z{fs[a][2]:+5.1f}N |f|{np.linalg.norm(fs[a]):4.1f}"
                for a in range(4)) + f"  Σz={sum(f[2] for f in fs):+.1f}", flush=True)
            if args.qp and alloc_dbg:
                print(f"   목표렌치 힘({w_dbg[0]:+.1f},{w_dbg[1]:+.1f},"
                      f"{w_dbg[2]:+.1f})N 토크({w_dbg[3]:+.2f},{w_dbg[4]:+.2f},"
                      f"{w_dbg[5]:+.2f})Nm", flush=True)
                for a in sorted(alloc_dbg):
                    o = off_acc[a] * 1000
                    sat = " ★클립포화" if np.abs(off_acc[a]).max() >= 0.0149 else ""
                    print(f"   {ARMS[a]}: 배분({alloc_dbg[a][0]:+.1f},"
                          f"{alloc_dbg[a][1]:+.1f},{alloc_dbg[a][2]:+.1f})N "
                          f"실측z{fm_dbg[a][2]:+.1f}N 치우침mm({o[0]:+.1f},"
                          f"{o[1]:+.1f},{o[2]:+.1f}){sat}", flush=True)
        if t < 260 and t % 40 == 0:
            tps = [robots[a // 2].get_dofs_position(
                pris[a // 2][(a % 2) * 3:(a % 2) * 3 + 3])[0].cpu().numpy().round(3)
                for a in range(4)]
            print(f"[tip t={t}] " + " | ".join(
                f"{ARMS[a]}{'A' if att[t, a] else '-'}{list(tps[a])}"
                for a in range(4)), flush=True)
        if q[2] < z_drop and fell is None:
            fell = t
        if demo_f[t] >= 0:
            pos_err.append(np.linalg.norm(q[:3] + Rw @ cen - ref_pos[t]))
            d = abs(float(np.dot(q[3:7], ref_quat[t])))
            rot_err.append(2 * np.arccos(min(1.0, d)))
            th_err.append(abs(thw - float(ref_theta[t])))
            z_act.append(float((q[:3] + Rw @ cen)[2]))
            z_ref.append(float(ref_pos[t, 2]))
        if cam is not None and t % 4 == 0:
            frames.append(cam.render()[0])

    pe, re_, te = map(np.asarray, (pos_err, rot_err, th_err))
    dfv = demo_f[demo_f >= 0]
    print("\n오차 시간 곡선 (데모 프레임: 위치cm / 회전°):")
    for k in range(0, len(pe), max(1, len(pe) // 14)):
        print(f"  f{dfv[k]:4d}: {pe[k]*100:5.1f}cm  {np.degrees(re_[k]):6.1f}°")
    print(f"\n=== 사다리 2단(맨몸 접촉) 판정 ({args.seq}) ===")
    print(f"낙하: {'없음' if fell is None else f'step {fell} 낙하!!'}")
    print(f"위치 오차: 평균 {pe.mean()*100:.1f}cm / 최대 {pe.max()*100:.1f}cm "
          f"/ 마지막 {pe[-1]*100:.1f}cm")
    print(f"회전 오차: 평균 {np.degrees(re_.mean()):.1f}° / 최대 "
          f"{np.degrees(re_.max()):.1f}°")
    print(f"뚜껑 각 오차: 최대 {np.degrees(te.max()):.1f}°")
    za, zr = np.asarray(z_act), np.asarray(z_ref)
    z0 = zr.min()
    print(f"들어올리기 정직 검사: 데모 최대 들림 {100*(zr.max()-z0):.1f}cm vs "
          f"실물 최대 들림 {100*(za.max()-z0):.1f}cm")
    lift = zr - z0 > 0.03                      # 데모가 3cm 이상 든 순간들
    if lift.any():
        print(f"  데모 들림(>3cm) 구간에서 실물 평균 들림: "
              f"{100*(za[lift]-z0).mean():.1f}cm (데모 평균 "
              f"{100*(zr[lift]-z0).mean():.1f}cm)")
    np.savez(osp.join(REPO, "data/contraction_dyn/replay",
                      f"{args.subj}__{args.seq}_contact_log.npz"),
             pos_err=pe, rot_err=re_, theta_err=te,
             fell=-1 if fell is None else fell)
    if cam is not None and frames:
        import imageio
        vp = osp.join(REPO, "md_claude", "dexmachina",
                      f"contact_replay_{args.seq}.mp4")
        imageio.mimsave(vp, frames, fps=15)
        print(f"비디오: {vp}")


if __name__ == "__main__":
    main()
