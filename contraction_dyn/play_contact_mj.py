"""사다리 2단 접촉 재생 — MuJoCo 포팅판.

제어·힘배분·서보·장부·판정은 play_contact.py(Genesis판)와 동일 코드.
물리 뒷단만 교체: 장면 = MJCF 생성(ARCTIC 박스 decomp 메시 + 질점 팁 4개),
접촉 = 파지용 조율(타원 원뿔, impratio 10, solref 4ms, noslip 5 —
scratchpad/mj_friction_test.py 실측: 끌기 비 0.73, 정하중 표류 0.00mm/s).

실행: ./venv/bin/python -m contraction_dyn.play_contact_mj box_grab_01 [--video]
"""
import argparse
import os.path as osp
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)
DECOMP = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets",
                  "arctic", "box", "decomp")

import mujoco                             # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

KP, KV, F_MAX = 1500.0, 40.0, 30.0  # (D_MAX 가 이 둘에서 유도됨)   # 구동기 여유: 20→30N (배분 쥐기 지시 23N이 20에서 잘리던 것)
SPHERE_R = 0.015
TIP_M = 0.1
RAMP_UP, RAMP_DN, HOLD = 18, 12, 30
D_MAX = F_MAX / KP            # 운반 당김 한도 = 힘한계÷강성 (= 2cm) — 구동기가
# 어차피 못 내는 당김은 요청하지 않음 (유도 묶기, 08-16)
FRIC = 0.6
ARMS = ["left_A", "left_B", "right_A", "right_B"]
N_SUB, DT = 10, 1.0 / 600                 # 제어 60Hz = 물리 1.667ms × 10


def quat_R(q):
    w, x, y, zz = q
    return np.array([
        [1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)],
        [2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)],
        [2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(u, a):
    """축 u(단위벡터) 둘레로 각 a 만큼의 회전 행렬 (로드리게스 공식)."""
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def _inertial_from_urdf(link):
    # URDF 관성은 rpy 회전된 국소 프레임 기준 → 몸체 프레임으로 회전해 전달
    ine = link.find("inertial")
    pos = ine.find("origin").get("xyz")
    rpy = [float(v) for v in ine.find("origin").get("rpy").split()]
    R = Rotation.from_euler("xyz", rpy).as_matrix()
    mass = ine.find("mass").get("value")
    I = ine.find("inertia")
    g = {k: float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    Im = np.array([[g["ixx"], g["ixy"], g["ixz"]],
                   [g["ixy"], g["iyy"], g["iyz"]],
                   [g["ixz"], g["iyz"], g["izz"]]])
    Ib = R @ Im @ R.T
    full = (f"{Ib[0,0]} {Ib[1,1]} {Ib[2,2]} "
            f"{Ib[0,1]} {Ib[0,2]} {Ib[1,2]}")
    return (f'<inertial pos="{pos}" mass="{mass}" fullinertia="{full}"/>')


def build_mjcf(q0, plat, park):
    urdf = ET.parse(osp.join(DECOMP, "box_decomp.urdf")).getroot()
    links = {lk.get("name"): lk for lk in urdf.findall("link")}
    meshes, geoms = [], {"bottom": [], "top": []}
    for nm, lk in links.items():
        for i, col in enumerate(lk.findall("collision")):
            f = col.find("geometry/mesh").get("filename")
            mid = f"{nm}_{i}"
            meshes.append(f'<mesh name="{mid}" file="{f}"/>')
            geoms[nm].append(
                f'<geom type="mesh" mesh="{mid}" friction="{FRIC} 0.005 0.0001"'
                f' rgba="1 .42 .04 1"/>')
    tips = []
    for a, arm in enumerate(ARMS):
        p = park[a]
        tips.append(f"""
    <body name="tip_{arm}" pos="{p[0]} {p[1]} {p[2]}">
      <joint name="{arm}_x" type="slide" axis="1 0 0" damping="{KV}"/>
      <joint name="{arm}_y" type="slide" axis="0 1 0" damping="{KV}"/>
      <joint name="{arm}_z" type="slide" axis="0 0 1" damping="{KV}"/>
      <geom type="sphere" size="{SPHERE_R}" mass="{TIP_M}"
            friction="{FRIC} 0.005 0.0001" rgba=".8 .8 .8 1"/>
    </body>""")
    acts = "\n".join(
        f'<position joint="{arm}_{ax}" kp="{KP}" forcerange="-{F_MAX} {F_MAX}"/>'
        for arm in ARMS for ax in ("x", "y", "z"))
    look = np.array([plat[0], plat[1], plat[2] + 0.15])
    cpos = look + np.array([0.1, -1.6, 0.5])
    zax = (cpos - look) / np.linalg.norm(cpos - look)
    return f"""
<mujoco model="contact_replay">
  <compiler autolimits="true" meshdir="{DECOMP}"/>
  <option timestep="{DT}" impratio="10" cone="elliptic"
          noslip_iterations="5"/>
  <default>
    <geom solref="0.004 1"/>
  </default>
  <visual><global offwidth="800" offheight="600"/></visual>
  <asset>{''.join(meshes)}</asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <camera name="cam" pos="{cpos[0]} {cpos[1]} {cpos[2]}"
            zaxis="{zax[0]} {zax[1]} {zax[2]}" fovy="32"/>
    <geom type="plane" size="3 3 .1" rgba=".35 .4 .5 1"/>
    <geom type="box" pos="{plat[0]} {plat[1]} {plat[2] - 0.05}"
          size=".45 .45 .05" friction="{FRIC} 0.005 0.0001"
          rgba=".92 .92 .92 1"/>
    <body name="box_bottom" pos="{q0[0]} {q0[1]} {q0[2]}"
          quat="{q0[3]} {q0[4]} {q0[5]} {q0[6]}">
      <freejoint name="box_free"/>
      {_inertial_from_urdf(links['bottom'])}
      {''.join(geoms['bottom'])}
      <body name="box_top" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 0 -1" range="0 2.6"
               damping="0.05" frictionloss="1.5"/>
        {_inertial_from_urdf(links['top'])}
        {''.join(geoms['top'])}
      </body>
    </body>
    {''.join(tips)}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--qp", action="store_true", default=True)
    ap.add_argument("--no-qp", dest="qp", action="store_false")
    ap.add_argument("--sg", type=float, default=0.036951158159670316, help="서보 적분 이득")
    ap.add_argument("--ffc", type=float, default=2.8006712639830047, help="되먹임 힘 축별 클립 [N]")
    ap.add_argument("--tfc", type=float, default=2.717526380636423, help="되먹임 토크 상한 [N·m]")
    ap.add_argument("--dro", type=float, default=0.886670732144944, help="회전 감쇠")
    ap.add_argument("--dr-mass", type=float, default=1.0,
                    help="박스 질량 배율 (물성 무작위화용)")
    ap.add_argument("--dr-fric", type=float, default=0.6,
                    help="접촉 마찰 계수 (물성 무작위화용)")
    ap.add_argument("--dr-com", type=float, default=0.0,
                    help="본체 질량중심 y 편차 [m] (물성 무작위화용)")
    ap.add_argument("--lever", type=float, default=0.12375890747223402,
                    help="회전력↔힘 환율 겸 회전 우선 가중 [m]")
    ap.add_argument("--carrot", type=float, default=17.513066072600232,
                    help="당김 회전 접근 한도 [도]")
    ap.add_argument("--press-scale", type=float, default=0.5677910466786238,
                    help="평형 예압 배율")
    args = ap.parse_args()   # 기본값 = CEM+DR 우승, 전체 정밀도 필수 (4자리 반올림도 낙하 유발 — 08-16)
    z = np.load(osp.join(REPO, "data/contraction_dyn/replay",
                         f"{args.subj}__{args.seq}.npz"), allow_pickle=True)
    targets, att = z["targets"], z["attached"]
    a_part, a_q, a_n = z["anchor_part"], z["anchor_q"], z["anchor_n"]
    ref_pos, ref_quat = z["ref_pos"], z["ref_quat"]
    ref_theta, demo_f = z["ref_theta"], z["demo_frame"]
    q0, plat, cen = z["q0"], z["platform"], z["cen"]
    T = len(targets)
    global ARMS
    if "arms" in z.files:
        ARMS = [str(x) for x in z["arms"]]
    NA = len(ARMS)
    press = z["press_N"] * args.press_scale if "press_N" in z.files else None
    appr = z["approach_u"] if "approach_u" in z.files else -np.ones((T, NA), np.float32)
    env = np.zeros((T, NA), np.float32)
    for a in range(NA):
        col = att[:, a].astype(np.int8)
        d = np.diff(col)
        starts = list(np.flatnonzero(d == 1) + 1) + ([0] if col[0] else [])
        ends = list(np.flatnonzero(d == -1) + 1) + ([T] if col[-1] else [])
        for s in sorted(starts):
            e = min((x for x in sorted(ends) if x > s), default=T)
            h = min(HOLD, e - s)
            up = min(RAMP_UP, e - s - h)
            env[s + h:s + h + up, a] = np.linspace(0, 1, max(up, 1),
                                                   endpoint=False)
            env[s + h + up:max(s + h + up, e - RAMP_DN), a] = 1.0
            dn = min(RAMP_DN, e - s)
            env[e - dn:e, a] = np.linspace(1, 0, dn)
    delta = env * np.clip(press, 0, F_MAX * 0.8) / KP

    m = mujoco.MjModel.from_xml_string(build_mjcf(q0, plat, targets[0]))
    for bn in ("box_bottom", "box_top"):        # 물성 무작위: 질량 배율
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        m.body_mass[bid] *= args.dr_mass
        m.body_inertia[bid] *= args.dr_mass
        if bn == "box_bottom":
            m.body_ipos[bid][1] += args.dr_com
    m.geom_friction[:, 0] = args.dr_fric        # 마찰 계수 일괄
    d = mujoco.MjData(m)
    jq = {nm: m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in
        ["box_free", "hinge"] + [f"{arm}_{ax}" for arm in ARMS
                                 for ax in ("x", "y", "z")]}
    tip_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"tip_{arm}")
               for arm in ARMS]              # 구동기 순서 = ARMS × xyz (선언순)
    d.qpos[jq["box_free"]:jq["box_free"] + 7] = q0[:7]
    d.qpos[jq["hinge"]] = q0[7]
    for a, arm in enumerate(ARMS):
        for k, ax in enumerate(("x", "y", "z")):
            d.qpos[jq[f"{arm}_{ax}"]] = 0.0    # 몸체 원점 = park 위치
        d.ctrl[3 * a:3 * a + 3] = 0.0
    park0 = targets[0].copy()

    def tip_ctrl(a, world_p):
        d.ctrl[3 * a:3 * a + 3] = world_p - park0[a]

    def tip_pos(a):
        return park0[a] + np.array(
            [d.qpos[jq[f"{ARMS[a]}_{ax}"]] for ax in ("x", "y", "z")])

    jv = {nm: m.jnt_dofadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in
        [f"{arm}_{ax}" for arm in ARMS for ax in ("x", "y", "z")]}

    def tip_force(a):
        # 박스에 실제 가해지는 힘 = 구동기 힘 − 관절 감쇠 소모 (KV·속도).
        # 구동기 힘만 쓰면 팁이 움직이는 동안 감쇠 몫(속도 0.1m/s 당 4N)이
        # 과대 계상되어 서보가 실제보다 세게 민다고 착각함 (08-14 진단)
        act = np.array(d.actuator_force[3 * a:3 * a + 3])
        vel = np.array([d.qvel[jv[f"{ARMS[a]}_{ax}"]] for ax in ("x", "y", "z")])
        return act - KV * vel

    mujoco.mj_forward(m, d)
    rend = None
    if args.video:
        rend = mujoco.Renderer(m, 600, 800)
    frames = []

    # ── QP 층 (play_contact.py 와 동일) ──
    KP_O, DP_O = 500.0, 30.0
    KR_O, DR_O = 5.0, args.dro
    F_FB_CAP, T_FB_CAP = args.ffc, args.tfc
    MU_QP = 0.5
    # 진짜 질량중심 (box_decomp.urdf 관성 원점, 질량 가중; 뚜껑은 θ 회전 동반).
    # cen(템플릿 표면 평균)과 2.4cm 차이 → 중력 보상을 cen 에 걸면 0.26N·m
    # 상시 기생 토크 발생 (08-14 진단). 추적점은 cen 유지, 중력만 질량중심에.
    MB, MT = 0.9641835, 0.13
    CB = np.array([0.06401, 0.12395, 0.07234])
    CT = np.array([0.11218, 0.07129, 0.07301])
    MG = (MB + MT) * 9.81
    CARROT_ANG = np.radians(args.carrot)   # 당김 회전 접근 한도 (CEM 변수;
    # 유도값 7.6° 는 5회 평가 기각 — 08-16)
    N_RAY = 8
    MU_RAY = MU_QP * np.cos(np.pi / N_RAY)
    prev_c, prev_er = None, None
    off_acc = np.zeros((NA, 3))
    alloc_dbg, fm_dbg, w_dbg = {}, {}, np.zeros(6)

    qp_x = [None]                 # 웜스타트: 이전 스텝 해 (변수 48개, 300회로는
    # 미수렴 — 종이 3000회 순z 13.2N vs 시뮬 300회 6.4N, 08-14 용량 시험)

    def qp_forces(P, Nin, w_des, floor_n=None):
        lever = args.lever   # 회전력↔힘 환율 겸 회전 우선 가중 (CEM 변수;
        # 실팔길이 유도는 5회 평가 기각 — 08-16)
        cols = []
        for p, n in zip(P, Nin):
            t1 = np.cross(n, [0, 0, 1.0])
            nt1 = np.linalg.norm(t1)
            t1 = t1 / nt1 if nt1 > 1e-6 else np.array([1.0, 0, 0])
            t2 = np.cross(n, t1)
            for k in range(N_RAY):
                ang = 2 * np.pi * k / N_RAY
                s = np.cos(ang) * t1 + np.sin(ang) * t2
                e = (n + MU_RAY * s) / np.linalg.norm(n + MU_RAY * s)
                cols.append(np.concatenate([e, np.cross(p, e) / lever]))
        A = np.array(cols).T
        w = np.concatenate([w_des[:3], w_des[3:] / lever])
        # 최소 쥐기 하한: 팁 i 의 좌표들에 균일 바닥을 깔면 법선력 = floor_n[i],
        # 대칭 접선은 상쇄. 균형은 하한 위 여유분에서 최적화가 맞춤
        lo = np.zeros(len(cols))
        if floor_n is not None:
            e_n = 1.0 / np.sqrt(1.0 + MU_RAY ** 2)   # 광선의 법선 성분
            for i, fn in enumerate(floor_n):
                lo[N_RAY * i:N_RAY * i + N_RAY] = max(fn, 0.0) / (N_RAY * e_n)
        x = qp_x[0] if qp_x[0] is not None and len(qp_x[0]) == len(cols) \
            else np.zeros(len(cols))
        L = np.linalg.norm(A, 2) ** 2
        for _ in range(1500):
            x = np.clip(x - (A.T @ (A @ x - w)) / L, lo, 15.0)
        qp_x[0] = x.copy()
        return np.array([np.array(cols[N_RAY * i:N_RAY * i + N_RAY])[:, :3].T
                         @ x[N_RAY * i:N_RAY * i + N_RAY]
                         for i in range(len(P))])

    pos_err, rot_err, th_err, z_act, z_ref = [], [], [], [], []
    # 무게 게이트 = 지지 틈 기하 판정 (08-16 ⓑ): 컴파일러가 저장한
    # "기준 자세 물체 최저점 − 받침대 윗면" 틈으로 직접 계산. 분위수·수평
    # 속도 어림 제거. 0.4초 선행(들기 직전 미리 팽팽해지기)은 유지
    LOOKAHEAD = 24
    if "ref_gap" in z.files:
        gap = z["ref_gap"]
        gap_ahead = np.array([gap[t:t + LOOKAHEAD + 1].max()
                              for t in range(T)])
        g_lift = np.clip(gap_ahead / 0.02, 0.0, 1.0)
    else:                                    # 옛 계획 파일 호환 (분위수 어림)
        z_rest = float(np.percentile(ref_pos[:, 2], 5))
        z_ahead = np.array([ref_pos[t:t + LOOKAHEAD + 1, 2].max()
                            for t in range(T)])
        g_lift = np.clip((z_ahead - z_rest) / 0.02, 0.0, 1.0)
    z_drop = float(ref_pos[:, 2].min()) - 0.15
    fell = None
    for t in range(T):
        q = np.concatenate([d.qpos[jq["box_free"]:jq["box_free"] + 7],
                            [d.qpos[jq["hinge"]]]])
        Rw, tw, thw = quat_R(q[3:7]), q[:3], float(q[7])
        step_t = targets[t].copy()
        ancs, nrms = {}, {}
        for a in range(NA):
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
                cn = np.clip(cn, -0.003, float(delta[t, a]))
                step_t[a] = anc + (SPHERE_R - delta[t, a] + cn) * nrm + ct
            elif appr[t, a] >= 0:
                u = float(appr[t, a])
                step_t[a] = anc + (SPHERE_R + (1 - u) * 0.08) * nrm

        on = [a for a in range(NA) if att[t, a] and a in ancs]
        if args.qp and len(on) >= 2:
            c_act = q[:3] + Rw @ cen
            R_ref = quat_R(ref_quat[t])
            e_p = ref_pos[t] - c_act
            Rrel = R_ref @ Rw.T
            ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
            e_r = np.zeros(3) if ang < 1e-6 else ang / (2 * np.sin(ang)) * \
                np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0],
                          Rrel[1, 0] - Rrel[0, 1]])
            v_c = np.zeros(3) if prev_c is None else (c_act - prev_c) * 60
            v_r = np.zeros(3) if prev_er is None else (e_r - prev_er) * 60
            prev_c, prev_er = c_act, e_r
            # 축별 클립 (노름 상한은 z 포화 시 수평 보정 몫이 0이 되어
            # 수평 밀림이 무보정 누적 — 08-14 왼쪽 이탈 영상의 원인)
            fb_f = np.clip(KP_O * e_p - DP_O * v_c, -F_FB_CAP, F_FB_CAP)
            fb_t = KR_O * e_r + DR_O * v_r
            nt_ = np.linalg.norm(fb_t)
            if nt_ > T_FB_CAP:
                fb_t *= T_FB_CAP / nt_
            com_p = (MB * CB + MT * (Rz(-thw) @ CT)) / (MB + MT)
            r_g = Rw @ (com_p - cen)          # 질량중심 − 추적점 (세계 좌표)
            # 무게 게이트: 기준 궤적이 탁자 위(정지 높이)면 무게는 탁자 몫 —
            # 탁자 위인데 무게 전부를 끌어올리려던 것이 초기 옆 밀림의 원인
            gMG = g_lift[t] * MG
            w_des = np.concatenate(
                [fb_f + np.array([0, 0, gMG]),
                 fb_t + np.cross(r_g, np.array([0, 0, gMG]))])
            P = np.array([ancs[a] - c_act for a in on])
            Nin = np.array([-nrms[a] for a in on])
            env_t = np.array([max(delta[t, a], 1e-6) /
                              max(delta[:, a].max(), 1e-6) for a in on])
            # 최소 쥐기 = 배분 문제의 하한 제약 (법선만 더하면 합력이 깨짐 —
            # 균형 접선까지 포함한 배분은 최적화가 하한 위에서 스스로 찾음)
            # 당근 거리 제한: 당김 목표용 기준 자세를 "현재 자세에서 시연
            # 쪽으로 회전 ≤20°, 위치 ≤2cm 만 접근한 중간 자세"로 — 오차가
            # 그 안이면 시연 자세 그대로(앞부분 무변화), 크게 벌어진 꼬리
            # 구간에서만 발동해 먼 목표로의 직선 당김 오발을 막음
            er_n = np.linalg.norm(e_r)
            if er_n > 1e-6:
                R_lim = rot_axis(e_r / er_n, min(er_n, CARROT_ANG)) @ Rw
            else:
                R_lim = R_ref
            dp_ = ref_pos[t] - c_act
            np_ = np.linalg.norm(dp_)
            p_lim = c_act + dp_ * (min(np_, D_MAX) / np_ if np_ > 1e-9 else 0.0)
            t_lim = p_lim - R_lim @ cen
            # 최소 쥐기의 국면 스케줄: 정지 국면(게이트 0)엔 예압 전체(밀림
            # 방지), 이동 국면(게이트 1)엔 유지 2.5N — 윗팁이 이동 내내 눌러
            # 박스를 물어버리던 상충(15N 상시·x 상쇄·사출)의 뿌리 수정
            g_ = float(g_lift[t])
            fl = (np.array([press[t, a] for a in on]) * (1 - g_) + 2.5 * g_)
            f = qp_forces(P, Nin, w_des, floor_n=fl)
            f *= env_t[:, None]
            if demo_f[t] in (585, 598, 611, 624, 637, 650):  # 꼬리: 회전 사슬
                fs_ = np.array([tip_force(a) for a in on])
                tq_a = sum(np.cross(P[k], f[k]) for k in range(len(on)))
                tq_r = sum(np.cross(P[k], fs_[k]) for k in range(len(on)))
                spd = max(np.linalg.norm(
                    [d.qvel[jv[f"{ARMS[a]}_{ax}"]] for ax in ("x", "y", "z")])
                    for a in on)
                print(f"[꼬리 f={demo_f[t]}] 회전오차={np.degrees(np.linalg.norm(e_r)):5.1f}° "
                      f"토크: 요구{np.linalg.norm(w_des[3:]):4.2f} "
                      f"배분{np.linalg.norm(tq_a):4.2f} 실현{np.linalg.norm(tq_r):4.2f}Nm "
                      f"팁최고속도={spd:.2f}m/s 박스z={100*c_act[2]:.1f}cm "
                      f"게이트={g_:.2f}", flush=True)
            alloc_dbg.clear(); fm_dbg.clear(); w_dbg = w_des.copy()
            for k, a in enumerate(on):
                fm = tip_force(a)
                # 전방 지시: 원하는 힘/강성 만큼 치우침을 즉시 부여 (순수
                # 적분기의 느린 축적과 진동 제거). 적분기는 잔차만 소이득 보정
                off_acc[a] = np.clip(
                    off_acc[a] + args.sg * (f[k] - fm) / KP, -0.010, 0.010)
                # 운반 당김: 당근 제한 자세(R_lim, t_lim) 기준의 팁 목표 쪽으로
                # 최대 2cm 앞서 당김 — 집게 상쇄를 뚫는 위치 통로. 오차가
                # 작으면 시연 자세 기준과 동일
                Rp_l = R_lim @ Rz(-float(ref_theta[t])) if a_part[t, a] \
                    else R_lim
                tgt_c = Rp_l @ a_q[t, a] + t_lim + SPHERE_R * (Rp_l @ a_n[t, a])
                carry = tgt_c - (ancs[a] + SPHERE_R * nrms[a])
                nc_ = np.linalg.norm(carry)
                if nc_ > D_MAX:
                    carry *= D_MAX / nc_
                step_t[a] = (ancs[a] + SPHERE_R * nrms[a]
                             + f[k] / KP + off_acc[a] + carry)
                alloc_dbg[a], fm_dbg[a] = f[k].copy(), fm.copy()
        for a in range(NA):
            tip_ctrl(a, step_t[a])
            d.xfrc_applied[tip_bid[a], 2] = TIP_M * 9.81   # 중력 보상
        for _ in range(N_SUB):
            mujoco.mj_step(m, d)
        if 180 < t < 420 and t % 30 == 0:
            fs = [tip_force(a) for a in range(NA)]
            print(f"[힘 t={t}] " + " | ".join(
                f"{ARMS[a]} z{fs[a][2]:+5.1f}N |f|{np.linalg.norm(fs[a]):4.1f}"
                for a in range(NA)) + f"  Σz={sum(f_[2] for f_ in fs):+.1f}",
                flush=True)
            if args.qp and alloc_dbg:
                print(f"   목표렌치 힘({w_dbg[0]:+.1f},{w_dbg[1]:+.1f},"
                      f"{w_dbg[2]:+.1f})N 토크({w_dbg[3]:+.2f},{w_dbg[4]:+.2f},"
                      f"{w_dbg[5]:+.2f})Nm", flush=True)
                for a in sorted(alloc_dbg):
                    o = off_acc[a] * 1000
                    sat = " ★클립포화" if np.abs(off_acc[a]).max() >= 0.0199 \
                        else ""
                    print(f"   {ARMS[a]}: 배분({alloc_dbg[a][0]:+.1f},"
                          f"{alloc_dbg[a][1]:+.1f},{alloc_dbg[a][2]:+.1f})N "
                          f"실측z{fm_dbg[a][2]:+.1f}N 치우침mm({o[0]:+.1f},"
                          f"{o[1]:+.1f},{o[2]:+.1f}){sat}", flush=True)
        if q[2] < z_drop and fell is None:
            fell = t
        if demo_f[t] >= 0:
            pos_err.append(np.linalg.norm(q[:3] + Rw @ cen - ref_pos[t]))
            dq = abs(float(np.dot(q[3:7], ref_quat[t])))
            rot_err.append(2 * np.arccos(min(1.0, dq)))
            th_err.append(abs(thw - float(ref_theta[t])))
            z_act.append(float((q[:3] + Rw @ cen)[2]))
            z_ref.append(float(ref_pos[t, 2]))
        if rend is not None and t % 4 == 0:
            rend.update_scene(d, camera="cam")
            frames.append(rend.render())

    pe, re_, te = map(np.asarray, (pos_err, rot_err, th_err))
    dfv = demo_f[demo_f >= 0]
    print("\n오차 시간 곡선 (데모 프레임: 위치cm / 회전°):")
    for k in range(0, len(pe), max(1, len(pe) // 14)):
        print(f"  f{dfv[k]:4d}: {pe[k]*100:5.1f}cm  {np.degrees(re_[k]):6.1f}°")
    print(f"\n=== 사다리 2단 MuJoCo판 판정 ({args.seq}) ===")
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
    lift = zr - z0 > 0.03
    if lift.any():
        print(f"  데모 들림(>3cm) 구간에서 실물 평균 들림: "
              f"{100*(za[lift]-z0).mean():.1f}cm (데모 평균 "
              f"{100*(zr[lift]-z0).mean():.1f}cm)")
    if frames:
        import imageio
        vp = osp.join(REPO, "md_claude", "dexmachina",
                      f"contact_mj_replay_{args.seq}.mp4")
        imageio.mimsave(vp, frames, fps=15)
        print(f"비디오: {vp}")


if __name__ == "__main__":
    main()
