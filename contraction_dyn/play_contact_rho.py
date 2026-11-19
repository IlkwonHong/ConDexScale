"""ρ판 폐루프 재생기 — 키프레임 접촉 조건 아래의 사람 경로 추종 + ℓ∞ 배분 여유 힘 배분.

설계: md_claude/plans/rho_continuous_control_v1.md. play_contact_mj.py(옛 재생기)에서 바뀐 것:
  ① 요구 렌치에 시연 가속 하중 전방지시 (m·a_ref, I·α_ref + ω×Iω) 추가. 되먹임(임피던스) 상수는 CEM 값 유지.
  ② 힘 배분 = alloc_margin.MarginAllocator (영공간 분해 + 배분 여유 r_req). 미리 쥐는 힘 표·바닥·저울 상수 없음.
  ③ 전환 = 순간이동: 계획의 swap 사건에서 시연 시계 정지, 박스는 유령 홀드로 현재 자세 고정, 옮기는 팁은
     새 앵커(레이캐스트 표면 정렬)로 즉시 이동, 상한 램프로 힘 투입, 유령 해제 후 재개. (비물리적 단순화 — 명시)
  ④ 진단 기록: 오차, ρ(t)(계획 파일), 확보 여유 r(t), 배분 단계, 실측 힘의 여유, 팁 힘.

실행: ./venv/bin/python -m contraction_dyn.play_contact_rho box_grab_01 [--plan-tag __nomove]
        [--r-req 0.15] [--no-ff] [--video] [--tag 이름]
출력: data/contraction_dyn/eval_rho/{subj}__{seq}__{tag}.npz (+ 표준출력 판정, --video 시 mp4)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")     # 화면 없는 환경에서의 영상 렌더링
import mujoco  # noqa: E402
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.alloc_margin import MarginAllocator      # noqa: E402
from contraction_dyn.box_model import BoxModel                # noqa: E402

DECOMP = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp"
OUT = ROOT / "data/contraction_dyn/eval_rho"

KP, KV, F_MAX = 1500.0, 40.0, 30.0
SPHERE_R, TIP_M = 0.015, 0.1
D_MAX = F_MAX / KP                    # 운반 당김 한도 (2cm)
FRIC = 0.6
N_SUB, DT = 10, 1.0 / 600
# 임피던스 층 — CEM 우승값 유지 (08-16). 전방지시가 얹혀도 손대지 않음 (그릴링 Q3)
KP_O, DP_O, KR_O = 500.0, 30.0, 5.0
DEFAULTS = dict(sg=0.036951158159670316, ffc=2.8006712639830047, tfc=2.717526380636423,
                dro=0.886670732144944, carrot=17.513066072600232)
# 배분기
MU_ALLOC = 0.5                        # 시뮬 μ 0.6 보다 보수적 (옛 QP 와 동일)
# 전환(유령 홀드) — static_hold_test 와 동일 이득
GHOST_KP, GHOST_KV, GHOST_KR, GHOST_KW = 600.0, 30.0, 25.0, 1.5
CAP_FLOOR = 0.5                       # 램프 시작 상한 [N]


def quat_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(u, a):
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def rotvec_err(R_ref, Rw):
    """actual → ref 회전 오차 벡터 (play_contact_mj 와 동일)."""
    Rrel = R_ref @ Rw.T
    ang = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array(
        [Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])


def build_mjcf(q0, plat, park, arms, bm, kv=KV, force_tip=False):
    import xml.etree.ElementTree as ET
    urdf = ET.parse(DECOMP / "box_decomp.urdf").getroot()
    links = {lk.get("name"): lk for lk in urdf.findall("link")}
    meshes, geoms = [], {"bottom": [], "top": []}
    for nm, lk in links.items():
        for i, col in enumerate(lk.findall("collision")):
            f = col.find("geometry/mesh").get("filename")
            meshes.append(f'<mesh name="{nm}_{i}" file="{f}"/>')
            geoms[nm].append(
                f'<geom type="mesh" mesh="{nm}_{i}" group="0" friction="{FRIC} 0.005 0.0001"'
                f' rgba="1 .42 .04 1"/>')
    tips = []
    for a, arm in enumerate(arms):
        p = park[a]
        tips.append(f"""
    <body name="tip_{arm}" pos="{p[0]} {p[1]} {p[2]}">
      <joint name="{arm}_x" type="slide" axis="1 0 0" damping="{kv}"/>
      <joint name="{arm}_y" type="slide" axis="0 1 0" damping="{kv}"/>
      <joint name="{arm}_z" type="slide" axis="0 0 1" damping="{kv}"/>
      <geom name="tipg_{arm}" type="sphere" size="{SPHERE_R}" mass="{TIP_M}" group="2"
            friction="{FRIC} 0.005 0.0001" rgba=".8 .8 .8 1"/>
    </body>""")
    acts = "" if force_tip else "\n".join(
        f'<position joint="{arm}_{ax}" kp="{KP}" forcerange="-{F_MAX} {F_MAX}"/>'
        for arm in arms for ax in ("x", "y", "z"))
    look = np.array([plat[0], plat[1], plat[2] + 0.15])
    cpos = look + np.array([0.1, -1.6, 0.5])
    zax = (cpos - look) / np.linalg.norm(cpos - look)
    return f"""
<mujoco model="contact_replay_rho">
  <compiler autolimits="true" meshdir="{DECOMP}"/>
  <option timestep="{DT}" impratio="10" cone="elliptic" noslip_iterations="5"/>
  <default><geom solref="0.004 1"/></default>
  <visual><global offwidth="800" offheight="600"/></visual>
  <asset>{''.join(meshes)}</asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <camera name="cam" pos="{cpos[0]} {cpos[1]} {cpos[2]}" zaxis="{zax[0]} {zax[1]} {zax[2]}" fovy="32"/>
    <geom type="plane" size="3 3 .1" group="3" rgba=".35 .4 .5 1"/>
    <geom type="box" pos="{plat[0]} {plat[1]} {plat[2] - 0.05}" size=".45 .45 .05" group="3"
          friction="{FRIC} 0.005 0.0001" rgba=".92 .92 .92 1"/>
    <body name="box_bottom" pos="{q0[0]} {q0[1]} {q0[2]}" quat="{q0[3]} {q0[4]} {q0[5]} {q0[6]}">
      <freejoint name="box_free"/>
      {bm.inertial_str('bottom')}
      {''.join(geoms['bottom'])}
      <body name="box_top" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 0 -1" range="0 2.6" damping="0.05" frictionloss="1.5"/>
        {bm.inertial_str('top')}
        {''.join(geoms['top'])}
      </body>
    </body>
    {''.join(tips)}
  </worldbody>
  {"<actuator>" + acts + "</actuator>" if acts else ""}
</mujoco>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--plan-tag", default="", help="재생 계획 변형 (예: __nomove)")
    ap.add_argument("--tag", default=None, help="출력 이름 (기본: 인자에서 생성)")
    ap.add_argument("--r-req", type=float, default=0.15, help="요구 여유 (ρ 정규화 단위)")
    ap.add_argument("--no-ff", action="store_true", help="가속 하중 전방지시 끄기")
    ap.add_argument("--kv", type=float, default=None, help="팁 관절 감쇠 [N·s/m] (기본 40; 힘 인가 모드 기본 0)")
    ap.add_argument("--force-tip", action="store_true",
                    help="힘 인가 팁(토크 제어 팔 흉내): 위치 구동기 없이 팁 몸체에 힘을 직접 가함 = 배분된 힘(피드포워드) "
                         "+ 팔별 부드러운 스프링 K_p(기준 자세 위 앵커 − 팁) + K_d(기준 속도 − 팁 속도) + 팁 중력 보상. "
                         "물체 수준 임피던스 렌치는 끔(--ft-objfb 로 유지). 속도 전방지시·어드미턴스·당김·적분 보정은 쓰지 않음")
    ap.add_argument("--ft-kp", type=float, default=300.0, help="힘 인가 팁의 팔별 스프링 강성 [N/m] (기본 300)")
    ap.add_argument("--ft-kd", type=float, default=None, help="힘 인가 팁의 팔별 감쇠 [N·s/m] (기본 2√(K_p·팁 질량))")
    ap.add_argument("--ft-cap", type=float, default=10.0, help="힘 인가 팁의 스프링 힘 상한 [N] (기본 10)")
    ap.add_argument("--ft-objfb", action="store_true",
                    help="힘 인가 팁에서도 물체 수준 임피던스 렌치를 배분기 입력에 유지 (기본: 끔, 팔별 스프링만)")
    ap.add_argument("--kpo", type=float, default=KP_O, help="임피던스 병진 강성 [N/m] (기본 500)")
    ap.add_argument("--dpo", type=float, default=DP_O, help="임피던스 병진 감쇠 [N·s/m] (기본 30)")
    ap.add_argument("--kro", type=float, default=KR_O, help="임피던스 회전 강성 [N·m/rad] (기본 5)")
    ap.add_argument("--damp-comp", action="store_true",
                    help="감쇠 보상: 팁 목표에 KV·v_tip/KP 를 더해 감쇠가 먹는 힘을 상쇄")
    ap.add_argument("--hybrid", action="store_true",
                    help="혼합 제어: 접선 = 순수 위치 제어(당김: 목표 자세의 접촉점으로), 법선 = 어드미턴스(배분 누르는 힘). "
                         "배분기에는 당김이 낼 접선 하중을 알려 쥐기를 정하고, 여유 불가면 당김 걸음을 반으로 줄임(최대 3회). "
                         "임피던스 되먹임 렌치는 쓰지 않음(중력+전방지시만)")
    ap.add_argument("--hybrid-throttle", action="store_true",
                    help="혼합 제어에서 여유 불가 시 당김 걸음을 반으로 줄이는 용량 인식 걸음 (기본 꺼짐)")
    ap.add_argument("--lam-cont", type=float, default=0.0,
                    help="배분기 연속성 가중치: 목적에 w·‖f − 직전 틱 힘‖² 추가 (0 = 없음)")
    ap.add_argument("--norm-adm", action="store_true",
                    help="법선 어드미턴스: 접촉력(힘 센서 역할)으로 잰 누르는 힘을 배분 목표에 맞추도록 법선 목표를 물리 스텝마다 넣고 뺌")
    ap.add_argument("--adm-tau", type=float, default=0.05, help="법선 어드미턴스 시간 상수 [s] (작을수록 빠름)")
    ap.add_argument("--rel-damp", action="store_true",
                    help="임피던스 병진 감쇠를 기준 속도 상대(ẋ_ref − ẋ)로: 기준을 따라 움직일 때 감쇠가 운동을 거스르지 않음 (가설 1 판별)")
    ap.add_argument("--vel-ff", action="store_true",
                    help="기준 속도 전방지시: 물리 스텝마다 팁 목표를 접촉점의 기준 속도 × 경과시간만큼 옮기고(갱신 지연 상쇄), "
                         "감쇠 × 기준 속도 ÷ 강성만큼 앞당김(감쇠 상쇄). 기준 속도는 시연 궤적에서 미리 계산")
    ap.add_argument("--shape", action="store_true",
                    help="용량 인식 요구 다듬기: 중력 몫은 반드시, 되먹임+전방지시 몫은 여유 r_req 를 지키는 최대 배율 s 로 축소")
    ap.add_argument("--carry-aware", action="store_true",
                    help="운반 당김 힘(1500×Δ)을 배분기에 이미 정해진 힘으로 알려 줌 (총 힘에 각뿔·여유 제약)")
    ap.add_argument("--no-carry", action="store_true",
                    help="운반 당김(팁 목표를 기준 자세 쪽으로 최대 2cm 옮기는 위치 항) 끄기 — 물체를 끄는 힘은 되먹임 렌치 경로만")
    ap.add_argument("--video", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--dr-mass", type=float, default=1.0)
    ap.add_argument("--dr-fric", type=float, default=FRIC)
    ap.add_argument("--dr-com", type=float, default=0.0)
    ap.add_argument("--stop-at", type=int, default=None, help="이 스텝까지만 진행 (진단용 조기 종료)")
    ap.add_argument("--depen-iters", type=int, default=0,
                    help="순간이동 뒤 침투 해소 반복 횟수 (바깥 법선으로 밀어 올림; 기본 0 = 안 함. 오목한 이음새에서는 오히려 더 파고든 사례 있음)")
    ap.add_argument("--debug-contacts", type=int, nargs=2, default=None,
                    help="이 스텝 구간에서 팁마다 접촉 목록(상대 geom, 깊이, 법선, 힘 벡터) 출력")
    ap.add_argument("--debug-steps", type=int, nargs=2, default=None,
                    help="이 스텝 구간의 상태(박스·유령·팁 힘·접촉)를 매 스텝 출력")
    args = ap.parse_args(argv)
    if args.kv is None:
        args.kv = 0.0 if args.force_tip else KV
    if args.ft_kd is None:
        args.ft_kd = 2.0 * np.sqrt(args.ft_kp * TIP_M)
    tag = args.tag or (f"r{args.r_req:.2f}" + ("_noff" if args.no_ff else "")
                       + ("_nocarry" if args.no_carry else "") + ("_velff" if args.vel_ff else "")
                       + ("_shape" if args.shape else "") + ("_caware" if args.carry_aware else "")
                       + ("_reldamp" if args.rel_damp else "") + ("_adm" if args.norm_adm else "")
                       + (f"_cont{args.lam_cont:g}" if args.lam_cont > 0 else "") + ("_hybrid" if args.hybrid else "")
                       + (f"_ftip{args.ft_kp:g}" + ("_objfb" if args.ft_objfb else "") if args.force_tip else "")
                       + args.plan_tag.replace("__", "_"))

    z = np.load(ROOT / "data/contraction_dyn/replay_rho" / f"{args.subj}__{args.seq}{args.plan_tag}.npz",
                allow_pickle=True)
    ARMS = [str(x) for x in z["arms"]]
    NA = len(ARMS)
    T = len(z["demo_frame"])
    demo_f, phase, ghost_g = z["demo_frame"], z["swap_phase"], z["ghost_gain"]
    swap_move, att = z["swap_move"], z["attached"]
    a_part, a_q, a_n = z["anchor_part"], z["anchor_q"], z["anchor_n"]
    ref_pos, ref_quat, ref_theta = z["ref_pos"], z["ref_quat"], z["ref_theta"]
    ref_acc, ref_om, ref_al = z["ref_acc"], z["ref_omega"], z["ref_alpha"]
    g_lift, rho_t = z["g_lift"], z["rho_t"]
    # 기준 추적점 속도 v_ref [m/s] — 실행 스텝(demo_frame ≥ 0)에서만 중심 차분, 멈춘 스텝은 0
    v_ref = np.zeros_like(ref_pos, dtype=float)
    _valid = demo_f >= 0
    for _t in range(1, T - 1):
        if _valid[_t - 1] and _valid[_t] and _valid[_t + 1]:
            v_ref[_t] = (ref_pos[_t + 1].astype(float) - ref_pos[_t - 1].astype(float)) * 30.0
    _k = np.ones(3) / 3
    v_ref = np.stack([np.convolve(v_ref[:, i], _k, mode="same") for i in range(3)], 1)
    v_ref[~_valid] = 0.0
    q0, plat, cen = z["q0"], z["platform"], z["cen"]
    r_obj = float(z["r_obj"])
    bm = BoxModel()
    MASS = bm.mass
    MG = MASS * 9.81
    CARROT = np.radians(args.carrot)

    # 대기 위치: 받침대 둘레 반지름 0.4m, 높이 +0.3m
    park = np.array([[plat[0] + 0.4 * np.cos(2 * np.pi * k / NA),
                      plat[1] + 0.4 * np.sin(2 * np.pi * k / NA), plat[2] + 0.3] for k in range(NA)])
    KVr = float(args.kv)
    m = mujoco.MjModel.from_xml_string(build_mjcf(q0, plat, park, ARMS, bm, kv=KVr, force_tip=args.force_tip))
    for bn in ("box_bottom", "box_top"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        m.body_mass[bid] *= args.dr_mass
        m.body_inertia[bid] *= args.dr_mass
        if bn == "box_bottom":
            m.body_ipos[bid][1] += args.dr_com
    m.geom_friction[:, 0] = args.dr_fric
    d = mujoco.MjData(m)
    jq = {nm: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
          for nm in ["box_free", "hinge"] + [f"{arm}_{ax}" for arm in ARMS for ax in "xyz"]}
    jv = {nm: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
          for nm in ["box_free"] + [f"{arm}_{ax}" for arm in ARMS for ax in "xyz"]}
    BOX_BID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box_bottom")
    tip_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"tip_{arm}") for arm in ARMS]
    d.qpos[jq["box_free"]:jq["box_free"] + 7] = q0[:7]
    d.qpos[jq["hinge"]] = q0[7]
    mujoco.mj_forward(m, d)

    def tip_ctrl(a, p):
        if args.force_tip:
            return
        d.ctrl[3 * a:3 * a + 3] = p - park[a]

    def tip_pos(a):
        return np.array([d.qpos[jq[f"{ARMS[a]}_{ax}"]] for ax in "xyz"]) + park[a]

    def tip_set(a, p):
        for k, ax in enumerate("xyz"):
            d.qpos[jq[f"{ARMS[a]}_{ax}"]] = p[k] - park[a][k]
            d.qvel[jv[f"{ARMS[a]}_{ax}"]] = 0.0
        tip_ctrl(a, p)

    def tip_vel(a):
        return np.array([d.qvel[jv[f"{ARMS[a]}_{ax}"]] for ax in "xyz"])

    def tip_force(a):
        """팁이 박스에 가하는 힘 [N, 세계 좌표]. 위치 서보 = 구동기 힘 − 감쇠; 힘 인가 모드 = 접촉력 벡터(힘 센서 역할)."""
        if args.force_tip:
            return contact_force_vec(a)
        act = np.array(d.actuator_force[3 * a:3 * a + 3])
        return act - KVr * tip_vel(a)

    tip_gid_of = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"tipg_{arm}"): a for a, arm in enumerate(ARMS)}
    _f6 = np.zeros(6)

    def contact_fn(a):
        """팁 a 가 박스에서 받는 접촉 법선력 합 [N] (MuJoCo 접촉력 — 실물의 힘 센서에 해당)."""
        tot = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            g_other = None
            if tip_gid_of.get(c.geom1) == a:
                g_other = c.geom2
            elif tip_gid_of.get(c.geom2) == a:
                g_other = c.geom1
            if g_other is None or m.geom_group[g_other] != 0:
                continue
            mujoco.mj_contactForce(m, d, ci, _f6)
            tot += abs(float(_f6[0]))
        return tot

    def contact_on_box(ci, a):
        """접촉 ci(팁 a 와 박스)에서 박스가 받는 힘 [N, 세계 좌표].
        mj_contactForce 의 법선 성분은 항상 ≥ 0 이고 접촉 좌표계 법선 축을 따르는데, 그 축이 팁→박스 인지 박스→팁 인지는
        충돌 쌍 종류마다 달라(공·상자 시험과 공·메시 시험에서 반대) geom 순서로 판단하지 않는다. 대신 기하로 정한다:
        박스가 받는 힘은 팁 중심에서 접촉점을 향하는 쪽(팁이 박스를 미는 방향)이어야 한다.
        주의: d.contact[ci].frame/.pos 는 이 판(2.3.7)에서 0 을 돌려주므로 d.contact.frame[ci]/.pos[ci] 를 쓴다."""
        mujoco.mj_contactForce(m, d, ci, _f6)
        fr = np.asarray(d.contact.frame[ci]).reshape(3, 3)
        w = fr.T @ _f6[:3]
        outward = np.asarray(d.contact.pos[ci]) - tip_pos(a)          # 팁 중심 → 접촉점
        return w if float(fr[0] @ outward) >= 0 else -w

    def contact_force_vec(a):
        """팁 a 가 박스에 가하는 접촉력 벡터 [N, 세계 좌표] = 박스가 받는 힘 (힘 센서 역할)."""
        tot = np.zeros(3)
        for ci in range(d.ncon):
            g1, g2 = int(d.contact.geom1[ci]), int(d.contact.geom2[ci])
            if (tip_gid_of.get(g1) == a and m.geom_group[g2] == 0) or (tip_gid_of.get(g2) == a and m.geom_group[g1] == 0):
                tot += contact_on_box(ci, a)
        return tot

    adm_n = np.zeros(NA)                              # 법선 어드미턴스 상태: 접촉점에서 안쪽으로 넣은 거리 [m]
    fn_target = np.zeros(NA)
    ray_grp = np.zeros(6, dtype=np.uint8)
    ray_grp[0] = 1                                   # 박스 geom(group 0)만
    gid = np.zeros(1, dtype=np.int32)

    def surface_align(anchor_w, n_in):
        """앵커 법선 선상에서 충돌 메시 표면 교점 (static_hold_test 규약)."""
        for back in (0.05, 0.03, 0.015, 0.008):
            origin = anchor_w - n_in * back
            dist = mujoco.mj_ray(m, d, origin, n_in, ray_grp, 1, -1, gid)
            if dist >= 0 and gid[0] >= 0:
                h = origin + n_in * dist
                if np.linalg.norm(h - anchor_w) < 0.04:
                    return h, True
        return anchor_w, False

    ghost_pose = None

    def ghost_hold(gain):
        if gain <= 0 or ghost_pose is None:
            d.xfrc_applied[BOX_BID, :] = 0
            return
        p_ref, q_ref = ghost_pose
        p = d.qpos[jq["box_free"]:jq["box_free"] + 3]
        q = d.qpos[jq["box_free"] + 3:jq["box_free"] + 7]
        v = d.qvel[jv["box_free"]:jv["box_free"] + 3]
        w = d.qvel[jv["box_free"] + 3:jv["box_free"] + 6]
        Rerr = quat_R(q) @ quat_R(q_ref).T
        rvec = Rotation.from_matrix(Rerr).as_rotvec()
        d.xfrc_applied[BOX_BID, :3] = gain * (GHOST_KP * (p_ref - p) - GHOST_KV * v)
        d.xfrc_applied[BOX_BID, 3:] = gain * (-GHOST_KR * rvec - GHOST_KW * w)

    if args.hybrid:
        args.norm_adm = True
        if args.lam_cont <= 0:
            args.lam_cont = 20.0
    alloc = MarginAllocator(mu=MU_ALLOC, f_cap=F_MAX, r_req=args.r_req, cont_w=args.lam_cont)
    attach_step = np.full(NA, -10 ** 6)
    off_acc = np.zeros((NA, 3))
    prev_c, prev_er = None, None
    rend = mujoco.Renderer(m, 600, 800) if args.video else None
    frames = []

    # 진단 기록
    D = dict(pos_err=np.full(T, np.nan), rot_err=np.full(T, np.nan), th_err=np.full(T, np.nan),
             r_used=np.full(T, np.nan), r_max=np.full(T, np.nan), stage=np.zeros(T, np.int8),
             r_meas=np.full(T, np.nan), resid=np.full(T, np.nan),
             w_des=np.full((T, 6), np.nan), f_alloc=np.full((T, NA, 3), np.nan),
             f_meas=np.full((T, NA, 3), np.nan), box_z=np.full(T, np.nan), align_fail=np.zeros(T, np.int8),
             swap_pen_mm=np.zeros(T), s_scale=np.full(T, np.nan), fn_meas=np.full((T, NA), np.nan), ft_meas=np.full((T, NA), np.nan),
             fn_contact=np.full((T, NA), np.nan), fn_target=np.full((T, NA), np.nan), f0_norm=np.full(T, np.nan), fN_norm=np.full(T, np.nan),
             f0_normal_mean=np.full(T, np.nan), fN_normal_mean=np.full(T, np.nan), eq_resid=np.full(T, np.nan), ft_spring_N=np.full((T, NA), np.nan),
             **{k: np.full((T, NA), np.nan) for k in ("gap_carry_N", "gap_off_N", "gap_damp_N", "gap_lag_N",
                                                      "gap_diff_N", "gap_diff_normal_N")})
    z_drop = float(ref_pos[:, 2].min()) - 0.15
    fell = None
    N_SETTLE = int((phase == 1).sum() / max(1, ((phase == 1) & (np.roll(phase, 1) != 1)).sum()))

    for t in range(T):
        q = np.concatenate([d.qpos[jq["box_free"]:jq["box_free"] + 7], [d.qpos[jq["hinge"]]]])
        Rw, tw, thw = quat_R(q[3:7]), q[:3].copy(), float(q[7])
        Rzt = Rz(-thw)
        ancs, nouts = {}, {}
        for a in range(NA):
            if not att[t, a] or np.isnan(a_q[t, a, 0]):
                continue
            Rp = Rw @ Rzt if a_part[t, a] else Rw
            ancs[a] = Rp @ a_q[t, a] + tw
            nouts[a] = Rp @ a_n[t, a]

        # ---- 전환 사건: 유령 홀드 + 순간이동 ----
        if phase[t] == 1 and (t == 0 or phase[t - 1] != 1):
            ghost_pose = (q[:3].copy(), q[3:7].copy())
            moved_now = []
            for a in range(NA):
                if swap_move[t, a] and a in ancs:
                    hit, ok = surface_align(ancs[a], -nouts[a])
                    D["align_fail"][t] += (not ok)
                    tip_set(a, hit + SPHERE_R * nouts[a])
                    # 침투 해소: 표면점 + 반지름 자리에서도 다른 조각과 겹치면(오목한 이음새) 바깥 법선으로 밀어 올린다
                    for _it in range(args.depen_iters):
                        mujoco.mj_forward(m, d)
                        deep = min([float(d.contact.dist[ci]) for ci in range(d.ncon)
                                    if tip_gid_of.get(int(d.contact.geom1[ci])) == a or tip_gid_of.get(int(d.contact.geom2[ci])) == a] + [0.0])
                        if deep > -0.001:
                            break
                        tip_set(a, tip_pos(a) + nouts[a] * (-deep + 0.0005))
                    attach_step[a] = t
                    off_acc[a] = 0.0
                    moved_now.append((a, hit, ok))
            # 순간이동 직후 침투 진단: 옮긴 팁의 가장 깊은 접촉 깊이 (음수 = 파고듦)
            mujoco.mj_forward(m, d)
            tip_gid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"tipg_{ARMS[a]}"): a for a in range(NA)}
            deepest = {a: 0.0 for a, _h, _ok in moved_now}
            for ci in range(d.ncon):
                c = d.contact[ci]
                for g in (c.geom1, c.geom2):
                    if g in tip_gid and tip_gid[g] in deepest:
                        deepest[tip_gid[g]] = min(deepest[tip_gid[g]], float(c.dist))
            for a, hit, ok in moved_now:
                msg = (f"  [전환 t={t} f={demo_f[max(0, t - 1)]}] {ARMS[a]}: 정렬 {'OK' if ok else '실패'} "
                       f"앵커→표면 {100 * np.linalg.norm(hit - ancs[a]):.1f}cm, 접촉 깊이 {1000 * deepest[a]:+.1f}mm")
                if deepest[a] < -0.004:
                    msg += "  ◀ 침투 경고"
                print(msg, flush=True)
            D["swap_pen_mm"][t] = 1000 * min(deepest.values()) if deepest else 0.0
        gg = float(ghost_g[t]) if phase[t] else 0.0
        # 정착 단계(phase 1)는 박스를 문자 그대로 멈춰 둔다 (위치·속도 고정 = 시뮬 정지).
        # PD 유령 홀드(600 N/m)는 팁 스프링 6개(각 1500 N/m)보다 약해, 움직이던 박스에 순간이동한
        # 팁들이 파고들며 회전을 유발하는 것을 막지 못했다 (2026-09-05 __always 판 f292 낙하 진단).
        # 해제 단계(phase 2)에서만 PD 유령을 램프로 푼다.
        freeze = phase[t] == 1
        ghost_hold(0.0 if freeze else gg)
        track_w = 1.0 - gg                                # 유령이 잡고 있는 동안 추종 되먹임 끔

        on = [a for a in range(NA) if a in ancs]
        step_t = {a: park[a].copy() for a in range(NA)}
        ft_des, ft_ff = {}, {}                            # 힘 인가 팁: a → (목표 위치, 목표 속도, 되먹임 가중), a → 피드포워드 힘
        for a in on:
            step_t[a] = ancs[a] + SPHERE_R * nouts[a]

        if len(on) >= 2:
            c_act = tw + Rw @ cen
            R_ref = quat_R(ref_quat[t])
            e_p = ref_pos[t] - c_act
            e_r = rotvec_err(R_ref, Rw)
            v_c = np.zeros(3) if prev_c is None else (c_act - prev_c) * 60
            v_r = np.zeros(3) if prev_er is None else (e_r - prev_er) * 60
            prev_c, prev_er = c_act, e_r
            v_rel = (v_c - v_ref[t]) if args.rel_damp else v_c        # 감쇠 기준 속도 (절대 / 기준 상대)
            fb_f = np.clip(args.kpo * e_p - args.dpo * v_rel, -args.ffc, args.ffc) * track_w
            fb_t = (args.kro * e_r + args.dro * v_r) * track_w
            nt_ = np.linalg.norm(fb_t)
            if nt_ > args.tfc:
                fb_t *= args.tfc / nt_
            if args.force_tip and not args.ft_objfb:
                fb_f, fb_t = np.zeros(3), np.zeros(3)     # 오차 보상은 팔별 부드러운 스프링이 맡음 (배분기 밖)
            _, com_c = bm.mass_com(thw)
            com_w = tw + Rw @ com_c
            gMG = float(g_lift[t]) * MG
            if args.no_ff:
                ff_f, ff_t = np.zeros(3), np.zeros(3)
            else:
                Iw = Rw @ bm.inertia(thw) @ Rw.T
                om = ref_om[t].astype(float)
                ff_f = MASS * ref_acc[t]
                ff_t = Iw @ ref_al[t] + np.cross(om, Iw @ om)
            w_des = np.concatenate([fb_f + np.array([0, 0, gMG]) + ff_f,
                                    fb_t + np.cross(c_act - com_w, fb_f) + ff_t])
            caps = np.array([F_MAX * np.clip((t - attach_step[a]) / max(N_SETTLE, 1), CAP_FLOOR / F_MAX, 1.0)
                             for a in on])
            P = np.array([ancs[a] for a in on])
            Nin = np.array([-nouts[a] for a in on])
            # 운반 당김 (당근 제한 기준 자세) — play_contact_mj 와 동일. 배분 전에 계산해 두어
            # --carry-aware 이면 그 힘(KP×Δ)을 배분기에 이미 정해진 힘으로 넘긴다.
            er_n = np.linalg.norm(e_r)
            R_lim = rot_axis(e_r / er_n, min(er_n, CARROT)) @ Rw if er_n > 1e-6 else R_ref
            dp_ = ref_pos[t] - c_act
            np_ = np.linalg.norm(dp_)
            p_lim = c_act + dp_ * (min(np_, D_MAX) / np_ if np_ > 1e-9 else 0.0)
            t_lim = p_lim - R_lim @ cen
            carry_of = {}
            for a in on:
                Rp_l = R_lim @ Rz(-float(ref_theta[t])) if a_part[t, a] else R_lim
                tgt_c = Rp_l @ a_q[t, a] + t_lim + SPHERE_R * (Rp_l @ a_n[t, a])
                cv = (tgt_c - (ancs[a] + SPHERE_R * nouts[a])) * track_w
                nc_ = np.linalg.norm(cv)
                if nc_ > D_MAX:
                    cv *= D_MAX / nc_
                carry_of[a] = np.zeros(3) if args.no_carry else cv
            alloc.setup(P, Nin, com_w, r_obj, cap=caps)
            carry_scale = 1.0
            if args.hybrid:
                # 접선 하중 = 당김(위치 제어)이 낼 힘의 접선 성분. 배분기는 중력+전방지시만 요구받고, 그 하중을 고정 힘으로 알아
                # 쥐기(법선)를 정한다. 여유 불가(1단계 실패)면 당김 걸음을 반으로 줄여 재시도 (용량 인식 걸음).
                w_des = np.concatenate([np.array([0, 0, gMG]) + ff_f, ff_t])
                n_try = 4 if args.hybrid_throttle else 1
                for _try in range(n_try):
                    f_fixed = np.array([KP * (carry_of[a] - (carry_of[a] @ (-nouts[a])) * (-nouts[a])) * carry_scale for a in on])
                    sol = alloc.solve(w_des, f_fixed=f_fixed)
                    if sol["stage"] == 1 or _try == n_try - 1:
                        break
                    carry_scale *= 0.5
                f = sol["f"] - f_fixed                 # (접선 치우침에는 쓰지 않음 — 법선 목표만 사용)
            else:
                f_fixed = (np.array([KP * carry_of[a] for a in on]) if args.carry_aware else None)
                w_base = np.array([0, 0, gMG, 0, 0, 0.0]) if args.shape else None
                sol = alloc.solve(w_des, f_fixed=f_fixed, w_base=w_base)
                f = sol["f"]
                if f_fixed is not None:
                    f = f - f_fixed                    # 목표 치우침으로 실현할 몫 = 총 힘 − 당김 힘 (당김은 carry 로 따로 실현)
            fm_all = np.zeros((len(on), 3))
            for k, a in enumerate(on):
                fm = tip_force(a)
                fm_all[k] = fm
                if not args.force_tip:
                    off_acc[a] = np.clip(off_acc[a] + args.sg * (sol["f"][k] - fm) / KP, -0.010, 0.010)
                carry = carry_of[a]
                dcomp = (KVr * tip_vel(a) / KP) if args.damp_comp else np.zeros(3)
                if args.hybrid:
                    n_in_a = -nouts[a]
                    fn_target[a] = float(sol["f"][k] @ n_in_a)          # 배분기가 정한 누르는 힘 (총 힘의 법선)
                    carry_t = (carry - (carry @ n_in_a) * n_in_a) * carry_scale   # 접선 위치 제어 몫 (걸음 축소 반영)
                    if attach_step[a] == t:
                        adm_n[a] = fn_target[a] / KP
                    step_t[a] = ancs[a] + SPHERE_R * nouts[a] + carry_t + dcomp + adm_n[a] * n_in_a
                    D["gap_carry_N"][t, a] = KP * np.linalg.norm(carry_t)
                elif args.norm_adm:
                    n_in_a = -nouts[a]
                    fn_target[a] = float(sol["f"][k] @ n_in_a)          # 배분기가 정한 누르는 힘
                    f_tan = f[k] - (f[k] @ n_in_a) * n_in_a              # 접선 몫만 치우침으로
                    off_t = off_acc[a] - (off_acc[a] @ n_in_a) * n_in_a
                    if attach_step[a] == t:                              # 새로 붙은 팁: 전방지시 초기값
                        adm_n[a] = fn_target[a] / KP
                    step_t[a] = (ancs[a] + SPHERE_R * nouts[a] + f_tan / KP + off_t + carry + dcomp
                                 + adm_n[a] * n_in_a)
                else:
                    step_t[a] = ancs[a] + SPHERE_R * nouts[a] + f[k] / KP + off_acc[a] + carry + dcomp
                D["f_alloc"][t, a] = sol["f"][k]
                D["f_meas"][t, a] = fm
                # 실현 간극 분해 (N 단위): 당김·적분 보정이 만드는 스프링 힘, 감쇠, 접촉점 대비 팁 위치 지연
                tp = np.array([d.qpos[jq[f"{ARMS[a]}_{ax}"]] for ax in "xyz"]) + park[a]
                tv = np.array([d.qvel[jv[f"{ARMS[a]}_{ax}"]] for ax in "xyz"])
                if not args.hybrid:
                    D["gap_carry_N"][t, a] = KP * np.linalg.norm(carry)
                D["gap_off_N"][t, a] = KP * np.linalg.norm(off_acc[a])
                D["gap_damp_N"][t, a] = KVr * np.linalg.norm(tv)
                D["gap_lag_N"][t, a] = KP * np.linalg.norm((ancs[a] + SPHERE_R * nouts[a]) - tp)
                D["gap_diff_N"][t, a] = np.linalg.norm(fm - f[k])
                D["gap_diff_normal_N"][t, a] = float((fm - f[k]) @ (-nouts[a]))
                fn_m = float(fm @ (-nouts[a]))                       # 실측 누르는 힘 (법선 성분)
                D["fn_meas"][t, a] = fn_m
                D["fn_contact"][t, a] = contact_fn(a)                 # 접촉력으로 잰 누르는 힘
                D["fn_target"][t, a] = float(sol["f"][k] @ (-nouts[a]))
                D["ft_meas"][t, a] = float(np.linalg.norm(fm + fn_m * nouts[a]))   # 실측 접선 힘 크기
            # 배분 내부 진단: 특수해 f₀(요구 렌치만 만드는 최소 힘) vs 영공간 성분 Nλ(맞누르는 힘)
            f0 = alloc.particular(w_des).reshape(len(on), 3)
            fN = f - f0
            D["f0_norm"][t] = np.linalg.norm(f0)
            D["fN_norm"][t] = np.linalg.norm(fN)
            D["f0_normal_mean"][t] = float((f0 * Nin).sum(1).mean())
            D["fN_normal_mean"][t] = float((fN * Nin).sum(1).mean())
            D["eq_resid"][t] = float(np.linalg.norm(alloc.G @ f.reshape(-1) - w_des))
            D["r_used"][t], D["stage"][t] = sol["r"], sol["stage"]
            D["s_scale"][t] = carry_scale if args.hybrid else sol.get("s", 1.0)
            D["r_max"][t] = sol["r_max"] if sol["r_max"] is not None else np.nan
            D["resid"][t] = sol["resid"]
            D["r_meas"][t] = alloc.margin_of(fm_all, caps)
            D["w_des"][t] = w_des
            if args.force_tip:
                # 팔별 스프링의 목표 = 기준 자세(시연) 위의 앵커 + 반지름, 목표 속도 = 그 점의 기준 속도
                t_ref = ref_pos[t].astype(float) - R_ref @ cen
                om_r = ref_om[t].astype(float)
                for k, a in enumerate(on):
                    Rp_r = R_ref @ Rz(-float(ref_theta[t])) if a_part[t, a] else R_ref
                    p_des = Rp_r @ a_q[t, a] + t_ref + SPHERE_R * (Rp_r @ a_n[t, a])
                    v_des = v_ref[t] + np.cross(om_r, p_des - ref_pos[t].astype(float))
                    p_act = ancs[a] + SPHERE_R * nouts[a]                 # 지금 박스 위의 앵커 (전환 중 목표)
                    ft_des[a] = ((1.0 - track_w) * p_act + track_w * p_des, v_des * track_w, 1.0)
                    ft_ff[a] = sol["f"][k].copy()

        vff = {}
        if args.vel_ff and len(on) >= 2:
            om_r = ref_om[t].astype(float)
            c_now = tw + Rw @ cen
            for a in on:
                vff[a] = (v_ref[t] + np.cross(om_r, ancs[a] - c_now)) * track_w   # 접촉점의 기준 속도
        for a in range(NA):
            tip_ctrl(a, step_t[a] + (KVr * vff[a] / KP if a in vff else 0.0))
            d.xfrc_applied[tip_bid[a], 2] = TIP_M * 9.81
        adm_on = args.norm_adm and len(on) >= 2
        if args.force_tip:
            for j_sub in range(N_SUB):
                for a in range(NA):
                    if a in ft_des:
                        p_des, v_des, wgt = ft_des[a]
                        p_des = p_des + v_des * ((j_sub + 0.5) * DT)        # 틱 안 기준점 이동(갱신 지연 상쇄)
                    else:
                        p_des, v_des, wgt = step_t[a], np.zeros(3), 1.0    # 붙지 않은 팁: 대기/현재 앵커 유지
                    f_sp = (args.ft_kp * (p_des - tip_pos(a)) + args.ft_kd * (v_des - tip_vel(a))) * wgt
                    n_sp = np.linalg.norm(f_sp)
                    if n_sp > args.ft_cap:
                        f_sp *= args.ft_cap / n_sp
                    if j_sub == 0 and a in ft_des:
                        D["ft_spring_N"][t, a] = n_sp if n_sp < args.ft_cap else args.ft_cap
                    d.xfrc_applied[tip_bid[a], :3] = f_sp + ft_ff.get(a, 0.0) + np.array([0, 0, TIP_M * 9.81])
                if freeze and ghost_pose is not None:
                    d.qpos[jq["box_free"]:jq["box_free"] + 3] = ghost_pose[0]
                    d.qpos[jq["box_free"] + 3:jq["box_free"] + 7] = ghost_pose[1]
                    d.qvel[jv["box_free"]:jv["box_free"] + 6] = 0.0
                mujoco.mj_step(m, d)
        for j_sub in range(0 if args.force_tip else N_SUB):
            if adm_on:
                for a in on:                                    # 법선 어드미턴스: 힘 오차 → 법선 목표 보정 (600Hz)
                    n_in_a = -nouts[a]
                    err = fn_target[a] - contact_fn(a)
                    adm_n[a] = float(np.clip(adm_n[a] + err / KP * (DT / args.adm_tau), -0.005, 0.03))
                    base = step_t[a] - (step_t[a] - (ancs[a] + SPHERE_R * nouts[a])) @ n_in_a * n_in_a  # 법선 성분 제거
                    step_t[a] = base + adm_n[a] * n_in_a
                    tip_ctrl(a, step_t[a] + ((vff[a] * ((j_sub + 0.5) * DT) + KVr * vff[a] / KP) if a in vff else 0.0))
            if vff and not adm_on:
                for a in vff:                                   # 갱신 지연 상쇄: 스텝 중앙 시각의 접촉점 위치
                    tip_ctrl(a, step_t[a] + vff[a] * ((j_sub + 0.5) * DT) + KVr * vff[a] / KP)
            if freeze and ghost_pose is not None:
                d.qpos[jq["box_free"]:jq["box_free"] + 3] = ghost_pose[0]
                d.qpos[jq["box_free"] + 3:jq["box_free"] + 7] = ghost_pose[1]
                d.qvel[jv["box_free"]:jv["box_free"] + 6] = 0.0
            mujoco.mj_step(m, d)
        if args.debug_contacts and args.debug_contacts[0] <= t <= args.debug_contacts[1]:
            for a in range(NA):
                if a not in ancs:
                    continue
                rows_ = []
                for ci in range(d.ncon):
                    g1, g2 = int(d.contact.geom1[ci]), int(d.contact.geom2[ci])
                    if tip_gid_of.get(g1) == a or tip_gid_of.get(g2) == a:
                        other = g2 if tip_gid_of.get(g1) == a else g1
                        fw = contact_on_box(ci, a)
                        rows_.append(f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)} 깊이{1000 * d.contact.dist[ci]:+.1f}mm "
                                     f"법선{np.round(np.asarray(d.contact.frame[ci])[:3], 2)} 박스가받는힘{np.round(fw, 1)}")
                tp_ = tip_pos(a)
                print(f"  [접촉 t={t} {ARMS[a]}] 팁−앵커 {np.round(1000 * (tp_ - ancs[a] - SPHERE_R * nouts[a]), 1)}mm, 피드포워드 {np.round(ft_ff.get(a, np.zeros(3)), 1)}: "
                      + (" | ".join(rows_) if rows_ else "접촉 없음"), flush=True)
        if args.debug_steps and args.debug_steps[0] <= t <= args.debug_steps[1]:
            gh = d.xfrc_applied[BOX_BID]
            bp = d.qpos[jq["box_free"]:jq["box_free"] + 3]
            bv = d.qvel[jv["box_free"]:jv["box_free"] + 6]
            fs = {ARMS[a]: np.round(tip_force(a), 1) for a in range(NA) if a in ancs}
            ncon_box = sum(1 for ci in range(d.ncon)
                           if m.geom_bodyid[d.contact[ci].geom1] in (BOX_BID, BOX_BID + 1)
                           or m.geom_bodyid[d.contact[ci].geom2] in (BOX_BID, BOX_BID + 1))
            print(f"  [dbg t={t} ph={phase[t]} gg={gg:.2f}] box p={np.round(bp, 3)} v={np.round(bv[:3], 2)} "
                  f"w={np.round(bv[3:], 1)} 유령F={np.round(gh[:3], 1)} 유령T={np.round(gh[3:], 2)} "
                  f"접촉수={ncon_box} 단계={D['stage'][t]} 팁힘={fs}", flush=True)

        D["box_z"][t] = float(d.qpos[jq["box_free"] + 2])
        if q[2] < z_drop and fell is None:
            fell = t
        if demo_f[t] >= 0:
            D["pos_err"][t] = np.linalg.norm(tw + Rw @ cen - ref_pos[t])
            dq = abs(float(np.dot(q[3:7], ref_quat[t])))
            D["rot_err"][t] = 2 * np.arccos(min(1.0, dq))
            D["th_err"][t] = abs(thw - float(ref_theta[t]))
        if rend is not None and t % 4 == 0:
            rend.update_scene(d, camera="cam")
            frames.append(rend.render())
        if args.stop_at is not None and t >= args.stop_at:
            print(f"  [조기 종료 --stop-at {args.stop_at}]", flush=True)
            break
        if t % 300 == 0:
            r_s = f"r={D['r_used'][t]:.3f} 단계{D['stage'][t]}" if len(on) >= 2 else "접촉<2"
            print(f"  [t={t:4d} f={demo_f[t]:4d}] 팔{len(on)} {r_s} 위치오차 "
                  f"{100 * np.nan_to_num(D['pos_err'][t]):.1f}cm", flush=True)

    # ---- 판정 ----
    ok = demo_f >= 0
    pe, re_, te = D["pos_err"][ok], D["rot_err"][ok], D["th_err"][ok]
    dfv = demo_f[ok]
    win = (dfv >= 48) & (dfv < 668)               # 첫 파지 ~ 사람이 놓기 전
    st = D["stage"][ok]
    summary = dict(
        seq=args.seq, tag=tag, r_req=args.r_req, no_ff=bool(args.no_ff), plan_tag=args.plan_tag,
        fell_step=fell, pos_mean_cm=100 * float(np.nanmean(pe)), pos_max_cm=100 * float(np.nanmax(pe)),
        pos_last_cm=100 * float(pe[-1]), rot_mean_deg=float(np.degrees(np.nanmean(re_))),
        rot_max_deg=float(np.degrees(np.nanmax(re_))),
        pos_mean_win_cm=100 * float(np.nanmean(pe[win])), rot_mean_win_deg=float(np.degrees(np.nanmean(re_[win]))),
        th_max_deg=float(np.degrees(np.nanmax(te))),
        stage_frac={int(k): float((st == k).mean()) for k in (1, 2, 3)},
        r_used_min=float(np.nanmin(D["r_used"])), r_meas_min=float(np.nanmin(D["r_meas"])),
        rho_t_min=float(np.nanmin(rho_t)), align_fail=int(D["align_fail"].sum()))
    print(f"\n=== ρ판 재생 판정 ({args.seq} / {tag}) ===")
    print(f"낙하: {'없음' if fell is None else f'step {fell} 낙하!!'}")
    print(f"위치 오차: 평균 {summary['pos_mean_cm']:.1f}cm / 최대 {summary['pos_max_cm']:.1f}cm / 마지막 {summary['pos_last_cm']:.1f}cm"
          f"  (파지 창 48~668 평균 {summary['pos_mean_win_cm']:.1f}cm)")
    print(f"회전 오차: 평균 {summary['rot_mean_deg']:.1f}° / 최대 {summary['rot_max_deg']:.1f}°"
          f"  (파지 창 평균 {summary['rot_mean_win_deg']:.1f}°)")
    print(f"뚜껑 각 오차: 최대 {summary['th_max_deg']:.1f}°")
    s_ok = D["s_scale"][ok]
    s_ok = s_ok[np.isfinite(s_ok)]
    summary["stage_frac"][4] = float((st == 4).mean())
    summary["s_mean"] = float(np.mean(s_ok)) if len(s_ok) else 1.0
    summary["s_lt1_frac"] = float(np.mean(s_ok < 0.999)) if len(s_ok) else 0.0
    print(f"요구 다듬기: 배율 s 평균 {summary['s_mean']:.3f}, s<1 인 스텝 비율 {summary['s_lt1_frac']:.2f}, 4단계 비율 {summary['stage_frac'][4]:.2f}")
    if args.force_tip:
        sp = D["ft_spring_N"][ok]; fa = np.linalg.norm(D["f_alloc"][ok], axis=2)
        mk = np.isfinite(sp) & np.isfinite(fa) & (fa > 0.5)
        summary["ft_spring_median_N"] = float(np.nanmedian(sp[mk])); summary["ft_share_median"] = float(np.nanmedian(sp[mk] / fa[mk]))
        summary["ft_cap_frac"] = float(np.mean(sp[mk] >= args.ft_cap - 1e-6))
        print(f"힘 인가 팁: 스프링 힘 중앙값 {summary['ft_spring_median_N']:.2f}N, 피드포워드 대비 비율 중앙값 {summary['ft_share_median']:.2f}, "
              f"상한({args.ft_cap:g}N) 도달 비율 {summary['ft_cap_frac']:.2f}  (K_p {args.ft_kp:g}, K_d {args.ft_kd:.1f}, 관절 감쇠 {KVr:g})")
    print(f"배분 단계 비율: 1단계 {summary['stage_frac'][1]:.2f} / 2단계 {summary['stage_frac'][2]:.2f} / 3단계 {summary['stage_frac'][3]:.2f}"
          f"  확보 여유 r 최소 {summary['r_used_min']:.3f}  실측 힘 여유 최소 {summary['r_meas_min']:.3f}"
          f"  ρ(t) 최소 {summary['rho_t_min']:.3f}  표면 정렬 실패 {summary['align_fail']}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{args.subj}__{args.seq}__{tag}.npz"
    np.savez_compressed(p, demo_frame=demo_f, rho_t=rho_t, ghost_gain=ghost_g, summary=np.array([summary], dtype=object),
                        **D)
    print(f"진단 저장: {p}")
    if frames:
        import imageio
        vp = ROOT / "md_claude/dexmachina" / f"contact_rho_{args.seq}_{tag}.mp4"
        imageio.mimsave(vp, frames, fps=15)
        print(f"비디오: {vp}")
    return summary


if __name__ == "__main__":
    main()
