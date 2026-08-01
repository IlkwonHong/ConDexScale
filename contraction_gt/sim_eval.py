"""P5: MuJoCo 시뮬 검증 — 구형 팁 N개로 물체를 쥐고 외란 wrench 를 램프해
실측 내하중 α_sim 을 얻는다 (계획서 D1/D10).

프로토콜 (Dexonomy eval_fc 의 구형 팁·capacity 변형):
  1. 각 팁 = 법선 방향 슬라이드 1자유도 강체(측방향은 무한강성 = 팔이 잡아줌),
     모터로 명령 누름힘 F_cmd 인가. 물체는 자유 강체, 중력 0 (D5: 질량 무관
     capacity 라벨 — 외란은 xfrc 로 직접 가한다).
  2. 정착 후 기준 자세 기록.
  3. 외란 wrench 를 α 그리드로 단조 증가시키며 각 단계를 유지, 자세 이탈
     (이동 > trans_tol·r_obj 또는 회전 > rot_tol)이 나면 직전 단계가 α_sim.
  4. 외란 정규화는 metric.direction_set 과 동일: 힘 = α·F_cmd·d̂,
     토크 = α·F_cmd·r_obj·d̂ → α_sim 을 LP 의 α 와 직접 비교할 수 있다.

마찰은 LP 와 같은 값(μ_t, μ_tor)을 팁·물체 geom 에 넣는다 (D4 원칙,
MuJoCo condim=4 의 두 번째 마찰계수 = 비틀림, 길이 단위).
"""

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class SimConfig:
    f_cmd: float = 5.0            # 팁당 접촉힘 상한 [N] — LP 의 "앵커당 압력 ≤ 1"에 대응
    kp: float = 2e4               # 위치 서보 강성. LP 의미론(접촉점 고정 + 힘 상한) 재현:
                                  # 순수 힘 제어는 압착 축 위치 강성이 0이라 전체가 표류함
    preload_frac: float = 3.0     # 서보 목표를 힘 상한 너머로 → 예압이 상한에 포화.
                                  # (접촉 침투 스프링과의 균형 때문에 여유 있게 줘야 하며,
                                  #  실제 힘은 forcerange 가 f_cmd 로 절삭한다)
                                  # LP 는 "필요 시 상한까지 조이는 최적 배분"을 가정하므로
                                  # (Dexonomy contact-aware control 에 대응) 포화 예압이 등가.
                                  # 포화여도 위치 서보라 힘은 아래 방향으로 즉시 재분배됨.
    joint_damping: float = 60.0
    tip_radius: float = 0.015     # [m]
    mu_t: float = 0.6
    mu_tor: float = 0.006         # [m] — search 의 mu_tor_from_pad 값과 일치시킬 것
    active: bool = False          # 능동 등급: 수평 2자유도 + 외란별 LP 힘 명령
    sensor_std: float = 1.5       # 힘 명령 오차 [N] — Franka 관절토크 기반 추정 수준
    lat_stiffness: float = 2000.0 # 수평 임피던스 스프링 (팔의 위치 강성 모델)
    lat_damping: float = 20.0
    timestep: float = 2e-3
    settle_steps: int = 400
    hold_steps: int = 250
    alpha_step: float = 0.1       # 램프 해상도
    alpha_max: float = 4.0
    trans_tol_rel: float = 0.10   # 실패 판정: 이동 > 0.10 · r_obj
    rot_tol_deg: float = 20.0
    start_gap: float = 0.002      # 팁 초기 표면 간격 [m]
    density: float = 200.0        # 물체 밀도 (중력 0 이라 관성 스케일만 결정)


def _object_xml(object_spec):
    """object_spec -> (asset_xml, geom_xml). coacd 는 convex 조각별 geom 으로
    구성한다 (MuJoCo 는 오목 mesh 단일 geom 을 convex hull 로 뭉개므로)."""
    kind = object_spec[0]
    if kind == "sphere":
        return "", f'<geom name="obj_g" type="sphere" size="{object_spec[1]}"/>'
    if kind == "box":
        s = object_spec[1]
        return "", f'<geom name="obj_g" type="box" size="{s[0]} {s[1]} {s[2]}"/>'
    if kind == "mesh":
        sc = object_spec[2]
        return (f'<mesh name="obj_mesh" file="{object_spec[1]}" scale="{sc} {sc} {sc}"/>',
                '<geom name="obj_g" type="mesh" mesh="obj_mesh"/>')
    if kind == "coacd":
        from pathlib import Path
        sc = object_spec[2]
        files = sorted(Path(object_spec[1]).glob("*.obj"))
        assets = "".join(
            f'<mesh name="objp{k}" file="{f}" scale="{sc} {sc} {sc}"/>'
            for k, f in enumerate(files))
        geoms = "".join(
            f'<geom type="mesh" mesh="objp{k}"/>' for k in range(len(files)))
        return assets, geoms
    raise ValueError(f"unknown object spec: {kind}")


def build_scene_xml(object_spec, anchor_pos, anchor_normal, cfg: SimConfig):
    """씬 MJCF 생성. anchor_pos/normal: [N,3] (물체 프레임 = 월드 초기 프레임)."""
    anchor_pos = np.asarray(anchor_pos, dtype=np.float64)
    anchor_normal = np.asarray(anchor_normal, dtype=np.float64)
    friction = f"{cfg.mu_t} {cfg.mu_tor} 0.0001"
    mesh_asset, obj_geoms = _object_xml(object_spec)

    tips, actuators = [], []
    for i, (p, n) in enumerate(zip(anchor_pos, anchor_normal)):
        n = n / np.linalg.norm(n)          # n = 안쪽 법선 (누름 방향, 파이프라인 규약)
        start = p - n * (cfg.tip_radius + cfg.start_gap)   # 팁은 표면 바깥에서 시작
        # 몸체 z축 = n (누름 방향). 슬라이드 joint 축 = +z → ctrl 양수가 누름.
        lat = ""
        if cfg.active:
            # 수평 2자유도: 임피던스 스프링(팔 위치 강성) + 힘 모터(LP 접선힘 명령)
            lat = (f'<joint name="latx{i}" type="slide" axis="1 0 0" '
                   f'stiffness="{cfg.lat_stiffness}" damping="{cfg.lat_damping}"/>'
                   f'<joint name="laty{i}" type="slide" axis="0 1 0" '
                   f'stiffness="{cfg.lat_stiffness}" damping="{cfg.lat_damping}"/>')
        tips.append(
            f'<body name="tip{i}" pos="{start[0]} {start[1]} {start[2]}" '
            f'zaxis="{n[0]} {n[1]} {n[2]}">'
            f'<joint name="press{i}" type="slide" axis="0 0 1" damping="{cfg.joint_damping}"/>'
            f'{lat}'
            f'<geom name="tip_g{i}" type="sphere" size="{cfg.tip_radius}" '
            f'friction="{friction}" condim="4" mass="0.05"/>'
            f'</body>')
        # 위치 서보 + 접촉힘 상한: 위치는 강성으로 잡고, 낼 수 있는 힘은 0~f_cmd
        actuators.append(
            f'<position joint="press{i}" kp="{cfg.kp}" forcerange="0 {cfg.f_cmd}"/>')
    if cfg.active:
        for i in range(len(anchor_pos)):
            actuators.append(f'<motor joint="latx{i}" gear="1"/>')
            actuators.append(f'<motor joint="laty{i}" gear="1"/>')

    return f"""
<mujoco>
  <compiler autolimits="true"/>
  <option gravity="0 0 0" timestep="{cfg.timestep}" integrator="implicitfast" impratio="10"/>
  <asset>{mesh_asset}</asset>
  <default><geom friction="{friction}" condim="4" density="{cfg.density}"
    solref="0.004 1" solimp="0.9 0.99 0.0005"/></default>
  <worldbody>
    <body name="object">
      <freejoint/>
      {obj_geoms}
    </body>
    {''.join(tips)}
  </worldbody>
  <actuator>{''.join(actuators)}</actuator>
</mujoco>"""


def _pose_deviation(data, obj_qpos_ref):
    dp = float(np.linalg.norm(data.qpos[:3] - obj_qpos_ref[:3]))
    q0, q1 = obj_qpos_ref[3:7], data.qpos[3:7]
    dot = float(np.clip(abs(q0 @ q1), -1.0, 1.0))
    return dp, np.degrees(2.0 * np.arccos(dot))


def measure_capacity(object_spec, anchor_pos, anchor_normal, wrench_dir, r_obj,
                     cfg: SimConfig = None):
    """외란 방향 wrench_dir(6, metric 규약)에 대한 실측 내하중 α_sim.

    반환: (alpha_sim, info dict). 정착 실패(팁이 못 잡음)면 alpha_sim = 0.
    """
    cfg = cfg or SimConfig()
    xml = build_scene_xml(object_spec, anchor_pos, anchor_normal, cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # 서보 목표 = 접촉 지점(start_gap 만큼 전진) + 예압분 침투.
    # 팁 시작점 구성상 표면까지의 이동량이 정확히 start_gap 이다.
    data.ctrl[:] = cfg.start_gap + cfg.preload_frac * cfg.f_cmd / cfg.kp
    for _ in range(cfg.settle_steps):
        mujoco.mj_step(model, data)
    ref = data.qpos[:7].copy()

    # 정착 판정: 앵커 배치가 무외란 평형 자체를 못 만들면(예: 같은 면 쌍)
    # 물체가 정착 중에 도망간다 → 내하중 0, 즉시 반환.
    drift = float(np.linalg.norm(ref[:3]))
    if drift > cfg.trans_tol_rel * r_obj or data.ncon == 0:
        return 0.0, {"settled": False, "settle_drift": drift}

    w = np.asarray(wrench_dir, dtype=np.float64)
    force_part, torque_part = w[:3], w[3:]
    sustained = 0.0
    levels = np.arange(cfg.alpha_step, cfg.alpha_max + 1e-9, cfg.alpha_step)
    for a in levels:
        data.xfrc_applied[obj_body, :3] = a * cfg.f_cmd * force_part
        data.xfrc_applied[obj_body, 3:] = a * cfg.f_cmd * r_obj * torque_part
        for _ in range(cfg.hold_steps):
            mujoco.mj_step(model, data)
        dp, dr = _pose_deviation(data, ref)
        if dp > cfg.trans_tol_rel * r_obj or dr > cfg.rot_tol_deg:
            break
        sustained = float(a)
    return sustained, {"settled": True, "levels_tested": len(levels),
                       "final_dev": _pose_deviation(data, ref)}


def measure_capacity_multi(object_spec, anchor_pos, anchor_normal, dirs, levels,
                           r_obj, cfg: SimConfig = None, preload_forces=None,
                           force_fn=None, seed=0):
    """여러 외란 방향을 한 모델로 측정 (모델 빌드·메쉬 로드를 방향 간 재사용).

    dirs: [D, 6] (metric 규약), levels: [D, L] 방향별 테스트 α 그리드 (오름차순).
    preload_forces: [N] 팁별 목표 예압 [N 단위]. None 이면 전 팁 상한 포화(구형
    대칭 배치용). 비대칭 다중 앵커는 균등 포화가 물체를 밀어버리므로(파일럿
    진단) LP 평형 배분을 넘겨야 한다. 목표 힘은 접촉 강성을 몰라도 되도록
    정착 중 피드백(측정 힘과의 차이로 서보 목표를 보정)으로 맞춘다.
    반환: (alpha_sim [D], settled bool, 예압 달성도).
    """
    cfg = cfg or SimConfig()
    xml = build_scene_xml(object_spec, anchor_pos, anchor_normal, cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    n_tips = len(anchor_pos)
    nrm_u = np.asarray(anchor_normal, dtype=np.float64)
    nrm_u = nrm_u / np.linalg.norm(nrm_u, axis=1, keepdims=True)
    active = bool(cfg.active and force_fn is not None)
    rng = np.random.default_rng(seed)

    lat_settle = np.zeros(2 * n_tips)
    tan_settle = None
    if preload_forces is None:
        press_ctrl = np.full(n_tips, cfg.start_gap + cfg.preload_frac * cfg.f_cmd / cfg.kp)
        target_f = None
    else:
        pf = np.asarray(preload_forces, dtype=np.float64)
        if pf.ndim == 2:                       # [N,3] 힘 벡터 (능동 모드: 접선 포함)
            target_f = np.clip((pf * nrm_u).sum(axis=1), 0.02 * cfg.f_cmd, cfg.f_cmd)
            tan_settle = pf - target_f[:, None] * nrm_u
        else:
            target_f = np.clip(pf, 0.02 * cfg.f_cmd, cfg.f_cmd)
        press_ctrl = cfg.start_gap + target_f / cfg.kp   # 초기 추정 (접촉 강성 미반영)

    # 팁 몸체 좌표축 (수평 모터 축의 월드 방향) — 슬라이드 관절뿐이라 불변
    mujoco.mj_forward(model, data)
    tip_axes = []
    for i in range(n_tips):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"tip{i}")
        R = data.xmat[bid].reshape(3, 3)
        tip_axes.append((R[:, 0].copy(), R[:, 1].copy()))
    if active and tan_settle is not None:
        for i, (ax, ay) in enumerate(tip_axes):
            lat_settle[2 * i] = float(tan_settle[i] @ ax)
            lat_settle[2 * i + 1] = float(tan_settle[i] @ ay)

    dirs = np.asarray(dirs, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    out = np.zeros(len(dirs))
    settled = True
    preload_frac_meas = 0.0            # 정착 직후 팁 힘 / 상한 (메쉬 불일치 진단용)

    def settle():
        mujoco.mj_resetData(model, data)
        data.ctrl[:n_tips] = press_ctrl
        if model.nu > n_tips:
            data.ctrl[n_tips:] = lat_settle
        steps = int(cfg.settle_steps * cfg.timestep / model.opt.timestep)
        for _ in range(steps):
            mujoco.mj_step(model, data)

    def _warn_count():
        return int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    def trial(d, a):
        """독립 시행: 리셋 → 정착 → 외란 a 인가·유지 → 버텼는지.

        단조 램프는 레벨마다 남은 크리프가 누적되어 레벨 수에 비례하는
        과소평가 편향을 만든다(파일럿 v2 로 확인) — 시행마다 리셋한다.
        수치 폭발(가속도 경고의 증가분으로 감지 — 카운터 리셋 여부에 무관)은
        실패와 구분해 시간 간격을 절반으로 줄여 재시도하고, 두 번 줄여도
        터지면 '측정 불가'(unstable)를 반환한다.
        """
        dt0 = model.opt.timestep
        for attempt in range(3):
            w_pre = _warn_count()
            settle()
            ref = data.qpos[:7].copy()
            if _warn_count() > w_pre:
                model.opt.timestep *= 0.5
                continue
            if float(np.linalg.norm(ref[:3])) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
                model.opt.timestep = dt0
                return None            # 정착 실패 (물리적)
            if active:
                # 외란별 LP 힘 배분을 명령 (센서 오차 σ 를 힘 벡터에 주입)
                fv = force_fn(d, a) + rng.normal(0.0, cfg.sensor_std, (n_tips, 3))
                fn = np.clip((fv * nrm_u).sum(axis=1), 0.0, cfg.f_cmd)
                ft = fv - fn[:, None] * nrm_u
                data.ctrl[:n_tips] = np.clip(
                    press_cal + (fn - F_cal) / k_eff,
                    0.0, cfg.start_gap + 5.0 * cfg.f_cmd / cfg.kp)
                for i, (ax, ay) in enumerate(tip_axes):
                    data.ctrl[n_tips + 2 * i] = float(ft[i] @ ax)
                    data.ctrl[n_tips + 2 * i + 1] = float(ft[i] @ ay)
            data.xfrc_applied[obj_body, :3] = a * cfg.f_cmd * dirs[d, :3]
            data.xfrc_applied[obj_body, 3:] = a * cfg.f_cmd * r_obj * dirs[d, 3:]
            w_hold = _warn_count()
            steps = int(cfg.hold_steps * dt0 / model.opt.timestep)
            for _ in range(steps):
                mujoco.mj_step(model, data)
            if _warn_count() > w_hold:
                model.opt.timestep *= 0.5
                continue
            dp, dr = _pose_deviation(data, ref)
            model.opt.timestep = dt0
            # 파이썬 bool 로 강제 — numpy bool 은 `is True` 판정을 통과하지 못함
            return bool(dp <= cfg.trans_tol_rel * r_obj and dr <= cfg.rot_tol_deg)
        model.opt.timestep = dt0
        return "unstable"

    # 목표 예압 피드백 보정 (접촉 강성을 몰라도 목표 힘에 수렴)
    # + 유효 강성 k_eff 추정 (능동 모드의 시행별 원샷 힘 명령에 사용)
    k_eff = np.full(n_tips, cfg.kp)
    if target_f is not None:
        prev_c = prev_F = None
        for _ in range(4):
            settle()
            F = data.actuator_force[:n_tips].copy()
            if prev_c is not None:
                dc = press_ctrl - prev_c
                mask = np.abs(dc) > 1e-8
                k_est = (F - prev_F) / np.where(mask, dc, 1.0)
                k_eff = np.clip(np.where(mask & (k_est > 0), k_est, k_eff),
                                500.0, cfg.kp)
            prev_c, prev_F = press_ctrl.copy(), F
            press_ctrl = press_ctrl + 0.8 * (target_f - F) / cfg.kp
        achieved = data.actuator_force[:n_tips] / np.maximum(target_f, 1e-9)
    # 정착 가능성 + 예압 계측 (1회)
    settle()
    if target_f is None:
        achieved = data.actuator_force[:n_tips] / cfg.f_cmd
    F_cal = data.actuator_force[:n_tips].copy()
    press_cal = press_ctrl.copy()
    preload_frac_meas = float(np.mean(achieved))
    if float(np.linalg.norm(data.qpos[:3])) > cfg.trans_tol_rel * r_obj or data.ncon == 0:
        return out, False, preload_frac_meas, 0

    n_unstable = 0
    for d in range(len(dirs)):
        lo, hi = 0.0, float(levels[d].max())
        r = trial(d, hi)
        if r is None:
            settled = False
            break
        if r == "unstable":
            out[d] = np.nan            # 측정 불가 — 실패로 계산하지 않음
            n_unstable += 1
            continue
        if r is True:
            out[d] = hi                # 상한까지 버팀 (측정 상한 도달)
            continue
        bad = False
        for _ in range(5):             # 이분탐색: 해상도 hi/32
            mid = 0.5 * (lo + hi)
            r = trial(d, mid)
            if r is None or r == "unstable":
                bad = True
                break
            if r is True:
                lo = mid
            else:
                hi = mid
        if bad:
            out[d] = np.nan
            n_unstable += 1
        else:
            out[d] = lo
    return out, settled, preload_frac_meas, n_unstable
