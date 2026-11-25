"""MuJoCo 마찰 단위시험 — Genesis 측정(servo_unit_test3~5)과 동일 항목.

  ① 누름 10N 유지한 채 1.5cm/s 등속 끌기 → 접선/법선 비 (쿨롱이면 μ=0.6 근처)
  ② 정하중: 법선 10N + 접선 4N 정력 (한도 5.8N 미만) → 표류 속도 (쿨롱이면 0)
설정 축: timestep 2ms/8.3ms × noslip 반복 0/5
비교 기준(Genesis 최선, substeps16+tc0.002): 끌기비 0.53, 표류 8.7mm/s
실행: ./venv/bin/python scratchpad/mj_friction_test.py
"""
import mujoco
import numpy as np

KP, KV, F_MAX = 1500.0, 40.0, 20.0
R = 0.015

XML = """
<mujoco>
  <compiler autolimits="true"/>
  <option timestep="{dt}" {extra} {noslip}/>
  <default><geom solref="{solref} 1"/></default>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <geom type="box" pos="0.3 0 0.1" size=".1 .1 .1" friction="0.6 0.005 0.0001"/>
    <body name="tip" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" damping="{kv}"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="{kv}"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="{kv}"/>
      <geom type="sphere" size="{r}" friction="0.6 0.005 0.0001" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="jx" kp="{kp}" forcerange="-{fm} {fm}"/>
    <position joint="jy" kp="{kp}" forcerange="-{fm} {fm}"/>
    <position joint="jz" kp="{kp}" forcerange="-{fm} {fm}"/>
  </actuator>
</mujoco>
"""


def run_case(dt, noslip_iter, impratio=1, elliptic=False, solref=0.02,
             tag=""):
    ns = f'noslip_iterations="{noslip_iter}"' if noslip_iter else ""
    extra = f'impratio="{impratio}"'
    if elliptic:
        extra += ' cone="elliptic"'
    m = mujoco.MjModel.from_xml_string(XML.format(
        dt=dt, noslip=ns, extra=extra, solref=solref, r=R, kp=KP, kv=KV,
        fm=F_MAX))
    d = mujoco.MjData(m)
    # 중력 보상: 팁 질량 0.1kg — z 목표에 보정 불필요 (PD가 감당, 0.98N)
    touch_z = 0.2 + R
    press_z = touch_z - 0.0067          # 법선 ≈ 10N

    def cmd(x, y, z):
        d.ctrl[:] = [x, y, z]

    def steps(n):
        for _ in range(n):
            mujoco.mj_step(m, d)

    d.qpos[:] = [0.3, 0.0, touch_z + 0.1]
    cmd(0.3, 0.0, touch_z + 0.1)
    steps(int(1.0 / dt))
    cmd(0.3, 0.0, press_z)
    steps(int(1.5 / dt))
    fz0 = abs(d.actuator_force[2])

    # ① 등속 끌기 1.5cm/s
    x, fx_log = 0.3, []
    n_drag = int(2.0 / dt)
    for i in range(n_drag):
        x += 0.015 * dt
        cmd(x, 0.0, press_z)
        mujoco.mj_step(m, d)
        if i >= n_drag // 2:
            fx_log.append(abs(d.actuator_force[0]))
    ratio = float(np.mean(fx_log)) / fz0 if fz0 > 1 else float("nan")

    # ② 정하중 4N: x 는 힘으로 밀고(외력), 위치 서보는 y,z 만
    cmd(x, 0.0, press_z)
    steps(int(0.5 / dt))
    m.actuator_gainprm[0, 0] = 0.0      # x 위치 서보 무력화
    m.actuator_biasprm[0, :] = 0.0
    x0 = float(d.qpos[0])
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip")
    for _ in range(int(2.0 / dt)):
        d.xfrc_applied[body_id, 0] = 4.0
        mujoco.mj_step(m, d)
    drift = (float(d.qpos[0]) - x0) / 2.0 * 1000
    print(f"[MuJoCo {tag}] 법선 {fz0:.1f}N | "
          f"끌기 비 {ratio:.3f} | 정하중 4N 표류 {drift:+.2f} mm/s")


run_case(0.002, 0, tag="기본 (dt2ms)")
run_case(0.002, 5, tag="기본+noslip5")
run_case(0.002, 0, elliptic=True, impratio=10, tag="타원원뿔 impratio10")
run_case(0.002, 0, elliptic=True, impratio=10, solref=0.004,
         tag="타원 impratio10 solref4ms")
run_case(0.002, 5, elliptic=True, impratio=10, solref=0.004,
         tag="타원 impratio10 solref4ms +noslip5")
