"""모서리 미끄러짐 경계 측정 — 현 물리(MuJoCo 파지 조율)에서.

팁을 윗면의 모서리에서 d 만큼 떨어진 지점에 10N으로 누른 채, 모서리 방향으로
접선 힘을 0→8N 램프. 평면 한복판이라면 마찰 한계 0.6×10=6N까지 버텨야 정상.
모서리 근처에서 얼마나 일찍 굴러떨어지는지 → 벌점 곡선(바닥·스케일)의 근거.
실행: ./venv/bin/python scratchpad/mj_edge_slip_test.py
"""
import mujoco
import numpy as np

KP, KV, F_MAX = 1500.0, 40.0, 30.0
R = 0.015
XML = """
<mujoco>
  <compiler autolimits="true"/>
  <option timestep="0.001667" impratio="10" cone="elliptic" noslip_iterations="5"/>
  <default><geom solref="0.004 1"/></default>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <geom type="box" pos="0.3 0 0.1" size=".1 .1 .1" friction="0.6 0.005 0.0001"/>
    <body name="tip" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0" damping="40"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="40"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="40"/>
      <geom type="sphere" size="0.015" friction="0.6 0.005 0.0001" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="jx" kp="1500" forcerange="-30 30"/>
    <position joint="jy" kp="1500" forcerange="-30 30"/>
    <position joint="jz" kp="1500" forcerange="-30 30"/>
  </actuator>
</mujoco>"""

EDGE_X = 0.4                     # 윗면의 +x 모서리
for d_mm in (0, 3, 5, 8, 10, 15, 25):
    m = mujoco.MjModel.from_xml_string(XML)
    dd = mujoco.MjData(m)
    x0 = EDGE_X - d_mm / 1000.0
    touch = np.array([x0, 0.0, 0.2 + R])
    dd.qpos[:] = touch + [0, 0, 0.1]
    dd.ctrl[:] = dd.qpos[:3]
    for _ in range(300):
        mujoco.mj_step(m, dd)
    dd.ctrl[:] = [x0, 0.0, 0.2 + R - 0.0067]      # 10N 누름
    for _ in range(600):
        mujoco.mj_step(m, dd)
    # 접선 램프: x 는 힘 제어(서보 무력화), z 는 누름 유지
    m.actuator_gainprm[0, 0] = 0.0
    m.actuator_biasprm[0, :] = 0.0
    esc = None
    for i in range(1800):                          # 3초간 0→8N
        fx = 8.0 * (i + 1) / 1800
        dd.xfrc_applied[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip"),
                        0] = fx
        mujoco.mj_step(m, dd)
        if dd.qpos[0] > EDGE_X + R or dd.qpos[2] < 0.2 - 0.01:
            esc = fx
            break
    print(f"모서리 거리 {d_mm:2d}mm: " +
          (f"이탈 @ 접선 {esc:.1f}N" if esc else "8N까지 버팀"), flush=True)
