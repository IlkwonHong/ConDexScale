#!/usr/bin/env python3
"""Generate minimal URDF files for "point-contact arm" robots for DexMachina.

One robot entity = k floating spheres; each sphere = one arm's end-effector
implementing a single contact force. Each sphere hangs on its own 6-DoF
chain (3 prismatic tx/ty/tz + 3 revolute roll/pitch/yaw) so DexMachina's
PD position control can move it freely, exactly like their floating-hand
wrists.

Naming follows DexMachina conventions (hand_proc/add_wrist_dof.py and
assets/allegro_hand/*_6dof.urdf) so their code auto-detects wrist DoFs:
  - joints contain the substring 'forearm' (robot.py wrist_dof_idxs)
  - tx/ty/tz/roll/pitch/yaw substrings drive init-limit margins
    (demo_data.get_joint_init_limits)
  - chain: {P}{i}_forearm_tx_link -(tx_link_joint)-> ty_link -> ... ->
    yaw_link -(yaw_link_joint)-> {P}{i}_tip_link (collision sphere)

Axis conventions copied from add_wrist_dof.py _FOREARM_DOFS:
  tx (1,0,0)  ty (0,1,0)  tz (0,0,1)  roll (0,0,1)  pitch (1,0,0)  yaw (0,-1,0)

Usage:
  python scripts/gen_point_urdf.py --out-dir extlib/dexmachina/dexmachina/assets/point_arm
"""

import argparse
import os

DOFS = [
    ("tx", "prismatic", "1 0 0", -5.0, 5.0),
    ("ty", "prismatic", "0 1 0", -5.0, 5.0),
    ("tz", "prismatic", "0 0 1", -5.0, 5.0),
    ("roll", "revolute", "0 0 1", -6.2, 6.2),
    ("pitch", "revolute", "1 0 0", -6.2, 6.2),
    ("yaw", "revolute", "0 -1 0", -6.2, 6.2),
]

DUMMY_LINK = """  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
"""

JOINT = """  <joint name="{name}" type="{jtype}">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <axis xyz="{axis}"/>
    <limit lower="{lo}" upper="{hi}" effort="100" velocity="5"/>
  </joint>
"""

FIXED_JOINT = """  <joint name="{name}" type="fixed">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""

TIP_LINK = """  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <geometry><sphere radius="{radius}"/></geometry>
    </visual>
    <collision>
      <geometry><sphere radius="{radius}"/></geometry>
    </collision>
  </link>
"""

ROOT_LINK = """  <link name="root_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
"""


def sphere_chain(prefix, idx, radius, tip_mass):
    """One sphere's 6-DoF chain. prefix e.g. 'L', idx 1-based."""
    p = f"{prefix}{idx}"
    parts = []
    # fixed joint from shared root to this chain's first link
    first = f"{p}_forearm_tx_link"
    parts.append(FIXED_JOINT.format(name=f"{p}_root_joint", parent="root_link", child=first))

    link_names = [f"{p}_forearm_{d[0]}_link" for d in DOFS] + [f"{p}_tip_link"]
    for i, (dof, jtype, axis, lo, hi) in enumerate(DOFS):
        parent, child = link_names[i], link_names[i + 1]
        parts.append(DUMMY_LINK.format(name=parent))
        parts.append(JOINT.format(
            name=f"{parent}_joint", jtype=jtype, parent=parent, child=child,
            axis=axis, lo=lo, hi=hi))
    parts.append(TIP_LINK.format(name=f"{p}_tip_link", mass=tip_mass, radius=radius))
    return "".join(parts)


def build_urdf(side, k, radius, tip_mass):
    prefix = "L" if side == "left" else "R"
    body = ROOT_LINK
    for i in range(1, k + 1):
        body += sphere_chain(prefix, i, radius, tip_mass)
    return f'<?xml version="1.0"?>\n<robot name="point{k}_{side}">\n{body}</robot>\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "extlib", "dexmachina", "dexmachina", "assets", "point_arm"))
    ap.add_argument("--radius", type=float, default=0.015)
    ap.add_argument("--tip-mass", type=float, default=0.2)
    ap.add_argument("--spheres", type=int, nargs="+", default=[1, 2],
                    help="sphere counts per entity to generate")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for k in args.spheres:
        for side in ["left", "right"]:
            path = os.path.join(args.out_dir, f"point{k}_{side}.urdf")
            with open(path, "w") as f:
                f.write(build_urdf(side, k, args.radius, args.tip_mass))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
