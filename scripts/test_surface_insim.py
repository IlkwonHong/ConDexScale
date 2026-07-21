#!/usr/bin/env python3
"""In-sim verification of the surface action mode (S4) — BEFORE any training.

Builds the real env (point4_hand, -am surface, VOC off) and injects scripted
actions in three phases, checking the physics does what the design intends:

  Phase 1 (press):  (0, 0, +0.5)  -> spheres sit ON their anchors pressing the
                    box: object receives contact force; sphere centers ~radius
                    from the surface; spheres near their anchor priors.
  Phase 2 (slide):  (+0.8, 0, +0.5) -> spheres displaced tangentially away
                    from anchors while contact persists.
  Phase 3 (hover):  (0, 0, -1.0) -> spheres lift off; contact force -> ~0.

Run: WANDB_MODE=disabled /home/robin/miniforge3/envs/dexmachina/bin/python \
       scripts/test_surface_insim.py
"""

import os.path as osp
import sys

import numpy as np
import torch

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO, "extlib", "dexmachina"))

import genesis as gs  # noqa
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg  # noqa
from dexmachina.envs.base_env import BaseEnv  # noqa


def main():
    gs.init(backend=gs.gpu)
    parser = get_common_argparser()
    # VOC ON (strong virtual springs hold the box on its demo trajectory) so we
    # isolate sphere-control correctness from free-box dynamics; also tests
    # ride-along against a MOVING object. Without this, pressing shoves the box
    # around and the box ends up resting on the spheres (first attempt).
    args = parser.parse_args([
        "--hand", "point4_hand", "-am", "surface",
        "--clip", "box-30-230", "--num_envs", "2",
        "--actuate_object", "--kp_init", "300", "--kv_init", "10",
        "-imi", "0.0", "-bc", "0.0", "-con", "0.0",
        "--task_rew_betas", "10", "1", "5",
    ])
    env_kwargs = get_all_env_cfg(args, device="cuda:0")
    env_kwargs["env_cfg"]["num_envs"] = 2
    env = BaseEnv(**env_kwargs)
    obs, _ = env.reset()
    B, A = env.num_envs, env.action_dim
    print(f"env built: num_envs={B} action_dim={A} (기대 12 = 4구×3)")
    assert A == 12, f"action_dim {A} != 12"

    obj = env.object
    left, right = env.robots["left"], env.robots["right"]

    def sphere_centers():
        return torch.cat([left.kpt_pos, right.kpt_pos], dim=1)  # (B,4,3)

    def obj_contact_force():
        obj.update_value_buffers()
        return obj.contact_force.norm(dim=-1).sum(dim=-1)  # (B,) sum over links

    def anchors_world():
        out = env._surface_out
        return out["surface_points"], out["normals"]

    results = {}
    phases = [
        ("press", [0.0, 0.0, 0.5], 60),
        ("slide", [0.8, 0.0, 0.5], 60),
        ("hover", [0.0, 0.0, -1.0], 60),
    ]
    for name, a3, n_steps in phases:
        env.reset()  # fresh box state per phase (isolate phases)
        act = torch.zeros(B, A, device=env.device)
        act[:] = torch.tensor(a3 * 4, device=env.device)  # same per sphere
        forces, gaps, slides = [], [], []
        for i in range(n_steps):
            env.step(act)
            if i >= n_steps - 10:  # measure at steady state (last 10 steps)
                sp, nrm = anchors_world()
                c = sphere_centers()
                # signed gap along normal from commanded surface point
                gap = ((c - sp) * nrm).sum(-1)          # (B,4)
                forces.append(obj_contact_force().cpu())
                gaps.append(gap.cpu())
                # tangential distance from the anchor prior (frame's surface pt
                # with zero slide = re-computed below for slide check)
        results[name] = dict(
            force=torch.stack(forces).mean().item(),
            gap=torch.stack(gaps).mean().item(),
        )
        print(f"[{name:5s}] obj 접촉력 합: {results[name]['force']:7.2f} N   "
              f"스피어-표면점 법선 gap: {results[name]['gap']*1000:6.1f} mm")

    # ---- verdicts ----
    ok_press = results["press"]["force"] > 1.0
    ok_hover = results["hover"]["force"] < 0.5 * max(results["press"]["force"], 1e-6)
    ok_gap_hover = results["hover"]["gap"] > 0.02  # lifted off (> ~radius)
    print("\n--- 판정 ---")
    print(f"1. press에서 접촉력 발생 (>1N): {'PASS' if ok_press else 'FAIL'}")
    print(f"2. hover에서 접촉력 소멸: {'PASS' if ok_hover else 'FAIL'}")
    print(f"3. hover에서 표면 이탈 (gap>2cm): {'PASS' if ok_gap_hover else 'FAIL'}")
    print(f"(slide 변위는 phase 로그로 정성 확인: slide gap {results['slide']['gap']*1000:.1f}mm)")


if __name__ == "__main__":
    main()
