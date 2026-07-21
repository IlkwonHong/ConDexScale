"""Genesis 배치 평가기 — 폐루프 임피던스 롤아웃으로 θ 후보 채점 (dexscale_mvp.md §2).

Stage 2(TO 롤아웃)와 Stage 3(배포)가 이 코드를 공유한다 (train = deploy).
씬: plane + cardboard(고정 받침) + box URDF(자유 base + 힌지) + point2 로봇 2개.
스피어 순서 [L1, L2, R1, R2] = reference.py와 동일.

임피던스 = Genesis 내장 PD (scripts/test_genesis_impedance_smoke.py 6/6 검증):
  F = kp(x_ref − x) − kv·ẋ,  |F| ≤ F_MAX (set_dofs_force_range)
fork 지뢰 (설계문서 §3): batch_dofs_info/batch_links_info 필수,
질량 DR은 set_links_inertial_mass(절대값), set_mass_shift 금지.
"""
import os.path as osp

import numpy as np
import torch

import genesis as gs

from dexscale.reference import SPHERE_R

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
ASSETS = osp.join(REPO, "extlib", "dexmachina", "dexmachina", "assets")

CARD_POS, CARD_SIZE = (0.0, -0.08, 0.90), (0.2, 0.2, 0.1)   # base_env.py와 동일
KP, KV, F_MAX = 1500.0, 40.0, 20.0        # 그릴링 Q6: 20N/팔
REV_KP, REV_KV, REV_FMAX = 5.0, 0.5, 10.0
# 비용 가중 — 선형 오차 (2026-07-15 수정: 제곱(m²)은 낙하 패널티 대비 너무 작아
# "포기하고 안 건드리기" exploit 유발. M3-A에서 관측)
# W_ARTI 0.5→1.5 (2026-07-18): E5~E7 연속으로 "운반 중 뚜껑 방치→벌컥→낙하" 패턴.
# 이 시점 이후 비용 숫자는 이전 실험과 직접 비교 불가 (물리 지표로만 비교).
W_POS, W_ROT, W_ARTI = 1.0, 0.3, 1.5      # ‖Δp‖[m] / 회전각[rad] / |Δθ|[rad]
DROP_BELOW, C_DROP = 0.15, 1.0            # 실제 낙하 = 데모 최저 z보다 15cm 아래

_gs_ready = False


def _ensure_gs():
    global _gs_ready
    if not _gs_ready:
        gs.init(backend=gs.gpu, logging_level="warning")
        _gs_ready = True


def _quat_to_R(q):
    """(N,4) wxyz → (N,3,3) 회전행렬."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _Rz_neg(a):
    """(N,) 관절각 → Rz(−a) (N,3,3). pose_apply의 top part 규약."""
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(a.shape[0], 3, 3, device=a.device, dtype=a.dtype)
    R[:, 0, 0], R[:, 0, 1] = c, s
    R[:, 1, 0], R[:, 1, 1] = -s, c
    R[:, 2, 2] = 1.0
    return R


class DexScaleEvaluator:
    def __init__(self, ref, n_dr=4, chunk=8, dt=1.0 / 60.0, substeps=2,
                 show_viewer=False, with_camera=False, cam_res=(640, 480)):
        """n_envs = chunk(동시 후보 수) × n_dr(물성 샘플 수). env 배치는 후보-major."""
        _ensure_gs()
        self.ref, self.n_dr, self.chunk = ref, n_dr, chunk
        self.n_envs = n_dr * chunk

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps,
                                              gravity=(0, 0, -9.81)),
            rigid_options=gs.options.RigidOptions(
                dt=dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=100,
                batch_dofs_info=True,
                batch_links_info=True,
            ),
            show_viewer=show_viewer,
        )
        self.plane = scene.add_entity(gs.morphs.Plane())
        self.card = scene.add_entity(
            gs.morphs.Box(pos=CARD_POS, size=CARD_SIZE, fixed=True))
        q0 = ref.demo_qpos0()
        self.obj = scene.add_entity(
            gs.morphs.URDF(
                file=osp.join(ASSETS, "arctic", "box", "decomp", "box_decomp.urdf"),
                pos=tuple(q0[:3]), quat=tuple(q0[3:7]), fixed=False,
                convexify=True, recompute_inertia=False,
            ))
        self.robots = []
        for side in ("left", "right"):
            r = scene.add_entity(
                gs.morphs.URDF(
                    file=osp.join(ASSETS, "point_arm", f"point2_{side}.urdf"),
                    pos=(0.0, 0.0, 0.0), fixed=True, merge_fixed_links=False,
                    convexify=True, recompute_inertia=True,
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
            self.robots.append(r)
        self.camera = None
        if with_camera:   # build 전에 추가해야 함 (base_env._add_camera와 동일 규약)
            self.camera = scene.add_camera(
                pos=(0.0, -1.5, 1.35), lookat=(0.0, -0.08, 0.95),
                res=cam_res, fov=30, GUI=False)
        scene.build(n_envs=self.n_envs, env_spacing=(3.0, 3.0))
        self.scene = scene

        # dof 인덱스: 이름순 정렬 = [S1_tx, S1_ty, S1_tz, S2_tx, S2_ty, S2_tz]
        self.pris, self.rev = [], []
        for r in self.robots:
            pj = sorted((j for j in r.joints if j.type == gs.JOINT_TYPE.PRISMATIC),
                        key=lambda j: j.name)
            rj = sorted((j for j in r.joints if j.type == gs.JOINT_TYPE.REVOLUTE),
                        key=lambda j: j.name)
            assert len(pj) == 6 and len(rj) == 6, [j.name for j in r.joints]
            for base in (0, 3):
                assert "_tx_" in pj[base].name and "_ty_" in pj[base + 1].name \
                    and "_tz_" in pj[base + 2].name, [j.name for j in pj]
            self.pris.append([j.dof_idx_local for j in pj])
            self.rev.append([j.dof_idx_local for j in rj])

        hinge = [j for j in self.obj.joints
                 if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
        assert len(hinge) == 1, [j.name for j in self.obj.joints]
        self.hinge_dof = hinge[0].dof_idx_local

        self.DEV = self.robots[0].get_dofs_position(self.pris[0]).device
        N = self.n_envs
        for r, pidx, ridx in zip(self.robots, self.pris, self.rev):
            r.set_dofs_kp(torch.full((N, 6), KP, device=self.DEV), pidx)
            r.set_dofs_kv(torch.full((N, 6), KV, device=self.DEV), pidx)
            fr = torch.full((N, 6), F_MAX, device=self.DEV)
            r.set_dofs_force_range(-fr, fr, pidx)
            r.set_dofs_kp(torch.full((N, 6), REV_KP, device=self.DEV), ridx)
            r.set_dofs_kv(torch.full((N, 6), REV_KV, device=self.DEV), ridx)
            fr2 = torch.full((N, 6), REV_FMAX, device=self.DEV)
            r.set_dofs_force_range(-fr2, fr2, ridx)

        self.obj_mass0 = self.obj.get_links_inertial_mass()[0:1].clone()  # (1, n_links)

        # demo 궤적 텐서 (비용 계산용)
        self.demo_pos = torch.as_tensor(ref.obj_pos, dtype=torch.float32, device=self.DEV)
        self.demo_quat = torch.as_tensor(ref.obj_quat, dtype=torch.float32, device=self.DEV)
        self.demo_arti = torch.as_tensor(ref.obj_arti, dtype=torch.float32, device=self.DEV)
        self._q0 = torch.as_tensor(ref.demo_qpos0(), dtype=torch.float32,
                                   device=self.DEV)[None].repeat(N, 1)
        # 절대 낙하 임계: 데모 물체 최저 높이 − DROP_BELOW (데모-상대 z갭은 안전한
        # 지연 상태도 플래그하는 버그였음 — M3-A 진단)
        self._z_drop = float(ref.obj_pos[:, 2].min()) - DROP_BELOW
        start = ref.start_positions()   # (4,3) 표면 접촉 위치
        self._start = [
            torch.as_tensor(start[2 * i:2 * i + 2].reshape(-1), dtype=torch.float32,
                            device=self.DEV)[None].repeat(N, 1)
            for i in range(2)
        ]

    # ---------------- DR / reset ----------------
    def set_physics(self, mass_scale, fric_ratio, hinge_damp):
        """각 인자 (n_envs,) np/torch. hinge_damp는 절대값 [N·m·s]."""
        ms = torch.as_tensor(mass_scale, dtype=torch.float32, device=self.DEV)
        self.obj.set_links_inertial_mass(self.obj_mass0 * ms[:, None])
        fr = torch.as_tensor(fric_ratio, dtype=torch.float32, device=self.DEV)[:, None]
        self.obj.set_friction_ratio(fr.repeat(1, self.obj.n_links))
        for r in self.robots:   # 접촉쌍 양쪽에 같은 비율 (fork 결합규칙 방어)
            r.set_friction_ratio(fr.repeat(1, r.n_links))
        hd = torch.as_tensor(hinge_damp, dtype=torch.float32, device=self.DEV)[:, None]
        self.obj.set_dofs_damping(hd, [self.hinge_dof])

    def reset(self, start_positions=None):
        """start_positions: (N,4,3) 초기 구체 위치. None이면 구경로 기본값."""
        N = self.n_envs
        self.obj.set_qpos(self._q0)   # zero_velocity=True 기본
        z6 = torch.zeros(N, 6, device=self.DEV)
        for i, r in enumerate(self.robots):
            if start_positions is None:
                st = self._start[i]
            else:
                st = start_positions[:, 2 * i:2 * i + 2].reshape(N, 6)
            r.set_dofs_position(st, self.pris[i])
            r.set_dofs_position(z6, self.rev[i])
            r.control_dofs_position(st, self.pris[i])
            r.control_dofs_position(z6, self.rev[i])

    # ---------------- rollout ----------------
    def rollout(self, refs_env, record_every=0, log_states=False):
        """refs_env: (n_envs, T, 4, 3) torch(DEV). 반환: 스텝평균 비용 (n_envs,) np.

        record_every>0 이면 (카메라 필요) 그 간격으로 프레임 캡처 + env0의
        스텝별 추적오차 trace를 diag에 담는다 (시각화/M1 눈검증용).
        log_states=True 이면 env0의 (구체 위치, 제어힘, 물체 qpos)를 매 스텝 기록
        (onset 해부 프로브용).
        """
        N, T = self.n_envs, refs_env.shape[1]
        costs = torch.zeros(N, device=self.DEV)
        dropped = torch.zeros(N, dtype=torch.bool, device=self.DEV)
        frames, pos_trace, arti_trace = [], [], []
        slog = {"sphere_pos": [], "force": [], "obj_qpos": []} if log_states else None
        for t in range(T):
            for i, r in enumerate(self.robots):
                tgt = refs_env[:, t, 2 * i:2 * i + 2].reshape(N, 6)
                r.control_dofs_position(tgt, self.pris[i])
            self.scene.step()
            q = self.obj.get_qpos()                         # (N, 8)
            perr = (q[:, :3] - self.demo_pos[t]).norm(dim=1)
            dot = (q[:, 3:7] * self.demo_quat[t]).sum(1).abs().clamp(max=1.0)
            rerr = 2.0 * torch.acos(dot)
            aerr = (q[:, 7] - self.demo_arti[t]).abs()
            dropped |= q[:, 2] < self._z_drop
            costs += W_POS * perr + W_ROT * rerr + W_ARTI * aerr \
                + C_DROP * dropped.float()
            if record_every and t % record_every == 0:
                assert self.camera is not None, "with_camera=True로 생성해야 함"
                frame = self.camera.render()[0]
                frames.append(np.asarray(frame))
                pos_trace.append(float(perr[0]))
                arti_trace.append(float(aerr[0]))
            if log_states:
                sp = torch.cat([r.get_dofs_position(self.pris[i]).reshape(N, 2, 3)
                                for i, r in enumerate(self.robots)], dim=1)
                fc = torch.cat([r.get_dofs_control_force(self.pris[i]).reshape(N, 2, 3)
                                for i, r in enumerate(self.robots)], dim=1)
                slog["sphere_pos"].append(sp[0].cpu().numpy())
                slog["force"].append(fc[0].cpu().numpy())
                slog["obj_qpos"].append(q[0].cpu().numpy())
        diag = {"dropped": dropped.cpu().numpy(),
                "final_pos_err": (q[:, :3] - self.demo_pos[T - 1]).norm(dim=1).cpu().numpy(),
                "final_arti_err": (q[:, 7] - self.demo_arti[T - 1]).abs().cpu().numpy()}
        if record_every:
            diag.update(frames=frames, pos_trace=np.array(pos_trace),
                        arti_trace=np.array(arti_trace), record_every=record_every)
        if log_states:
            diag["log"] = {k: np.array(v) for k, v in slog.items()}
        return (costs / T).cpu().numpy(), diag

    # ---------------- 하이브리드 (물체 기준 + 운반) ----------------
    def plan_to_tensors(self, plans):
        """plan(dict of numpy) 리스트(chunk개) → env 차원으로 확장된 torch dict."""
        out = {}
        for k in ("can_p", "can_n", "delta", "dmax", "tilt_c", "demo_anchor", "start_pos"):
            arr = np.stack([p[k] for p in plans])          # (chunk, ...)
            t = torch.as_tensor(arr, dtype=torch.float32, device=self.DEV)
            out[k] = t.repeat_interleave(self.n_dr, dim=0)  # (n_envs, ...)
        top = np.stack([p["is_top"] for p in plans])
        out["is_top"] = torch.as_tensor(top, dtype=torch.bool, device=self.DEV
                                        ).repeat_interleave(self.n_dr, dim=0)
        return out

    def rollout_hybrid(self, plan, record_every=0, log_states=False):
        """물체 기준 reference를 매 스텝 실물 pose로 조립해 재생.

        plan: plan_to_tensors 결과 (env 차원 포함).
        x_ref = 실물 위 앵커 + (r−δ)·법선 + clip(데모 앵커 − 실물 앵커, d_max)
        """
        N = self.n_envs
        T = plan["can_p"].shape[1]
        costs = torch.zeros(N, device=self.DEV)
        dropped = torch.zeros(N, dtype=torch.bool, device=self.DEV)
        perr_sum = torch.zeros(N, device=self.DEV)
        aerr_sum = torch.zeros(N, device=self.DEV)
        frames, pos_trace, arti_trace = [], [], []
        slog = {"sphere_pos": [], "force": [], "obj_qpos": []} if log_states else None
        for t in range(T):
            q = self.obj.get_qpos()                          # 현재 실물 상태로 ref 조립
            Rb = _quat_to_R(q[:, 3:7])
            Ra = _Rz_neg(q[:, 7])
            p_c, n_c = plan["can_p"][:, t], plan["can_n"][:, t]
            t_c = plan["tilt_c"][:, t]
            top = plan["is_top"][:, t].unsqueeze(-1)
            p_eff = torch.where(top, torch.einsum("nij,nsj->nsi", Ra, p_c), p_c)
            n_eff = torch.where(top, torch.einsum("nij,nsj->nsi", Ra, n_c), n_c)
            t_eff = torch.where(top, torch.einsum("nij,nsj->nsi", Ra, t_c), t_c)
            anchor = torch.einsum("nij,nsj->nsi", Rb, p_eff) + q[:, None, :3]
            normal = torch.einsum("nij,nsj->nsi", Rb, n_eff)
            tilt_w = torch.einsum("nij,nsj->nsi", Rb, t_eff)
            grip = anchor + (SPHERE_R - plan["delta"][:, t]).unsqueeze(-1) * normal \
                + tilt_w
            gap = plan["demo_anchor"][:, t] - anchor
            gl = gap.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            offset = gap * torch.clamp(plan["dmax"][:, t, None, None] / gl, max=1.0)
            x_ref = grip + offset                             # (N,4,3)
            for i, r in enumerate(self.robots):
                r.control_dofs_position(x_ref[:, 2 * i:2 * i + 2].reshape(N, 6),
                                        self.pris[i])
            self.scene.step()
            q2 = self.obj.get_qpos()
            perr = (q2[:, :3] - self.demo_pos[t]).norm(dim=1)
            dot = (q2[:, 3:7] * self.demo_quat[t]).sum(1).abs().clamp(max=1.0)
            rerr = 2.0 * torch.acos(dot)
            aerr = (q2[:, 7] - self.demo_arti[t]).abs()
            dropped |= q2[:, 2] < self._z_drop
            costs += W_POS * perr + W_ROT * rerr + W_ARTI * aerr \
                + C_DROP * dropped.float()
            perr_sum += perr
            aerr_sum += aerr
            if record_every and t % record_every == 0:
                assert self.camera is not None, "with_camera=True로 생성해야 함"
                frames.append(np.asarray(self.camera.render()[0]))
                pos_trace.append(float(perr[0]))
                arti_trace.append(float(aerr[0]))
            if log_states:
                sp = torch.cat([r.get_dofs_position(self.pris[i]).reshape(N, 2, 3)
                                for i, r in enumerate(self.robots)], dim=1)
                fc = torch.cat([r.get_dofs_control_force(self.pris[i]).reshape(N, 2, 3)
                                for i, r in enumerate(self.robots)], dim=1)
                slog["sphere_pos"].append(sp[0].cpu().numpy())
                slog["force"].append(fc[0].cpu().numpy())
                slog["obj_qpos"].append(q2[0].cpu().numpy())
        diag = {"dropped": dropped.cpu().numpy(),
                "final_pos_err": (q2[:, :3] - self.demo_pos[T - 1]).norm(dim=1).cpu().numpy(),
                "final_arti_err": (q2[:, 7] - self.demo_arti[T - 1]).abs().cpu().numpy(),
                "mean_pos_err": (perr_sum / T).cpu().numpy(),
                "mean_arti_err": (aerr_sum / T).cpu().numpy()}
        if record_every:
            diag.update(frames=frames, pos_trace=np.array(pos_trace),
                        arti_trace=np.array(arti_trace), record_every=record_every)
        if log_states:
            diag["log"] = {k: np.array(v) for k, v in slog.items()}
        return (costs / T).cpu().numpy(), diag

    def evaluate_hybrid(self, ref, thetas, dr, n_knots, w_worst=0.5, tilt=False):
        """하이브리드 채점: thetas (P, theta_dim_hybrid) → (P,) 집계 비용."""
        P = len(thetas)
        agg = np.empty(P)
        mass_t = np.tile(dr["mass_scale"], self.chunk)
        fric_t = np.tile(dr["fric_ratio"], self.chunk)
        hinge_t = np.tile(dr["hinge_damp"], self.chunk)
        for c0 in range(0, P, self.chunk):
            idx = list(range(c0, min(c0 + self.chunk, P)))
            padded = idx + [idx[-1]] * (self.chunk - len(idx))
            plans = [ref.build_hybrid_plan(thetas[i], n_knots, tilt=tilt) for i in padded]
            pt = self.plan_to_tensors(plans)
            self.set_physics(mass_t, fric_t, hinge_t)
            self.reset(start_positions=pt["start_pos"])
            costs, _ = self.rollout_hybrid(pt)
            cm = costs.reshape(self.chunk, self.n_dr)
            for j, i in enumerate(idx):
                agg[i] = cm[j].mean() + w_worst * cm[j].max()
        return agg

    # ---------------- CEM 인터페이스 (구경로) ----------------
    def evaluate(self, thetas, dr, n_knots, w_worst=0.5):
        """thetas (P,D), dr = dict(mass_scale/fric_ratio/hinge_damp 각 (n_dr,)).
        반환: (P,) 집계 비용 = mean + w_worst·max over 물성."""
        P = len(thetas)
        agg = np.empty(P)
        mass_t = np.tile(dr["mass_scale"], self.chunk)
        fric_t = np.tile(dr["fric_ratio"], self.chunk)
        hinge_t = np.tile(dr["hinge_damp"], self.chunk)
        for c0 in range(0, P, self.chunk):
            idx = list(range(c0, min(c0 + self.chunk, P)))
            padded = idx + [idx[-1]] * (self.chunk - len(idx))
            refs = np.stack([self.ref.build(thetas[i], n_knots) for i in padded])
            refs_env = torch.as_tensor(refs, dtype=torch.float32, device=self.DEV
                                       ).repeat_interleave(self.n_dr, dim=0)
            self.set_physics(mass_t, fric_t, hinge_t)
            self.reset()
            costs, _ = self.rollout(refs_env)
            cm = costs.reshape(self.chunk, self.n_dr)
            for j, i in enumerate(idx):
                agg[i] = cm[j].mean() + w_worst * cm[j].max()
        return agg
