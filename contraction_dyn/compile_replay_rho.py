"""재생 계획 컴파일러 (ρ판) — ρ 눌러앉기 시간표 → 스텝 단위 재생 계획.

설계: md_claude/plans/rho_continuous_control_v1.md §4·§5. compile_replay.py(비행 전환판)와의 차이:
  * 전환 = 순간이동 사건: 앵커가 바뀌는 경계에서 시연 시계를 멈추고(frozen) 유령 홀드 정착
    N_SETTLE 스텝 + 해제 램프 N_RELEASE 스텝. 비행 경로·순서 탐색 없음. 첫 파지도 같은 사건.
  * 기준 가속 하중: 시연 질량중심 궤적의 가속도 a_ref, 각속도 ω_ref, 각가속도 α_ref (5프레임 평활,
    30fps 미분). 멈춘 스텝에서는 0.
  * 오프라인 ρ(t): 스텝별 접촉 집합·자세·명목 렌치(중력 게이트 + m·a + I·α + ω×Iω)로
    재배분 허용 여유 ρ 를 미리 계산해 저장 (진단 곡선용).
  * 미리 쥐는 힘 표(press_N) 없음 — 새 배분기가 매 틱 결정.

출력: data/contraction_dyn/replay_rho/{subj}__{seq}[__nomove].npz
실행: ./venv/bin/python -m contraction_dyn.compile_replay_rho box_grab_01 [--never-move]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contraction_dyn.hand_contacts import SeqData          # noqa: E402
from contraction_dyn.box_model import BoxModel             # noqa: E402
from contraction_gt.metric import compute_rho              # noqa: E402

G = 9.81
FPS = 30.0
SLOW = 2                                  # 30fps 시연 → 60Hz 제어 (실시간)
N_SETTLE, N_RELEASE = 60, 45              # 전환: 정착 1.0s + 유령 해제 0.75s
SMOOTH = 5                                # 미분용 이동 평균 창 (프레임)
LOOKAHEAD, GATE_H = 24, 0.02              # 무게 게이트 (play_contact_mj 와 동일)
MOVE_EPS = 0.02                           # 이보다 멀면 "옮긴 팔"
ARMS = ["left_A", "left_B", "left_C", "right_A", "right_B", "right_C"]
OUT = ROOT / "data/contraction_dyn/replay_rho"


def smooth(x, k=SMOOTH):
    """이동 평균 (가장자리는 끝값 복제 — 0 채움은 끝 프레임에 가짜 가속을 만든다)."""
    if k <= 1:
        return x
    ker = np.ones(k) / k
    h = k // 2
    xp = np.pad(x, ((h, h), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], ker, mode="valid") for i in range(x.shape[1])], 1)


def demo_kinematics(sd, bm):
    """프레임별 (R, t, θ, 중심 c_w, 질량중심 com_w, 속도, 가속도, ω, α)."""
    T = len(sd.obj)
    Rs, ts, ths, cen_w, com_w = [], [], [], [], []
    for f in range(T):
        R, t, th, v, c = sd.obj_frame(f)
        _, com_c = bm.mass_com(th)
        Rs.append(R)
        ts.append(t)
        ths.append(th)
        cen_w.append(R @ c + t)
        com_w.append(R @ com_c + t)
    Rs, com_w = np.array(Rs), np.array(com_w)
    p = smooth(com_w)
    v = np.gradient(p, 1 / FPS, axis=0)
    a = smooth(np.gradient(v, 1 / FPS, axis=0))
    w = np.zeros((T, 3))
    for f in range(1, T - 1):
        w[f] = Rotation.from_matrix(Rs[f + 1] @ Rs[f - 1].T).as_rotvec() * FPS / 2
    w = smooth(w)
    al = smooth(np.gradient(w, 1 / FPS, axis=0))
    return dict(R=Rs, t=np.array(ts), theta=np.array(ths), cen_w=np.array(cen_w),
                com_w=com_w, v=v, a=a, omega=w, alpha=al)


def run(seq, subj, never_move, device, always_move=False, slow=1.0):
    """slow: 균일 감속 배율 k (1.0 = 실시간). 스텝/프레임 = round(SLOW·k), 기준 가속·각속도·각가속도는 1/k², 1/k, 1/k²."""
    sched_tag = "__nomove" if never_move else ("__always" if always_move else "")
    tag = sched_tag + (f"__slow{slow:g}" if slow != 1.0 else "")
    steps_per_frame = int(round(SLOW * slow))
    k_eff = steps_per_frame / SLOW
    sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule_rho"
                           / f"{subj}__{seq}{sched_tag}.json"))
    rows = sched["rows"]
    sd = SeqData(subj, seq)
    bm = BoxModel()
    kin = demo_kinematics(sd, bm)
    T_demo = len(sd.obj)

    # 받침대 윗면 (compile_replay 와 동일 정의) + 기준 자세 지지 틈
    R0, t0, th0, v0, c0 = sd.obj_frame(0)
    plat_top = float(((R0 @ v0.T).T + t0)[:, 2].min()) - 0.002
    top_mask = sd.parts == 0

    def ref_gap_of(f):
        R, t, th, _v, _c = sd.obj_frame(f)
        v = sd.tverts
        if abs(th) > 1e-8:
            v = v.copy()
            v[top_mask] = v[top_mask] @ Rotation.from_rotvec([0, 0, -th]).as_matrix().T
        return max(float(((v @ R.T)[:, 2] + t[2]).min()) - plat_top, 0.0)

    gaps = np.array([ref_gap_of(f) for f in range(T_demo)])

    # ---- 스텝 나열 ----
    cols = dict(demo_frame=[], swap_phase=[], ghost_gain=[], swap_move=[], attached=[],
                anchor_part=[], anchor_q=[], anchor_n=[], ref_pos=[], ref_quat=[],
                ref_theta=[], ref_gap=[], ref_acc=[], ref_omega=[], ref_alpha=[])
    cur = {a: None for a in ARMS}         # arm → (part, q, n_out)
    attached = {a: False for a in ARMS}
    n_swaps = 0

    def push(f, frozen, phase=0, gain=0.0, move=None):
        cols["demo_frame"].append(-1 if frozen else f)
        cols["swap_phase"].append(phase)
        cols["ghost_gain"].append(gain)
        cols["swap_move"].append([bool(move and a in move) for a in ARMS])
        cols["attached"].append([attached[a] for a in ARMS])
        cols["anchor_part"].append([int(cur[a][0]) if cur[a] else 0 for a in ARMS])
        cols["anchor_q"].append([cur[a][1] if cur[a] else np.full(3, np.nan) for a in ARMS])
        cols["anchor_n"].append([cur[a][2] if cur[a] else np.full(3, np.nan) for a in ARMS])
        R, t = kin["R"][f], kin["t"][f]
        cols["ref_pos"].append(kin["cen_w"][f])
        q = Rotation.from_matrix(R).as_quat()
        cols["ref_quat"].append([q[3], q[0], q[1], q[2]])
        cols["ref_theta"].append(kin["theta"][f])
        cols["ref_gap"].append(gaps[f])
        z3 = np.zeros(3)
        cols["ref_acc"].append(z3 if frozen else kin["a"][f] / k_eff ** 2)
        cols["ref_omega"].append(z3 if frozen else kin["omega"][f] / k_eff)
        cols["ref_alpha"].append(z3 if frozen else kin["alpha"][f] / k_eff ** 2)

    for row in rows:
        s, e = row["start"], row["end"]
        new = row.get("anchors")
        if new:
            moving = []
            for a, d in new.items():
                rec = (bool(d["part"]), np.asarray(d["pos_part"], float),
                       np.asarray(d["nrm_part"], float))
                old = cur[a]
                if (old is None or old[0] != rec[0]
                        or np.linalg.norm(old[1] - rec[1]) > MOVE_EPS):
                    moving.append(a)
                cur[a] = rec
                attached[a] = True
            if moving:
                n_swaps += 1
                for k in range(N_SETTLE + N_RELEASE):
                    if k < N_SETTLE:
                        push(s, True, phase=1, gain=1.0, move=moving if k == 0 else None)
                    else:
                        push(s, True, phase=2,
                             gain=1.0 - (k - N_SETTLE + 1) / N_RELEASE)
        for f in range(s, e):
            for _ in range(steps_per_frame):
                push(f, False)

    arr = {k: np.array(v) for k, v in cols.items()}
    T = len(arr["demo_frame"])

    # ---- 오프라인 ρ(t): 명목 렌치(게이트 중력 + m·a + I·α + ω×Iω), 접촉 집합·자세별 ----
    gap = arr["ref_gap"]
    gap_ahead = np.array([gap[t:t + LOOKAHEAD + 1].max() for t in range(T)])
    g_lift = np.clip(gap_ahead / GATE_H, 0.0, 1.0)
    mass = bm.mass
    rho_t = np.full(T, np.nan)
    r_max_t = np.full(T, np.nan)
    wnom = np.zeros((T, 6))
    pos_l, nrm_l, com_l, w_l, idx_l = [], [], [], [], []
    for t in range(T):
        on = [a for a in range(len(ARMS)) if arr["attached"][t, a]]
        f = arr["demo_frame"][t]
        if f < 0:                                  # 멈춘 스텝: 직전 실행 프레임 자세
            f = next((arr["demo_frame"][u] for u in range(t, -1, -1) if arr["demo_frame"][u] >= 0), 0)
        R, tr, th = kin["R"][f], kin["t"][f], kin["theta"][f]
        _, com_c = bm.mass_com(th)
        com_w = R @ com_c + tr
        Iw = R @ bm.inertia(th) @ R.T
        om, al = arr["ref_omega"][t], arr["ref_alpha"][t]
        w = np.concatenate([[0, 0, g_lift[t] * mass * G] + mass * arr["ref_acc"][t],
                            Iw @ al + np.cross(om, Iw @ om)])
        wnom[t] = w
        if len(on) < 2:
            continue
        Rz = Rotation.from_rotvec([0, 0, -th]).as_matrix() if abs(th) > 1e-8 else np.eye(3)
        P, N = [], []
        for a in on:
            Rp = R @ Rz if arr["anchor_part"][t, a] else R
            P.append(Rp @ arr["anchor_q"][t, a] + tr)
            N.append(-(Rp @ arr["anchor_n"][t, a]))      # 안쪽 법선
        pos_l.append(np.array(P))
        nrm_l.append(np.array(N))
        com_l.append(com_w)
        w_l.append(w)
        idx_l.append(t)
    row0 = next(r for r in rows if r.get("anchors"))
    if idx_l:
        # 접촉 수가 같은 스텝끼리 묶어 배치 계산
        by_n = {}
        for k, t in enumerate(idx_l):
            by_n.setdefault(len(pos_l[k]), []).append(k)
        td = dict(dtype=torch.float64, device=device)
        for n, ks in by_n.items():
            for c0_ in range(0, len(ks), 512):
                kk = ks[c0_:c0_ + 512]
                out = compute_rho(torch.tensor(np.stack([pos_l[k] for k in kk]), **td),
                                  torch.tensor(np.stack([nrm_l[k] for k in kk]), **td),
                                  torch.tensor(np.stack([w_l[k] for k in kk]), **td),
                                  com=torch.tensor(np.stack([com_l[k] for k in kk]), **td),
                                  r_obj=row0["r_obj"], f_cap=row0["f_cap"],
                                  mu_t=row0["mu_t"], mu_tor=row0["mu_tor"], backend="batched")
                rho = out["rho"].cpu().numpy()
                for j, k in enumerate(kk):
                    rho_t[idx_l[k]] = rho[j]

    qq = Rotation.from_matrix(R0).as_quat()
    q0 = [t0[0], t0[1], t0[2], qq[3], qq[0], qq[1], qq[2], th0]
    wv = (R0 @ v0.T).T + t0
    plat_c = wv.mean(0)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{subj}__{seq}{tag}.npz"
    np.savez_compressed(
        p, q0=np.array(q0, np.float64),
        demo_frame=arr["demo_frame"].astype(np.int32),
        swap_phase=arr["swap_phase"].astype(np.int8),
        ghost_gain=arr["ghost_gain"].astype(np.float32),
        swap_move=arr["swap_move"].astype(bool),
        attached=arr["attached"].astype(bool),
        anchor_part=arr["anchor_part"].astype(np.int64),
        anchor_q=arr["anchor_q"].astype(np.float32),
        anchor_n=arr["anchor_n"].astype(np.float32),
        ref_pos=arr["ref_pos"].astype(np.float32),
        ref_quat=arr["ref_quat"].astype(np.float32),
        ref_theta=arr["ref_theta"].astype(np.float32),
        ref_gap=arr["ref_gap"].astype(np.float64),
        ref_acc=arr["ref_acc"].astype(np.float32),
        ref_omega=arr["ref_omega"].astype(np.float32),
        ref_alpha=arr["ref_alpha"].astype(np.float32),
        g_lift=g_lift.astype(np.float32),
        w_nom=wnom.astype(np.float32),
        rho_t=rho_t.astype(np.float32), r_max_t=r_max_t.astype(np.float32),
        platform=np.array([plat_c[0], plat_c[1], plat_top], np.float64),
        cen=sd.tverts.mean(0), arms=np.array(ARMS),
        mass=np.float64(mass), r_obj=np.float64(row0["r_obj"]),
        mu_t=np.float64(row0["mu_t"]), mu_tor=np.float64(row0["mu_tor"]),
        f_cap=np.float64(row0["f_cap"]))
    n_frozen = int((arr["demo_frame"] < 0).sum())
    fin = np.isfinite(rho_t)
    print(f"저장: {p}\n  균일 감속 배율 {k_eff:g} (프레임당 {steps_per_frame}스텝)  총 {T}스텝 (전환 {n_swaps}회, 멈춘 스텝 {n_frozen}), "
          f"ρ(t) 계산 {fin.sum()}스텝: 최소 {np.nanmin(rho_t):+.3f} / 중앙값 {np.nanmedian(rho_t):+.3f}")
    # 구간별 ρ(t) 최소 요약
    for row in rows:
        if not row.get("anchors"):
            continue
        m = (arr["demo_frame"] >= row["start"]) & (arr["demo_frame"] < row["end"]) & fin
        if m.any():
            am = np.linalg.norm(arr["ref_acc"][m], axis=1).max()
            print(f"  구간 {row['seg']:2d} f{row['start']:4d}-{row['end']:4d}: ρ(t) 최소 {rho_t[m].min():+.3f} "
                  f"(정적 worst-ρ {row['rho_worst']:+.3f}), |a|max {am:.2f} m/s², "
                  f"|α|max {np.degrees(np.linalg.norm(arr['ref_alpha'][m], axis=1).max()):5.0f} °/s²")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=["box_grab_01"])
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--never-move", action="store_true")
    ap.add_argument("--always-move", action="store_true")
    ap.add_argument("--slow", type=float, default=1.0, help="균일 감속 배율 (1 = 실시간, 2 = 절반 속도)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in args.seqs:
        run(s, args.subj, args.never_move, dev, always_move=args.always_move, slow=args.slow)
