"""사다리 재생 계획 컴파일러 — 앵커 시간표 + 전환 계획 → 스텝 단위 재생 파일.

08-08 수정 3종 (사용자 프레임 판독 → 초기 부착 붕괴 진단):
  ① 평형 예압: 스텝별 부착 팔 집합에 대해 LP 자기평형 배분(합력·합토크 0)으로
     팁별 목표 힘(press_N)을 계산 — 혼자 붙은 팁은 자동으로 0 (박스 안 밀림)
  ② 착지 실물화 표시: 접근 마지막 구간(approach_u 0→1)을 표시해 재생기가
     실물 물체 기준으로 착지하게 함 (비행 중 앵커 부품 좌표도 미리 실림)
  ③ 테이블 지지 전환은 이동 팔 동시 비행 (하나씩 밀치는 것 방지 + 시간 단축)

좌표 규약: 앵커는 부품(본체/뚜껑) 고정 좌표, 세계 = R(t)@[Rz(−θ)·]q + t.
출력: data/contraction_dyn/replay/{subj}__{seq}.npz
실행: ./venv/bin/python -m contraction_dyn.compile_replay box_grab_01 [box_use_02]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_dyn.hand_contacts import SeqData
from contraction_gt.pilot_sim import _equilibrium_preload
from contraction_gt.search import mu_tor_from_pad

SLOW = 2          # 실시간 (30fps 시연 → 60Hz)
SLOW_WINDOWS = []                # 빈 목록 = 전 구간 실시간   # 구간별 감속 배율 (시작f, 끝f, 배): 빠른
# 굴리기 구간(회전 최고 234°/초, 팁 속도 1m/s급 → 감쇠 소모가 축 한계 잠식)
# 만 2배 느리게 — 설계의 "구간 감속 배율(CEM 변수)" 수동 선행 (08-15)


def slow_of(f):
    m = 1
    for s0, e0, k in SLOW_WINDOWS:
        if s0 <= f < e0:
            m = max(m, k)
    return SLOW * m
SPHERE_R = 0.015
LIFT = 0.08
LEGS = (24, 36, 24)
MU, F_CMD = 0.6, 12.0          # 평형 예압: 마찰, 팁 힘 캡 [N] (18N 시험: 들림 0.3→0.0 악화 — 힘 부족 아님)
POST_ATTACH_WAIT = 48          # 부착 후 시연 시계 정지 스텝 수 = play_contact 의
# 힘 램프(HOLD 30 + RAMP_UP 18). 램프 중 기준 궤적이 도망가 회전 오차가
# 쌓이던 문제(08-12 기준점검 로그: 0.2초에 z+3.6cm/회전+6.6°)의 수정
ARMS = ["left_A", "left_B", "left_C", "right_A", "right_B", "right_C"]
OUT = ROOT / "data/contraction_dyn/replay"


def run(seq, subj="s01"):
    sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule"
                           / f"{subj}__{seq}.json"))["rows"]
    trans = json.load(open(ROOT / "data/contraction_dyn/transitions"
                           / f"{subj}__{seq}.json"))["transitions"]
    trans_at = {t["at_frame"]: t for t in trans}
    sd = SeqData(subj, seq)

    def anchor_pqn(row, arm):
        d = row["anchors"][arm]
        return bool(d["part"]), np.asarray(d["pos_part"]), np.asarray(d["nrm_part"])

    def part_R(f, part):
        R, t, th, _v, _c = sd.obj_frame(f)
        if part and abs(th) > 1e-8:
            return R @ Rotation.from_rotvec([0, 0, -th]).as_matrix()
        return R

    def world(f, part, q, nq):
        R, t, th, _v, _c = sd.obj_frame(f)
        Rp = part_R(f, part)
        pw, nw = Rp @ q + t, Rp @ nq
        return pw + SPHERE_R * nw, nw

    R0_, t0_, th0_, v0_, _c0 = sd.obj_frame(0)
    plat_top = float(((R0_ @ v0_.T).T + t0_)[:, 2].min()) - 0.002

    def zc(p):
        q = np.array(p, np.float64)
        q[2] = max(q[2], plat_top + 0.025)
        return q

    targets, rev_targets = [], []
    ref_pos, ref_quat, ref_theta, demo_f, att_log = [], [], [], [], []
    ref_gap = []
    anchor_part, anchor_q, anchor_n, approach_u = [], [], [], []
    events, events_part = [], []
    cur_t = {a: None for a in ARMS}
    attached = {a: False for a in ARMS}
    part_of = {a: False for a in ARMS}
    R_att = {a: np.eye(3) for a in ARMS}
    cur_anchor = {a: (0, np.full(3, np.nan), np.full(3, np.nan)) for a in ARMS}
    appr = {a: -1.0 for a in ARMS}
    wrist_prev = {a: np.zeros(3) for a in ARMS}

    def wrist(f, a):
        if not attached[a]:
            wrist_prev[a] = np.zeros(3)
            return np.zeros(3)
        rel = part_R(f, part_of[a]) @ R_att[a].T
        z, x, y = Rotation.from_matrix(rel).as_euler("ZXY")
        raw = np.array([z, x, -y])
        raw += np.round((wrist_prev[a] - raw) / (2 * np.pi)) * 2 * np.pi
        wrist_prev[a] = raw
        return raw

    gap_cache = {}

    def ref_gap_of(f):
        """기준 자세에서 물체 최저점과 받침대 윗면 사이 틈 [m] (지지 판정용).

        분위수·속도 어림 대신 기하로 직접 계산 — 어림의 오판 사례(집 모형
        높이에서 공중에 든 것을 '얹힘'으로 판정)를 없애는 교체 (08-16 ⓑ)
        """
        if f not in gap_cache:
            R, t, th, _v, _c = sd.obj_frame(f)
            v = sd.tverts
            if abs(th) > 1e-8:
                v = v.copy()
                top = sd.parts == 0
                v[top] = v[top] @ Rotation.from_rotvec(
                    [0, 0, -th]).as_matrix().T
            zw = (v @ R.T)[:, 2] + t[2]
            gap_cache[f] = max(float(zw.min()) - plat_top, 0.0)
        return gap_cache[f]

    def push(step_targets, f, frozen=False):
        targets.append([step_targets[a] for a in ARMS])
        rev_targets.append([wrist(f, a) for a in ARMS])
        anchor_part.append([cur_anchor[a][0] for a in ARMS])
        anchor_q.append([cur_anchor[a][1] for a in ARMS])
        anchor_n.append([cur_anchor[a][2] for a in ARMS])
        approach_u.append([appr[a] for a in ARMS])
        R, t, th, _v, c = sd.obj_frame(f)
        ref_pos.append(R @ c + t)
        q = Rotation.from_matrix(R).as_quat()
        ref_quat.append([q[3], q[0], q[1], q[2]])
        ref_theta.append(th)
        ref_gap.append(ref_gap_of(f))
        demo_f.append(-1 if frozen else f)
        att_log.append([attached[a] for a in ARMS])

    def flight_ways(arm, row, prow, s):
        pt, q, nq = anchor_pqn(row, arm)
        to_pw, to_nw = world(s, pt, q, nq)
        fr_pw, fr_nw = cur_t[arm], to_nw
        if prow is not None and arm in (prow.get("anchors") or {}):
            pt0, q0_, nq0 = anchor_pqn(prow, arm)
            fr_pw, fr_nw = world(s, pt0, q0_, nq0)
        Rf, tf, thf, _vf, cf = sd.obj_frame(s)
        c_w = Rf @ cf + tf
        r_o = float(np.linalg.norm(sd.tverts - sd.tverts.mean(0), axis=1).max())

        def sdir(p):
            d = p - c_w
            return d / max(np.linalg.norm(d), 1e-9)
        d0, d1 = sdir(fr_pw), sdir(to_pw)
        o0 = c_w + d0 * (r_o + 0.12)
        o1 = c_w + d1 * (r_o + 0.12)
        way = []
        for k in range(LEGS[0]):
            u = (k + 1) / LEGS[0]
            way.append((zc(fr_pw * (1 - u) + o0 * u), -1.0))
        for k in range(LEGS[1]):
            u = (k + 1) / LEGS[1]
            d = d0 * (1 - u) + d1 * u
            n_ = np.linalg.norm(d)
            d = d / n_ if n_ > 0.2 else np.array([0, 0, 1.0])
            way.append((zc(c_w + d * (r_o + 0.12)), -1.0))
        for k in range(LEGS[2]):
            u = (k + 1) / LEGS[2]
            p = zc(o1 * (1 - u) + to_pw * u) if u < 1.0 else to_pw
            way.append((p, u))                 # 마지막 레그 = 접근 (실물화 표시)
        return (pt, q, nq), way

    first = next(r for r in sched if r.get("anchors"))
    for k, a in enumerate(ARMS):
        if a in first["anchors"]:
            pt_, q_, nq_ = anchor_pqn(first, a)
            pw_, nw_ = world(first["start"], pt_, q_, nq_)
            cur_t[a] = zc(pw_ + 0.15 * nw_)
        else:                       # 이 구간에 몫이 없는 팔: 물체 주변 대기 지점
            ang = 2 * np.pi * k / len(ARMS)
            cur_t[a] = np.array([t0_[0] + 0.4 * np.cos(ang),
                                 t0_[1] + 0.4 * np.sin(ang), plat_top + 0.3])

    for row in sched:
        s, e = row["start"], row["end"]
        tr = trans_at.get(s)
        if tr is not None:
            prow = next((r for r in sched if r["end"] == s and r.get("anchors")),
                        None)
            move_arms = list(tr["order"])
            plans = {}
            for arm in move_arms:
                pqn, way = flight_ways(arm, row, prow, s)
                plans[arm] = (pqn, way)
            if tr.get("table_supported"):
                # ③ 동시 비행: 전 이동 팔 분리 → 같이 날기 → 같이 부착
                for arm in move_arms:
                    ai = ARMS.index(arm)
                    if attached[arm]:
                        events.append((len(targets), ai, 0))
                        events_part.append(int(part_of[arm]))
                        attached[arm] = False
                    cur_anchor[arm] = (int(plans[arm][0][0]), plans[arm][0][1],
                                       plans[arm][0][2])
                n_step = len(plans[move_arms[0]][1])
                for k in range(n_step):
                    for arm in move_arms:
                        cur_t[arm], appr[arm] = plans[arm][1][k]
                    push(dict(cur_t), s, frozen=True)
                for arm in move_arms:
                    ai = ARMS.index(arm)
                    appr[arm] = -1.0
                    events.append((len(targets), ai, 1))
                    events_part.append(int(plans[arm][0][0]))
                    attached[arm] = True
                    part_of[arm] = bool(plans[arm][0][0])
                    R_att[arm] = part_R(s, part_of[arm])
            else:
                for arm in move_arms:
                    ai = ARMS.index(arm)
                    if attached[arm]:
                        events.append((len(targets), ai, 0))
                        events_part.append(int(part_of[arm]))
                        attached[arm] = False
                    cur_anchor[arm] = (int(plans[arm][0][0]), plans[arm][0][1],
                                       plans[arm][0][2])
                    for p, u in plans[arm][1]:
                        cur_t[arm], appr[arm] = p, u
                        push(dict(cur_t), s, frozen=True)
                    appr[arm] = -1.0
                    events.append((len(targets), ai, 1))
                    events_part.append(int(plans[arm][0][0]))
                    attached[arm] = True
                    part_of[arm] = bool(plans[arm][0][0])
                    R_att[arm] = part_R(s, part_of[arm])
        anchors_c = {a: anchor_pqn(row, a) for a in ARMS
                     if row.get("anchors") and a in row["anchors"]}
        if tr is not None:
            # ④ 예압 대기: 힘 램프가 끝날 때까지 첫 프레임에 멈춰 기다림
            for a, (pt, q, nq) in anchors_c.items():
                if attached[a]:
                    cur_t[a] = world(s, pt, q, nq)[0]
            for _ in range(POST_ATTACH_WAIT):
                push(dict(cur_t), s, frozen=True)
        for f in range(s, e):
            for a, (pt, q, nq) in anchors_c.items():
                if attached[a]:
                    cur_t[a] = world(f, pt, q, nq)[0]
            for _ in range(slow_of(f)):
                push(dict(cur_t), f)

    # ── ① 평형 예압: 스텝별 부착 집합 → 팁별 목표 힘 [N] ──
    T = len(targets)
    att_arr = np.array(att_log, bool)
    ap_arr = np.array(anchor_part, np.int64)
    aq_arr = np.array(anchor_q, np.float64)
    an_arr = np.array(anchor_n, np.float64)
    th_arr = np.array(ref_theta, np.float64)
    press = np.zeros((T, len(ARMS)), np.float32)
    cache = {}
    for t in range(T):
        on = np.flatnonzero(att_arr[t])
        if len(on) == 0:
            continue
        key = (round(float(th_arr[t]), 1),
               tuple((int(a), int(ap_arr[t, a]))
                     + tuple(np.round(aq_arr[t, a], 3)) for a in on))
        if key not in cache:
            th = th_arr[t]
            Ra = Rotation.from_rotvec([0, 0, -th]).as_matrix()
            v = sd.tverts.copy()
            top = sd.parts == 0
            if abs(th) > 1e-8:
                v[top] = v[top] @ Ra.T
            com = v.mean(0)
            r_o = float(np.linalg.norm(v - com, axis=1).max())
            pos, nin = [], []
            for a in on:
                qa, na = aq_arr[t, a], an_arr[t, a]
                if ap_arr[t, a] and abs(th) > 1e-8:
                    qa, na = Ra @ qa, Ra @ na
                pos.append((qa - com) / r_o)
                nin.append(-na)                       # 안쪽 법선
            mu_tor = mu_tor_from_pad(0.10 * r_o, MU, 0.5) / r_o
            try:
                pl, _x, _w = _equilibrium_preload(
                    np.array(pos), np.array(nin), MU, mu_tor, F_CMD)
            except Exception:
                pl = np.zeros(len(on))
            cache[key] = np.asarray(pl, np.float64)
        press[t, list(on)] = cache[key]
    k = np.ones(12) / 12                              # 12스텝 평활 (배분 전환 완충)
    for a in range(len(ARMS)):
        press[:, a] = np.convolve(press[:, a], k, mode="same")

    R0_, t0_, th0_, v0_, _c0 = sd.obj_frame(0)
    qq = Rotation.from_matrix(R0_).as_quat()
    q0 = [t0_[0], t0_[1], t0_[2], qq[3], qq[0], qq[1], qq[2], th0_]
    wv = (R0_ @ v0_.T).T + t0_
    plat_c = wv.mean(0)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{subj}__{seq}.npz"
    np.savez_compressed(
        p, q0=np.array(q0, np.float64),
        targets=np.array(targets, np.float32),
        rev_targets=np.array(rev_targets, np.float32),
        ref_pos=np.array(ref_pos, np.float32),
        ref_quat=np.array(ref_quat, np.float32),
        ref_theta=np.array(ref_theta, np.float32),
        demo_frame=np.array(demo_f, np.int32),
        attached=att_arr,
        events=np.array(events, np.int64).reshape(-1, 3),
        events_part=np.array(events_part, np.int64),
        anchor_part=ap_arr, anchor_q=np.array(anchor_q, np.float32),
        anchor_n=np.array(anchor_n, np.float32),
        approach_u=np.array(approach_u, np.float32),
        press_N=press,
        platform=np.array([plat_c[0], plat_c[1], plat_top], np.float64),
        cen=sd.tverts.mean(0),
        arms=np.array(ARMS),
        ref_gap=np.array(ref_gap, np.float64))
    n_tr = sum(1 for d in demo_f if d < 0)
    print(f"저장: {p} — 총 {T}스텝 (전환 {n_tr}), 이벤트 {len(events)}개, "
          f"예압 구성 {len(cache)}종 (팁 힘 범위 "
          f"{press[press > 0].min() if (press > 0).any() else 0:.1f}~"
          f"{press.max():.1f}N)")


if __name__ == "__main__":
    for seq in sys.argv[1:] or ["box_grab_01"]:
        run(seq)
