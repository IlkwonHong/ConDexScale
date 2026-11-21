"""전환 계획기 v1 (잠정판) — 동적 v1 구현 3번 (설계 §2-③, 수정 예상 표시됨).

앵커 시간표의 인접 구간 쌍마다: 이동 팔 목록 → "한 번에 한 팔" 순서의 순열 전수
→ 각 중간 상태(팔 하나 빠진 접촉 집합)의 min-α 를 경계 프레임의 물체 자세로 채점
→ 전 상태 ≥ 임계인 순서 중 최악 중간 α 최대인 순서 채택.

테이블 지지 감면: 경계 시점에 물체가 테이블 위(높이 여유 1cm 미만)면 중간 상태
임계를 면제한다 — 7월 실측 "반대손 자유 전환의 대부분 = 최초 파지/해제, 환경 지지"
와 정합. 최초 파지(팔 0→4)가 이 감면의 대표 사례.

지지 삽입: 감면 없이 모든 순서가 실패할 때, 노는 팔(앵커 없는 팔)이 있으면 전 표면
LP 훑기로 임시 지지점을 찾는다. v1 에서는 해제 규칙 때문에 노는 팔이 드물어
실패 시 "불가"로 기록될 가능성이 큼 — 잠정판의 알려진 한계.

출력: data/contraction_dyn/transitions/{subj}__{seq}.json
실행: ./venv/bin/python -m contraction_dyn.transitions box_grab_01 [box_use_02 ...]
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.search import _batched_alpha, mu_tor_from_pad
from contraction_dyn.hand_contacts import SeqData
from contraction_dyn.anchors import canonical_smp, rot_view, from_part_frame

MU_T, PAD_REL, THR = 0.6, 0.10, 0.15
MOVE_EPS_REL = 0.02          # 이보다 가까우면 "같은 자리" (재스냅 잡음 허용)
TABLE_EPS = 0.01             # 테이블 지지 판정: 높이 여유 [m]
OUT = ROOT / "data/contraction_dyn/transitions"


def set_alpha(smp_r, idxs, mu_tor):
    """접촉 집합(표면 인덱스 리스트)의 min-α. 빈/1점 집합은 0."""
    if len(idxs) == 0:
        return 0.0
    a = _batched_alpha(smp_r, np.asarray([idxs]), MU_T,
                       np.array([mu_tor]), "cuda")[0]
    return float(np.nanmin(a))


def plan_transition(smp, R, mu_tor, hold, moves, thr):
    """hold: {팔: idx} 유지 팔들. moves: [(팔, from_idx|None, to_idx)].

    반환: (채택 순서 리스트, 단계별 기록, 최악 중간 α, 성립 여부)
    """
    smp_r = rot_view(smp, R)
    best = None
    for order in permutations(range(len(moves))):
        pos = {a: i for a, i in hold.items()}
        for a, fr, _to in moves:
            if fr is not None:
                pos[a] = fr
        steps, worst = [], np.inf
        for oi in order:
            a, fr, to = moves[oi]
            if fr is not None:
                del pos[a]                          # 떼기 → 중간 상태
            s = set_alpha(smp_r, list(pos.values()), mu_tor)
            steps.append((f"{a} 떼기" if fr is not None else f"{a} 접근",
                          len(pos), round(s, 3)))
            worst = min(worst, s)
            pos[a] = to                             # 새 자리 붙이기
            s2 = set_alpha(smp_r, list(pos.values()), mu_tor)
            steps.append((f"{a} 붙이기", len(pos), round(s2, 3)))
        ok = worst >= thr
        key = (ok, worst)
        if best is None or key > best[0]:
            best = (key, [moves[i][0] for i in order], steps, worst)
    (ok, _), order, steps, worst = best
    return order, steps, float(worst), ok


def run(seq, subj="s01"):
    sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule"
                           / f"{subj}__{seq}.json"))
    sd = SeqData(subj, seq)
    rows = sched["rows"]

    # 지지 판정용 받침대 윗면 = 첫 프레임 물체 최저점 − 2mm (컴파일러와 동일
    # 정의). 분위수 어림 대신 기하 (08-16 ⓑ)
    from scipy.spatial.transform import Rotation as _Rot
    R0_, t0_, th0_, _v0, _c0 = sd.obj_frame(0)
    plat_top = float(((R0_ @ sd.tverts.T).T + t0_)[:, 2].min()) - 0.002

    def support_gap(f):
        """프레임 f 기준 자세에서 물체 최저점 − 받침대 윗면 [m]."""
        R, t, th, _v, _c = sd.obj_frame(f)
        v = sd.tverts
        if abs(th) > 1e-8:
            v = v.copy()
            top = sd.parts == 0
            v[top] = v[top] @ _Rot.from_rotvec([0, 0, -th]).as_matrix().T
        return max(float(((v @ R.T)[:, 2] + t[2]).min()) - plat_top, 0.0)
    rest_z = plat_top                       # 기록 호환용

    smp_cache = {}

    def smp_for(theta):
        key = round(theta, 1)
        if key not in smp_cache:
            smp_cache[key] = canonical_smp(sd, key)
        return smp_cache[key]

    def snap(smp, pos):
        return int(np.linalg.norm(smp.points - np.asarray(pos), axis=1).argmin())

    out = []
    for prev, nxt in zip(rows[:-1], rows[1:]):
        if nxt.get("anchors") is None:
            continue
        f = nxt["start"]
        theta = nxt["theta"]
        smp = smp_for(theta)
        mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
        R, t, th, v, c = sd.obj_frame(f)
        table = bool(support_gap(f) < 0.005)   # 기하 판정: 최저점 틈 5mm 이내

        pa = prev.get("anchors") or {}
        na = nxt["anchors"]
        hold, moves = {}, []
        for arm, d in na.items():
            # 부품 좌표 기준 — 뚜껑 위 앵커는 뚜껑이 돌아도 "안 움직인 것"
            to = from_part_frame(smp, bool(d["part"]), np.asarray(d["pos_part"]))
            if arm in pa:
                p0 = pa[arm]
                dist = (np.linalg.norm(np.asarray(d["pos_part"]) - p0["pos_part"])
                        if p0.get("part") == d["part"] else np.inf)
                if dist < MOVE_EPS_REL * smp.r_obj:
                    hold[arm] = to
                else:
                    fr = from_part_frame(smp, bool(p0["part"]),
                                         np.asarray(p0["pos_part"]))
                    moves.append((arm, fr, to))
            else:
                moves.append((arm, None, to))       # 신규 부착
        if not moves:
            continue
        thr = 0.0 if table else THR
        order, steps, worst, ok = plan_transition(smp, R, mu_tor, hold, moves, thr)
        verdict = ("성립" if ok else "불가") + (" (테이블 지지 감면)" if table else "")
        out.append(dict(at_frame=f, boundary=f"{prev['seg']}→{nxt['seg']}",
                        moving=[m[0] for m in moves], order=order,
                        steps=steps, worst_mid_alpha=round(worst, 3),
                        table_supported=table, ok=bool(ok)))
        print(f"[전환 {prev['seg']:2d}→{nxt['seg']:2d}] 프레임 {f:4d} "
              f"이동 {len(moves)}팔 {[m[0] for m in moves]} → 순서 {order} | "
              f"중간 최악 α={worst:.3f} | {verdict}")
        for name, n, s in steps:
            print(f"        {name:14s} 접촉 {n}개  α={s:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj, rest_z=rest_z, transitions=out),
              open(p, "w"), ensure_ascii=False, indent=1)
    n_ok = sum(t["ok"] for t in out)
    print(f"\n전환 {len(out)}개 중 성립 {n_ok}개 / 불가 {len(out) - n_ok}개")
    print(f"저장: {p}")


if __name__ == "__main__":
    for seq in sys.argv[1:] or ["box_grab_01"]:
        print(f"===== {seq} =====")
        run(seq)
