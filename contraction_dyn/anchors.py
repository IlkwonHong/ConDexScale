"""앵커 시간표 생성 — 동적 v1 구현 2번 후반부 (설계 §2-② contraction).

hold 구간마다: 구간 합산 접촉 → 손 소유 그룹(손별 [엄지|나머지]) → 다중 시작
좌표 상승(LP min-α 최대화) + 눌러앉기·해제 규칙 → 팔 4대 앵커 시간표.

규칙 (설계 확정본):
  눌러앉기: 이전 앵커 배치의 α ≥ max(임계 0.15, 새 탐색 최고의 80%) 이면 유지
  해제 이벤트: 사람이 손을 떼도 그 손 몫 팔들은 기존 앵커 유지
  worst-α: 표본 프레임들의 물체 자세(중력 방향)별 재채점의 최솟값 — 구간 내내 버티는가

좌표: 앵커·마스크는 canonical 물체 좌표, α 채점 시 프레임 회전 R 로 세계 정렬.
출력: data/contraction_dyn/anchor_schedule/{subj}__{seq}.json
실행: ./venv/bin/python -m contraction_dyn.anchors box_grab_01 [box_use_02 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.surface import SurfaceSamples, sample_surface
from contraction_gt.search import _batched_alpha, mu_tor_from_pad
from contraction_dyn.hand_contacts import SeqData, UNITS

MU_T, PAD_REL = 0.6, 0.10
RHO_REL, MARGIN_REL = 0.15, 0.05
THR, STICKY = 0.15, 0.8
N_SAMPLE_F, N_STARTS, ROUNDS = 5, 4, 2
OUT = ROOT / "data/contraction_dyn/anchor_schedule"
ARMS = [("left", "A"), ("left", "B"), ("left", "C"),
        ("right", "A"), ("right", "B"), ("right", "C")]   # 손당 3팔 (08-14)


def rot_view(smp, R):
    return SurfaceSamples(points=smp.points @ R.T, normals=smp.normals @ R.T,
                          r_obj=smp.r_obj, com=R @ smp.com)


def canonical_smp(sd, theta, n=2048):
    """관절각 theta 의 canonical(정점평균 원점) 표면 표본.

    부가 속성: smp.part_top[n] (가동부 여부), smp.cen (정점평균), smp.theta.
    """
    from scipy.spatial.transform import Rotation
    v = sd.tverts.copy()
    top = sd.parts == 0
    if abs(theta) > 1e-8:
        v[top] = v[top] @ Rotation.from_rotvec([0, 0, -theta]).as_matrix().T
    c = v.mean(0)
    mesh = trimesh.Trimesh(vertices=v - c, faces=sd.tfaces, process=False)
    smp = sample_surface(mesh, n)
    bad = ((-smp.normals) * (smp.points - smp.com)).sum(1) < 0
    smp.normals[bad] *= -1        # 1차: 중심 규약 (볼록 외피에서 유효)
    # 2차: 포함성(레이 패리티) 판정 — 중심 규약은 속 빈 용기의 안쪽 면
    # (이 박스는 표면의 ~절반)에서 부호가 반전된다 (2026-09-04, use_02
    # 열린 뚜껑 구간에서 시뮬 접촉 법선 대조로 확정). 바깥 후보(−n) 3mm
    # 프로브가 메시 내부이고 안쪽(+n) 프로브가 외부면 반전으로 확정하고
    # 뒤집는다. 둘 다 내부/외부인 병리 표본(쐐기 모서리·얇은 벽)은 유지.
    out_p = mesh.contains(smp.points - smp.normals * 0.003)
    in_p = mesh.contains(smp.points + smp.normals * 0.003)
    smp.normals[out_p & ~in_p] *= -1
    nn = np.linalg.norm((smp.points + c)[:, None] - v[None], axis=-1).argmin(1)
    smp.part_top = top[nn]
    smp.cen = c
    smp.theta = float(theta)
    # 날카로운 모서리 선분 (이면각 40° 이상) — 모서리 여유 제약용
    ang = mesh.face_adjacency_angles
    ve = mesh.face_adjacency_edges[ang > np.deg2rad(40)]
    smp.edge_segs = mesh.vertices[ve] if len(ve) else np.zeros((0, 2, 3))
    return smp


def to_part_frame(smp, i):
    """표본점 i → (part_top, 부품 좌표 위치, 부품 좌표 바깥 법선)."""
    from scipy.spatial.transform import Rotation
    v_art = smp.points[i] + smp.cen
    n_out = -smp.normals[i]
    if smp.part_top[i] and abs(smp.theta) > 1e-8:
        Rz = Rotation.from_rotvec([0, 0, smp.theta]).as_matrix()
        return True, Rz @ v_art, Rz @ n_out
    return bool(smp.part_top[i]), v_art, n_out


def from_part_frame(smp, part_top, q):
    """부품 좌표 위치 → 이 표본(smp.theta)의 canonical 최근접 표면점 idx."""
    from scipy.spatial.transform import Rotation
    v = q
    if part_top and abs(smp.theta) > 1e-8:
        v = Rotation.from_rotvec([0, 0, -smp.theta]).as_matrix() @ q
    return int(np.linalg.norm(smp.points - (v - smp.cen), axis=1).argmin())


def alpha_of(smp_r, idx_arr, mu_tor):
    a = _batched_alpha(smp_r, np.asarray(idx_arr), MU_T,
                       np.full(len(idx_arr), mu_tor), "cuda")
    return a


def anchor0_unit(con, smp):
    """엄지, 없으면 대립 부위(나머지의 개수 가중 평균 법선과 가장 마주보는 부위)."""
    present = [u for u in range(6) if con["mask"][u]]
    if UNITS.index("thumb") in present:
        return UNITS.index("thumb")
    nn = ((con["cpos"][:, None] - smp.points[None]) ** 2).sum(-1).argmin(1)
    un = {u: smp.normals[nn[con["cunit"] == u]].mean(0) for u in present}
    best, score = present[0], np.inf
    for u in present:
        others = [v for v in present if v != u]
        w = np.array([(con["cunit"] == v).sum() for v in others], np.float64)
        mean = sum(wi * un[v] for wi, v in zip(w, others)) / max(w.sum(), 1e-9)
        s = float(np.dot(un[u], mean))
        if s < score:
            best, score = u, s
    return best


def group_masks(con, smp, blocked=None):
    """손 하나의 그룹 3개: A=[앵커0 부위], B/C=[나머지 접촉점 주축 2분할].

    나머지 접촉점을 가장 퍼진 방향(주축)으로 반씩 나눠 팔 2개에 배정.
    나머지가 빈약하면 그룹 수가 2 또는 1로 줄 수 있음 (호출부가 앞에서부터 배정).
    blocked: [2048] bool — 환경 차단 표면점 (테이블에 눌린 바닥면 등) 제외.
    """
    a0 = anchor0_unit(con, smp)
    sel0 = np.isin(con["cunit"], [a0])
    pts_groups = [con["cpos"][sel0]]
    rp = con["cpos"][~sel0]
    if len(rp) >= 2:
        u_ = rp - rp.mean(0)
        ax = np.linalg.svd(u_, full_matrices=False)[2][0]
        proj = u_ @ ax
        med = float(np.median(proj))
        lo, hi = rp[proj <= med], rp[proj > med]
        if len(hi) == 0:                     # 전부 같은 자리인 경우 반반
            half = len(rp) // 2
            lo, hi = rp[:half], rp[half:]
        pts_groups += [g for g in (lo, hi) if len(g)]
    elif len(rp) == 1:
        pts_groups.append(rp)
    out = []
    for pts in pts_groups:
        if len(pts) == 0:
            continue
        center = pts.mean(0)
        spread = float(np.linalg.norm(pts - center, axis=1).max())
        radius = max(spread, RHO_REL * smp.r_obj) + MARGIN_REL * smp.r_obj
        d = np.linalg.norm(smp.points - center, axis=1)
        ok = d <= radius
        if blocked is not None:
            ok &= ~blocked
        cand = np.where(ok)[0]
        grp_pts = pts
        if len(cand) == 0:
            far = d.copy()
            if blocked is not None:
                far[blocked] = np.inf
            cand = np.argsort(far)[:8]
        init_d = d.copy()
        init_d[~np.isin(np.arange(len(d)), cand)] = np.inf
        out.append(dict(center=center, cand=cand,
                        init=int(np.argmin(init_d)),
                        pts=grp_pts))          # 그룹 소속 접촉점 (측지 후보용)
    return out or None


def ascent(smp_r, slots, active, mu_tor, starts, edge_w=None):
    """slots: {arm: idx or None}. active 팔들만 각자의 후보에서 좌표 상승.

    edge_w: [2048] 모서리 벌점 (0.85~1). 점수 = α × min(구성원 w) — 약한 고리
    기준. 금지 대신 감점이라 얇은 부품에서도 후보가 살아남음."""
    if edge_w is None:
        edge_w = np.ones(len(smp_r.points))
    arms = [a for a in ARMS if slots.get(a) is not None or a in active]
    best = None
    for st in starts:
        cur = {a: (st[a] if a in st else slots[a]) for a in arms}
        for _ in range(ROUNDS):
            for a in active:
                cand = active[a]["cand"]
                taken = {cur[x] for x in arms if x != a}   # 팔끼리 같은 점 금지
                cand = np.array([c for c in cand if c not in taken],
                                dtype=np.int64)
                if len(cand) == 0:
                    continue
                combos = np.tile([cur[x] for x in arms], (len(cand), 1))
                combos[:, arms.index(a)] = cand
                w_fix = min((edge_w[cur[x]] for x in arms if x != a),
                            default=1.0)
                w_row = np.minimum(w_fix, edge_w[cand])
                sc = alpha_of(smp_r, combos, mu_tor).min(1) * w_row
                cur[a] = int(cand[int(np.nanargmax(sc))])
        a14 = alpha_of(smp_r, [[cur[x] for x in arms]], mu_tor)[0]
        s = float(np.nanmin(a14)) * min(edge_w[cur[x]] for x in arms)
        if best is None or s > best[0]:
            best = (s, dict(cur))
    return best


def run(seq, subj="s01"):
    kf = json.load(open(ROOT / "data/contraction_dyn/keyframes" / f"{subj}__{seq}.json"))
    sd = SeqData(subj, seq)
    rng = np.random.default_rng(0)
    smp_cache = {}

    def smp_for(theta):
        key = round(theta, 1)
        if key not in smp_cache:
            smp_cache[key] = canonical_smp(sd, key)
        return smp_cache[key]

    # 환경 차단 마스크: 테이블 지지 자세에서 물체 최저면 +2cm 안의 표면점은
    # 팁이 물리적으로 못 들어감 (사다리 1단 실측: 구체가 테이블 밑에 걸림).
    # 테이블 지지 구간들의 대표 자세 합집합으로 계산해 전 구간 공통 제외 (v1 보수).
    zs = []
    for f in range(0, len(sd.obj), 10):
        R, t, th, v, c = sd.obj_frame(f)
        zs.append((R @ c + t)[2])
    rest_z = float(np.percentile(zs, 5))
    CLEAR = 0.02
    blocked_by_key = {}

    EDGE_W0, EDGE_D = 0.85, 0.008   # 모서리 벌점 바닥/스케일 (정적 실측 근거; 동적 검증은 진행 중)
                               # (사다리 2단 실측: 테두리 앵커에서 구체가 미끄러져 낙하)

    def blocked_for(theta_key, frames):
        """(차단 마스크, 모서리 벌점 w) 반환.

        차단(자격 박탈)은 물리적 불가능뿐: 테이블에 눌린 바닥면.
        모서리는 금지 대신 **벌점** — 실측(mj_edge_slip_test, 현 물리) 근거:
        모서리 0mm 에서도 접선 7.1N 버팀 (평면 ~8N, 위험 ≤15%), 8mm 밖은
        평면과 동일 → w = 0.85 + 0.15×(모서리거리/8mm), 0.85~1.0.
        (옛 2cm 금지는 점성 마찰 시절 유물 — 얇은 부품 후보 전멸의 원인)
        """
        if theta_key not in blocked_by_key:
            smp_ = smp_cache[theta_key]
            segs = smp_.edge_segs
            if len(segs):
                a, b = segs[:, 0], segs[:, 1]
                ab = b - a
                P = smp_.points
                t_ = ((P[:, None] - a[None]) * ab[None]).sum(-1) \
                    / (ab * ab).sum(-1).clip(1e-12)
                t_ = np.clip(t_, 0, 1)
                proj = a[None] + t_[..., None] * ab[None]
                dmin = np.linalg.norm(P[:, None] - proj, axis=-1).min(1)
                blocked_by_key[theta_key] = \
                    EDGE_W0 + (1 - EDGE_W0) * np.clip(dmin / EDGE_D, 0, 1)
            else:
                blocked_by_key[theta_key] = np.ones(2048)
        blocked = np.zeros(2048, bool)
        for f in frames:
            R, t, th, v, c = sd.obj_frame(f)
            if (R @ c + t)[2] - rest_z < 0.01:            # 테이블 지지 자세
                zw = (smp_cache[theta_key].points @ R.T)[:, 2]
                blocked |= zw < zw.min() + CLEAR
        return blocked, blocked_by_key[theta_key]

    slots = {a: None for a in ARMS}          # 팔 → canonical 표면점 idx
    rows = []
    last_key = None
    for si, seg in enumerate(kf["segments"]):
        s, e = seg["start"], seg["end"]
        frames = sorted(set(np.linspace(s, e - 1, min(N_SAMPLE_F, e - s))
                            .round().astype(int).tolist()))
        theta = seg["arti_mean"]
        key = round(theta, 1)
        if last_key is not None and key != last_key:
            # 관절 각이 바뀌면 부품 좌표를 경유해 재스냅 — 뚜껑 앵커는 뚜껑과 동승
            old = smp_for(last_key)
            new = smp_for(key)
            for a in ARMS:
                if slots[a] is not None:
                    pt, q, _n = to_part_frame(old, slots[a])
                    slots[a] = from_part_frame(new, pt, q)
        last_key = key
        smp = smp_for(theta)
        mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
        rep = frames[len(frames) // 2]
        R_rep = sd.obj_frame(rep)[0]
        smp_rep = rot_view(smp, R_rep)
        blk, edge_w = blocked_for(key, frames)

        note = []
        for side in ("left", "right"):
            con = sd.segment_contacts(side, frames)
            hand_arms = [(side, "A"), (side, "B"), (side, "C")]
            if con is None:
                held_now = [a for a in hand_arms if slots[a] is not None]
                if held_now:
                    # 조건부 해제: 남는 팔들만으로 α ≥ 임계면 이 손 몫을 놓아줌
                    others = [a for a in ARMS
                              if slots[a] is not None and a not in hand_arms]
                    s_o = 0.0
                    if others:
                        a14o = alpha_of(smp_rep, [[slots[a] for a in others]],
                                        mu_tor)[0]
                        s_o = float(np.nanmin(a14o))
                    if s_o >= THR:
                        for a in held_now:
                            slots[a] = None
                        note.append(f"{side}: 조건부 해제(남은 α={s_o:.2f})")
                    else:
                        note.append(f"{side}: 해제 무시(남은 α={s_o:.2f}<임계)")
                continue
            gm = group_masks(con, smp, blk)
            if gm is None:
                note.append(f"{side}: 그룹 구성 불가")
                continue
            active = {hand_arms[k]: gm[k]
                      for k in range(min(len(gm), len(hand_arms)))}
            starts = []
            prev = {a: slots[a] for a in hand_arms}
            prev_blocked = any(v is not None and blk[v] for v in prev.values())
            if prev_blocked:
                note.append(f"{side}: 차단 앵커 강제 이동")
                prev = {a: None for a in hand_arms}
            if all(v is not None for v in prev.values()):
                starts.append(prev)                       # warm start
            starts.append({a: active[a]["init"] for a in active})
            for _ in range(N_STARTS - len(starts)):
                starts.append({a: int(rng.choice(active[a]["cand"])) for a in active})
            fresh_s, fresh = ascent(smp_rep, slots, active, mu_tor, starts,
                                    edge_w)
            if all(v is not None for v in prev.values()):
                kept = dict(slots)
                a14k = alpha_of(smp_rep, [[kept[a] if kept[a] is not None else fresh[a]
                                           for a in ARMS if kept[a] is not None
                                           or a in fresh]], mu_tor)[0]
                kept_s = float(np.nanmin(a14k))
                if kept_s >= max(THR, STICKY * fresh_s):
                    note.append(f"{side}: 눌러앉기({kept_s:.2f}≥{fresh_s:.2f}×0.8)")
                    continue
                note.append(f"{side}: 이동({kept_s:.2f}<{fresh_s:.2f})")
            else:
                note.append(f"{side}: 신규({fresh_s:.2f})")
            for a in active:
                slots[a] = fresh[a]

        held = [a for a in ARMS if slots[a] is not None]
        if not held:
            rows.append(dict(seg=si, start=s, end=e, hand=seg["hand_state"],
                             theta=theta, anchors=None, note="앵커 없음"))
            continue
        idxs = [[slots[a] for a in held]]
        a_rep = alpha_of(smp_rep, idxs, mu_tor)[0]
        worst = min(float(np.nanmin(alpha_of(rot_view(smp, sd.obj_frame(f)[0]),
                                             idxs, mu_tor)[0]))
                    for f in frames)
        rows.append(dict(
            seg=si, start=s, end=e, hand=seg["hand_state"], theta=theta,
            anchors={f"{a[0]}_{a[1]}": (lambda pt_q_n: dict(
                idx=int(slots[a]),
                pos=[round(float(x), 4) for x in smp.points[slots[a]]],
                part=int(pt_q_n[0]),
                pos_part=[round(float(x), 5) for x in pt_q_n[1]],
                nrm_part=[round(float(x), 5) for x in pt_q_n[2]]))(
                    to_part_frame(smp, slots[a]))
                     for a in held},
            min_alpha=round(float(np.nanmin(a_rep)), 3),
            worst_alpha=round(worst, 3),
            gravity_alpha=round(float(a_rep[5]), 3),
            note="; ".join(note)))
        flag = "" if worst >= THR else "  ◀ 임계 미달"
        print(f"[{si:2d}] {s:4d}-{e:4d} {seg['hand_state']:5s} 팔{len(held)} "
              f"min-α={rows[-1]['min_alpha']:.3f} worst={worst:.3f} "
              f"중력={rows[-1]['gravity_alpha']:.3f}{flag}  {rows[-1]['note']}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj, rows=rows), open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"저장: {out}")


if __name__ == "__main__":
    for seq in sys.argv[1:] or ["box_grab_01"]:
        print(f"===== {seq} =====")
        run(seq)
