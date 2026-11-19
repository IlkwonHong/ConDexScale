"""키프레임 정지 파지 용량 연구 — air-hold worst-ρ 채점의 구간별 독립 앵커 생성.

그릴링 확정 (2026-09-03~04):
  * 명제: 모든 hold 구간에서 앵커 배치가 공중에서 mg 를 지고 정지 유지 (지지 감면
    없음, w_req = 자세별 중력 만재 — rho_eval 의 들림 게이트를 쓰지 않음).
  * 눌러앉기·해제 규칙 없음 — 구간별 완전 독립: 사람 접촉 → group_masks 후보
    (contraction 충실도 = 후보 반경) → worst-ρ(표본 자세 5 × 정점 64) 좌표 상승.
  * 한 손 구간 그대로 포함 — 예정된 실패(ρ<0)는 "요구사항 초과"로 기록.
  * 기준선 = 기존 anchor_schedule(α 탐색 배치)를 같은 air-hold ρ 로 재채점.
  * 산출물은 재생용 시간표가 아니라 정적 용량 연구 — 시뮬 검증(static_hold_test)
    과 시각화의 입력.

점수 규약: score = worst-ρ × min(모서리 벌점 w)  (ρ>0 일 때만 가중 — ρ<0 은
부족분 신호라 감점 곱셈이 부호를 왜곡하므로 원값 사용).

실행: ./venv/bin/python -m contraction_dyn.static_capacity box_grab_01 [box_use_02]
출력: data/contraction_dyn/static_capacity/{subj}__{seq}.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.metric import compute_rho, vertex_direction_set  # noqa: E402
from contraction_gt.search import mu_tor_from_pad                    # noqa: E402
from contraction_dyn.anchors import (canonical_smp, group_masks,     # noqa: E402
                                     to_part_frame, MU_T, PAD_REL,
                                     N_SAMPLE_F, N_STARTS, ROUNDS,
                                     RHO_REL, MARGIN_REL)
from contraction_dyn.rho_eval import urdf_inertials, mass_com, vertex_sig  # noqa: E402
from contraction_dyn.hand_contacts import SeqData                    # noqa: E402

G = 9.81
OUT = ROOT / "data/contraction_dyn/static_capacity"
# 환경 상수 — anchors.py 의 blocked_for 와 동일 값 (탐색 조건 정렬)
CLEAR = 0.02
EDGE_W0, EDGE_D = 0.85, 0.008
# 후보 규약 개정 (2026-09-05, 사용자 지시):
#  ① 반경 = 유클리드가 아니라 **표면 측지 거리** — 유클리드 공은 얇은 벽의
#     안팎을 한 덩어리로 묶어 안쪽 면이 후보로 새던 원인.
#  ② 후보는 **바깥 면 전용** — 닫힌(θ=0) 배치에서 바깥 법선 레이가 메시에
#     다시 맞지 않고 탈출하는 면 (= 밖에서 만질 수 있는 면).
GEO_RADIUS_REL = RHO_REL + MARGIN_REL   # 접촉 무리로부터의 측지 반경 (r_obj 배)


def surface_graph(smp):
    """표면 측지용 kNN 그래프. 간선 조건: 같은 부품 + 법선 호환(내적 > −0.5)
    + 3cm 이내 — 얇은 벽 안↔밖(법선 반대)과 부품 간 점프를 차단해 거리가
    표면을 따라가게 한다."""
    from scipy.sparse import csr_matrix
    P = smp.points
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, 1:13]
    rows, cols, vals = [], [], []
    for i in range(len(P)):
        for j in idx[i]:
            if smp.part_top[i] != smp.part_top[j]:
                continue
            if float(smp.normals[i] @ smp.normals[j]) < -0.5:
                continue
            dist = float(np.sqrt(d2[i, j]))
            if dist > 0.03:
                continue
            rows.append(i)
            cols.append(int(j))
            vals.append(dist)
    return csr_matrix((vals, (rows, cols)), shape=(len(P), len(P)))


def exterior_mask(smp, mesh0):
    """바깥 면 판정: 표본을 닫힌(θ=0) 배치로 되돌린 뒤, 바깥 법선으로 2mm
    띄워 쏜 레이가 메시에 다시 맞지 않으면(탈출) 바깥 면."""
    from scipy.spatial.transform import Rotation
    pts = smp.points + smp.cen
    n_out = -smp.normals
    if abs(smp.theta) > 1e-8:
        Rz = Rotation.from_rotvec([0, 0, smp.theta]).as_matrix()
        pts, n_out = pts.copy(), n_out.copy()
        pts[smp.part_top] = pts[smp.part_top] @ Rz.T
        n_out[smp.part_top] = n_out[smp.part_top] @ Rz.T
    hit = mesh0.ray.intersects_any(pts + n_out * 0.002, n_out)
    return ~np.asarray(hit)


class AirHoldScorer:
    """구간 하나의 air-hold worst-ρ 채점기.

    combos [C, A] (표면 표본 인덱스) → worst-ρ [C]:
    표본 프레임 자세마다 (앵커·법선·질량중심을 그 자세로 회전, w_req = +mg·ẑ world)
    ρ 를 풀고 프레임 최솟값. 비가능 배치는 ρ = −부족분 (연속 신호, ascent 가능).
    """

    def __init__(self, smp, Rs, com_ref, mass, f_cap, mu_tor, device):
        self.smp, self.mass, self.f_cap = smp, mass, f_cap
        self.mu_tor, self.device = mu_tor, device
        self.Rs = np.stack(Rs)                                # [F, 3, 3]
        self.com_f = np.einsum("fij,j->fi", self.Rs, com_ref)  # [F, 3]
        self.com_ref = com_ref
        self.F = len(Rs)

    def _batch(self, combos):
        C, A = combos.shape
        F = self.F
        pts = self.smp.points[combos]                          # [C, A, 3]
        nrs = self.smp.normals[combos]
        pos = np.einsum("caj,fij->cfai", pts, self.Rs)         # [C, F, A, 3]
        nrm = np.einsum("caj,fij->cfai", nrs, self.Rs)
        com = np.broadcast_to(self.com_f, (C, F, 3))
        wreq = np.zeros((C, F, 6))
        wreq[..., 2] = self.mass * G
        td = dict(dtype=torch.float64, device=self.device)
        return compute_rho(
            torch.tensor(pos.reshape(C * F, A, 3), **td),
            torch.tensor(nrm.reshape(C * F, A, 3).copy(), **td),
            torch.tensor(wreq.reshape(C * F, 6), **td),
            com=torch.tensor(com.reshape(C * F, 3).copy(), **td),
            r_obj=self.smp.r_obj, f_cap=self.f_cap,
            mu_t=MU_T, mu_tor=self.mu_tor, backend="batched")

    def rho(self, combos):
        """[C, A] → worst-ρ [C]."""
        out = self._batch(np.asarray(combos, dtype=np.int64))
        C = len(combos)
        return out["rho"].reshape(C, self.F).min(dim=1).values.cpu().numpy()

    def detail(self, idxs):
        """단일 배치 → 프레임별 ρ·가능성, 최악 프레임, 최약 정점."""
        out = self._batch(np.asarray([idxs], dtype=np.int64))
        rho = out["rho"].reshape(self.F).cpu().numpy()
        feas = out["feasible"].reshape(self.F).cpu().numpy()
        fi = int(rho.argmin())
        verts = vertex_direction_set(torch.float64, self.device).cpu().numpy()
        weak = (vertex_sig(verts[int(out["weak_vertex"][fi])])
                if bool(feas[fi]) else "비가능")
        return dict(rho_frames=[round(float(x), 4) for x in rho],
                    feasible_frames=[bool(x) for x in feas],
                    rho_worst=round(float(rho.min()), 4),
                    worst_frame_i=fi, weak_vertex=weak)


def ascent_rho(scorer, active, starts, edge_w, keep_current=False):
    """anchors.ascent 와 동일한 다중 시작 좌표 상승 — 점수만 air-hold worst-ρ.

    keep_current=False (기본, 2026-09-05 저녁 개정): 후보는 사람 접촉 무리의 측지 반경 안으로
    **강제**된다 — 현 위치가 후보 밖이면 유지 선택지가 없다 (논문 ‖p_i − p̂_i‖ ≤ ε_p 조건).
    keep_current=True: 옛 동작 — 현 위치를 항상 후보에 포함. 기준선에서 물려받은 반경 밖
    앵커(사람 접촉에서 20cm)가 여유가 크면 그대로 남아 사람 접촉 충실도가 깨지던 원인
    (재생 분할 영상 대조로 확정 — 구간 5 이후 오른팔이 사람 오른손을 따르지 않음).
    """
    arms = list(active.keys())
    best = None
    for st in starts:
        cur = dict(st)
        for _ in range(ROUNDS):
            for a in arms:
                cand = (np.concatenate([[cur[a]], active[a]["cand"]]) if keep_current
                        else np.asarray(active[a]["cand"], dtype=np.int64))
                taken = {cur[x] for x in arms if x != a}
                cand = np.unique(np.array(
                    [c for c in cand if c not in taken], dtype=np.int64))
                if len(cand) == 0:
                    continue
                combos = np.tile([cur[x] for x in arms], (len(cand), 1))
                combos[:, arms.index(a)] = cand
                rho = scorer.rho(combos)
                w_fix = min((edge_w[cur[x]] for x in arms if x != a),
                            default=1.0)
                w_row = np.minimum(w_fix, edge_w[cand])
                sc = np.where(rho > 0, rho * w_row, rho)
                cur[a] = int(cand[int(np.nanargmax(sc))])
        rho1 = float(scorer.rho([[cur[x] for x in arms]])[0])
        w1 = min(edge_w[cur[x]] for x in arms)
        s = rho1 * w1 if rho1 > 0 else rho1
        if best is None or s > best[0]:
            best = (s, dict(cur), rho1)
    return best


def placement_dict(smp, arm_idx, scorer, edge_w, score=None):
    """{팔이름: 표본 idx} → JSON 직렬화 가능한 배치 기록 + air-hold 채점."""
    arms = list(arm_idx.keys())
    idxs = [arm_idx[a] for a in arms]
    det = scorer.detail(idxs)
    out = dict(
        arms={a: dict(
            idx=int(i),
            pos=[round(float(x), 5) for x in smp.points[i]],
            nrm_in=[round(float(x), 5) for x in smp.normals[i]],
            part=int(to_part_frame(smp, i)[0]),
            edge_w=round(float(edge_w[i]), 3))
            for a, i in zip(arms, idxs)},
        **det)
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def run(seq, subj, f_cap, device, allow_baseline_start=False):
    kf = json.load(open(ROOT / "data/contraction_dyn/keyframes"
                        / f"{subj}__{seq}.json"))
    sched = {r["seg"]: r for r in json.load(open(
        ROOT / "data/contraction_dyn/anchor_schedule"
        / f"{subj}__{seq}.json"))["rows"]}
    sd = SeqData(subj, seq)
    iner = urdf_inertials()
    mass = sum(m for m, _ in iner.values())
    rng = np.random.default_rng(0)
    smp_cache, edge_w_cache = {}, {}
    import trimesh
    from scipy.sparse.csgraph import dijkstra
    mesh0 = trimesh.Trimesh(vertices=sd.tverts, faces=sd.tfaces, process=False)
    graph_cache, ext_cache = {}, {}

    def geo_ext_for(smp):
        key = round(smp.theta, 1)
        if key not in graph_cache:
            graph_cache[key] = surface_graph(smp)
            ext_cache[key] = exterior_mask(smp, mesh0)
        return graph_cache[key], ext_cache[key]

    def smp_for(theta):
        key = round(theta, 1)
        if key not in smp_cache:
            smp_cache[key] = canonical_smp(sd, key)
        return smp_cache[key]

    # 테이블 차단 기준 정지 높이 (anchors.py 와 동일 정의)
    zs = []
    for f in range(0, len(sd.obj), 10):
        R, t, th, v, c = sd.obj_frame(f)
        zs.append((R @ c + t)[2])
    rest_z = float(np.percentile(zs, 5))

    def blocked_and_edgew(smp, frames):
        """anchors.blocked_for 복제 — 차단(테이블 눌린 면) + 모서리 벌점 w."""
        key = round(smp.theta, 1)
        if key not in edge_w_cache:
            segs = smp.edge_segs
            if len(segs):
                a, b = segs[:, 0], segs[:, 1]
                ab = b - a
                P = smp.points
                t_ = ((P[:, None] - a[None]) * ab[None]).sum(-1) \
                    / (ab * ab).sum(-1).clip(1e-12)
                t_ = np.clip(t_, 0, 1)
                proj = a[None] + t_[..., None] * ab[None]
                dmin = np.linalg.norm(P[:, None] - proj, axis=-1).min(1)
                edge_w_cache[key] = \
                    EDGE_W0 + (1 - EDGE_W0) * np.clip(dmin / EDGE_D, 0, 1)
            else:
                edge_w_cache[key] = np.ones(len(smp.points))
        blocked = np.zeros(len(smp.points), bool)
        for f in frames:
            R, t, th, v, c = sd.obj_frame(f)
            if (R @ c + t)[2] - rest_z < 0.01:
                zw = (smp.points @ R.T)[:, 2]
                blocked |= zw < zw.min() + CLEAR
        return blocked, edge_w_cache[key]

    rows = []
    print(f"--- {subj}/{seq}  air-hold ρ  질량 {mass:.3f}kg (mg={mass*G:.2f}N), "
          f"f_cap {f_cap:.0f}N, 자세 {N_SAMPLE_F} × 정점 64 ---")
    for si, seg in enumerate(kf["segments"]):
        s, e = seg["start"], seg["end"]
        frames = sorted(set(np.linspace(s, e - 1, min(N_SAMPLE_F, e - s))
                            .round().astype(int).tolist()))
        theta = seg["arti_mean"]
        smp = smp_for(theta)
        mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
        com_ref = mass_com(iner, smp.theta) - smp.cen
        Rs = [sd.obj_frame(f)[0] for f in frames]
        scorer = AirHoldScorer(smp, Rs, com_ref, mass, f_cap, mu_tor, device)
        blk, edge_w = blocked_and_edgew(smp, frames)

        row = dict(seg=si, start=s, end=e, hand=seg["hand_state"], theta=theta,
                   frames=frames, mass=mass, f_cap=f_cap,
                   r_obj=float(smp.r_obj), mu_t=MU_T, mu_tor=float(mu_tor),
                   cen=[round(float(x), 5) for x in smp.cen],
                   com_ref=[round(float(x), 5) for x in com_ref],
                   R_frames=[np.round(R, 6).tolist() for R in Rs])

        # ---- 사람 접촉 + 후보 그룹 (구간 독립, 규칙 없음) ----
        active, contacts = {}, {}
        for side in ("left", "right"):
            con = sd.segment_contacts(side, frames)
            if con is None:
                continue
            contacts[side] = [[round(float(x), 4) for x in p]
                              for p in con["cpos"]]
            gm = group_masks(con, smp, blk)
            if gm is None:
                continue
            # 후보 재구성: 접촉 무리에서의 표면 측지 거리 ≤ GEO_RADIUS_REL·r_obj
            # ∧ 바깥 면 ∧ 비차단. 비면 측지 최근접 바깥면 8개(그것도 없으면
            # 유클리드 최근접 바깥면 8개) 폴백.
            G_, ext = geo_ext_for(smp)
            usable = ext & ~blk
            for k, g in enumerate(gm[:3]):
                src = np.unique(np.linalg.norm(
                    g["pts"][:, None] - smp.points[None], axis=-1).argmin(1))
                D = dijkstra(G_, directed=False, indices=src, min_only=True)
                cand = np.where((D <= GEO_RADIUS_REL * smp.r_obj) & usable)[0]
                if len(cand) == 0:
                    Dm = np.where(usable, D, np.inf)
                    cand = np.argsort(Dm)[:8]
                    if not np.isfinite(Dm[cand]).any():
                        de = np.linalg.norm(smp.points - g["center"], axis=1)
                        de[~usable] = np.inf
                        cand = np.argsort(de)[:8]
                g["cand"] = np.asarray(cand, dtype=np.int64)
                g["init"] = int(g["cand"][int(np.argmin(
                    np.nan_to_num(D[g["cand"]], nan=np.inf, posinf=1e9)))])
                active[f"{side}_{'ABC'[k]}"] = g
        row["contacts"] = contacts

        # ---- 기준선: 기존 α 시간표 배치를 air-hold ρ 로 재채점 ----
        # (해제 무시 규칙 덕에 접촉 없는 구간에도 기준선 6팔이 존재 — 항상 채점)
        base, b_idx = None, None
        srow = sched.get(si)
        if srow and srow.get("anchors"):
            b_idx = {}
            for arm, a in srow["anchors"].items():
                i = int(a["idx"])
                p = np.asarray(a["pos"], dtype=np.float64)
                if np.linalg.norm(smp.points[i] - p) > 1e-3:
                    i = int(np.linalg.norm(smp.points - p, axis=1).argmin())
                b_idx[arm] = i
            base = placement_dict(smp, b_idx, scorer, edge_w)
            base["alpha_worst"] = srow.get("worst_alpha")
        row["baseline"] = base

        if not active:
            row.update(tuned=None, note="접촉 없음")
            rows.append(row)
            b_str = (f"기준 ρ={base['rho_worst']:+.3f}" if base else "기준 없음")
            print(f"[{si:2d}] {s:4d}-{e:4d} {seg['hand_state']:5s} "
                  f"접촉 없음 ({b_str})")
            continue

        # ---- ρ 파인튜닝: 다중 시작 좌표 상승 ----
        # 기준선 배치를 시작점에 포함 (팔 구성이 같고 **전원 바깥 면**일 때만 —
        # 안쪽 앵커가 '유지' 선택지로 살아남으면 바깥면 강제가 깨짐).
        _, ext = geo_ext_for(smp)
        starts = [{a: active[a]["init"] for a in active}]
        if b_idx is not None and set(b_idx) == set(active):
            if allow_baseline_start:
                ok_b = all(ext[i] for i in b_idx.values())
            else:   # 강제 반경: 기준선 앵커가 전부 자기 팔의 후보 집합 안일 때만 시작점으로
                ok_b = all(int(b_idx[a]) in set(active[a]["cand"].tolist()) for a in active)
            if ok_b:
                starts.insert(0, dict(b_idx))
        for _ in range(N_STARTS - len(starts) + 1):
            starts.append({a: int(rng.choice(active[a]["cand"]))
                           for a in active})
        score, tuned_idx, rho1 = ascent_rho(scorer, active, starts, edge_w,
                                            keep_current=allow_baseline_start)
        row["tuned"] = placement_dict(smp, tuned_idx, scorer, edge_w,
                                      score=score)
        # 사람 접촉 충실도: 팔별 앵커 → 그 손 접촉점 최근접 거리 [cm]
        fid = {}
        for a, i in tuned_idx.items():
            pts = np.asarray(contacts.get(a.split("_")[0], []), dtype=np.float64)
            fid[a] = (round(float(np.linalg.norm(pts - smp.points[i], axis=1).min()) * 100, 2)
                      if len(pts) else None)
        row["tuned"]["fidelity_cm"] = fid
        rows.append(row)

        b_str = (f"기준 ρ={base['rho_worst']:+.3f}" if base else "기준 없음")
        t = row["tuned"]
        print(f"[{si:2d}] {s:4d}-{e:4d} {seg['hand_state']:5s} 팔{len(active)} "
              f"{b_str} → 튜닝 ρ={t['rho_worst']:+.3f} "
              f"({t['rho_worst']*f_cap:+.1f}N) 최약 {t['weak_vertex']} 접촉거리최대 "
              f"{max(v for v in t['fidelity_cm'].values() if v is not None):.1f}cm"
              f"{'  ◀ 비가능' if not all(t['feasible_frames']) else ''}")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj, mass=mass, f_cap=f_cap,
                   n_vertices=64,
                   note="air-hold worst-ρ (만재) · 후보 = 표면 측지 반경 + 바깥면 전용 "
                        "(2026-09-05 개정)" + (" · 기준선 시작 허용(옛 동작)" if allow_baseline_start
                                              else " · 후보 반경 강제 (2026-09-05 저녁 개정)"),
                   rows=rows), open(out, "w"), ensure_ascii=False, indent=1)
    print(f"저장: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=["box_grab_01"])
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--f-cap", type=float, default=30.0)
    ap.add_argument("--allow-baseline-start", action="store_true",
                    help="옛 동작: 기준선 시작 + 현 위치 유지 후보 (반경 밖 앵커 잔류 허용)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seq in args.seqs or ["box_grab_01"]:
        run(seq, args.subj, args.f_cap, dev, allow_baseline_start=args.allow_baseline_start)
