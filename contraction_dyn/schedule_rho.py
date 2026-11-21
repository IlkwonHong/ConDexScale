"""ρ 눌러앉기 시간표 — static_capacity 의 구간별 독립 튜닝 배치를 재생용 시간표로 엮기.

규칙 (그릴링 확정 2026-09-05, 설계 md_claude/plans/rho_continuous_control_v1.md Q2):
  * 첫 6팔 튜닝 배치로 시작.
  * 이후 구간마다 "현재 배치"를 그 구간의 표본 자세들로 재채점(air-hold worst-ρ, 만재 중력).
      유지 ρ ≥ max(RHO_FLOOR, STICKY × 새 튜닝 ρ)  → 유지
      아니면 (새 튜닝 배치가 6팔일 때만)             → 이동
  * 한 손 구간(3팔 튜닝, 비가능 또는 여유 ≈ 0)과 접촉 없는 구간은 자동으로 유지.
  * --never-move: 첫 배치를 끝까지 유지 (대조 실험 ③).
  * --always-move: 눌러앉기 규칙 없이 6팔 튜닝 배치가 있는 구간마다 무조건 교체 (사람 재파지 충실 추종).

출력: data/contraction_dyn/anchor_schedule_rho/{subj}__{seq}[__nomove].json
  rows[*].anchors[arm] = {idx, pos(canonical 중심), nrm_in, part, pos_part, nrm_part(바깥)}
  — compile_replay 계열이 읽는 anchor_schedule 형식과 호환 + rho_worst, decision.

실행: ./venv/bin/python -m contraction_dyn.schedule_rho box_grab_01 [--never-move]
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
from contraction_gt.metric import compute_rho   # noqa: E402

G = 9.81
RHO_FLOOR, STICKY = 0.15, 0.8
OUT = ROOT / "data/contraction_dyn/anchor_schedule_rho"


def worst_rho(row, plc, device):
    """배치 plc 를 구간 row 의 표본 자세들로 재채점한 air-hold worst-ρ (만재 중력)."""
    arms = sorted(plc["arms"])
    pos = np.array([plc["arms"][a]["pos"] for a in arms])
    nrm = np.array([plc["arms"][a]["nrm_in"] for a in arms])
    Rs = np.array(row["R_frames"])
    F = len(Rs)
    com = np.array(row["com_ref"])
    P = np.einsum("aj,fij->fai", pos, Rs)
    N = np.einsum("aj,fij->fai", nrm, Rs)
    C = np.einsum("fij,j->fi", Rs, com)
    w = np.zeros((F, 6))
    w[:, 2] = row["mass"] * G
    td = dict(dtype=torch.float64, device=device)
    out = compute_rho(torch.tensor(P, **td), torch.tensor(N, **td), torch.tensor(w, **td),
                      com=torch.tensor(C, **td), r_obj=row["r_obj"], f_cap=row["f_cap"],
                      mu_t=row["mu_t"], mu_tor=row["mu_tor"], backend="batched")
    return float(out["rho"].min())


def part_frame(arm_rec, cen, theta):
    """canonical(중심) 앵커 → (part, 부품 좌표 위치, 부품 좌표 바깥 법선). anchors.to_part_frame 와 동일."""
    v_art = np.asarray(arm_rec["pos"]) + np.asarray(cen)
    n_out = -np.asarray(arm_rec["nrm_in"])
    part = bool(arm_rec["part"])
    if part and abs(theta) > 1e-8:
        Rz = Rotation.from_rotvec([0, 0, theta]).as_matrix()
        return part, Rz @ v_art, Rz @ n_out
    return part, v_art, n_out


def placement_record(plc, cen, theta):
    out = {}
    for arm, rec in plc["arms"].items():
        part, q, nq = part_frame(rec, cen, theta)
        out[arm] = dict(idx=int(rec["idx"]),
                        pos=[round(float(x), 5) for x in rec["pos"]],
                        nrm_in=[round(float(x), 5) for x in rec["nrm_in"]],
                        part=int(part),
                        pos_part=[round(float(x), 5) for x in q],
                        nrm_part=[round(float(x), 5) for x in nq])
    return out


def run(seq, subj, never_move, device, always_move=False):
    cap = json.load(open(ROOT / "data/contraction_dyn/static_capacity" / f"{subj}__{seq}.json"))
    rows = cap["rows"]
    cur, cur_seg = None, None
    out_rows, n_move = [], 0
    print(f"--- {subj}/{seq} ρ 눌러앉기 시간표 (바닥 {RHO_FLOOR}, 유지 배율 {STICKY}"
          f"{', 이동 금지' if never_move else ''}{', 눌러앉기 없음(항상 교체)' if always_move else ''}) ---")
    for r in rows:
        t = r.get("tuned")
        six = bool(t) and len(t["arms"]) == 6
        if cur is None:
            if not six:
                out_rows.append(dict(seg=r["seg"], start=r["start"], end=r["end"],
                                     hand=r["hand"], theta=r["theta"], anchors=None,
                                     rho_worst=None, decision="앵커 없음"))
                print(f"[{r['seg']:2d}] {r['start']:4d}-{r['end']:4d} {r['hand']:5s} 앵커 없음")
                continue
            cur, cur_seg = t, r["seg"]
            rho, dec = t["rho_worst"], "신규"
        else:
            keep = worst_rho(r, cur, device)
            if six and not never_move and (
                    always_move or keep < max(RHO_FLOOR, STICKY * t["rho_worst"])):
                moved = [a for a in t["arms"]
                         if np.linalg.norm(np.array(t["arms"][a]["pos"])
                                           - np.array(cur["arms"][a]["pos"])) > 0.02]
                cur, cur_seg = t, r["seg"]
                rho, dec = t["rho_worst"], f"이동 {len(moved)}팔 {moved} (유지였다면 {keep:+.3f})"
                if not moved:
                    rho, dec = keep, "유지 (같은 배치)"
                else:
                    n_move += 1
            else:
                rho, dec = keep, f"유지 (구간 {cur_seg} 배치)"
                if six and never_move:
                    dec += f" [이동 금지; 튜닝 {t['rho_worst']:+.3f}]"
        out_rows.append(dict(seg=r["seg"], start=r["start"], end=r["end"], hand=r["hand"],
                             theta=r["theta"], anchors=placement_record(cur, r["cen"], r["theta"]),
                             source_seg=cur_seg, rho_worst=round(rho, 4), decision=dec,
                             mass=r["mass"], f_cap=r["f_cap"], r_obj=r["r_obj"],
                             mu_t=r["mu_t"], mu_tor=r["mu_tor"],
                             cen=r["cen"], com_ref=r["com_ref"]))
        print(f"[{r['seg']:2d}] {r['start']:4d}-{r['end']:4d} {r['hand']:5s} ρ={rho:+.3f}  {dec}")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "__nomove" if never_move else ("__always" if always_move else "")
    p = OUT / f"{subj}__{seq}{tag}.json"
    worst = min(x["rho_worst"] for x in out_rows if x["rho_worst"] is not None)
    json.dump(dict(seq=seq, subject=subj, rule=dict(rho_floor=RHO_FLOOR, sticky=STICKY,
                                                    never_move=never_move, always_move=always_move),
                   n_moves=n_move, rho_worst_all=round(worst, 4), rows=out_rows),
              open(p, "w"), ensure_ascii=False, indent=1)
    print(f"이동 {n_move}번, 전 구간 최악 ρ {worst:+.3f}  저장: {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=["box_grab_01"])
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--never-move", action="store_true")
    ap.add_argument("--always-move", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for s in args.seqs:
        run(s, args.subj, args.never_move, dev, always_move=args.always_move)
