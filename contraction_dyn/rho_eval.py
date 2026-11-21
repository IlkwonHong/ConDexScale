"""앵커 시간표의 ℓ∞ 강건성 여유 ρ 재채점 — 논문 III-B 정식화의 평가판.

기존 14방향 α (명목 렌치 0, 축별 외란 프로브)와 달리, 논문의 ρ 는
  ① 명목 렌치 w_req (준정적: 들림 게이트 × m·g·ẑ 중력 몫) 실현을 전제로
  ② 그 주변 가중 ℓ∞ 박스 정점 64개(w_req + ρ·D·v)가 전부 렌치 집합 W_c 에
     들어가는 최대 ρ 를 구한다 — 축 결합 외란까지 커버하는 보수적(2-노름
     구 포함) 강건성 여유.

시간표 생성(anchors.py)은 건드리지 않는다 — 산출된 스케줄 JSON 을 읽어
같은 표면 표본·같은 프레임 표집으로 재채점만 한다.

단위: ρ 는 정규화 단위 (앵커당 법선력 상한 f_cap 배율, 토크 지렛대 r_obj).
물리 환산 = 힘 ρ·f_cap [N], 토크 ρ·f_cap·r_obj [N·m].

토크 기준점 = URDF 질량중심(관절각 반영, 2026-08-30) — 이 기준에서 중력 토크가
정확히 0 이라 w_req 토크 = 0 이 근사가 아닌 정확식. 남는 오차는 URDF 모델 자체와
실물의 차이뿐 (저울질 = 질량·중심 역산으로 상환 예정).

실행: ./venv/bin/python -m contraction_dyn.rho_eval box_grab_01 [box_use_02 ...]
      [--f-cap 30] [--mass 자동(URDF)] [--force-only]
출력: data/contraction_dyn/rho_eval/{subj}__{seq}.json + 표
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_gt.metric import (compute_rho, vertex_direction_set,
                                   ray_wrench_matrix, alpha_batched)
from contraction_gt.search import mu_tor_from_pad
from contraction_dyn.anchors import canonical_smp, MU_T, PAD_REL, N_SAMPLE_F
from contraction_dyn.hand_contacts import SeqData

G = 9.81
GATE_H = 0.02           # 들림 게이트 램프 폭 (시뮬 무게 전방힘 게이트와 동일 2cm)
OUT = ROOT / "data/contraction_dyn/rho_eval"
URDF = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box/decomp/box_decomp.urdf"


def urdf_inertials(path=URDF):
    """링크별 (질량, 질량중심 xyz) — 시뮬과 같은 출처.

    좌표계 검증 (2026-08-30): decomp obj bbox = canonical 부품 bbox (단위 m,
    경첩 = 원점·−z 축) → URDF 관성 원점을 canonical 좌표로 그대로 사용 가능.
    """
    root = ET.parse(path).getroot()
    out = {}
    for lk in root.findall("link"):
        ine = lk.find("inertial")
        out[lk.get("name")] = (
            float(ine.find("mass").get("value")),
            np.array([float(x) for x in ine.find("origin").get("xyz").split()]))
    return out


def mass_com(inertials, theta):
    """관절각 theta 의 합성 질량중심 (canonical frame).

    top 링크(가동부, parts==0)는 canonical_smp 와 동일한 경첩 변환
    (원점 기준 rotvec [0,0,−θ])으로 회전시킨다.
    """
    from scipy.spatial.transform import Rotation
    mb, cb = inertials["bottom"]
    mt, ct = inertials["top"]
    if abs(theta) > 1e-8:
        ct = Rotation.from_rotvec([0, 0, -theta]).as_matrix() @ ct
    return (mb * cb + mt * ct) / (mb + mt)


def vertex_sig(v):
    s = lambda x: "+" if x > 0 else ("-" if x < 0 else "0")
    f, t = "".join(s(x) for x in v[:3]), "".join(s(x) for x in v[3:])
    return f"f({f})τ({t})"


def eval_seq(seq, subj, mass, f_cap, force_only, device, inertials=None):
    sched = json.load(open(ROOT / "data/contraction_dyn/anchor_schedule"
                           / f"{subj}__{seq}.json"))
    sd = SeqData(subj, seq)
    smp_cache = {}

    def smp_for(theta):
        key = round(theta, 1)
        if key not in smp_cache:
            smp_cache[key] = canonical_smp(sd, key)
        return smp_cache[key]

    # 정지 높이 (anchors.py 와 동일한 정의) — 들림 게이트의 기준
    zs = []
    for f in range(0, len(sd.obj), 10):
        R, t, th, v, c = sd.obj_frame(f)
        zs.append((R @ c + t)[2])
    rest_z = float(np.percentile(zs, 5))

    verts = vertex_direction_set(torch.float64, device, force_only).cpu().numpy()
    rows_out = []
    print(f"--- {subj}/{seq}  질량 {mass:.3f}kg (mg={mass*G:.2f}N), "
          f"f_cap {f_cap:.0f}N, 정점 {len(verts)}개 ---")
    for row in sched["rows"]:
        if not row.get("anchors"):
            rows_out.append(dict(seg=row["seg"], start=row["start"],
                                 end=row["end"], rho=None, note="앵커 없음"))
            continue
        s, e = row["start"], row["end"]
        frames = sorted(set(np.linspace(s, e - 1, min(N_SAMPLE_F, e - s))
                            .round().astype(int).tolist()))
        smp = smp_for(row["theta"])
        mu_tor = mu_tor_from_pad(PAD_REL * smp.r_obj, MU_T, 0.5)
        # 토크 기준점: URDF 질량중심(관절각 반영) — 중력 토크가 정확히 0 이 되는
        # 유일한 기준. 없으면 부피 중심 폴백 (box 실측 편차 1.66cm).
        com_ref = (mass_com(inertials, smp.theta) - smp.cen) if inertials \
            else smp.com

        # 저장된 앵커 → 표면 표본 인덱스 (idx 검증, 어긋나면 pos 최근접 폴백)
        idxs = []
        for arm, a in row["anchors"].items():
            i = int(a["idx"])
            p = np.asarray(a["pos"], dtype=np.float64)
            if np.linalg.norm(smp.points[i] - p) > 1e-3:
                i = int(np.linalg.norm(smp.points - p, axis=1).argmin())
            idxs.append(i)
        idxs = np.asarray(idxs)

        # 프레임 배치: world 정렬 자세 + 들림 게이트 중력 몫
        pos_f, nrm_f, com_f, wreq_f, gates = [], [], [], [], []
        for f in frames:
            R, t, th, v, c = sd.obj_frame(f)
            lift = (R @ c + t)[2] - rest_z
            gate = float(np.clip(lift / GATE_H, 0.0, 1.0))
            pos_f.append(smp.points[idxs] @ R.T)
            nrm_f.append(smp.normals[idxs] @ R.T)
            com_f.append(R @ com_ref)
            wreq_f.append([0.0, 0.0, gate * mass * G, 0.0, 0.0, 0.0])
            gates.append(gate)
        td = dict(dtype=torch.float64, device=device)
        out = compute_rho(
            torch.tensor(np.stack(pos_f), **td), torch.tensor(np.stack(nrm_f), **td),
            torch.tensor(wreq_f, **td), com=torch.tensor(np.stack(com_f), **td),
            r_obj=smp.r_obj, f_cap=f_cap, mu_t=MU_T, mu_tor=mu_tor,
            force_only=force_only, backend="batched")

        # 방향 불문(보수) 채점 — 자세 표집에 의존하지 않는 인증.
        # 중력 렌치는 어떤 자세든 크기 mg 의 순수 힘(질량중심 기준 토크 0)이므로
        # 중력 몫은 힘 축에만 부과한다. 항상 들려 있다고 가정(게이트 없음, 최보수).
        # 음수 = 인증 불가(실패의 증명은 아님). λ = mg/f_cap.
        #   8정점(힘 외란만): 같은 반직선이라 뺄셈이 정확 — r = ρ0(힘만) − λ.
        #   64정점(힘+토크 외란): 박스 [힘 λ+r, 토크 r] 의 꼭짓점이 λ·v_f + r·v
        #   이므로 정점별 w_req = λ·v_f 로 LP 를 풀어 r = min_v t_v.
        #   (ρ0 − λ 일괄 뺄셈은 중력이 안 쓰는 토크 예산까지 깎는 과잉 — 폐기)
        lam = mass * G / f_cap
        pos0 = torch.tensor(smp.points[idxs][None], **td)
        nrm0 = torch.tensor(smp.normals[idxs][None], **td)
        com0 = torch.tensor(com_ref[None], **td)
        o8 = compute_rho(pos0, nrm0, torch.zeros(1, 6, **td), com=com0,
                         r_obj=smp.r_obj, f_cap=f_cap, mu_t=MU_T, mu_tor=mu_tor,
                         force_only=True, backend="batched")
        anyg = {True: float(o8["rho"][0]) - lam}
        if anyg[True] < 0:
            anyg[False] = anyg[True]   # 힘 코너부터 비가능 — 정점 LP 전제 미충족
        else:
            V = vertex_direction_set(torch.float64, device)
            vf = V.clone()
            vf[:, 3:] = 0.0
            W0 = ray_wrench_matrix((pos0 - com0[:, None]) / smp.r_obj, nrm0,
                                   torch.zeros(1, 3, **td), 1.0, MU_T,
                                   mu_tor / smp.r_obj)
            t64 = alpha_batched(W0.expand(len(V), *W0.shape[1:]),
                                -V.unsqueeze(1), w_req=lam * vf)
            anyg[False] = float(t64.min())

        rho = out["rho"].cpu().numpy()
        fi = int(rho.argmin())
        feas_all = bool(out["feasible"].all())
        weak = verts[int(out["weak_vertex"][fi])] if out["feasible"][fi] else None
        rows_out.append(dict(
            seg=row["seg"], start=s, end=e, hand=row.get("hand"),
            n_arms=len(idxs), gate=[round(g, 2) for g in gates],
            alpha_min=row.get("min_alpha"), alpha_worst=row.get("worst_alpha"),
            feasible=feas_all,
            rho_worst=round(float(rho.min()), 4),
            rho_frames=[round(float(x), 4) for x in rho],
            rho_worst_N=round(float(rho.min()) * f_cap, 2),
            wreq_norm=round(float(out["wreq_norm"].max()), 4),
            weak_vertex=vertex_sig(weak) if weak is not None else "비가능",
            rho_anyg_64=round(anyg[False], 4),
            rho_anyg_8=round(anyg[True], 4),
        ))
        r = rows_out[-1]
        flag = "" if feas_all and r["rho_worst"] > 0 else "  ◀ 주의"
        print(f"[{r['seg']:2d}] {s:4d}-{e:4d} 팔{r['n_arms']} "
              f"게이트{min(gates):.1f}~{max(gates):.1f} "
              f"α_worst={r['alpha_worst']:.3f} → ρ={r['rho_worst']:.3f} "
              f"({r['rho_worst_N']:.1f}N) 최약 {r['weak_vertex']} "
              f"| 방향불문 64:{anyg[False]:+.3f} 8:{anyg[True]:+.3f}{flag}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{subj}__{seq}.json"
    json.dump(dict(seq=seq, subject=subj, mass=mass, f_cap=f_cap, gate_h=GATE_H,
                   force_only=force_only, n_vertices=len(verts), rows=rows_out),
              open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=["box_grab_01"])
    ap.add_argument("--subj", default="s01")
    ap.add_argument("--mass", type=float, default=None,
                    help="물체 질량 kg (기본: box URDF 링크 합)")
    ap.add_argument("--f-cap", type=float, default=30.0,
                    help="앵커당 법선력 상한 N (시뮬 F_MAX)")
    ap.add_argument("--force-only", action="store_true",
                    help="힘 정점 8개만 (기본: 6D 정점 64개)")
    args = ap.parse_args()
    iner = urdf_inertials()
    mass = args.mass if args.mass is not None else sum(m for m, _ in iner.values())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seq in args.seqs or ["box_grab_01"]:
        eval_seq(seq, args.subj, mass, args.f_cap, args.force_only, dev,
                 inertials=iner)
