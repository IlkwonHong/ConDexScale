"""박스 들기 앵커의 게이트 시뮬 검산 — LP 주장 α의 25%(base)/50%(margin) 하중 실물 시험.

box_lift_probe 가 뽑은 마스크 제한 최적 앵커(키프레임 3개 × N=2/3a/3b/4)를
데이터셋 게이트와 동일 프로토콜(_worker: 예압 평형 → 방향별 하중 유지)로 시험.
메시: coacd 미설치 → 부품 2개(트레이·뚜껑)를 그대로 내보내 MuJoCo 가 부품별
convex hull 로 취급 (뚜껑 닫힘 + 외면 접촉만이므로 타당한 근사).
N=2 는 팁 패드 0.10r/0.05r 두 종 시험 (패드 민감도 실물 확인).
실행: ./venv/bin/python scratchpad/box_gate_sim.py   (spawn+가드 — 08-02 OOM 교훈)
"""
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))

import box_lift_probe as bp
from contraction_gt.search import mu_tor_from_pad
from contraction_gt.metric import direction_set
from contraction_gt.pilot_sim import _equilibrium_preload
from contraction_gt.gate_runner import _worker

GAMMAS = {"base": 0.25, "margin": 0.50}
FLOOR = 0.01
MESH_OUT = ROOT / "scratchpad/box_meshes"
_MDIR = ROOT / "extlib/dexmachina/dexmachina/assets/arctic/box"
PARTS = ["bottom_watertight_tiny_box.obj", "top_watertight_tiny_box.obj"]


def export_parts(fidx):
    """키프레임의 세계 정렬 프레임으로 부품 2개를 내보내고 디렉토리 반환."""
    j = json.load(open(ROOT / "data/results_arctic/s01/box_grab_01/Robotiq/mu0.9"
                       / f"{fidx}_Robotiq_0_mu_0.9.json"))
    g = json.load(open(ROOT / "data/arctic_cache/sample_graphs"
                       / f"{j['file_name']}_graph.json"))
    R = np.array(g["obj_rotation"], dtype=np.float64)
    cen = np.array(g["canonical_centroid"], dtype=np.float64)
    out = MESH_OUT / f"kf{fidx}"
    out.mkdir(parents=True, exist_ok=True)
    for name in PARTS:
        m = trimesh.load(_MDIR / name, force="mesh")
        m = trimesh.Trimesh(vertices=(R @ (m.vertices - cen).T).T,
                            faces=m.faces, process=False)
        m.export(out / name)
    return out


def main():
    jobs, meta = [], []
    for fidx in (3, 10, 21):
        mdir = export_parts(fidx)
        smp, contacts, _ = bp.load_frame(fidx)
        parts = bp.partitions_of(contacts)
        configs = [(n, 0.10) for n in parts] + [("N=2  [좌손|우손]", 0.05)]
        for name, pad in configs:
            groups = parts[name]
            masks, init = bp.masks_for(smp, groups, contacts)
            bp.MU_T = 0.6
            best = bp.coordinate_ascent(smp, masks, init, pad)
            a14 = bp.alpha_of(smp, [best], pad)[0]
            pos, nrm = smp.points[best], smp.normals[best]
            pos_n = torch.as_tensor((pos - smp.com) / smp.r_obj,
                                    dtype=torch.float64).unsqueeze(0)
            dirs_all = direction_set(pos_n)[0].numpy()
            dsel = sorted(set(np.argsort(a14)[:3].tolist()) | set(range(6)))
            mu_tor = mu_tor_from_pad(pad * smp.r_obj, bp.MU_T, 0.5)
            preload, _x, _w = _equilibrium_preload(
                (pos - smp.com) / smp.r_obj, nrm, bp.MU_T, mu_tor / smp.r_obj, 5.0)
            for kind, gam in GAMMAS.items():
                levels = np.maximum(a14[dsel] * gam, FLOOR)[:, None]
                jobs.append(dict(
                    idx=len(jobs), coacd=str(mdir), scale=1.0,
                    pos=pos, nrm=nrm, dirs=dirs_all[dsel], levels=levels,
                    r_obj=float(smp.r_obj), mu_tor=float(mu_tor),
                    preload=preload, tip_radius=max(pad * smp.r_obj, 0.002)))
                meta.append(dict(kf=fidx, name=name, pad=pad, kind=kind,
                                 min_a=float(np.nanmin(a14))))

    print(f"\n시뮬 작업 {len(jobs)}개 → 6워커", flush=True)
    t0 = time.time()
    verd = {}
    with get_context("spawn").Pool(6) as pool:
        for k, (idx, v, _nu) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            verd[idx] = v
            print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== 게이트 시뮬 결과 (LP 주장 대비 실물 유지 여부) ===")
    rows = {}
    for jm, m in enumerate(meta):
        rows.setdefault((m["kf"], m["name"], m["pad"], m["min_a"]), {})[m["kind"]] = verd.get(jm)
    print(f"{'키프레임':>6} {'배치':34s} {'패드':>5} {'min-α':>6}  base(25%)  margin(50%)")
    for (kf, name, pad, ma), r in rows.items():
        print(f"{kf:>6} {name:34s} {pad:>5.2f} {ma:>6.3f}  {r.get('base','?'):>9s}  {r.get('margin','?'):>10s}")


if __name__ == "__main__":
    main()
