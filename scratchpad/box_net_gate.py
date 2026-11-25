"""네트워크 (2,2) 앵커의 게이트 시뮬 검산 + 모서리 근접 진단 (박스).

각 키프레임: 네트워크 손별 2팔 앵커 4개 → ① 날카로운 모서리까지 거리 표
② MuJoCo 게이트 (주장 α×25%/50%) — 튐(정착실패/불안정) 여부 확인.
실행: ./venv/bin/python scratchpad/box_net_gate.py
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "box_grab_01"]          # box_net_eval 이 argv 를 읽으므로 고정

import box_net_eval as E
from arctic_scene import scene
from contraction_gt.surface import sample_surface
from contraction_gt.search import mu_tor_from_pad
from contraction_gt.metric import direction_set
from contraction_gt.pilot_sim import _equilibrium_preload
from contraction_gt.gate_runner import _worker
from contraction_net.partition_vocab import PartitionVocab
from contraction_net.stage2_model import Stage2Net

FRAMES = [90, 300, 630]
GAMMAS = {"base": 0.25, "margin": 0.50}
FLOOR = 0.01
MESH_OUT = ROOT / "scratchpad/net_gate_meshes"


def sharp_edges(mesh, deg=40.0):
    """이면각이 deg 이상 꺾이는 모서리 선분들 [E,2,3]."""
    ang = mesh.face_adjacency_angles
    sel = ang > np.deg2rad(deg)
    ve = mesh.face_adjacency_edges[sel]
    return mesh.vertices[ve]


def dist_to_segments(p, segs):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1).clip(1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(p - proj, axis=1).min())


def main():
    vocab = PartitionVocab.load()
    ck = torch.load(ROOT / "checkpoints/stage2_featnorm/best.pt",
                    map_location=E.DEV, weights_only=False)
    model = Stage2Net(vocab, d_pf=11, d_ptf=10).to(E.DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    batches = torch.load(ROOT / "scratchpad/net_batches_box_grab_01.pt",
                         weights_only=False)

    jobs, meta = [], []
    for fr in FRAMES:
        smp, m = scene("box_grab_01", fr, sample_surface)
        mesh_w = trimesh.Trimesh(
            vertices=smp.points, faces=[], process=False)   # 자리표시
        # 모서리는 원본 메시에서
        from arctic_scene import load_template
        V, F, parts = load_template("box")
        Vw = (m["R"] @ (V - m["cen"]).T).T
        segs = sharp_edges(trimesh.Trimesh(vertices=Vw, faces=F, process=False))
        mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, E.MU_T, 0.5)

        cand_sets, blocks_all = [], []
        for side in ("left", "right"):
            b = batches[f"{fr}_{side}"]
            blocks, cands = E.hand_candidates(model, vocab, b, smp, 2)
            cand_sets += cands
            blocks_all.append(blocks)
        idx, a14 = E.best_combo(smp, cand_sets, mu_tor)
        print(f"\n[{fr:05d}] min-α={np.nanmin(a14):.3f}  파티션 좌{blocks_all[0]} 우{blocks_all[1]}")
        for k, i in enumerate(idx):
            d = dist_to_segments(smp.points[i], segs)
            print(f"  앵커{k}: 모서리까지 {d*100:.1f}cm"
                  + ("  ◀ 모서리 근접(<팁반경 2.2cm)" if d < 0.10 * smp.r_obj else ""))

        # 시뮬 부품 내보내기 (부품별 볼록 껍질용 — 원본 두 파트)
        out = MESH_OUT / f"kf{fr}"
        out.mkdir(parents=True, exist_ok=True)
        for tag, sel in [("bottom", parts != 0), ("top", parts == 0)]:
            vm = np.zeros(len(V), dtype=bool)
            vm[sel] = True
            fs = F[np.all(vm[F], axis=1)]
            used = np.unique(fs)
            remap = np.zeros(len(V), np.int64)
            remap[used] = np.arange(len(used))
            trimesh.Trimesh(vertices=Vw[used], faces=remap[fs],
                            process=False).export(out / f"{tag}.obj")

        pos, nrm = smp.points[idx], smp.normals[idx]
        pos_n = torch.as_tensor((pos - smp.com) / smp.r_obj,
                                dtype=torch.float64).unsqueeze(0)
        dirs_all = direction_set(pos_n)[0].numpy()
        dsel = sorted(set(np.argsort(a14)[:3].tolist()) | set(range(6)))
        preload, _x, _w = _equilibrium_preload(
            (pos - smp.com) / smp.r_obj, nrm, E.MU_T, mu_tor / smp.r_obj, 5.0)
        for kind, gam in GAMMAS.items():
            levels = np.maximum(a14[dsel] * gam, FLOOR)[:, None]
            jobs.append(dict(idx=len(jobs), coacd=str(out), scale=1.0,
                             pos=pos, nrm=nrm, dirs=dirs_all[dsel], levels=levels,
                             r_obj=float(smp.r_obj), mu_tor=float(mu_tor),
                             preload=preload, tip_radius=0.10 * smp.r_obj))
            meta.append((fr, kind, float(np.nanmin(a14))))

    print(f"\n게이트 시뮬 {len(jobs)}개 → 4워커")
    verd = {}
    with get_context("spawn").Pool(4) as pool:
        for i, v, _nu in pool.imap_unordered(_worker, jobs):
            verd[i] = v
    print("\n=== 네트워크 앵커 게이트 판정 ===")
    for i, (fr, kind, ma) in enumerate(meta):
        print(f"  {fr:05d} {kind:6s} (주장 {ma:.3f}) → {verd.get(i)}")


if __name__ == "__main__":
    main()
