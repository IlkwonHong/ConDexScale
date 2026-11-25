"""대조 실험: 00300에서 수학 탐색 (2,2) 앵커를 같은 메시로 게이트 시뮬.

네트워크 앵커의 fail_settle 이 앵커 배치 탓인지 메시 출처 탓인지 분리.
실행: ./venv/bin/python scratchpad/box_net_gate_ctrl.py
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "box_grab_01"]

import box_net_eval as E
from box_net_gate import sharp_edges, dist_to_segments
from arctic_scene import scene, load_template
from contraction_gt.surface import sample_surface
from contraction_gt.search import mu_tor_from_pad
from contraction_gt.metric import direction_set
from contraction_gt.pilot_sim import _equilibrium_preload
from contraction_gt.gate_runner import _worker
from contraction_net.partition_vocab import PartitionVocab
from contraction_net.stage2_model import Stage2Net
import trimesh

FR = 300
GAMMAS = {"base": 0.25, "margin": 0.50}
FLOOR = 0.01


def main():
    vocab = PartitionVocab.load()
    ck = torch.load(ROOT / "checkpoints/stage2_featnorm/best.pt",
                    map_location=E.DEV, weights_only=False)
    model = Stage2Net(vocab, d_pf=11, d_ptf=10).to(E.DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    batches = torch.load(ROOT / "scratchpad/net_batches_box_grab_01.pt",
                         weights_only=False)
    smp, m = scene("box_grab_01", FR, sample_surface)
    mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, E.MU_T, 0.5)

    h22 = {}
    for side in ("left", "right"):
        b = batches[f"{FR}_{side}"]
        h22[side] = E.hand_candidates(model, vocab, b, smp, 2)
        h22[side + "_batch"] = b
    ma, idx, a14 = E.math_search(smp, h22, mu_tor)

    V, F_, parts = load_template("box")
    Vw = (m["R"] @ (V - m["cen"]).T).T
    segs = sharp_edges(trimesh.Trimesh(vertices=Vw, faces=F_, process=False))
    print(f"수학 탐색 (2,2) @{FR:05d}: min-α={ma:.3f}")
    for k, i in enumerate(idx):
        print(f"  앵커{k}: 모서리까지 {dist_to_segments(smp.points[i], segs)*100:.1f}cm")

    out = ROOT / "scratchpad/net_gate_meshes" / f"kf{FR}"   # 게이트와 같은 부품 재사용
    pos, nrm = smp.points[idx], smp.normals[idx]
    pos_n = torch.as_tensor((pos - smp.com) / smp.r_obj,
                            dtype=torch.float64).unsqueeze(0)
    dirs_all = direction_set(pos_n)[0].numpy()
    dsel = sorted(set(np.argsort(a14)[:3].tolist()) | set(range(6)))
    preload, _x, _w = _equilibrium_preload(
        (pos - smp.com) / smp.r_obj, nrm, E.MU_T, mu_tor / smp.r_obj, 5.0)
    jobs = []
    for kind, gam in GAMMAS.items():
        levels = np.maximum(a14[dsel] * gam, FLOOR)[:, None]
        jobs.append(dict(idx=len(jobs), coacd=str(out), scale=1.0, pos=pos, nrm=nrm,
                         dirs=dirs_all[dsel], levels=levels, r_obj=float(smp.r_obj),
                         mu_tor=float(mu_tor), preload=preload,
                         tip_radius=0.10 * smp.r_obj))
    with get_context("spawn").Pool(2) as pool:
        for i, v, _nu in pool.imap_unordered(_worker, jobs):
            print(f"  {list(GAMMAS)[i]:6s} → {v}")


if __name__ == "__main__":
    main()
