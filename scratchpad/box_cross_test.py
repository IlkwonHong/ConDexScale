"""교차 시험: N=4가 통과한 절대 하중(0.147)을 팔 2대 배치에 걸면 어떻게 되나.
게이트 하중이 '자기 주장 비례'임을 절대 하중으로 체감시키는 1회성 실측.
실행: ./venv/bin/python scratchpad/box_cross_test.py
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))

import box_lift_probe as bp
from box_gate_sim import export_parts
from contraction_gt.search import mu_tor_from_pad
from contraction_gt.metric import direction_set
from contraction_gt.pilot_sim import _equilibrium_preload
from contraction_gt.gate_runner import _worker


def main():
    mdir = export_parts(3)
    smp, contacts, _ = bp.load_frame(3)
    parts = bp.partitions_of(contacts)
    jobs, meta = [], []
    for name, lvl in [("N=2  [좌손|우손]", 0.147), ("N=2  [좌손|우손]", 0.0195),
                      ("N=4  [좌엄|좌4|우엄|우4]", 0.147)]:
        groups = parts[name]
        masks, init = bp.masks_for(smp, groups, contacts)
        best = bp.coordinate_ascent(smp, masks, init, 0.10)
        a14 = bp.alpha_of(smp, [best], 0.10)[0]
        pos, nrm = smp.points[best], smp.normals[best]
        pos_n = torch.as_tensor((pos - smp.com) / smp.r_obj,
                                dtype=torch.float64).unsqueeze(0)
        dirs_all = direction_set(pos_n)[0].numpy()
        dsel = sorted(set(np.argsort(a14)[:3].tolist()) | set(range(6)))
        mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, 0.6, 0.5)
        preload, _x, _w = _equilibrium_preload(
            (pos - smp.com) / smp.r_obj, nrm, 0.6, mu_tor / smp.r_obj, 5.0)
        jobs.append(dict(idx=len(jobs), coacd=str(mdir), scale=1.0, pos=pos, nrm=nrm,
                         dirs=dirs_all[dsel], levels=np.full((len(dsel), 1), lvl),
                         r_obj=float(smp.r_obj), mu_tor=float(mu_tor), preload=preload,
                         tip_radius=0.10 * smp.r_obj))
        meta.append((name, lvl, float(np.nanmin(a14))))
    with get_context("spawn").Pool(3) as pool:
        for idx, v, _nu in pool.imap_unordered(_worker, jobs):
            n, l, a = meta[idx]
            print(f"{n:28s} 주장 min-α={a:.3f}  절대 하중 {l:.4f} 전방향 → {v}",
                  flush=True)


if __name__ == "__main__":
    main()
