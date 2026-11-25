"""하중 사다리 실측: 팔 2대/4대 배치가 시뮬에서 실제로 버티는 한계 용량.

LP 보증치(min-α)와 시뮬 실측 용량을 방향별로 나란히 비교한다.
measure_capacity_multi 의 levels 그리드(오름차순)가 방향별 실측 한계를 반환.
실행: ./venv/bin/python scratchpad/box_capacity_ladder.py
"""
import sys
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
from contraction_gt.sim_eval import SimConfig, measure_capacity_multi

LADDER = np.array([0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.90, 1.30])


def main():
    mdir = export_parts(3)
    smp, contacts, _ = bp.load_frame(3)
    parts = bp.partitions_of(contacts)
    for name in ("N=2  [좌손|우손]", "N=4  [좌엄|좌4|우엄|우4]"):
        groups = parts[name]
        masks, init = bp.masks_for(smp, groups, contacts)
        best = bp.coordinate_ascent(smp, masks, init, 0.10)
        a14 = bp.alpha_of(smp, [best], 0.10)[0]
        pos, nrm = smp.points[best], smp.normals[best]
        pos_n = torch.as_tensor((pos - smp.com) / smp.r_obj,
                                dtype=torch.float64).unsqueeze(0)
        dirs_all = direction_set(pos_n)[0].numpy()
        dsel = sorted(set(np.argsort(a14)[:3].tolist()) | {5})   # 최약 3 + 중력
        mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, 0.6, 0.5)
        preload, _x, _w = _equilibrium_preload(
            (pos - smp.com) / smp.r_obj, nrm, 0.6, mu_tor / smp.r_obj, 5.0)
        cfg = SimConfig(mu_tor=float(mu_tor), tip_radius=0.10 * smp.r_obj,
                        timestep=1e-3, settle_steps=600, hold_steps=300)
        levels = np.tile(LADDER, (len(dsel), 1))
        a_sim, settled, pre, nu = measure_capacity_multi(
            ("coacd", str(mdir), 1.0), pos, nrm, dirs_all[dsel], levels,
            float(smp.r_obj), cfg, preload_forces=preload, seed=0)
        print(f"\n{name}  (정착 {settled}, 예압 달성 {pre:.2f}, 불안정 방향 {nu})")
        for k, di in enumerate(dsel):
            print(f"  {bp.DIR_NAMES[di]:10s} LP 보증 α={a14[di]:.3f}  →  "
                  f"시뮬 실측 한계 {a_sim[k]:.2f}"
                  + ("  (사다리 최상단까지 버팀)" if a_sim[k] >= LADDER[-1] else ""))


if __name__ == "__main__":
    main()
