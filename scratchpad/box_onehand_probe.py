"""한 손 지지 구간의 contraction 유지력 검사 — box_grab_01, N=2/3/4.

한 손 접촉만으로 수축한 팔 배치가 14방향 하중(특히 중력·전도)을 버티는지.
프레임: 00180(오른손만), 00450(오른손만, 들림 13cm), 00570(왼손만, 들림 12cm).
방법: 부위별 접촉 정점 평균 → 파티션 전수(enumerate_valid) × 다중 시작 좌표 상승
      → 파티션별 최고 min-α. 판정선 = 절대 임계 0.15 (설계 확정본 Q6).
실행: ./venv/bin/python scratchpad/box_onehand_probe.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scratchpad"))
sys.path.insert(0, str(ROOT))

import box_lift_probe as bp
from arctic_scene import scene
from contraction_gt.surface import sample_surface
from contraction_net.partition_vocab import enumerate_valid

CASES = [(180, "right"), (450, "right"), (570, "left")]
UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]
DIRN = bp.DIR_NAMES
THR = 0.15


def best_for_partition(smp, part, contacts, rng):
    masks, init0 = bp.masks_for(smp, part, contacts)
    best = None
    for trial in range(4):
        init = init0 if trial == 0 else [int(rng.choice(c)) for c in masks]
        cur = bp.coordinate_ascent(smp, masks, init, 0.10)
        a14 = bp.alpha_of(smp, [cur], 0.10)[0]
        s = float(np.nanmin(a14))
        if best is None or s > best[0]:
            best = (s, a14, cur)
    return best


def main():
    bp.MU_T = 0.6
    con = np.load(ROOT / "scratchpad/arctic_link_contacts_box_grab_01.npz")
    for fr, side in CASES:
        smp, _m = scene("box_grab_01", fr, sample_surface)
        cpos = con[f"{fr}_{side}_cpos"]
        cunit = con[f"{fr}_{side}_cunit"]
        present = [UNITS[u] for u in range(6) if (cunit == u).any()]
        contacts = {UNITS[u]: cpos[cunit == u].mean(0)
                    for u in range(6) if (cunit == u).any()}
        a0 = "thumb" if "thumb" in present else present[0]
        rng = np.random.default_rng(0)
        print(f"\n=== 프레임 {fr:05d} ({side} 단독, 접촉 부위 {present}) ===")
        for n in (2, 3, 4):
            parts = ([[[a0], [u for u in present if u != a0]]] if n == 2
                     else enumerate_valid(present, a0, n))
            if not parts:
                print(f"  팔 {n}대: 유효 파티션 없음")
                continue
            best = None
            for part in parts:
                r = best_for_partition(smp, [list(b) for b in part], contacts, rng)
                if best is None or r[0] > best[0][0]:
                    best = (r, part)
            (s, a14, cur), part = best
            worst = int(np.nanargmin(a14))
            verdict = "통과" if s >= THR else ("임계 미달" if s > 0.03 else "낙제")
            print(f"  팔 {n}대: min-α={s:.3f} ({verdict})  중력α={a14[5]:.3f}  "
                  f"최약={DIRN[worst]}  파티션={[list(b) for b in part]}")


if __name__ == "__main__":
    main()
