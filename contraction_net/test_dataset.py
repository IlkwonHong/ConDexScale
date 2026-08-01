"""ContractionGTDataset 검증 — dev 샘플 전수.

실행: ./venv/bin/python -m contraction_net.test_dataset [records_dir]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from contraction_net.dataset import ContractionGTDataset, MAX_ANCHORS, UNITS

RECORDS = Path(sys.argv[1] if len(sys.argv) > 1 else
               "data/contraction_gt_dev/search_records")


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise AssertionError(name)


def main():
    ds = ContractionGTDataset(RECORDS)
    print(f"레코드 {len(ds)}개, 타입 {len(ds.type_vocab)}종")

    # ── 전수 무결성 ──
    t0 = time.time()
    n_trusted = 0
    surf_dist_max = 0.0
    inward_ok = 0
    inward_tot = 0
    for i in range(len(ds)):
        s = ds[i]
        n = int(s["n_anchors"])
        for k, v in s.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                assert not torch.isnan(v).any(), f"[{i}] {k} 에 NaN"
        assert s["anchor_mask"].sum() == n
        assert (s["anchor_pos"][n:] == 0).all()
        groups = set(s["unit_group"][s["unit_group"] >= 0].tolist())
        assert groups == set(range(n)), f"[{i}] 그룹 {groups} != 0..{n-1}"
        assert abs(float(s["alpha"].min()) - float(s["min_alpha"])) < 1e-9
        # 앵커는 물체 표면점에서 뽑힘 → 표면과의 거리 ~0 이어야 함
        d = torch.cdist(s["anchor_pos"][:n], s["obj_points"]).min(dim=1).values
        surf_dist_max = max(surf_dist_max, float(d.max()))
        # 접촉 법선(반전 후)은 그 지점의 물체 캐시 법선(안쪽 기준)과 같은 방향이어야
        # 한다. 표면점 2048개는 간격이 ~5mm 라, 최근접점이 2mm 이내로 확실한
        # 접촉만 판정한다 (얇은 벽에서 반대면에 잡히는 기하 잡음 배제).
        m = s["contact_mask"]
        cp, cn = s["contact_pos"][m], s["contact_normal"][m]
        d = torch.cdist(cp, s["obj_points"])
        nn_d, nn = d.min(dim=1)
        sure = nn_d < 0.002
        dot = (cn[sure] * s["obj_normals"][nn[sure]]).sum(-1)
        inward_ok += int((dot > 0).sum())
        inward_tot += int(sure.sum())
        n_trusted += int(s["trusted"])
    dt = time.time() - t0
    print(f"전수 통과 ({dt:.1f}s, {len(ds)/dt:.0f} 레코드/s)")
    check(f"앵커-표면 최대거리 {surf_dist_max*1000:.3f}mm < 0.1mm", surf_dist_max < 1e-4)
    check(f"접촉 법선-물체 법선 일치 {inward_ok/inward_tot:.1%} > 95% "
          f"(판정 가능 {inward_tot}건)", inward_ok / inward_tot > 0.95)
    print(f"  신뢰 등급 레코드 {n_trusted}개 (전체의 {n_trusted/len(ds):.1%})")

    # ── 필터 ──
    f1 = ContractionGTDataset(RECORDS, min_alpha=0.1)
    check("min_alpha 필터", len(f1) < len(ds)
          and all(ds.rec["min_alpha"][f1.idx] >= 0.1))
    f2 = ContractionGTDataset(RECORDS, n_anchors=(2,))
    check("n_anchors 필터", all(ds.rec["n_anchors"][f2.idx] == 2))

    # ── train/val 분리: 물체 단위로 서로소 ──
    tr = ContractionGTDataset(RECORDS, split="train", val_fraction=0.1)
    va = ContractionGTDataset(RECORDS, split="val", val_fraction=0.1)
    obj = lambda d: {d.obj_of[g][1] for g in d.rec["grasp_id"][d.idx]}
    check(f"분리 서로소 (train {len(tr)} / val {len(va)})",
          len(tr) + len(va) == len(ds) and not (obj(tr) & obj(va)))

    # ── DataLoader 배치 (기본 collate, worker 2) ──
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    t0 = time.time()
    b = next(iter(dl))
    check("배치 형태", b["obj_points"].shape == (32, 2048, 3)
          and b["anchor_pos"].shape == (32, MAX_ANCHORS, 3)
          and b["unit_group"].shape == (32, len(UNITS))
          and len(b["grasp_id"]) == 32)
    print(f"  첫 배치 {time.time()-t0:.1f}s (worker 기동 포함)")

    # ── qpos 옵션 ──
    dq = ContractionGTDataset(RECORDS, load_qpos=True)
    check("qpos [29]", dq[0]["qpos"].shape == (29,))

    print("\n전부 통과")


if __name__ == "__main__":
    main()
