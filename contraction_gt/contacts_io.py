"""접촉 테이블 v2 로더 — INTERFACE.md 명세 + D8 합의 수정 4건 반영.

수정 4건 (계획서 D8):
1. threshold 3mm 의 finger_contact 플래그 사용 (raw 거리도 함께 노출).
2. 법선 부호 반전: 테이블은 바깥 방향, 파이프라인 규약은 안쪽.
3. 스케일은 npz 의 total_scale 사용 (scene_scale 단독 금지).
4. link_min_dist < -0.01 이상치(비수밀 판정 실패) 링크는 접촉에서 제외.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .units import UnitPatch

FINGER_TO_UNIT = {"index": "index", "middle": "middle", "ring": "ring",
                  "little": "little", "thumb": "thumb", "palm": "palm"}


@lru_cache(maxsize=1)
def load_meta(cache_root):
    meta = json.load(open(Path(cache_root) / "meta.json"))
    return meta


def load_grasp_contacts(cache_root, rel_npy_path, row):
    """grasp 1건 -> ({unit: UnitPatch}, total_scale).

    UnitPatch.points 는 해당 유닛의 접촉 링크별 patch centroid (패치 범위 추정용),
    patch_radius 는 max(링크 centroid 퍼짐, sqrt(면적/π)) — palm 처럼 링크가
    하나여도 면적으로 패치 크기를 근사한다.
    """
    cache_root = Path(cache_root)
    meta = load_meta(str(cache_root))
    d = np.load(cache_root / rel_npy_path.replace(".npy", ".npz"))
    i = int(row)
    total_scale = float(d["total_scale"][i])

    link_ok = d["link_contact"][i] & (d["link_min_dist"][i] > -0.01)   # 수정 4
    link_unit = np.asarray(meta["link_finger_id"])
    fingers = meta["finger_names"]

    patches = {}
    for f_idx, fname in enumerate(fingers):
        if not bool(d["finger_contact"][i, f_idx]):
            continue
        lmask = link_ok & (link_unit == f_idx)
        if not lmask.any():
            continue
        pts = d["link_patch_centroid"][i][lmask].astype(np.float64)
        nrms = -d["link_contact_normal"][i][lmask].astype(np.float64)   # 수정 2
        areas = d["link_contact_area"][i][lmask].astype(np.float64)
        w = areas / areas.sum() if areas.sum() > 0 else np.full(len(areas), 1 / len(areas))
        centroid = d["finger_patch_centroid"][i, f_idx].astype(np.float64)
        mean_n = -d["finger_contact_normal"][i, f_idx].astype(np.float64)
        nn = np.linalg.norm(mean_n)
        mean_n = mean_n / nn if nn > 1e-12 else nrms[0]
        spread = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) else 0.0
        area_r = float(np.sqrt(max(areas.sum(), 0.0) / np.pi))
        patches[FINGER_TO_UNIT[fname]] = UnitPatch(
            unit=FINGER_TO_UNIT[fname], points=pts, normals=nrms, areas=areas,
            centroid=centroid, mean_normal=mean_n,
            total_area=float(areas.sum()),
            patch_radius=max(spread, area_r),
        )
    return patches, total_scale
