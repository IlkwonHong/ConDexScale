"""P2: 물체 표면 캐시 — 물체당 한 번 샘플링해 npz 로 저장, 사용 시 스케일만 곱한다.

핵심 상각 구조:
- 캐시는 스케일 1 기준으로 물체당 1개. grasp 마다 다른 scene_scale 은 로드
  시점에 점·외접반경·질량중심에 곱하기만 하면 된다 (법선은 스케일 불변).
  → 물체 1만 개 캐시가 grasp 950만 개 × 스케일 전부에 상각된다.
- 메쉬 해석: <asset_root>/<dataset>/processed_data/<물체ID>/mesh/simplified.obj
  (없으면 coacd 조각들을 합쳐서 사용). dataset 은 grasp 의 scene_path 에서 나온다.

실행 (캐시 일괄 빌드):
  ./venv/bin/python -m contraction_gt.object_cache <asset_root> <out_dir> <id_list.txt>
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

from .surface import SurfaceSamples, sample_surface

DATASETS = ("objaverse_5k", "DGN_5k")


def parse_scene_path(scene_path):
    """grasp 레코드의 scene_path -> (dataset, object_id).

    예: 'assets/object/objaverse_5k/scene_cfg/<id>/floating/scale008.npy'
    """
    parts = Path(scene_path).parts
    for ds in DATASETS:
        if ds in parts:
            i = parts.index(ds)
            return ds, parts[i + 2]   # scene_cfg 다음이 물체 ID
    raise ValueError(f"scene_path 에서 dataset 을 찾지 못함: {scene_path}")


def resolve_mesh_path(asset_root, dataset, object_id):
    base = Path(asset_root) / dataset / "processed_data" / object_id
    for rel in ("mesh/simplified.obj", "mesh/normalized.obj"):
        p = base / rel
        if p.exists():
            return p
    mesh_dir = base / "urdf" / "meshes"
    if mesh_dir.exists():
        return mesh_dir            # coacd 조각 디렉토리 → load_mesh 가 합침
    raise FileNotFoundError(f"메쉬 없음: {base}")


def load_mesh(path):
    path = Path(path)
    if path.is_dir():
        pieces = [trimesh.load(p, force="mesh") for p in sorted(path.glob("*.obj"))]
        return trimesh.util.concatenate(pieces)
    return trimesh.load(path, force="mesh")


def cache_path(cache_dir, dataset, object_id):
    return Path(cache_dir) / dataset / f"{object_id}.npz"


def build_one(asset_root, cache_dir, dataset, object_id, count=2048, seed=0):
    out = cache_path(cache_dir, dataset, object_id)
    if out.exists():
        return out
    mesh = load_mesh(resolve_mesh_path(asset_root, dataset, object_id))
    s = sample_surface(mesh, count=count, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, points=s.points, normals=s.normals,
                        r_obj=s.r_obj, com=s.com)
    return out


def load_samples(cache_dir, dataset, object_id, scale=1.0) -> SurfaceSamples:
    """캐시 로드 + per-grasp scene_scale 적용."""
    d = np.load(cache_path(cache_dir, dataset, object_id))
    return SurfaceSamples(
        points=d["points"] * scale,
        normals=d["normals"],
        r_obj=float(d["r_obj"]) * scale,
        com=d["com"] * scale,
    )


def build_many(asset_root, cache_dir, pairs, count=2048, log_every=200):
    """pairs = [(dataset, object_id), ...] 일괄 빌드. 실패는 모아서 보고."""
    failed = []
    for i, (ds, oid) in enumerate(pairs):
        try:
            build_one(asset_root, cache_dir, ds, oid, count=count)
        except Exception as e:
            failed.append((ds, oid, repr(e)))
        if (i + 1) % log_every == 0:
            print(f"{i + 1}/{len(pairs)} 완료 (실패 {len(failed)})", flush=True)
    return failed


if __name__ == "__main__":
    asset_root, out_dir, id_list = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = []
    for line in Path(id_list).read_text().splitlines():
        if line.strip():
            ds, oid = line.split()
            pairs.append((ds, oid))
    failed = build_many(asset_root, out_dir, pairs)
    print(f"총 {len(pairs)}개 중 실패 {len(failed)}개")
    for f in failed[:20]:
        print("  실패:", f)
