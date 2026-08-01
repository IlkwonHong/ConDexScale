"""물체 패치 캐시 생성 — 표면점 2048개를 패치 128개로 분해해 저장.

물체 캐시(data/contraction_gt_pilot/object_cache, 단위 스케일·안쪽 법선)에서:
  - FPS(최원점 샘플링, 결정적: 0번 점 시작)로 패치 중심 128개 선정
  - 각 표면점을 최근접 중심에 귀속 (point_patch_id)
  - 패치 법선 = 소속점 법선의 평균(정규화, 안쪽 유지),
    곡률 = 1 − ‖평균 법선‖ (법선 분산 근사 — 구 DexYCB 패치 캐시와 같은 정의)
  - 패치 중심 간 k=6 최근접 이웃 엣지

출력: data/contraction_net/patch_cache/<dataset>/<물체ID>.npz
실행: ./venv/bin/python -m contraction_net.build_patch_cache [workers]
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "contraction_gt_pilot" / "object_cache"
DST = ROOT / "data" / "contraction_net" / "patch_cache"
N_PATCH = 128
KNN = 6


def fps(points, k):
    sel = [0]                                    # 결정적: 0번 점에서 시작
    d = np.linalg.norm(points - points[0], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.array(sel)


def build_one(src_path):
    rel = src_path.relative_to(SRC)
    out = DST / rel
    if out.exists():
        return 0
    with np.load(src_path) as z:
        pts, nrm = z["points"], z["normals"]
    idx = fps(pts, N_PATCH)
    centers = pts[idx]
    d2 = ((pts[:, None] - centers[None]) ** 2).sum(-1)      # [2048,128]
    pid = d2.argmin(1).astype(np.int16)
    normals = np.zeros((N_PATCH, 3))
    curv = np.zeros(N_PATCH)
    for p in range(N_PATCH):
        m = pid == p
        mean = nrm[m].mean(0) if m.any() else nrm[idx[p]]
        n = np.linalg.norm(mean)
        normals[p] = mean / max(n, 1e-9)
        curv[p] = 1.0 - n
    cd = ((centers[:, None] - centers[None]) ** 2).sum(-1)
    np.fill_diagonal(cd, np.inf)
    nn = cd.argsort(1)[:, :KNN]
    edge_src = np.repeat(np.arange(N_PATCH), KNN).astype(np.int16)
    edge_dst = nn.reshape(-1).astype(np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, centers=centers.astype(np.float32),
                        normals=normals.astype(np.float32),
                        curvature=curv.astype(np.float32),
                        center_idx=idx.astype(np.int16), point_patch_id=pid,
                        edge_src=edge_src, edge_dst=edge_dst)
    return 1


def main(workers=15, n_patch=None, dst_name=None):
    global N_PATCH, DST
    if n_patch:
        N_PATCH = n_patch          # fork 워커가 전역을 상속
    if dst_name:
        DST = ROOT / "data" / "contraction_net" / dst_name
    srcs = sorted(SRC.glob("*/*.npz"))
    print(f"물체 {len(srcs):,}개, 패치 {N_PATCH}, workers={workers} → {DST}", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        n_new = sum(pool.imap_unordered(build_one, srcs, chunksize=64))
    print(f"완료: 신규 {n_new:,} (기존 {len(srcs)-n_new:,}), {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15,
         int(sys.argv[2]) if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
