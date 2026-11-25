"""읽기 전용 집계: 물체 표면점을 128/256/512개로 줄였을 때 팔마다 탐색 범위 안에 후보가 몇 개 남는가.
선형계획·시뮬레이션 없음. 줄이는 방법은 서로 가장 먼 점을 차례로 고르는 방식(씨앗 0 고정)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path("/home/robin/a_workspace/condex")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scratchpad"))
import gt_seed_consistency as G
from contraction_gt.surface import _farthest_point_subsample
G.N_LIST = (3,)
GRIDS = (128, 256, 512)
grasps = G.load_grasps()
stored = G.load_stored({g for g, _, _ in grasps})
by_gid = {}
for k, v in stored.items():
    by_gid.setdefault(k[0], []).append(v)
cfg = G.SearchConfig()
cnt = {g: [] for g in GRIDS}; combos = {g: [] for g in GRIDS}; spacing = {g: [] for g in GRIDS}; snap = {g: [] for g in GRIDS}
full = []
for gid, ds, oid in grasps:
    rel, ridx = gid.rsplit("#", 1)
    patches, ts = G.load_grasp_contacts(str(ROOT / G.CONTACT_ROOT), rel, int(ridx))
    if len(patches) < 2: continue
    s = G.load_samples(ROOT / G.CACHE_ROOT, ds, oid, scale=ts)
    grid = {g: _farthest_point_subsample(s.points, g, seed=0) for g in GRIDS}
    for g in GRIDS:
        t = cKDTree(s.points[grid[g]])
        d, _ = t.query(s.points[grid[g]], k=2)
        spacing[g].append(np.median(d[:, 1]) / s.r_obj)
        for rec in by_gid.get(gid, []):
            dd, _ = t.query(rec["pos"])
            snap[g].extend((dd / s.r_obj).tolist())
    for part in G.enumerate_partitions(patches, 3):
        per = {g: [] for g in GRIDS}
        for group in part:
            pts = np.concatenate([patches[u].points for u in group])
            ar = np.concatenate([patches[u].areas for u in group])
            w = ar / ar.sum() if ar.sum() > 0 else None
            c = np.average(pts, axis=0, weights=w)
            nrm = np.average(np.concatenate([patches[u].normals for u in group]), axis=0, weights=w)
            rho = max(float(np.linalg.norm(pts - c, axis=1).max()), cfg.rho_rel * s.r_obj)
            mask = np.linalg.norm(s.points - c, axis=1) <= rho
            nm = s.normals @ nrm > 0.1
            if (mask & nm).any(): mask &= nm
            full.append(int(mask.sum()))
            for g in GRIDS:
                per[g].append(int(mask[grid[g]].sum()))
        for g in GRIDS:
            cnt[g].extend(per[g]); combos[g].append(int(np.prod(per[g])))
q = lambda x: np.percentile(x, [10, 50, 90])
print(f"팔 {len(full)}개, 파티션 {len(combos[128])}개. 지금(2,048점) 범위 안 점 수 10/50/90 백분위 {q(full)}")
for g in GRIDS:
    c, cb = np.array(cnt[g]), np.array(combos[g])
    print(f"[표면점 {g}개] 점 간격 중앙값 {np.median(spacing[g]):.3f}배")
    print(f"  팔마다 범위 안 후보 수 10/50/90 백분위 {q(c)} | 후보가 0개인 팔 {np.mean(c==0)*100:.1f}% | 1개 이하인 팔 {np.mean(c<=1)*100:.1f}%")
    print(f"  파티션마다 조합 수 10/50/90 백분위 {q(cb)} | 최대 {cb.max()} | 조합이 0개인 파티션 {np.mean(cb==0)*100:.1f}% | 1개 이하인 파티션 {np.mean(cb<=1)*100:.1f}% | 192개 넘는 파티션 {np.mean(cb>192)*100:.1f}%")
    sn = np.array(snap[g])
    print(f"  저장된 정답 점에서 가장 가까운 줄인 점까지 거리(물체 반경의 배수) 50/90 백분위 {np.percentile(sn,[50,90]).round(3)}")
