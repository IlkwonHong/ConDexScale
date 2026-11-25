"""읽기 전용: 물체 표면점 2048개의 실제 점 간격(가장 가까운 이웃까지 거리 ÷ 물체 반경)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path("/home/robin/a_workspace/condex")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scratchpad"))
import gt_seed_consistency as G
seen, nn1, nn_med_obj, within = set(), [], [], []
for gid, ds, oid in G.load_grasps():
    rel, ridx = gid.rsplit("#", 1)
    patches, ts = G.load_grasp_contacts(str(ROOT / G.CONTACT_ROOT), rel, int(ridx))
    key = (ds, oid, round(float(ts), 6))
    if key in seen: continue
    seen.add(key)
    s = G.load_samples(ROOT / G.CACHE_ROOT, ds, oid, scale=ts)
    t = cKDTree(s.points)
    d, _ = t.query(s.points, k=2)
    nn = d[:, 1] / s.r_obj
    nn_med_obj.append(np.median(nn)); nn1.append(nn)
    cnt = np.array([len(x) for x in t.query_ball_point(s.points, 0.078 * s.r_obj)])
    within.append(np.median(cnt))
nn1 = np.concatenate(nn1)
print(f"물체(크기 포함) {len(seen)}종, 표면점 수 {len(s.points)}")
print(f"가장 가까운 이웃까지 거리 ÷ 물체 반경: 전체 점 10/50/90 백분위 {np.percentile(nn1,[10,50,90]).round(3)}")
print(f"물체별 중앙값의 10/50/90 백분위 {np.percentile(nn_med_obj,[10,50,90]).round(3)}")
print(f"한 점에서 0.078배 안에 있는 표면점 수(자기 포함), 물체별 중앙값의 10/50/90 백분위 {np.percentile(within,[10,50,90])}")
