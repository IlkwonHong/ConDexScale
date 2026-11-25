"""읽기 전용 집계: (1) 정답 탐색에서 팔마다 범위 안 표면점 수와 씨앗별 32점 겹침,
(2) net2 평가 자료에서 단계별 '세 팔 모두 정답 한 칸 이내' 비율."""
import pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/robin/a_workspace/condex")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scratchpad"))
import gt_seed_consistency as G
from contraction_gt.surface import _farthest_point_subsample

# ---------- (2) net2 단계별 ----------
CELL = 0.078
rows = []
for f in G.IN_PKLS:
    d = pickle.load(open(ROOT / f, "rb"))
    mined = d["mined"] if isinstance(d, dict) else d
    for m in mined:
        pts, r = m["pts"].astype(np.float64), float(m["r_obj"])
        for p in m["parts"]:
            if p["n_g"] != 3: continue
            gt = p["gt_anchor"].astype(np.float64)
            top1 = [np.linalg.norm(pts[t[0]] - gt[g]) / r for g, t in enumerate(p["top8"])]
            top8 = [np.linalg.norm(pts[t] - gt[g], axis=1).min() / r for g, t in enumerate(p["top8"])]
            fin = np.linalg.norm(pts[p["combo"]] - gt, axis=1) / r
            rows.append((max(top1), max(top8), fin.max(), np.mean(np.array(top1) <= CELL)))
rows = np.array(rows)
print(f"[net2] 파티션 {len(rows)}개")
print(f"  팔 하나: 점수 1등 점이 한 칸 이내 {rows[:,3].mean()*100:.1f}%")
print(f"  세 팔 모두: 점수 1등 점 {np.mean(rows[:,0]<=CELL)*100:.1f}% | 상위 8점 안에 있음 {np.mean(rows[:,1]<=CELL)*100:.1f}% | 최종 조합(α 선택+주변 7점 다듬기 후) {np.mean(rows[:,2]<=CELL)*100:.1f}%")
has = rows[:,1] <= CELL
print(f"  상위 8점에 세 팔 모두 있었던 {has.sum()}개 중 최종 조합도 한 칸 이내 {int((rows[has,2]<=CELL).sum())}개")

# ---------- (1) 정답 탐색 범위 안 점 수 ----------
grasps = G.load_grasps()
cfg = G.SearchConfig()
n_in, rho_rel, ov, n_ref = [], [], [], []
for gid, ds, oid in grasps:
    rel, ridx = gid.rsplit("#", 1)
    patches, ts = G.load_grasp_contacts(str(ROOT / G.CONTACT_ROOT), rel, int(ridx))
    if len(patches) < 2: continue
    s = G.load_samples(ROOT / G.CACHE_ROOT, ds, oid, scale=ts)
    for part in G.enumerate_partitions(patches, 3):
        for group in part:
            pts = np.concatenate([patches[u].points for u in group])
            ar = np.concatenate([patches[u].areas for u in group])
            w = ar / ar.sum() if ar.sum() > 0 else None
            c = np.average(pts, axis=0, weights=w)
            nrm = np.average(np.concatenate([patches[u].normals for u in group]), axis=0, weights=w)
            rho = max(float(np.linalg.norm(pts - c, axis=1).max()), cfg.rho_rel * s.r_obj)
            dd = np.linalg.norm(s.points - c, axis=1); mask = dd <= rho
            nm = s.normals @ nrm > 0.1
            if (mask & nm).any(): mask &= nm
            idx = np.nonzero(mask)[0]
            n_in.append(len(idx)); rho_rel.append(rho / s.r_obj)
            if len(idx) > 32:
                a = set(idx[_farthest_point_subsample(s.points[idx], 32, seed=0)])
                b = set(idx[_farthest_point_subsample(s.points[idx], 32, seed=1)])
                ov.append(len(a & b) / 32)
                # 다듬기 반경(0.05 r_obj) 안의 점 수(자기 포함) — 씨앗 0 후보 32점 기준
                for i in list(a)[:8]:
                    m2 = (np.linalg.norm(s.points - s.points[i], axis=1) <= cfg.rho_rel * s.r_obj / cfg.refine_shrink) & (s.normals @ s.normals[i] > 0.1)
                    n_ref.append(int(m2.sum()))
n_in, rho_rel, ov, n_ref = map(np.array, (n_in, rho_rel, ov, n_ref))
q = lambda x: np.percentile(x, [10, 50, 90]).round(2)
print(f"[정답 탐색, 팔 3대] 팔(그룹) {len(n_in)}개")
print(f"  범위 반경 ÷ 물체 반경: 10/50/90 백분위 {q(rho_rel)}")
print(f"  범위 안 표면점 수: 10/50/90 백분위 {q(n_in)} | 32개 넘는 비율 {np.mean(n_in>32)*100:.1f}%")
print(f"  32개 넘는 팔에서 씨앗 0과 씨앗 1이 고른 32점이 겹치는 비율: 10/50/90 백분위 {q(ov)}")
print(f"  다듬기 반경 안의 점 수(자기 포함): 10/50/90 백분위 {q(n_ref)} | 자기 혼자인 비율 {np.mean(n_ref<=1)*100:.1f}% | 10개 넘는 비율 {np.mean(n_ref>10)*100:.1f}%")
