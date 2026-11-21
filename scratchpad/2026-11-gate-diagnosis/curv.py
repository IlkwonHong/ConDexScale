import pickle, sys, numpy as np, trimesh
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0,'.'); sys.path.insert(0,'/home/robin/.claude/jobs/a635eb86/tmp')
ROOT=Path('data/contraction_gt_300k'); T=Path('/home/robin/.claude/jobs/a635eb86/tmp')
def work(a):
    idx, d, pos, nrm, sc, tr = a
    from contraction_gt.pilot_sim import ASSET_ROOT
    base=ASSET_ROOT/d[0]/'processed_data'/d[1]
    mesh=trimesh.util.concatenate([trimesh.load(p,force='mesh') for p in sorted((base/'urdf/meshes').glob('*.obj'))]); mesh.apply_scale(sc)
    pts, fi = trimesh.sample.sample_surface_even(mesh, 40000, seed=0)
    fn = mesh.face_normals[fi]
    nu = nrm/np.linalg.norm(nrm,axis=1,keepdims=True)
    dev=[]; 
    for p,n in zip(pos,nu):
        sel=np.linalg.norm(pts-p,axis=1)<tr
        if sel.sum()<3: dev.append(np.nan); continue
        c=(-fn[sel]*n).sum(1)
        dev.append(np.degrees(np.arccos(np.clip(c,-1,1))).max())
    # 팁 구가 계획 자리(표면에서 반지름만큼 바깥)에 놓였을 때 물체를 파고드는 깊이
    tipc = pos - nu*tr
    pq = trimesh.proximity.ProximityQuery(mesh)
    pen = np.clip(tr - pq.signed_distance(tipc)*0 - np.abs(pq.signed_distance(tipc)), 0, None) if False else None
    sd = pq.signed_distance(tipc)   # 양수 = 메쉬 안쪽
    dist = trimesh.proximity.closest_point(mesh, tipc)[1]
    gap = dist/tr                    # 1 이면 계획대로 팁이 계획점에서만 닿음, <1 이면 다른 곳이 먼저 닿음
    return idx, np.array(dev), gap
def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    m=_load_records(ROOT)
    R=[r for r in pickle.load(open(ROOT/'wrench_similarity/realized_contacts.pkl','rb')) if r['verdict'] in ('pass','fail') and r['settled']]
    man=[l.split('\t') for l in (ROOT/'pilot_manifest.tsv').read_text().splitlines()[1:]]
    mmap={r[3]:(r[1],r[2]) for r in man}
    jobs={j['idx']:j for j in _prepare(np.array([r['idx'] for r in R]), m, ROOT)}
    args=[(r['idx'], mmap[str(m['grasp_id'][r['idx']]).split('#')[0]], jobs[r['idx']]['pos'], jobs[r['idx']]['nrm'], float(m['total_scale'][r['idx']]), jobs[r['idx']]['tip_radius']) for r in R]
    with get_context('spawn').Pool(8) as pool: out=pool.map(work,args,chunksize=4)
    O={o[0]:o for o in out}; V={r['idx']:r['verdict'] for r in R}
    hy=pickle.load(open(T/'hybrid_raw.pkl','rb'))
    fail8=[29093,62729,102750,129568,272962,342257,485907,518538]
    cfix=sorted({r['idx'] for r in hy if r['cond']=='C' and all(r['held'])}-{r['idx'] for r in hy if r['cond']=='O' and all(r['held'])})
    pas=sorted({r['idx'] for r in hy if r['cond']=='O' and all(r['held'])})
    from sklearn.metrics import roc_auc_score
    res={}
    for name,ids in (('C로 못 고친 8',fail8),('C로 고쳐진 8',cfix),('원래 합격 10',pas),('합격 전체',[i for i in V if V[i]=='pass']),('불합격 전체',[i for i in V if V[i]=='fail'])):
        dv=np.array([np.nanmax(O[i][1]) for i in ids]); gp=np.array([O[i][2].min() for i in ids])
        res[name]=(dv,gp)
        print(f"{name:14s} n={len(ids):3d} | 팁 반지름 안 법선 변화(파지별 최대) 중앙값 {np.nanmedian(dv):.1f}° | 계획 위치 팁 중심↔표면 최소거리/반지름 중앙값 {np.median(gp):.2f} (0.95 미만 비율 {np.mean(gp<0.95):.2f})")
    (dp,gp),(df,gf)=res['합격 전체'],res['불합격 전체']; y=np.r_[np.ones(len(dp)),np.zeros(len(df))]
    print(f"AUC: 법선 변화 {roc_auc_score(y,-np.nan_to_num(np.r_[dp,df])):.3f} | 팁 간섭(최소거리) {roc_auc_score(y,np.r_[gp,gf]):.3f}")
if __name__=='__main__': main()
