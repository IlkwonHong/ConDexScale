import pickle, sys, numpy as np, trimesh
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0,'.')
ROOT=Path('data/contraction_gt_300k'); T=Path('/home/robin/.claude/jobs/a635eb86/tmp')

def work(a):
    idx, d, pos, nrm, sc, tr = a
    from contraction_gt.object_cache import resolve_mesh_path, load_mesh
    from contraction_gt.pilot_sim import ASSET_ROOT
    base = ASSET_ROOT/d[0]/'processed_data'/d[1]
    plan = load_mesh(base/'mesh/simplified.obj' if (base/'mesh/simplified.obj').exists() else base/'mesh/normalized.obj')
    pieces = [trimesh.load(p, force='mesh') for p in sorted((base/'urdf/meshes').glob('*.obj'))]
    plan.apply_scale(sc); [p.apply_scale(sc) for p in pieces]
    cp, dp, fp = trimesh.proximity.closest_point(plan, pos)
    best = np.full(len(pos), np.inf); bn = np.zeros((len(pos),3))
    for p in pieces:
        c, dd, f = trimesh.proximity.closest_point(p, pos)
        sel = dd < best; best[sel] = dd[sel]; bn[sel] = p.face_normals[f[sel]]
    # 볼록 조각 법선(바깥) vs 계획 법선(안쪽) → 각도
    nu = nrm/np.linalg.norm(nrm,axis=1,keepdims=True)
    ang = np.degrees(np.arccos(np.clip((-bn*nu).sum(1),-1,1)))
    return idx, dp/tr, best/tr, ang

def main():
    from contraction_gt.gate_runner import _load_records, _prepare
    m=_load_records(ROOT)
    R=[r for r in pickle.load(open(ROOT/'wrench_similarity/realized_contacts.pkl','rb')) if r['verdict'] in ('pass','fail') and r['settled']]
    man=[l.split('\t') for l in (ROOT/'pilot_manifest.tsv').read_text().splitlines()[1:]]
    mmap={r[3]:(r[1],r[2]) for r in man}
    jobs={j['idx']:j for j in _prepare(np.array([r['idx'] for r in R]), m, ROOT)}
    args=[(r['idx'], mmap[str(m['grasp_id'][r['idx']]).split('#')[0]], jobs[r['idx']]['pos'], jobs[r['idx']]['nrm'],
           float(m['total_scale'][r['idx']]), jobs[r['idx']]['tip_radius']) for r in R]
    with get_context('spawn').Pool(8) as pool: out=pool.map(work,args,chunksize=4)
    V={r['idx']:r['verdict'] for r in R}
    hy=pickle.load(open(T/'hybrid_raw.pkl','rb'))
    fail8=[29093,62729,102750,129568,272962,342257,485907,518538]
    cfix=sorted({r['idx'] for r in hy if r['cond']=='C' and all(r['held'])}-{r['idx'] for r in hy if r['cond']=='O' and all(r['held'])})
    pas=sorted({r['idx'] for r in hy if r['cond']=='O' and all(r['held'])})
    O={o[0]:o for o in out}
    pickle.dump(O, open(T/'surface_gap.pkl','wb'))
    def show(name, ids):
        dp=np.concatenate([O[i][1] for i in ids]); dc=np.concatenate([O[i][2] for i in ids]); an=np.concatenate([O[i][3] for i in ids])
        mx=np.array([O[i][2].max() for i in ids])
        print(f"{name:16s} n={len(ids):3d} | 계획 표면까지 {np.median(dp):.3f} | 볼록 조각 표면까지 중앙값 {np.median(dc):.3f} p90 {np.percentile(dc,90):.3f} | 파지별 최대 중앙값 {np.median(mx):.3f} | 볼록 조각 법선 각도 중앙값 {np.median(an):.1f}° p90 {np.percentile(an,90):.1f}°")
        return mx
    print('(거리 단위 = 팁 반지름)')
    show('C로 못 고친 8',fail8); show('C로 고쳐진 8',cfix); show('원래 합격 10',pas)
    mp=show('합격 전체',[i for i in V if V[i]=='pass']); mf=show('불합격 전체',[i for i in V if V[i]=='fail'])
    from sklearn.metrics import roc_auc_score
    y=np.r_[np.ones(len(mp)),np.zeros(len(mf))]
    print(f"볼록 조각 표면 거리(파지별 최대)로 합격/불합격 가르기 AUC {roc_auc_score(y,-np.r_[mp,mf]):.3f}")
if __name__=='__main__': main()
