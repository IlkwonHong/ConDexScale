"""EXP-004 추가 분석 (읽기 전용): 선형계획을 m 번만 풀 수 있을 때, 네트워크가 추린 후보와 값싼 점수가 추린 조합 중
어느 쪽이 새 정답(A) 수준의 α 를 더 자주 찾는가.
  네트워크: 팔마다 점수 상위 k 개(앞 팔은 모델 1등으로 고정)의 곱 k^3 개 조합 중 최소 간격을 지키는 것
  값싼 점수: 세 점이 누르는 힘이 상쇄되는 정도가 좋은 순서로 앞에서 m = k^3 개 조합
둘 다 조합마다 선형계획으로 α 를 구해 가장 높은 것을 고른다. 기준은 새 정답의 α (후보 전체를 다 푼 최선)."""
import itertools, json, pickle, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scratchpad"))
import eval_c128_learnability as E
import gen_c128_records as M
from contraction_net.stage2_cond_dataset import Stage2CondDataset, canonical_order
from contraction_net.stage2_cond_model import eval_metrics
G, RB, N, KS = E.G, E.RB, 3, (1, 2, 3, 5)
dkw = dict(augment=False, contact_geom=False, com_feat=True, feat_norm=True, patch_interp=True, cand_from_records=True, only_n=N)
rec_dir = E.DATA / "search_records_c128_eval"
info = E.load_rows(rec_dir)
ds = Stage2CondDataset(split="val", records_dir=rec_dir, **dkw)
pairs = [(i, t) for i, (gid, _e) in enumerate(ds.samples) for t in ds.cond_tasks[gid]]
rows = [info[(ds.samples[i][0], json.dumps(t[0]))] for i, t in pairs]
alpha_a = np.array([r["alpha"] for r in rows])
cfg = M.SearchConfig()

# 물체·조합 (값싼 점수 순서) — 정답 생성과 같은 함수
ctx, cen, bad = {}, {}, []
for i, _t in pairs:
    gid = ds.samples[i][0]
    if gid not in ctx:
        d_, oid = ds.obj_of[gid]
        c = M.prepare(gid, d_, oid, cfg, cen, bad)
        by = {}
        for ps, p in c["parts"].items():
            b = json.loads(ps); o = canonical_order(b)
            by[json.dumps([b[j] for j in o])] = p["combos"]
        s = c["samples"]
        ctx[gid] = (s, c["mu_tor"], max(cfg.pad_radius_rel_multi[0] * s.r_obj, 0.002), by)

def best_alpha(lists):
    """레코드마다 조합 배열 [C,3] → 그중 최고 min α (조합이 없으면 0)."""
    entries, cnt = [], []
    for (i, _t), c in zip(pairs, lists):
        s, mu, _tip, _by = ctx[ds.samples[i][0]]
        cnt.append(len(c))
        if len(c): entries.append((s, np.asarray(c, np.int64), mu))
    a = RB._alpha_many(entries, cfg).min(1) if entries else np.zeros(0)
    out, ofs = [], 0
    for n in cnt:
        out.append(a[ofs:ofs + n].max() if n else 0.0); ofs += n
    return np.array(out), np.array(cnt)

p = lambda x: f"{100 * float(np.mean(x)):5.1f}%"
res = {}
print(f"평가 레코드 {len(rows)}개. 표의 값 = 고른 조합의 α 가 새 정답 α 의 0.95배 이상인 레코드 (괄호 안은 0.8배 이상)")
prox = {}
for k in KS:
    m = k ** 3
    lists = [ctx[ds.samples[i][0]][3][json.dumps(t[0])][:m] for i, t in pairs]
    a, cnt = best_alpha(lists)
    prox[k] = (float((a >= 0.95 * alpha_a).mean()), float((a >= 0.8 * alpha_a).mean()), float(cnt.mean()))
    print(f"  값싼 점수 순서로 앞에서 {m:3d}개: {p(a >= 0.95 * alpha_a)} ({p(a >= 0.8 * alpha_a)}), 실제로 푼 조합 수 평균 {cnt.mean():.1f}")
res["proxy"] = prox
loader = DataLoader(E.TaskView(ds, pairs), batch_size=96, num_workers=4)
for tag in "ab":
    model, ck = E.load_model(ROOT / f"checkpoints/stage2_cond_c128{tag}/best.pt", ds.vocab, "cuda")
    TOP, J = [], []
    with torch.no_grad():
        for batch in loader:
            J.append(batch.pop("_j").numpy())
            batch = {k_: v.to("cuda") if isinstance(v, torch.Tensor) else v for k_, v in batch.items()}
            with torch.amp.autocast("cuda"):
                out = model(batch)
                _, greedy = eval_metrics(model, out, batch)
                _, _, e_p, Tg = out
                top = torch.full((greedy.shape[0], 4, max(KS)), -1, dtype=torch.long, device="cuda")
                for k_, _rows, logp in model.chain_logits(e_p, Tg, batch, greedy):
                    v, ix = logp.topk(max(KS), -1)
                    top[:, k_] = torch.where(torch.isfinite(v), ix, torch.full_like(ix, -1))   # 후보가 아닌 점은 -1
            TOP.append(top.cpu().numpy())
    TOP = np.concatenate(TOP)[np.argsort(np.concatenate(J))][:, :N]
    r = {}
    for k in KS:
        lists = []
        for (i, _t), tp in zip(pairs, TOP):
            s, _mu, tip, _by = ctx[ds.samples[i][0]]
            cs = []
            for c in itertools.product(*[[x for x in tp[g, :k] if x >= 0] for g in range(N)]):
                q = s.points[list(c)]
                if min(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[0] - q[2]), np.linalg.norm(q[1] - q[2])) >= 2.4 * tip:
                    cs.append(c)
            lists.append(np.array(cs, np.int64).reshape(-1, N))
        a, cnt = best_alpha(lists)
        r[k] = (float((a >= 0.95 * alpha_a).mean()), float((a >= 0.8 * alpha_a).mean()), float(cnt.mean()))
        print(f"  모델 {tag.upper()}: 팔마다 상위 {k}개 (최대 {k ** 3:3d}개 조합): {p(a >= 0.95 * alpha_a)} ({p(a >= 0.8 * alpha_a)}), 실제로 푼 조합 수 평균 {cnt.mean():.1f}")
    res["model_" + tag] = r
    del model
pickle.dump(res, open(ROOT / "scratchpad/analyze_c128_topk.pkl", "wb"))
