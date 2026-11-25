"""EXP-004 모델 A 결과의 세부 분석 (읽기 전용: 저장된 체크포인트로 평가 레코드를 다시 디코딩해 나눠 센다)."""
import json, pickle, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scratchpad"))
import eval_c128_learnability as E
from contraction_net.stage2_cond_dataset import Stage2CondDataset, canonical_order
G, N = E.G, 3
model_tag = sys.argv[1] if len(sys.argv) > 1 else "a"
dkw = dict(augment=False, contact_geom=False, com_feat=True, feat_norm=True, patch_interp=True, cand_from_records=True, only_n=N)
rec_dir = E.DATA / "search_records_c128_eval"
info = E.load_rows(rec_dir)
ds = Stage2CondDataset(split="val", records_dir=rec_dir, **dkw)
pairs = [(i, t) for i, (gid, _e) in enumerate(ds.samples) for t in ds.cond_tasks[gid]]
rows = [info[(ds.samples[i][0], json.dumps(t[0]))] for i, t in pairs]
own_key = "a" if model_tag == "a" else "b"
OWN = np.stack([r[own_key] for r in rows]); ncand = np.stack([r["n_cand"] for r in rows])
model, ck = E.load_model(ROOT / f"checkpoints/stage2_cond_c128{model_tag}/best.pt", ds.vocab, "cuda")
greedy, top3, tgt, mc, J = E.decode(model, DataLoader(E.TaskView(ds, pairs), batch_size=96, num_workers=4), "cuda")
greedy = greedy[np.argsort(J)][:, :N]
hit = greedy == OWN
p = lambda x: f"{100 * float(np.mean(x)):5.1f}%"
print(f"[모델 {model_tag.upper()}] 평가 레코드 {len(rows)}개: 팔 하나 일치 {p(hit)}, 세 팔 모두 {p(hit.all(1))}")
print("팔 순서별 일치(엄지 쪽 팔 / 둘째 / 셋째):", ", ".join(p(hit[:, k]) for k in range(N)))
print("세 팔 중 맞힌 팔 수 분포(0/1/2/3개):", ", ".join(p(hit.sum(1) == k) for k in range(4)))
print("앞 팔이 맞았을 때 뒤 팔도 맞은 비율 — 첫째가 맞았을 때 둘째:", p(hit[hit[:, 0], 1]), "| 첫째·둘째가 맞았을 때 셋째:", p(hit[hit[:, 0] & hit[:, 1], 2]),
      "| (참고) 둘째 전체", p(hit[:, 1]), "셋째 전체", p(hit[:, 2]))
print("후보 수에 따른 팔 하나 일치:")
for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 15), (16, 200)):
    m = (ncand >= lo) & (ncand <= hi)
    if m.any(): print(f"  후보 {lo}~{hi}개인 팔 {int(m.sum()):5d}개: 일치 {p(hit[m])} (우연 수준 {p(1 / ncand[m])})")
# 틀린 팔이 고른 점과 정답 점의 거리
dist = []
for (i, _t), g, o in zip(pairs, greedy, OWN):
    gid, e = ds.samples[i]; d_, oid = ds.obj_of[gid]
    s = G.load_samples(ROOT / G.CACHE_ROOT, d_, oid, scale=e["total_scale"])
    dist.append(np.linalg.norm(s.points[g] - s.points[o], axis=1) / s.r_obj)
dist = np.array(dist); w = ~hit
print(f"틀린 팔 {int(w.sum())}개에서 고른 점과 정답 점의 거리(물체 반경의 배수): 중앙값 {np.median(dist[w]):.2f}, 0.2배 이내(이웃 후보) {p(dist[w] <= 0.2)}, 0.3배 이내 {p(dist[w] <= 0.3)}")
# α 가 비슷한 다른 조합이 있는 레코드와 없는 레코드 (EXP-003 의 규칙 나 결과)
e3 = pickle.load(open(ROOT / "scratchpad/gt_coarse_candidate_precheck.pkl", "rb"))["out128"]["recs"]
tie = {}
for (gid, ps, _pad), v in e3.items():
    b = json.loads(ps); o = canonical_order(b)
    if v["나"]["alpha"] is not None: tie[(gid, json.dumps([b[j] for j in o]))] = v["나"]["has_other"]
t = np.array([tie.get((ds.samples[i][0], json.dumps(tk[0])), False) for i, tk in pairs])
print(f"최선 말고도 α 가 0.95배 이상인 조합이 있는 레코드 {int(t.sum())}개: 세 팔 모두 일치 {p(hit[t].all(1))} | 없는 레코드 {int((~t).sum())}개: {p(hit[~t].all(1))}")
pickle.dump(dict(greedy=greedy, own=OWN, ncand=ncand, dist=dist, tie=t), open(ROOT / f"scratchpad/analyze_c128_model_{model_tag}.pkl", "wb"))
