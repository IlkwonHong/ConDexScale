"""실제 접촉 위치로 알파값 재계산 → 게이트 판정을 맞히는가 (원인 1 확정 시험, 판정형 대체판).

게이트는 방향마다 '계획 알파 x 0.25' 외란을 버텼는지만 저장했다. 그래서
'실제 접촉 알파 >= 시험 강도' 이면 합격 예측으로 두고 실제 판정과 대조한다.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from contraction_gt.gate_runner import _load_records, _prepare, GAMMA  # noqa: E402
from contraction_gt.metric import ray_wrench_matrix, alpha_reference  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
res = pickle.load(open(ROOT / "wrench_similarity/realized_contacts.pkl", "rb"))
res = [r for r in res if r["verdict"] in ("pass", "fail") and r["settled"]]
m = _load_records(ROOT)
idxs = np.array([r["idx"] for r in res])
jobs = {j["idx"]: j for j in _prepare(idxs, m, ROOT)}

rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
mmap = {r[3]: (r[1], r[2]) for r in rows}


def com_of(i):
    ds, oid = mmap[str(m["grasp_id"][i]).split("#")[0]]
    c = np.load(Path("data/contraction_gt_pilot/object_cache") / ds / f"{oid}.npz")
    return c["com"] * float(m["total_scale"][i])


def alpha(pos, nrm, com, r_obj, mu_tor, dirs):
    if len(pos) == 0:
        return np.zeros(len(dirs))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)
    W = ray_wrench_matrix(t(pos), t(nrm), t(com), r_obj, 0.6, mu_tor)
    return alpha_reference(W, t(dirs))[0].numpy()


out = []
for r in res:
    i, j = r["idx"], jobs[r["idx"]]
    com = com_of(i)
    n = len(j["pos"])
    a_plan = alpha(j["pos"], j["nrm"], com, j["r_obj"], j["mu_tor"], j["dirs"])
    # 팁별 합치기: 접촉 법선력 가중 평균 위치·법선, 닿지 않은 팁 제외
    P, N = [], []
    for t in range(n):
        cs = [c for c in r["contacts"] if c["tip"] == t]
        if not cs:
            continue
        w = np.array([max(c["f_normal"], 1e-9) for c in cs])
        P.append((w[:, None] * np.stack([c["pos_obj"] for c in cs])).sum(0) / w.sum())
        nv = (w[:, None] * np.stack([c["normal_obj"] for c in cs])).sum(0)
        N.append(nv / np.linalg.norm(nv))
    a_real = alpha(np.array(P).reshape(-1, 3), np.array(N).reshape(-1, 3), com, j["r_obj"], j["mu_tor"], j["dirs"])
    lv = j["levels"][:, 0]
    out.append(dict(v=r["verdict"], n_tip=n, n_touch=len(P), plan=a_plan, real=a_real, lv=lv,
                    stored=m["alpha"][i]))

# 검증: 계획 접촉 재계산 == 저장 알파 (게이트 방향 세트에서)
err = []
for o, r in zip(out, res):
    ds = sorted(set(np.argsort(o["stored"])[:3].tolist()) | set(range(6)))
    err.append(np.abs(o["plan"] - o["stored"][ds]).max())
print(f"재현 검증: 계획 알파 재계산 vs 저장값 최대 오차 {max(err):.2e} (중앙값 {np.median(err):.1e})")

pred_pass = np.array([(o["real"] >= o["lv"] - 1e-9).all() for o in out])
truth = np.array([o["v"] == "pass" for o in out])
ratio = np.array([(o["real"] / np.maximum(o["plan"], 1e-9)).min() for o in out])  # 최악 방향의 실제/계획
margin = np.array([(o["real"] / np.maximum(o["lv"], 1e-9)).min() for o in out])  # 최악 방향의 실제알파/시험강도
for name, mask in (("합격", truth), ("불합격", ~truth)):
    print(f"{name}: n={mask.sum()}  실제알파로 합격 예측 비율 {pred_pass[mask].mean():.2f} | "
          f"최악 방향 실제/계획 알파 중앙값 {np.median(ratio[mask]):.2f} (p25 {np.percentile(ratio[mask],25):.2f}) | "
          f"실제알파/시험강도 중앙값 {np.median(margin[mask]):.2f} | 닿은 팁 비율 "
          f"{np.mean([o['n_touch']/o['n_tip'] for o,k in zip(out,mask) if k]):.2f}")
acc = (pred_pass == truth).mean()
from sklearn.metrics import roc_auc_score  # noqa: E402
print(f"판정 일치율 {acc:.3f}  | AUC(실제알파/시험강도) {roc_auc_score(truth, margin):.3f} "
      f"| AUC(최악 실제/계획) {roc_auc_score(truth, ratio):.3f}")
np.savez(Path(sys.argv[1]) if len(sys.argv) > 1 else "realized_alpha_out.npz",
         truth=truth, pred=pred_pass, ratio=ratio, margin=margin)
err = np.array(err); bad = err > 1e-3
print(f"재현 실패 레코드 {bad.sum()}/{len(err)}")
ok = ~bad
for name, mask in (("합격", truth & ok), ("불합격", ~truth & ok)):
    print(f"[재현 성공만] {name}: n={mask.sum()} 합격 예측 {pred_pass[mask].mean():.2f} 실제/계획 중앙값 {np.median(ratio[mask]):.2f} 실제알파/시험강도 중앙값 {np.median(margin[mask]):.2f}")
print(f"[재현 성공만] AUC {roc_auc_score(truth[ok], margin[ok]):.3f}")
