"""사람 접촉 렌치 합 vs 정답 앵커 렌치 합 (2026-10-24, handoff_wrench_support.md 3-1). 실행: PYTHONPATH=. ./venv/bin/python scratchpad/wrench_sum_check.py"""
"""사람 접촉 렌치 합 vs 정답 앵커 렌치 합 — 검증 파지 240개, 팔 3대, 저장된 파티션의 정답 앵커 기준."""
import pickle, numpy as np
from scipy.stats import spearmanr
from contraction_net.stage2_cond_dataset import Stage2CondDataset
from contraction_net.stage1_dataset import _LINK_UNIT, _LINK_NAMES
M = pickle.load(open("scratchpad/net2_cond_eval_v1.pkl", "rb"))
ds = Stage2CondDataset(split="val", augment=False, contact_geom=False, com_feat=True, feat_norm=True, patch_interp=True)
def wr(p, n, com, r):   # 법선 렌치 [n, (p-com)/r × n]
    q = (p - com) / r
    return np.concatenate([n, np.cross(q, n)], -1)
rows = []
for m in M:
    gid = m["gid"]; rel, ridx = gid.rsplit("#", 1); ridx = int(ridx)
    con = ds._contacts.get(rel)
    lp, lc, la = con["link_contact_point"][ridx], con["link_contact"][ridx], con["link_area"][ridx]
    li = np.nonzero(lc)[0]
    pts, nrm, com, r = m["pts"].astype(np.float64), m["nrm"].astype(np.float64), m["com"], m["r_obj"]
    # 사람 접촉: 마디 접촉점 → 가장 가까운 표면점의 법선, 면적 가중 / 단순 합
    nn = np.linalg.norm(pts[None] - lp[li][:, None], axis=-1).argmin(1)
    W_h = wr(pts[nn], nrm[nn], com, r)
    w_area = la[li] / la[li].sum()
    H_sum, H_wsum = W_h.sum(0), (W_h * w_area[:, None]).sum(0) * len(li)
    for p in m["parts"]:
        gi = np.linalg.norm(pts[None] - p["gt_anchor"][:, None], axis=-1).argmin(1)
        G = wr(pts[gi], nrm[gi], com, r).sum(0)
        rows.append((np.linalg.norm(H_sum), np.linalg.norm(G), np.linalg.norm(H_sum - G), np.linalg.norm(H_wsum - G),
                     len(li), p["n_g"], p["rec_alpha"] if p["rec_verdict"] else np.nan))
R = np.array(rows)
q = lambda a: np.round(np.nanpercentile(a, [25, 50, 75]), 2)
print(f"파티션 {len(R)}개, 사람 접촉 마디 수 중앙값 {np.median(R[:,4]):.0f}")
print("‖사람 렌치 합‖ 분위:", q(R[:, 0]), " ‖정답 앵커 3개 렌치 합‖ 분위:", q(R[:, 1]))
print("‖사람 합 − 정답 합‖ (단순 합) 분위:", q(R[:, 2]), " (면적 가중 합×마디 수):", q(R[:, 3]))
print("참고: 법선 하나의 렌치 크기는 ≥ 1 (힘 부분이 단위 벡터)")
ok = ~np.isnan(R[:, 6])
print("‖사람 합 − 정답 합‖ 과 정답 min α 의 순위 상관 (파티션 간): %.2f" % spearmanr(R[ok, 2], R[ok, 6]).correlation)
print("‖정답 앵커 렌치 합‖ 과 정답 min α 의 순위 상관: %.2f" % spearmanr(R[ok, 1], R[ok, 6]).correlation)
