"""동급 배치(여러 벌 정답) 사이에서 사람 렌치 범위 덮음(lost_frac)이 얼마나 다른가 (2026-10-25, handoff_wrench_support.md 5-4 확인 1).
시험 생성 데이터 search_records_multi 의 팔 3대·2벌 이상 레코드. 사람 = 마디 접촉(link_contact_point/normal), 로봇 = 저장된 벌의 앵커.
실행: PYTHONPATH=. ./venv/bin/python scratchpad/mode_lost_frac.py
"""
import glob, collections, numpy as np, torch
from contraction_gt.object_cache import load_samples
from contraction_gt.runner_batched import CACHE_ROOT, CONTACT_ROOT
from contraction_gt.contacts_io import load_grasp_contacts
from contraction_gt.wrench_support import sample_directions, cone_edge_wrenches, support_max, similarity
z = collections.defaultdict(list)
for f in sorted(glob.glob("data/contraction_gt_300k/search_records_multi/shard_*.npz")):
    d = np.load(f)
    for k in ("grasp_id", "n_anchors", "n_modes", "mode_idx", "mode_alpha", "total_scale", "r_obj"): z[k].append(d[k])
z = {k: np.concatenate(v) for k, v in z.items()}
man = {}
for l in open("data/contraction_gt_300k/pilot_manifest.tsv").read().splitlines()[1:]:
    t = l.split("\t"); man[f"{t[3]}#{t[4]}"] = (t[1], t[2])
sel = np.nonzero((z["n_anchors"] == 3) & (z["n_modes"] >= 2))[0]
rng = np.random.default_rng(0); sel = rng.choice(sel, 1500, replace=False)
dirs = sample_directions(512)
rows = []
for i in sel:
    gid = str(z["grasp_id"][i]); rel, ridx = gid.rsplit("#", 1)
    try:
        patches, ts = load_grasp_contacts(CONTACT_ROOT, rel, int(ridx))
    except Exception:
        continue
    s = load_samples(CACHE_ROOT, *man[gid], scale=ts)
    hp = np.concatenate([p.points for p in patches.values()]); hn = np.concatenate([p.normals for p in patches.values()])
    com = torch.as_tensor(s.com)[None]
    Wh = cone_edge_wrenches(torch.as_tensor(hp)[None], torch.as_tensor(hn)[None], com, s.r_obj)
    sig_h = support_max(Wh, dirs)
    mi = z["mode_idx"][i]; ma = z["mode_alpha"][i].min(1); out = []
    for m in range(int(z["n_modes"][i])):
        idx = mi[m, :3].astype(int)
        Wr = cone_edge_wrenches(torch.as_tensor(s.points[idx])[None], torch.as_tensor(s.normals[idx])[None], com, s.r_obj)
        lf = similarity(sig_h, support_max(Wr, dirs))["lost_frac"][0].item()
        out.append((lf, ma[m] / ma[0]))
    rows.append(out)
print(f"팔 3대·2벌 이상 레코드 {len(rows)}개 (표본)")
lf_best = np.array([r[0][0] for r in rows]); lf_min = np.array([min(x[0] for x in r) for r in rows])
gain = lf_best - lf_min
print(f"최선(α) 벌의 lost_frac 중앙값 {np.median(lf_best):.3f} | 벌들 중 가장 작은 lost_frac 중앙값 {np.median(lf_min):.3f}")
print(f"최선 벌 → 가장 덮는 벌로 바꿨을 때 lost_frac 감소: 중앙값 {np.median(gain):.3f}, 0.05 이상 줄어드는 레코드 {np.mean(gain>=0.05):.0%}, 0.1 이상 {np.mean(gain>=0.1):.0%}")
alt = np.array([min(r, key=lambda x: x[0])[1] for r in rows])
print(f"그 벌의 α ÷ 최선 α: 중앙값 {np.median(alt):.2f} (저장 기준상 ≥ 0.8)")
sw = np.array([np.argmin([x[0] for x in r]) != 0 for r in rows]); print(f"가장 덮는 벌이 최선 벌과 다른 레코드 {sw.mean():.0%}")
