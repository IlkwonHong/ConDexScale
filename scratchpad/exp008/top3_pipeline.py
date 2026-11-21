"""안전장치 파이프라인 평가 (검증 파지, 저장된 GT 결과 사용 — 새 계산 없음).

절차: 후보 파티션 집합 S 에서 탐색 품질(레코드 min-α)이 가장 높은 파티션의 배치를 쓰고, 그 배치의 시뮬 게이트 판정을 본다.
S = Net1 상위 1 / 상위 3 / 무작위 3 / 전체 (탐색 상한). 판정 = 기존 서보 게이트(gate_records + gate_records_all).
지표: 고른 배치의 합격률, 고른 배치 α ÷ 전체 최선 α ≥ 0.95 비율, 탐색해야 하는 파티션 수.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from contraction_gt.gate_runner import _load_records  # noqa: E402
from contraction_net.stage1_dataset import Stage1Dataset  # noqa: E402
from contraction_net.stage1_model import Stage1Net  # noqa: E402

ROOT = Path("data/contraction_gt_300k")
va = Stage1Dataset(split="val", augment=False, contact_geom=False, sigma_max=0.08, n_patch=128)
m = _load_records(ROOT)
verdict = {}
for name in ("gate_records.npz", "gate_records_all.npz"):
    z = np.load(ROOT / name, allow_pickle=True)
    verdict.update(dict(zip(z["idx"].tolist(), z["verdict"].tolist())))
# (gid, N, 파티션 id) → 레코드 목록 (패드 반경별) : (α_min, 판정)
vgids = {g for g, _e in va.samples}
recs = defaultdict(list)
for i in range(len(m["grasp_id"])):
    g = str(m["grasp_id"][i])
    n = int(m["n_anchors"][i])
    if g in vgids and n >= 3:
        cid = va.vocab.id_of(json.loads(m["partition"][i]))
        recs[(g, n, cid)].append((float(m["min_alpha"][i]), verdict.get(i, "none")))

dev = "cuda"
model = Stage1Net(len(va.vocab), n_patch=128).to(dev)
model.load_state_dict(torch.load("checkpoints/exp005_f1.0_s0/best.pt", map_location=dev, weights_only=False)["model"])
model.eval()
order = {3: [], 4: []}
with torch.no_grad():
    for b in DataLoader(va, batch_size=256, num_workers=4):
        b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        lg, _r, _p = model(b)
        for n in (3, 4):
            o = lg.masked_fill(~b[f"valid_n{n}"], float("-inf")).argsort(-1, descending=True)[:, :5]
            order[n] += o.tolist()

rng = np.random.default_rng(0)
for n in (3, 4):
    out = defaultdict(list)
    for i, (g, e) in enumerate(va.samples):
        cands = [c for c in e[f"ids{n}"].tolist() if (g, n, c) in recs]
        if not cands:
            continue
        best_of = lambda c: max(recs[(g, n, c)], key=lambda t: t[0])       # 패드 반경 중 최선
        top_all = max(best_of(c)[0] for c in cands)
        sets = {"Net1 상위 1": [c for c in order[n][i] if c in cands][:1],
                "Net1 상위 3": [c for c in order[n][i] if c in cands][:3],
                "무작위 3": list(rng.choice(cands, min(3, len(cands)), replace=False)),
                "전체 탐색": cands}
        for name, S in sets.items():
            if not S:
                S = [cands[0]]
            a, v = max((best_of(c) for c in S), key=lambda t: t[0])
            # 시뮬로 확인하는 안전장치: S 의 파티션별 최선 배치를 모두 시뮬해 하나라도 합격하면 사용 가능
            any_pass = any(best_of(c)[1] == "pass" for c in S)
            n_sim = next((k + 1 for k, c in enumerate(sorted(S, key=lambda c: -best_of(c)[0]))
                          if best_of(c)[1] == "pass"), len(S))   # α 순서로 시뮬했을 때 합격까지 시뮬 횟수
            out[name].append((v == "pass", v in ("pass", "none", "unstable"), a >= 0.95 * top_all, len(S),
                              any_pass, n_sim))
        out["후보 수"].append((len(cands),))
    print(f"\n팔 {n}대: 검증 파지 {len(out['후보 수'])}건, 후보 파티션 수 중앙값 {np.median([x[0] for x in out['후보 수']]):.0f}")
    for name in ("Net1 상위 1", "Net1 상위 3", "무작위 3", "전체 탐색"):
        r = np.array(out[name], dtype=float)
        print(f"  {name:10s}: 시뮬 합격 {r[:, 0].mean() * 100:.1f}% (판정 없음 포함 {r[:, 1].mean() * 100:.1f}%) | "
              f"품질 ≥0.95×전체 최선 {r[:, 2].mean() * 100:.1f}% | 탐색할 파티션 수 평균 {r[:, 3].mean():.1f} | "
              f"시뮬 확인 후 하나라도 합격 {r[:, 4].mean() * 100:.1f}% (시뮬 횟수 평균 {r[:, 5].mean():.2f})")
