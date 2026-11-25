"""GT 데이터셋 생성 파지에서: 새 구현의 여유 vs 시뮬 실측 — 시뮬 도장 대체 판별.

표본: gate 도장(pass/fail/fail_settle)별 무작위 파지. 각 파지마다
  ① 옛 주장 = 저장된 14방향 α 의 최약값(min-α14)
  ② 새 구현 = 64정점 ρ0 (무중력 capacity 의미론, w_req=0 — GT 라벨과 동일 조건)
  ③ 시뮬 실측 = 옛 최약 축방향 + 새 최약 정점 방향의 α_sim (전체 램프,
     gate 와 같은 P5 기계: coacd 물체 + 구형 팁 + LP 평형 예압)

판별 질문:
  (a) fail 파지에서 시뮬/LP 비율이 얼마나 낮나 (LP 과대주장의 크기)
  (b) 새 ρ64 가 fail 을 수치만으로 미리 걸러내나 (pass/fail 의 ρ64 분리도)

실행: ./venv/bin/python scratchpad/gt_rho_vs_sim.py [표본수/그룹=12]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)   # gate_runner/pilot_sim 의 상대경로 관례

from contraction_gt.gate_runner import _load_records
from contraction_gt.pilot_sim import _equilibrium_preload, ASSET_ROOT
from contraction_gt.sim_eval import SimConfig, measure_capacity_multi
from contraction_gt.metric import (ray_wrench_matrix, alpha_batched,
                                   vertex_direction_set, direction_set)

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MU_T, F_CMD = 0.6, 5.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
td = dict(dtype=torch.float64, device=dev)
Vts = vertex_direction_set(torch.float64, dev)

m = _load_records("data/contraction_gt_300k")
g = np.load("data/contraction_gt_300k/gate_records_all.npz")
assert str(m["grasp_id"][g["idx"][0]]) == str(g["grasp_id"][0]), "idx 매핑 불일치"

rows = [l.split("\t") for l in
        Path("data/contraction_gt_300k/pilot_manifest.tsv").read_text().splitlines()[1:]]
mmap = {r[3]: (r[1], r[2]) for r in rows}
cache_root = Path("data/contraction_gt_pilot/object_cache")

rng = np.random.default_rng(7)
samples = []
for verdict in ("pass", "fail", "fail_settle"):
    pool = np.where(g["verdict"] == verdict)[0]
    n_take = N_PER if verdict != "fail_settle" else max(N_PER // 2, 3)
    samples += [(verdict, int(g["idx"][k]))
                for k in rng.choice(pool, n_take, replace=False)]

print(f"{'도장':11s} {'N':>2s} {'minα14':>7s} {'ρ64':>7s} "
      f"{'축:sim/LP':>9s} {'정점:sim/LP':>10s} {'정착':>4s}")
out = []
for verdict, i in samples:
    gid = str(m["grasp_id"][i])
    ds, oid = mmap[gid.split("#")[0]]
    n = int(m["n_anchors"][i])
    pos, nrm = m["anchor_pos"][i][:n], m["anchor_normal"][i][:n]
    r_obj = float(m["r_obj"][i])
    scale = float(m["total_scale"][i])
    mu_tor = float(m["mu_tor"][i])
    com = np.load(cache_root / ds / f"{oid}.npz")["com"] * scale
    pos_n = (pos - com) / r_obj

    pn = torch.tensor(pos_n[None], **td)
    nn_ = torch.tensor(nrm[None], **td)
    W = ray_wrench_matrix(pn, nn_, torch.zeros(1, 3, **td), 1.0, MU_T,
                          torch.full((1, n), mu_tor / r_obj, **td))
    # 새 64정점 ρ0 (외란 규약: ΣWx = −t·d, w=0 라 ± 동일)
    tv = alpha_batched(W.expand(64, *W.shape[1:]), Vts.unsqueeze(1))
    rho64 = float(tv.min())
    v_weak = Vts[int(tv.argmin())].cpu().numpy()
    a14 = m["alpha"][i]
    wd = int(np.argmin(a14))
    d_axis = direction_set(pn)[0, wd].cpu().numpy()
    claims = np.array([float(a14[wd]), rho64])

    dirs2 = np.stack([d_axis, v_weak])
    levels = np.maximum(claims[:, None] * np.linspace(0.1, 1.6, 16)[None], 0.01)
    preload = _equilibrium_preload(pos_n, nrm, MU_T, mu_tor / r_obj, F_CMD)[0]
    cfg = SimConfig(mu_tor=mu_tor, tip_radius=max(float(m["pad_rel"][i]) * r_obj, 0.002),
                    timestep=1e-3, settle_steps=600, hold_steps=300)
    try:
        a_sim, settled, _pre, _nu = measure_capacity_multi(
            ("coacd", str(ASSET_ROOT / ds / "processed_data" / oid / "urdf" / "meshes"),
             scale), pos, nrm, dirs2, levels, r_obj, cfg,
            preload_forces=preload, seed=i)
    except Exception as e:
        print(f"{verdict:11s} {n:2d}  에러: {e!r}")
        continue
    r_ax = a_sim[0] / claims[0] if settled and claims[0] > 1e-6 else np.nan
    r_vx = a_sim[1] / claims[1] if settled and claims[1] > 1e-6 else np.nan
    out.append(dict(verdict=verdict, min_a14=float(a14.min()), rho64=rho64,
                    r_ax=r_ax, r_vx=r_vx, settled=bool(settled)))
    print(f"{verdict:11s} {n:2d} {a14.min():7.3f} {rho64:7.3f} "
          f"{r_ax:9.2f} {r_vx:10.2f} {'O' if settled else 'X'}")

print("\n--- 그룹 요약 (중앙값) ---")
for v in ("pass", "fail", "fail_settle"):
    grp = [o for o in out if o["verdict"] == v]
    if not grp:
        continue
    med = lambda k: float(np.nanmedian([o[k] for o in grp]))
    settle_rate = np.mean([o["settled"] for o in grp])
    print(f"{v:11s} n={len(grp):2d} 재정착율 {settle_rate:.2f} | "
          f"minα14 {med('min_a14'):.3f}  ρ64 {med('rho64'):.3f} | "
          f"축 sim/LP {med('r_ax'):.2f}  정점 sim/LP {med('r_vx'):.2f}")
