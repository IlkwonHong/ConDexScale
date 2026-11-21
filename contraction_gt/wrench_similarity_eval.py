"""렌치 공간 닮음 — 전량 계산 (계획서 §5-3).

A: 사람 손 그림자 (마디 23개판 σ_h23, 접촉 부위 6개판 σ_h6) — 판정이 있는 파지 전부
B: 계획 앵커 그림자 σ_r — 시뮬 판정이 있는 기록 전부 (gate_records + gate_records_all)
닮음 요약(lost_frac, mean_short, paper_score)을 (h23 vs r), (h6 vs r) × β∈{0.1, 0.2} 로 저장.
파지 수준으로 (h23 vs h6) 도 저장 — "23마디 → 6부위" 단계의 손실.

실행: ./venv/bin/python -m contraction_gt.wrench_similarity_eval [--limit N] [--device cuda:0] [--save-sigma]
출력: data/contraction_gt_300k/wrench_similarity/{records.npz, grasps.npz, meta.json}

규약 (wrench_support.py): 접촉당 누르는 힘 1, 위치 (p−com)/r_obj, μ=0.6, 모서리 8, 방향 512(시드 0).
com = 물체 캐시 질량중심 × total_scale (gate_runner 와 동일), r_obj = 기록의 r_obj.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .gate_runner import _load_records
from .wrench_support import sample_directions, cone_edge_wrenches, support_max, similarity

ROOT = Path("data/contraction_gt_300k")
OUT = ROOT / "wrench_similarity"
CACHE_ROOT = Path("data/contraction_gt_pilot/object_cache")
CONTACT_ROOT = Path("data/dexonomy/contact_cache")
MU, N_EDGES, N_DIRS, SEED = 0.6, 8, 512, 0
BETAS = (0.1, 0.2)


def load_verdicts():
    out = {}
    for name in ("gate_records.npz", "gate_records_all.npz"):
        z = np.load(ROOT / name, allow_pickle=True)
        for i, v in zip(z["idx"].astype(int).tolist(), z["verdict"].tolist()):
            assert i not in out, f"idx {i} 중복"
            out[i] = v
    return out


def manifest_map():
    rows = [l.split("\t") for l in (ROOT / "pilot_manifest.tsv").read_text().splitlines()[1:]]
    return {r[3]: (r[1], r[2]) for r in rows}


def human_contacts(grasp_ids, m_rows):
    """파지별 사람 접촉 (마디 23 / 부위 6): 위치·안쪽 법선·활성 마스크. 파일 단위로 묶어 한 번만 연다."""
    G = len(grasp_ids)
    p23 = np.zeros((G, 23, 3), np.float64); n23 = np.zeros((G, 23, 3), np.float64); a23 = np.zeros((G, 23), bool)
    p6 = np.zeros((G, 6, 3), np.float64); n6 = np.zeros((G, 6, 3), np.float64); a6 = np.zeros((G, 6), bool)
    by_file = defaultdict(list)
    for gi, gid in enumerate(grasp_ids):
        rel, row = gid.split("#")
        by_file[rel].append((gi, int(row)))
    for k, (rel, items) in enumerate(by_file.items()):
        z = np.load(CONTACT_ROOT / rel.replace(".npy", ".npz"))
        for gi, row in items:
            a23[gi] = z["link_contact"][row]
            p23[gi] = z["link_contact_point"][row]
            n23[gi] = -z["link_contact_normal"][row]          # 표의 법선은 바깥 → 안쪽으로
            a6[gi] = z["finger_contact"][row]
            p6[gi] = z["finger_contact_point"][row]
            n6[gi] = -z["finger_contact_normal"][row]
        if (k + 1) % 5000 == 0:
            print(f"  접촉 표 {k + 1}/{len(by_file)} 파일", flush=True)
    return (p23, n23, a23), (p6, n6, a6)


def batched_support(pos, nrm, act, com, r_obj, dirs, device, bs=8192):
    """numpy [G,C,3]... → σ [G, D] float32 (GPU 배치)."""
    out = np.zeros((len(pos), dirs.shape[0]), np.float32)
    for s in range(0, len(pos), bs):
        e = min(s + bs, len(pos))
        W = cone_edge_wrenches(
            torch.as_tensor(pos[s:e], dtype=torch.float32, device=device),
            torch.as_tensor(nrm[s:e], dtype=torch.float32, device=device),
            torch.as_tensor(com[s:e], dtype=torch.float32, device=device),
            torch.as_tensor(r_obj[s:e], dtype=torch.float32, device=device),
            mu=MU, n_edges=N_EDGES,
            active=torch.as_tensor(act[s:e], device=device))
        out[s:e] = support_max(W, dirs).cpu().numpy()
    return out


def summarize(sig_h, sig_r, device, bs=65536):
    """[N,D] 두 그림자 → β 별 요약 dict (numpy)."""
    res = {f"{k}_b{int(b*100):02d}": np.zeros(len(sig_h), np.float32)
           for b in BETAS for k in ("lost_frac", "mean_short", "paper_score")}
    res["n_active"] = np.zeros(len(sig_h), np.int32)
    for s in range(0, len(sig_h), bs):
        e = min(s + bs, len(sig_h))
        h = torch.as_tensor(sig_h[s:e], device=device)
        r = torch.as_tensor(sig_r[s:e], device=device)
        for b in BETAS:
            sm = similarity(h, r, beta=b)
            tag = f"_b{int(b*100):02d}"
            res["lost_frac" + tag][s:e] = sm["lost_frac"].cpu().numpy()
            res["mean_short" + tag][s:e] = sm["mean_short"].cpu().numpy()
            res["paper_score" + tag][s:e] = sm["paper_score"].cpu().numpy()
            res["n_active"][s:e] = sm["n_active"].cpu().numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="판정 기록 앞 N 개만 (시험용)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-sigma", action="store_true", help="방향별 σ_r 전량 저장 (float16, 약 0.9GB)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    m = _load_records(ROOT)
    verdicts = load_verdicts()
    idx = np.array(sorted(verdicts))
    if args.limit:
        idx = idx[: args.limit]
    verdict = np.array([verdicts[i] for i in idx])
    print(f"기록 {len(m['min_alpha']):,}개 중 판정 {len(idx):,}개", flush=True)

    mmap = manifest_map()
    gids_rec = m["grasp_id"][idx].astype(str)
    grasp_ids, grasp_idx = np.unique(gids_rec, return_inverse=True)
    print(f"파지 {len(grasp_ids):,}개", flush=True)

    # 물체 질량중심 (스케일 1) 캐시
    com_cache = {}
    def com_of(gid):
        ds, oid = mmap[gid.split("#")[0]]
        key = (ds, oid)
        if key not in com_cache:
            com_cache[key] = np.load(CACHE_ROOT / ds / f"{oid}.npz")["com"]
        return com_cache[key]

    # 파지별 공통량: total_scale, r_obj, com (첫 기록에서 — 같은 파지의 기록은 동일)
    first_rec = np.zeros(len(grasp_ids), int)
    first_rec[grasp_idx[::-1]] = idx[::-1]
    g_scale = m["total_scale"][first_rec].astype(np.float64)
    g_robj = m["r_obj"][first_rec].astype(np.float64)
    g_com = np.stack([com_of(g) for g in grasp_ids]) * g_scale[:, None]
    print(f"물체 {len(com_cache):,}종, 준비 {time.time()-t0:.0f}s", flush=True)

    dirs = sample_directions(N_DIRS, seed=SEED, dtype=torch.float32, device=args.device)

    # A. 사람 손
    (p23, n23, a23), (p6, n6, a6) = human_contacts(grasp_ids, None)
    sig_h23 = batched_support(p23, n23, a23, g_com, g_robj, dirs, args.device)
    sig_h6 = batched_support(p6, n6, a6, g_com, g_robj, dirs, args.device)
    print(f"A 완료 {time.time()-t0:.0f}s  (마디 수 평균 {a23.sum(1).mean():.1f}, 부위 수 평균 {a6.sum(1).mean():.1f})", flush=True)

    # B. 계획 앵커
    n_anchors = m["n_anchors"][idx].astype(int)
    apos = np.nan_to_num(m["anchor_pos"][idx].astype(np.float64))
    anrm = np.nan_to_num(m["anchor_normal"][idx].astype(np.float64))
    aact = np.arange(4)[None, :] < n_anchors[:, None]
    sig_r = batched_support(apos, anrm, aact, g_com[grasp_idx], g_robj[grasp_idx], dirs, args.device)
    print(f"B 완료 {time.time()-t0:.0f}s", flush=True)

    # 요약
    rec = dict(idx=idx, grasp_idx=grasp_idx, verdict=verdict, n_anchors=n_anchors,
               min_alpha=m["min_alpha"][idx].astype(np.float32),
               alpha=m["alpha"][idx].astype(np.float32),
               r_obj=g_robj[grasp_idx].astype(np.float32),
               pad_rel=m["pad_rel"][idx].astype(np.float32),
               sig_r_max=sig_r.max(1), sig_r_mean=sig_r.mean(1))
    for name, sh in (("h23", sig_h23[grasp_idx]), ("h6", sig_h6[grasp_idx])):
        for k, v in summarize(sh, sig_r, args.device).items():
            rec[f"{name}__{k}"] = v
    if args.save_sigma:
        rec["sig_r"] = sig_r.astype(np.float16)
    np.savez_compressed(OUT / "records.npz", **rec)

    gr = dict(grasp_id=grasp_ids, type=m["type"][first_rec], total_scale=g_scale.astype(np.float32),
              r_obj=g_robj.astype(np.float32), com=g_com.astype(np.float32),
              n_links=a23.sum(1).astype(np.int16), n_units=a6.sum(1).astype(np.int16),
              sig_h23=sig_h23, sig_h6=sig_h6,
              sig_h23_max=sig_h23.max(1), sig_h6_max=sig_h6.max(1))
    for k, v in summarize(sig_h23, sig_h6, args.device).items():
        gr[f"h23_vs_h6__{k}"] = v
    np.savez_compressed(OUT / "grasps.npz", **gr)
    json.dump(dict(mu=MU, n_edges=N_EDGES, n_dirs=N_DIRS, seed=SEED, betas=list(BETAS),
                   n_records=int(len(idx)), n_grasps=int(len(grasp_ids)), limit=args.limit,
                   elapsed_s=round(time.time() - t0, 1)),
              open(OUT / "meta.json", "w"), indent=1)
    print(f"저장 완료 {OUT}  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
