"""조건부 net2 평가 (2026-10-17) — net2_viz_mine / net2_kshot_decompose 와 같은 검증 파지 240개(팔 3대).

파티션: net1 상위 3개 (지금 파이프라인과 같음). 그룹 순서: canonical_order (엄지 먼저).
디코딩 (파티션마다):
  greedy    : 그룹마다 앞 자리 조건 1등을 차례로
  beam_net  : 첫 그룹 상위 8 → 둘째 상위 8 씩(64개) → 결합 확률 상위 8 앞 자리 → 셋째 상위 8 씩 = 조합 64개, LP 최선
  beam_lp   : 앞 자리 8개까지는 같고, 셋째 그룹은 마스크 안 모든 점을 LP 로 → 최선
후보 6개 = 파티션마다 beam_lp 조합 중 min α 상위 2개(서로 다른 것)를 이웃 7점 다듬기 → LP 순서로 시뮬.
비교 지표: 다듬기 전 LP α ÷ GT 최선 (파티션 3개 중 최선), 1발·6발 성공률, 첫 자리 고정 비교.
실행: ./venv/bin/python scratchpad/net2_cond_eval.py <ckpt> <out.pkl>
"""
import itertools
import pickle
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from net2_viz_mine import (refine, build_sim_job, load_record_table, group_mask_and_gt,
                           canon_key_of_blocks, MU_T, PAD_REL, DEV, BATCH_KEYS, N_TARGET, M_GRASP)
from contraction_net.stage2_cond_dataset import Stage2CondDataset, canonical_order
from contraction_net.stage2_cond_model import Stage2CondNet
from contraction_net.stage2_dataset import UNITS
from contraction_net.stage1_dataset import N_UNIT
from contraction_gt.search import _batched_alpha, mu_tor_from_pad
from contraction_gt.surface import SurfaceSamples
from contraction_gt.gate_runner import _worker

BEAM = 8
SIM_WORKERS = 6


def main():
    ck_path = ROOT / sys.argv[1]
    out_path = ROOT / sys.argv[2]
    n_want = int(sys.argv[3]) if len(sys.argv) > 3 else M_GRASP
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    ds = Stage2CondDataset(split="val", augment=False, contact_geom=False, com_feat=a["com_feat"],
                           feat_norm=a["feat_norm"], patch_interp=a["patch_interp"])
    model = Stage2CondNet(ds.vocab, d_pf=11 if a["com_feat"] else 7, d_ptf=10 if a["com_feat"] else 6,
                          wrench_feat=a.get("wrench_feat", "none"))
    sup = model.wrench_feat == "support"
    model.load_state_dict(ck["model"])
    model.to(DEV).eval()
    keys = BATCH_KEYS + (("point_patch3", "point_patch3_w") if a["patch_interp"] else ())
    print(f"모델 {ck_path} (에폭 {ck['epoch']})", flush=True)

    val_gids = {g for g, _ in ds.samples}
    table, gt_pass = load_record_table(ROOT / "data/contraction_gt_300k/search_records", val_gids)
    t0 = time.time()
    mined = []
    for i in range(len(ds)):
        if len(mined) >= n_want:
            break
        gid, e = ds.samples[i]
        if gt_pass.get((gid, N_TARGET), 0.0) <= 0.0:
            continue
        item = ds[i]
        if not item[f"valid_n{N_TARGET}"].any():
            continue
        r_obj, ts = e["r_obj"], e["total_scale"]
        ridx = int(gid.rsplit("#", 1)[1])
        obj = ds._objects.get(ds.obj_of[gid])
        pts_m = (obj["pts"] * ts).astype(np.float64)
        samples = SurfaceSamples(points=pts_m, normals=obj["nrm"].astype(np.float64),
                                 r_obj=r_obj, com=(obj["com"] * ts).astype(np.float64))
        mu_tor = mu_tor_from_pad(PAD_REL * r_obj, MU_T)
        batch = {k: item[k].unsqueeze(0).to(DEV) for k in keys}
        pos = item["cond_pos"].unsqueeze(0).to(DEV)
        nrm = item["cond_nrm"].unsqueeze(0).to(DEV)
        with torch.no_grad():
            x, h = model.encode(batch)
            fu = model.phi_u(x[:, model.n_patch:model.n_patch + N_UNIT])
            T = torch.einsum("cgu,bud->bcgd", model.member.to(fu.dtype), fu)
            ctx = model.ctx_proj(h)[:, None, None, :].expand(-1, T.shape[1], 4, -1)
            size = model.gsize.to(fu.dtype)[None, :, :, None].expand(1, -1, -1, -1)
            g_ = model.phi_g(torch.cat([T, ctx, size], -1)) * model.gmask.to(fu.dtype)[None, :, :, None]
            logits = model.psi(g_.sum(2)).squeeze(-1)[0]
            if a["patch_interp"]:
                i3, w3 = batch["point_patch3"], batch["point_patch3_w"].to(x.dtype)
                xp = sum(torch.gather(x[:, :model.n_patch], 1, i3[..., j, None].expand(-1, -1, x.shape[-1]))
                         * w3[..., j, None] for j in range(i3.shape[-1]))
            else:
                xp = torch.gather(x[:, :model.n_patch], 1,
                                  batch["point_patch"][..., None].expand(-1, -1, x.shape[-1]))
            e_p = model.pt_mlp(torch.cat([xp, batch["point_feat"]], -1))
        valid = item[f"valid_n{N_TARGET}"].to(DEV)
        top3 = logits.masked_fill(~valid, float("-inf")).topk(min(3, int(valid.sum()))).indices.tolist()

        parts = []
        for cid in top3:
            blocks0 = ds.vocab.classes[cid]
            blocks = [list(blocks0[j]) for j in canonical_order([list(b) for b in blocks0])]
            masks, gt_anchor = group_mask_and_gt(ds, gid, ridx, blocks, pts_m, r_obj, ds.hm_tasks[gid])
            if masks is None:
                continue
            n_g = len(blocks)
            ug = np.full(N_UNIT, -1, np.int64)
            for gi, blk in enumerate(blocks):
                for u in blk:
                    ug[UNITS.index(u)] = gi
            ug_t = torch.from_numpy(ug).unsqueeze(0).to(DEV)
            mask_t = torch.from_numpy(masks).to(DEV)
            with torch.no_grad():
                G = F.one_hot(ug_t.clamp(min=0), 4).to(fu.dtype) * (ug_t >= 0)[..., None].to(fu.dtype)
                Tg = model.tok_proj(torch.einsum("bug,buh->bgh", G, fu))

                proj1 = model.wrench_proj({"cond_pos": pos, "cond_nrm": nrm}) if sup else None

                def logp_k(prefix, k):
                    """prefix: [M, k] 점 인덱스 → [M, P] log P_k."""
                    M_ = prefix.shape[0]
                    ep = e_p.expand(M_, -1, -1)
                    cum = torch.zeros_like(ep)
                    if sup:
                        proj = proj1.expand(M_, -1, -1)
                        h = torch.zeros(M_, proj.shape[2], dtype=proj.dtype, device=proj.device)
                    for j in range(k):
                        cum = cum + model.phi(ep, pos.expand(M_, -1, -1), nrm.expand(M_, -1, -1), prefix[:, j])
                        if sup:
                            h = model.support_update(h, proj, prefix[:, j])
                    z = ep + Tg[:, k, None] + cum
                    if sup:
                        z = z + model.support_term(proj, h, ep)
                    lg = model.sc_mlp(z).squeeze(-1).float()
                    return lg.masked_fill(~mask_t[k][None], float("-inf")).log_softmax(-1)

                # 탐욕
                gr = torch.zeros(1, 0, dtype=torch.long, device=DEV)
                for k in range(n_g):
                    gr = torch.cat([gr, logp_k(gr, k).argmax(-1, keepdim=True)], 1)
                # 빔: 첫 그룹 상위 8 → 둘째 확장 → 결합 상위 8 (팔 3대: 앞 자리 2개)
                lp0 = logp_k(torch.zeros(1, 0, dtype=torch.long, device=DEV), 0)[0]
                v0, i0 = lp0.topk(min(BEAM, int(mask_t[0].sum())))
                beams, scores = i0[:, None], v0
                for k in range(1, n_g - 1):
                    lpk = logp_k(beams, k)
                    v, ix = lpk.topk(min(BEAM, int(mask_t[k].sum())), -1)
                    cand = torch.cat([beams.repeat_interleave(v.shape[1], 0), ix.reshape(-1, 1)], 1)
                    sc = (scores[:, None] + v).reshape(-1)
                    keep = sc.topk(min(BEAM, len(sc))).indices
                    beams, scores = cand[keep], sc[keep]
                lpl = logp_k(beams, n_g - 1)                           # [8,P]
                vl, il = lpl.topk(min(8, int(mask_t[n_g - 1].sum())), -1)
            beams_np = beams.cpu().numpy()
            combos_net = np.concatenate([np.repeat(beams_np, il.shape[1], 0),
                                         il.cpu().numpy().reshape(-1, 1)], 1)
            last_pts = np.nonzero(masks[n_g - 1])[0]
            combos_lp = np.concatenate([np.repeat(beams_np, len(last_pts), 0),
                                        np.tile(last_pts, len(beams_np))[:, None]], 1)
            greedy = gr.cpu().numpy()[0]
            a_gr = _batched_alpha(samples, greedy[None].astype(np.int64), MU_T, np.full(1, mu_tor), DEV)[0]
            a_net = _batched_alpha(samples, combos_net.astype(np.int64), MU_T, np.full(len(combos_net), mu_tor), DEV)
            a_lp = _batched_alpha(samples, combos_lp.astype(np.int64), MU_T, np.full(len(combos_lp), mu_tor), DEV)
            key = canon_key_of_blocks(blocks)
            v, ma_rec = table.get(gid, {}).get(key, ("", np.nan))
            parts.append(dict(cid=cid, blocks=blocks, ug=ug, n_g=n_g, masks=masks, gt_anchor=gt_anchor,
                              greedy=greedy, greedy_min=float(a_gr.min()),
                              net_combos=combos_net, net_min=a_net.min(1),
                              lp_combos=combos_lp, lp_min=a_lp.min(1), lp_alpha=a_lp,
                              lp0=lp0.cpu().numpy(), rec_verdict=v, rec_alpha=ma_rec))
        if not parts:
            continue
        # 후보 6개: 파티션마다 beam_lp 상위 2개 다듬기
        cands = []
        for pi, p in enumerate(parts):
            order = np.argsort(-p["lp_min"])
            seen, taken = set(), 0
            for j in order:
                if taken >= 2:
                    break
                c0, a0 = p["lp_combos"][j], p["lp_alpha"][j]
                cr, ar = refine(samples, c0, mu_tor)
                if ar.min() < a0.min():
                    cr, ar = c0, a0
                kk = tuple(int(t) for t in cr)
                if kk in seen:
                    continue
                seen.add(kk)
                taken += 1
                cands.append(dict(part=pi, combo=np.asarray(cr, np.int32), alpha14=np.asarray(ar, np.float32),
                                  min_alpha=float(ar.min())))
        cands.sort(key=lambda c: -c["min_alpha"])
        mined.append(dict(gid=gid, ds_oid=ds.obj_of[gid], ts=ts, r_obj=r_obj, pts=pts_m.astype(np.float32),
                          nrm=obj["nrm"].astype(np.float16), com=samples.com.copy(), parts=parts,
                          cands=cands, gt_best_pass=gt_pass[(gid, N_TARGET)], mu_tor=mu_tor))
        if len(mined) % 40 == 0:
            print(f"  {len(mined)}/{n_want} ({time.time() - t0:.0f}s)", flush=True)

    # ── 다듬기 전 LP 비교 (파티션 3개 중 최선) ──
    def best_ratio(f):
        return np.array([max(f(p) for p in m["parts"]) / m["gt_best_pass"] for m in mined])
    for name, f in (("팔마다 탐욕 1등 조합", lambda p: p["greedy_min"]),
                    ("빔 + 셋째도 네트워크 상위 8 (64개)", lambda p: p["net_min"].max()),
                    ("빔 + 셋째는 마스크 전체 LP", lambda p: p["lp_min"].max())):
        r = best_ratio(f)
        print(f"{name}: α÷GT 최선 중앙값 {np.median(r):.2f}, 0.8 이상 {np.mean(r >= 0.8):.2f}")
    print("빔 + LP 조합 수 중앙값:", int(np.median([len(p["lp_combos"]) for m in mined for p in m["parts"]])))

    # ── 시뮬 ──
    jobs, idx = [], []
    for gi, m in enumerate(mined):
        s = SurfaceSamples(points=m["pts"].astype(np.float64), normals=m["nrm"].astype(np.float64),
                           r_obj=m["r_obj"], com=m["com"])
        for ci, c in enumerate(m["cands"]):
            jobs.append(build_sim_job(len(jobs), m["ds_oid"][0], m["ds_oid"][1], m["ts"], m["r_obj"], s,
                                      c["combo"], c["alpha14"].astype(np.float64), m["mu_tor"]))
            idx.append((gi, ci))
    print(f"시뮬 {len(jobs)}개", flush=True)
    with get_context("spawn").Pool(SIM_WORKERS) as pool:
        for k, verdict, _nu in pool.imap_unordered(_worker, jobs, chunksize=2):
            gi, ci = idx[k]
            mined[gi]["cands"][ci]["verdict"] = verdict
    for m in mined:
        for c in m["cands"]:
            c["success"] = c["verdict"] == "pass" and c["min_alpha"] >= 0.8 * m["gt_best_pass"]
    N = len(mined)
    for K in range(1, 7):
        k = sum(any(c["success"] for c in m["cands"][:K]) for m in mined)
        print(f"  {K}발: {k}/{N} = {100 * k / N:.1f}%")
    print("판정:", dict(Counter(c["verdict"] for m in mined for c in m["cands"])))
    with open(out_path, "wb") as f:
        pickle.dump(mined, f)
    print(f"저장 {out_path}, 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
