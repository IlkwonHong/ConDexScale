"""net2 시각화용 사례 채굴 — 1발 파이프라인(e2e) 재현 + 시뮬 판정 + 4유형 분류.

파이프라인: net1 top-3 파티션 → 파티션별 net2 히트맵 → 그룹당 top-8
→ 8^N 조합 배치 LP → 최선 조합 이웃(7점) 다듬기 → 전체 1등 = 1발 → 짧은 시뮬.

유형: 1 성공+GT근처 / 2 성공+다른자리 / 3 실패+분지미달 / 4 실패+선별오류 / 0 기타
주의: 학습과 달리 마스크에 GT 정답점을 강제 포함하지 않는다 (추론 공정성).
출력: scratchpad/net2_viz_cases.pkl
실행: ./venv/bin/python scratchpad/net2_viz_mine.py   (spawn+가드 필수 — 08-02 OOM 사고)
"""
import itertools
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contraction_net.stage2_dataset import (Stage2Dataset, UNITS, RHO_REL,
                                            MARGIN_REL)
from contraction_net.stage1_dataset import _LINK_NAMES, _LINK_UNIT, N_UNIT
from contraction_net.stage2_model import Stage2Net
from contraction_gt.search import _batched_alpha, mu_tor_from_pad
from contraction_gt.surface import SurfaceSamples
from contraction_gt.metric import direction_set
from contraction_gt.pilot_sim import _equilibrium_preload, ASSET_ROOT
from contraction_gt.gate_runner import _worker, GAMMA, LEVEL_FLOOR

MU_T = 0.6
PAD_REL = 0.10               # N>=3 파이프라인 규약 (팁 반경 무의미 판정)
N_TARGET = 3
M_GRASP = 240                # 채굴 대상 파지 수
SIM_WORKERS = 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_KEYS = ("patch_feat", "patch_local", "patch_local_mask", "unit_feat",
              "rel_type", "log_robj", "point_feat", "point_patch")


def canon_key_of_blocks(blocks):
    return tuple(sorted(tuple(sorted(b)) for b in blocks))


def load_record_table(records_dir, val_gids):
    """val 파지의 (파티션 → verdict·min_alpha) 표 + GT 최고 합격 α."""
    cols = {k: [] for k in ["grasp_id", "n_anchors", "partition", "pad_rel",
                            "min_alpha"]}
    for p in sorted(Path(records_dir).glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            for k in cols:
                cols[k].append(z[k])
    rec = {k: np.concatenate(v) for k, v in cols.items()}
    n_total = len(rec["grasp_id"])
    verdict = np.full(n_total, "", dtype=object)
    for name in ("gate_records.npz", "gate_records_all.npz"):
        gp = Path(records_dir).parent / name
        if gp.exists():
            with np.load(gp, allow_pickle=True) as z:
                verdict[z["idx"].astype(int)] = z["verdict"]
    table = defaultdict(dict)      # gid -> canon_key -> (verdict, min_alpha)
    gt_pass = defaultdict(float)   # (gid, n) -> 최고 합격 α
    for i in range(n_total):
        g = str(rec["grasp_id"][i])
        if g not in val_gids:
            continue
        n = int(rec["n_anchors"][i])
        if n < 3:
            continue
        key = canon_key_of_blocks(json.loads(rec["partition"][i]))
        table[g][key] = (str(verdict[i]), float(rec["min_alpha"][i]))
        if verdict[i] == "pass":
            gt_pass[(g, n)] = max(gt_pass[(g, n)], float(rec["min_alpha"][i]))
    return table, gt_pass


def group_mask_and_gt(ds, gid, ridx, blocks, pts_m, r_obj, rec_tasks):
    """파티션(blocks)의 그룹별 후보 마스크 + GT 앵커(내 그룹 순서로 정렬)."""
    con = ds._contacts.get(gid.rsplit("#", 1)[0])
    lp = con["link_contact_point"][ridx]
    lc = con["link_contact"][ridx]
    la = con["link_area"][ridx]
    # 이 파티션의 GT 레코드 찾기 (그룹 구성 집합이 같은 것)
    my_sets = [frozenset(b) for b in blocks]
    gt_anchor = None
    for ug_r, n_r, ap in rec_tasks:
        if n_r != len(blocks):
            continue
        rec_sets = {}
        for u in range(N_UNIT):
            if ug_r[u] >= 0:
                rec_sets.setdefault(int(ug_r[u]), set()).add(UNITS[u])
        rec_fs = [frozenset(s) for _, s in sorted(rec_sets.items())]
        if set(rec_fs) == set(my_sets):
            order = [rec_fs.index(s) for s in my_sets]
            gt_anchor = ap[order]
            break
    if gt_anchor is None:
        return None, None
    n_g = len(blocks)
    masks = np.zeros((n_g, len(pts_m)), bool)
    for g, blk in enumerate(blocks):
        members = [UNITS.index(u) for u in blk]
        li = [j for j in range(len(_LINK_NAMES))
              if _LINK_UNIT[j] in members and lc[j]]
        if li:
            w = la[li] / max(la[li].sum(), 1e-12)
            center = (lp[li] * w[:, None]).sum(0)
            spread = float(np.linalg.norm(lp[li] - center, axis=1).max())
        else:
            center, spread = gt_anchor[g], 0.0
        radius = max(spread, RHO_REL * r_obj) + MARGIN_REL * r_obj
        masks[g] = np.linalg.norm(pts_m - center, axis=1) <= radius
        if not masks[g].any():                     # 폴백: 최근접 32점
            masks[g][np.argsort(np.linalg.norm(pts_m - center, axis=1))[:32]] = True
    return masks, gt_anchor


def refine(samples, combo, mu_tor, k=7):
    """각 앵커의 최근접 k점(자기 포함)으로 조합 재채점 → 더 좋으면 교체."""
    sets_ = []
    for idx in combo:
        d = np.linalg.norm(samples.points - samples.points[idx], axis=1)
        sets_.append(np.argsort(d)[:k])
    combos = np.array(list(itertools.product(*sets_)), np.int64)
    a = _batched_alpha(samples, combos, MU_T, np.full(len(combos), mu_tor), DEV)
    b = int(a.min(1).argmax())
    return combos[b], a[b]


def build_sim_job(i, ds_name, oid, ts, r_obj, samples, combo, alpha14, mu_tor):
    pos = samples.points[combo]
    nrm = samples.normals[combo]
    pos_n = torch.as_tensor((pos - samples.com) / r_obj,
                            dtype=torch.float64).unsqueeze(0)
    dirs_all = direction_set(pos_n)[0].numpy()
    dir_set = sorted(set(np.argsort(alpha14)[:3].tolist()) | set(range(6)))
    levels = np.maximum(alpha14[dir_set] * GAMMA, LEVEL_FLOOR)[:, None]
    preload, _x, _W = _equilibrium_preload(
        (pos - samples.com) / r_obj, nrm, MU_T, mu_tor / r_obj, 5.0)
    return dict(
        idx=i,
        coacd=str(ASSET_ROOT / ds_name / "processed_data" / oid / "urdf" / "meshes"),
        scale=ts, pos=pos, nrm=nrm, dirs=dirs_all[dir_set], levels=levels,
        r_obj=r_obj, mu_tor=mu_tor, preload=preload,
        tip_radius=max(PAD_REL * r_obj, 0.002))


def main():
    ck_path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "checkpoints/stage2_featnorm/best.pt")
    out_path = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "scratchpad/net2_viz_cases.pkl")
    # 2026-11-08: 세 번째 인자 = 이미 쓴 파지 pkl. 그 안의 gid 는 건너뛰어 '새 파지'만 채굴한다 (시험용 묶음).
    exclude = set()
    if len(sys.argv) > 3:
        exclude = {m["gid"] for m in pickle.load(open(ROOT / sys.argv[3], "rb"))}
        print(f"제외 파지 {len(exclude)}개 ({sys.argv[3]})", flush=True)
    interp = bool(torch.load(ck_path, map_location="cpu", weights_only=False)["args"].get("patch_interp"))
    ds = Stage2Dataset(split="val", contact_geom=False, com_feat=True,
                       feat_norm=True, patch_interp=interp)
    keys = BATCH_KEYS + (("point_patch3", "point_patch3_w") if interp else ())
    print(f"모델 {ck_path}, 패치 섞기 {interp}", flush=True)
    print(f"val 파지 {len(ds)}개", flush=True)
    model = Stage2Net(ds.vocab, d_pf=11, d_ptf=10)
    ck = torch.load(ck_path, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(DEV).eval()

    val_gids = {g for g, _ in ds.samples}
    table, gt_pass = load_record_table(
        ROOT / "data/contraction_gt_300k/search_records", val_gids)
    print(f"판정표 파지 {len(table)}개", flush=True)

    mined = []
    for i in range(len(ds)):
        if len(mined) >= M_GRASP:
            break
        gid, e = ds.samples[i]
        if gid in exclude:
            continue
        if gt_pass.get((gid, N_TARGET), 0.0) <= 0.0:
            continue                      # 구제 불능(N=3) — 1발 분모 밖
        item = ds[i]
        if not item[f"valid_n{N_TARGET}"].any():
            continue
        r_obj, ts = e["r_obj"], e["total_scale"]
        rel, ridx = gid.rsplit("#", 1)
        ridx = int(ridx)
        obj = ds._objects.get(ds.obj_of[gid])
        pts_m = (obj["pts"] * ts).astype(np.float64)
        samples = SurfaceSamples(points=pts_m, normals=obj["nrm"].astype(np.float64),
                                 r_obj=r_obj, com=(obj["com"] * ts).astype(np.float64))
        mu_tor = mu_tor_from_pad(PAD_REL * r_obj, MU_T)

        batch = {k: item[k].unsqueeze(0).to(DEV) for k in keys}
        with torch.no_grad():
            x, h = model.encode(batch)
            units = x[:, model.n_patch:model.n_patch + N_UNIT]
            fu = model.phi_u(units)
            T = torch.einsum("cgu,bud->bcgd", model.member.to(fu.dtype), fu)
            ctx = model.ctx_proj(h)[:, None, None, :].expand(-1, T.shape[1], 4, -1)
            size = model.gsize.to(fu.dtype)[None, :, :, None].expand(1, -1, -1, -1)
            g_ = model.phi_g(torch.cat([T, ctx, size], -1))
            g_ = g_ * model.gmask.to(fu.dtype)[None, :, :, None]
            logits = model.psi(g_.sum(2)).squeeze(-1)[0]
            if interp:
                i3, w3 = batch["point_patch3"], batch["point_patch3_w"].to(x.dtype)
                xp = sum(torch.gather(x[:, :model.n_patch], 1, i3[..., j, None].expand(-1, -1, x.shape[-1]))
                         * w3[..., j, None] for j in range(i3.shape[-1]))
            else:
                xp = torch.gather(x[:, :model.n_patch], 1,
                                  batch["point_patch"][..., None].expand(-1, -1, x.shape[-1]))
            e_p = model.pt_mlp(torch.cat([xp, batch["point_feat"]], -1))

        valid = item[f"valid_n{N_TARGET}"].to(DEV)
        lg = logits.masked_fill(~valid, float("-inf"))
        top3 = lg.topk(min(3, int(valid.sum()))).indices.tolist()

        parts = []
        for cid in top3:
            blocks = ds.vocab.classes[cid]
            masks, gt_anchor = group_mask_and_gt(
                ds, gid, ridx, blocks, pts_m, r_obj, ds.hm_tasks[gid])
            if masks is None:
                continue
            n_g = len(blocks)
            ug = np.full(N_UNIT, -1, np.int8)
            for gi, blk in enumerate(blocks):
                for u in blk:
                    ug[UNITS.index(u)] = gi
            ug_t = torch.from_numpy(ug.astype(np.int64)).unsqueeze(0).to(DEV)
            with torch.no_grad():
                G = torch.nn.functional.one_hot(ug_t.clamp(min=0), 4).to(fu.dtype)
                G = G * (ug_t >= 0)[..., None].to(fu.dtype)
                Tg = torch.einsum("bug,buh->bgh", G, fu)
                heat = model.sc_mlp(
                    e_p[:, None] + model.tok_proj(Tg)[:, :, None]).squeeze(-1)[0]
            heat = heat[:n_g].cpu().numpy()
            top8 = []
            for g in range(n_g):
                hg = np.where(masks[g], heat[g], -np.inf)
                top8.append(np.argsort(hg)[::-1][:min(8, int(masks[g].sum()))])
            combos = np.array(list(itertools.product(*top8)), np.int64)
            a = _batched_alpha(samples, combos, MU_T,
                               np.full(len(combos), mu_tor), DEV)
            best = int(a.min(1).argmax())
            combo_r, alpha_r = refine(samples, combos[best], mu_tor)
            if alpha_r.min() < a[best].min():
                combo_r, alpha_r = combos[best], a[best]
            key = canon_key_of_blocks(blocks)
            v, ma_rec = table.get(gid, {}).get(key, ("", np.nan))
            parts.append(dict(cid=cid, ug=ug, n_g=n_g, masks=masks,
                              heat=heat.astype(np.float16),
                              top8=[t.astype(np.int32) for t in top8],
                              combo=combo_r.astype(np.int32),
                              alpha14=alpha_r.astype(np.float32),
                              min_alpha=float(alpha_r.min()),
                              gt_anchor=gt_anchor.astype(np.float32),
                              rec_verdict=v, rec_alpha=ma_rec))
        if not parts:
            continue
        sel = int(np.argmax([p["min_alpha"] for p in parts]))
        mined.append(dict(gid=gid, ds_oid=ds.obj_of[gid], ts=ts, r_obj=r_obj,
                          pts=pts_m.astype(np.float32),
                          nrm=obj["nrm"].astype(np.float16),
                          com=samples.com.copy(),
                          parts=parts, sel=sel,
                          gt_best_pass=gt_pass[(gid, N_TARGET)]))
        if len(mined) % 40 == 0:
            print(f"채굴 {len(mined)}/{M_GRASP}", flush=True)

    # ── 최종 선택 시뮬 판정 ──
    jobs = []
    for k, m in enumerate(mined):
        p = m["parts"][m["sel"]]
        jobs.append(build_sim_job(k, m["ds_oid"][0], m["ds_oid"][1], m["ts"],
                                  m["r_obj"], SurfaceSamples(
                                      points=m["pts"].astype(np.float64),
                                      normals=m["nrm"].astype(np.float64),
                                      r_obj=m["r_obj"], com=m["com"]),
                                  p["combo"], p["alpha14"].astype(np.float64),
                                  mu_tor_from_pad(PAD_REL * m["r_obj"], MU_T)))
    ctx = get_context("spawn")
    with ctx.Pool(SIM_WORKERS) as pool:
        for idx, verdict, _nu in pool.imap_unordered(_worker, jobs, chunksize=1):
            mined[idx]["verdict"] = verdict
    print("시뮬 완료", flush=True)

    # ── 4유형 분류 ──
    for m in mined:
        p = m["parts"][m["sel"]]
        br = 0.15 * m["r_obj"]
        anc = m["pts"][p["combo"]]
        d_anchor = np.linalg.norm(anc - p["gt_anchor"], axis=1)
        reach = all(np.linalg.norm(m["pts"][t] - p["gt_anchor"][g], axis=1).min() <= br
                    for g, t in enumerate(p["top8"]))
        success = (m["verdict"] == "pass" and
                   p["min_alpha"] >= 0.8 * m["gt_best_pass"])
        pool_has = any(q["rec_verdict"] == "pass" for q in m["parts"])
        if success:
            m["case_type"] = 1 if d_anchor.max() <= br else 2
        elif not reach:
            m["case_type"] = 3
        elif pool_has:
            m["case_type"] = 4
        else:
            m["case_type"] = 0
    from collections import Counter
    print("유형 분포:", Counter(m["case_type"] for m in mined))
    print("판정 분포:", Counter(m["verdict"] for m in mined))
    out = out_path
    with open(out, "wb") as f:
        pickle.dump(mined, f)
    print(f"저장: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
