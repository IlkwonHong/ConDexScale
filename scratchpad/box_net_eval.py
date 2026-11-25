"""박스 네트워크 수축 평가 — 손별 Stage1+2 추론 → 양손 합본 α 9칸 격자.

손별: net1(파티션 top-1, N=2는 규칙) → net2 히트맵 → 그룹당 상위 8점
합본: 왼손 그룹 + 오른손 그룹 → 조합 LP(총 4그룹 이하는 전수, 초과는 좌표 상승)
      → 6-이웃 다듬기(좌표 상승) → 14방향 min-α
비교 기준: 수학 탐색 팔4대 0.589/0.624/0.602 (키프레임별), 6월 k-means 0.02~0.04
실행: ./venv/bin/python scratchpad/box_net_eval.py
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from contraction_gt.surface import sample_surface
from contraction_gt.search import _batched_alpha, mu_tor_from_pad
from contraction_net.partition_vocab import PartitionVocab
from contraction_net.stage1_dataset import N_UNIT
from contraction_net.stage2_model import Stage2Net

import os
RANDOM_CANDS = bool(int(os.environ.get("RANDOM_CANDS", "0")))
UNITS = ["index", "middle", "ring", "little", "thumb", "palm"]
SEQ = sys.argv[1] if len(sys.argv) > 1 else "box_grab_01"
FRAMES = [int(a) for a in sys.argv[2:]] or [90, 300, 630]
K_TOP, KNN = 8, 6
RHO_REL, MARGIN_REL, MU_T = 0.15, 0.05, 0.6
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from arctic_scene import scene


def load_smp(fr):
    return scene(SEQ, fr, sample_surface)[0]


def math_search(smp, hands22, mu_tor):
    """(2,2) 그룹 정의 그대로, 전체 마스크 좌표 상승 = 마스크 물리 상한."""
    cand_sets, cur = [], []
    for side in ("left", "right"):
        blocks, _ = hands22[side]
        b = hands22[side + "_batch"]
        cpos, cunit = b["_cpos"].numpy(), b["_cunit"].numpy()
        for blk in blocks:
            mem = [UNITS.index(u) for u in blk]
            sel = np.isin(cunit, mem)
            center = cpos[sel].mean(0)
            spread = float(np.linalg.norm(cpos[sel] - center, axis=1).max())
            radius = max(spread, RHO_REL * smp.r_obj) + MARGIN_REL * smp.r_obj
            d = np.linalg.norm(smp.points - center, axis=1)
            cand = np.where(d <= radius)[0]
            cand_sets.append(cand if len(cand) else np.argsort(d)[:8])
            cur.append(int(np.argmin(d)))
    for _ in range(2):
        for gi, cand in enumerate(cand_sets):
            combos = np.tile(cur, (len(cand), 1))
            combos[:, gi] = cand
            a = _batched_alpha(smp, combos, MU_T,
                               np.full(len(combos), mu_tor), DEV).min(1)
            cur[gi] = int(cand[int(np.nanargmax(a))])
    a14 = _batched_alpha(smp, np.array([cur]), MU_T, np.array([mu_tor]), DEV)[0]
    return float(np.nanmin(a14)), cur, a14


def hand_candidates(model, vocab, b, smp, n):
    """한 손, 팔 n개: 파티션 선택 + 그룹별 히트맵 상위 8점. → (blocks, cand_sets)"""
    bb = {k: (v.unsqueeze(0).to(DEV) if isinstance(v, torch.Tensor) else v)
          for k, v in b.items() if not k.startswith("_")}
    with torch.no_grad():
        x, h = model.encode(bb)
        fu = model.phi_u(x[:, model.n_patch:model.n_patch + N_UNIT])
        xp = torch.gather(x[:, :model.n_patch], 1,
                          bb["point_patch"][..., None].expand(-1, -1, x.shape[-1]))
        e_p = model.pt_mlp(torch.cat([xp, bb["point_feat"]], -1))
        umask = np.array([bool((b["_cunit"] == u).any()) for u in range(N_UNIT)])
        present = [UNITS[u] for u in range(N_UNIT) if umask[u]]
        if n == 2:                                     # 규칙: 앵커0 | 나머지
            a0 = b["_anchor0"]
            blocks = [[a0], [u for u in present if u != a0]]
        else:
            vmask = b[f"valid_n{n}"].to(DEV)
            if not vmask.any():
                return None, None
            Tm = torch.einsum("cgu,bud->bcgd", model.member.to(fu.dtype), fu)
            ctx = model.ctx_proj(h)[:, None, None, :].expand(-1, Tm.shape[1], 4, -1)
            size = model.gsize.to(fu.dtype)[None, :, :, None].expand(1, -1, -1, -1)
            gg = model.phi_g(torch.cat([Tm, ctx, size], -1)) * \
                model.gmask.to(fu.dtype)[None, :, :, None]
            logits = model.psi(gg.sum(2)).squeeze(-1)[0]
            c = int(logits.masked_fill(~vmask, float("-inf")).argmax())
            blocks = [list(blk) for blk in vocab.classes[c]]

        cpos, cunit = b["_cpos"].numpy(), b["_cunit"].numpy()
        cands = []
        for blk in blocks:
            mem = [UNITS.index(u) for u in blk]
            sel = np.isin(cunit, mem)
            if not sel.any():
                return None, None
            center = cpos[sel].mean(0)
            spread = float(np.linalg.norm(cpos[sel] - center, axis=1).max())
            radius = max(spread, RHO_REL * smp.r_obj) + MARGIN_REL * smp.r_obj
            m = np.linalg.norm(smp.points - center, axis=1) <= radius
            G1 = torch.zeros(1, N_UNIT, 4, device=DEV, dtype=fu.dtype)
            for u in mem:
                G1[0, u, 0] = 1
            Tg = torch.einsum("bug,buh->bgh", G1, fu)
            heat = model.sc_mlp(e_p + model.tok_proj(Tg[:, 0])[:, None]).squeeze(-1)[0]
            hm = heat.float().cpu().numpy()
            hm[~m] = -np.inf
            k = min(K_TOP, int(m.sum()))
            if RANDOM_CANDS:                # 공정성 검사: 마스크 안 무작위 8점
                rseed = int(os.environ.get("RSEED", "0"))
                rng = np.random.default_rng(len(cands) * 977 + k + rseed * 10007)
                cands.append(rng.choice(np.where(m)[0], k, replace=False))
            else:
                cands.append(np.argsort(hm)[::-1][:k].copy())
        return blocks, cands


def best_combo(smp, cand_sets, mu_tor):
    """조합 선택: 총 조합 ≤32768 전수 LP, 초과는 좌표 상승. 이후 6-이웃 좌표 상승."""
    sizes = [len(c) for c in cand_sets]
    total = np.prod([float(s) for s in sizes])
    if total <= 32768:
        combos = np.array(list(product(*cand_sets)))
        a = _batched_alpha(smp, combos, MU_T, np.full(len(combos), mu_tor), DEV)
        cur = list(combos[int(np.nanmin(a, 1).argmax())])
    else:
        cur = [int(c[0]) for c in cand_sets]
        for _ in range(2):
            for gi, cand in enumerate(cand_sets):
                combos = np.tile(cur, (len(cand), 1))
                combos[:, gi] = cand
                a = _batched_alpha(smp, combos, MU_T,
                                   np.full(len(combos), mu_tor), DEV).min(1)
                cur[gi] = int(cand[int(np.nanargmax(a))])
    for _ in range(2):                                 # 6-이웃 다듬기
        for gi in range(len(cur)):
            nb = np.argsort(np.linalg.norm(smp.points - smp.points[cur[gi]],
                                           axis=1))[:KNN + 1]
            combos = np.tile(cur, (len(nb), 1))
            combos[:, gi] = nb
            a = _batched_alpha(smp, combos, MU_T,
                               np.full(len(combos), mu_tor), DEV).min(1)
            cur[gi] = int(nb[int(np.nanargmax(a))])
    a14 = _batched_alpha(smp, np.array([cur]), MU_T, np.array([mu_tor]), DEV)[0]
    return cur, a14


def main():
    vocab = PartitionVocab.load()
    ck = torch.load(ROOT / "checkpoints/stage2_featnorm/best.pt",
                    map_location=DEV, weights_only=False)
    model = Stage2Net(vocab, d_pf=11, d_ptf=10).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    bpath = ROOT / f"scratchpad/net_batches_{SEQ}.pt"
    if not bpath.exists():
        bpath = ROOT / "scratchpad/box_net_batches.pt"
    batches = torch.load(bpath, weights_only=False)

    for fr in FRAMES:
        smp = load_smp(fr)
        mu_tor = mu_tor_from_pad(0.10 * smp.r_obj, MU_T, 0.5)
        hands = {}
        for side in ("right", "left"):
            b = batches[f"{fr}_{side}"]
            for n in (2, 3, 4):
                hands[(side, n)] = hand_candidates(model, vocab, b, smp, n)
        h22 = {"left": hands[("left", 2)], "right": hands[("right", 2)],
               "left_batch": batches[f"{fr}_left"], "right_batch": batches[f"{fr}_right"]}
        base = (math_search(smp, h22, mu_tor)[0]
                if h22["left"][0] and h22["right"][0] else float("nan"))
        print(f"\n=== 키프레임 {fr:05d} (수학 탐색 팔4대 기준 {base:.3f}) ===")
        print("       왼손N=2      왼손N=3      왼손N=4")
        for nr in (2, 3, 4):
            row = [f"우{nr}"]
            for nl in (2, 3, 4):
                bl, cl = hands[("left", nl)]
                br, cr = hands[("right", nr)]
                if bl is None or br is None:
                    row.append("   불가   ")
                    continue
                idx, a14 = best_combo(smp, cl + cr, mu_tor)
                row.append(f"{np.nanmin(a14):.3f}/중력{a14[5]:.2f}")
            print("  " + "  ".join(f"{c:>12s}" for c in row))
        b22 = (hands[("left", 2)], hands[("right", 2)])
        blocks = [b[0] for b in b22]
        print(f"  (2,2) 파티션: 좌 {blocks[0]} | 우 {blocks[1]}")


if __name__ == "__main__":
    main()
