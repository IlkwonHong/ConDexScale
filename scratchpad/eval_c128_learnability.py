"""EXP-004 평가 — 후보 128개 정답으로 학습한 조건부 net2 가 정답을 그대로 내는가 (2026-11-14).

모델 A (새 정답으로 학습) 와 모델 B (저장된 정답을 가장 가까운 후보로 옮긴 것으로 학습) 를 평가용 검증 파지 480개의
팔 3대 레코드에서 채점한다. 파티션은 레코드의 것을 그대로 주고 (net1 을 거치지 않음), 팔마다 앞 팔의 선택을 보고
점수 1등 후보를 차례로 고른다 (stage2_cond_model.eval_metrics 의 탐욕 디코딩). 체크포인트가 없는 모델은 건너뛴다.

지표: 세 팔 모두 정답과 같은 점 (주 지표, 각자의 정답 기준) · 팔 하나 일치 · 팔마다 상위 3개 안 · 상대 정답과의 일치 ·
      고른 조합의 α ÷ 새 정답(A)의 α ≥ 0.95 (최소 간격을 어기면 미달) · 단순 규칙 · 우연 수준 · 학습 파지에서의 일치율.
읽는 기준 (계획서 고정값): A ≥ 50% 잘 낸다 / A < 25% 못 낸다,  A − B ≥ 15%p 일관성이 원인 / < 5%p 원인이 아니다.

자료 확인 (항상): 데이터셋이 만든 정답 번호 = 레코드의 c128_a_idx, 후보 표시의 점 수 = 레코드의 후보 수.

실행: ./venv/bin/python scratchpad/eval_c128_learnability.py [--smoke] [--models a,b] [--check-train]
계획: harness/research/IDEA-001-imitate-verified-gt/experiments/EXP-004-c128-learnability/plan.md
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

import gt_seed_consistency as G                                              # noqa: E402
import contraction_gt.runner_batched as RB                                   # noqa: E402
from contraction_gt.search import SearchConfig, mu_tor_from_pad              # noqa: E402
from contraction_net.stage2_cond_dataset import Stage2CondDataset, canonical_order   # noqa: E402
from contraction_net.stage2_cond_model import Stage2CondNet, eval_metrics    # noqa: E402

DATA = ROOT / "data/contraction_gt_300k"
N = 3
GOOD, BAD, GAP_YES, GAP_NO = 0.50, 0.25, 0.15, 0.05      # 계획서 고정값
ALPHA_OK = 0.95
N_TRAIN_FIT = 1000


def load_rows(rec_dir):
    """(gid, 규칙 순서로 정렬한 파티션 문자열) → 레코드의 c128_* 값 (팔 순서도 규칙 순서로)."""
    info = {}
    for p in sorted(rec_dir.glob("shard_*.npz")):
        with np.load(p, allow_pickle=False) as z:         # 열마다 한 번만 읽는다 (행마다 z[...] 를 부르면 매번 전체를 푼다)
            c = {k: z[k] for k in ("n_anchors", "c128_ok", "partition", "grasp_id", "c128_a_idx", "c128_b_idx",
                                   "c128_near_idx", "c128_prox_idx", "c128_cand", "c128_alpha", "c128_snap_alpha",
                                   "c128_near_alpha", "c128_prox_alpha")}
        for i in np.nonzero((c["n_anchors"] == N) & c["c128_ok"])[0]:
            blocks = json.loads(str(c["partition"][i]))
            order = canonical_order(blocks)
            key = (str(c["grasp_id"][i]), json.dumps([blocks[j] for j in order]))
            o = np.array(order)
            info[key] = dict(
                a=c["c128_a_idx"][i][:N][o].astype(np.int64), b=c["c128_b_idx"][i][:N][o].astype(np.int64),
                near=c["c128_near_idx"][i][:N][o].astype(np.int64),
                prox=c["c128_prox_idx"][i][:N][o].astype(np.int64),
                n_cand=(c["c128_cand"][i][:N][o] >= 0).sum(1),
                alpha=float(c["c128_alpha"][i]), snap_alpha=float(c["c128_snap_alpha"][i]),
                near_alpha=float(c["c128_near_alpha"][i]), prox_alpha=float(c["c128_prox_alpha"][i]))
    return info


class TaskView(Dataset):
    """(표본 번호, 과제) 쌍마다 그 과제를 강제로 고르게 해서 데이터셋 항목을 만든다."""

    def __init__(self, ds, pairs):
        self.ds, self.pairs = ds, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        i, task = self.pairs[j]
        gid = self.ds.samples[i][0]
        keep = self.ds.cond_tasks[gid]
        self.ds.cond_tasks[gid] = [task]
        try:
            item = self.ds[i]
        finally:
            self.ds.cond_tasks[gid] = keep
        item["_j"] = torch.tensor(j)
        return item


@torch.no_grad()
def decode(model, loader, device):
    """탐욕 조합 [R,4], 팔별 상위 3 [R,4,3], 데이터셋의 정답 번호 [R,4], 후보 표시의 점 수 [R,4], 항목 번호."""
    G_, T3, TG, MC, J = [], [], [], [], []
    model.eval()
    for batch in loader:
        J.append(batch.pop("_j").numpy() if "_j" in batch else np.zeros(0, int))
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast(device, enabled=device == "cuda"):
            out = model(batch)
            _, greedy = eval_metrics(model, out, batch)
            _, _, e_p, Tg = out
            top3 = torch.full((greedy.shape[0], 4, 3), -1, dtype=torch.long, device=greedy.device)
            for k, _rows, logp in model.chain_logits(e_p, Tg, batch, greedy):
                top3[:, k] = logp.topk(3, -1).indices
        G_.append(greedy.cpu().numpy())
        T3.append(top3.cpu().numpy())
        TG.append(batch["cond_tgt"][:, 0].cpu().numpy())
        MC.append(batch["cond_mask"].sum(-1).cpu().numpy())
    cat = lambda x: np.concatenate(x)
    return cat(G_), cat(T3), cat(TG), cat(MC), cat(J)


def load_model(path, vocab, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = Stage2CondNet(vocab, d_pf=11 if a["com_feat"] else 7, d_ptf=10 if a["com_feat"] else 6,
                          wrench_feat=a.get("wrench_feat", "none"), use_cond=not a.get("no_cond", False))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def pct(x):
    return f"{100 * float(np.mean(x)):5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="a,b")
    ap.add_argument("--check-train", action="store_true", help="학습용 레코드 폴더의 자료 확인만 하고 끝낸다")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    dkw = dict(augment=False, contact_geom=False, com_feat=True, feat_norm=True, patch_interp=True,
               cand_from_records=True, only_n=N)
    dirs = {"a": DATA / f"search_records_c128a{sfx}", "b": DATA / f"search_records_c128b{sfx}"}
    ckpt = {m: ROOT / ("checkpoints/_smoke_c128" + m if args.smoke else "checkpoints/stage2_cond_c128" + m) / "best.pt"
            for m in "ab"}

    def check_dir(rec_dir, split, label, key, limit=None):
        """데이터셋이 만든 정답 번호·후보 표시가 레코드의 값과 같은지 확인. 반환: 데이터셋, 쌍 목록, 레코드 값."""
        info = load_rows(rec_dir)
        ds = Stage2CondDataset(split=split, records_dir=rec_dir, **dkw)
        pairs = [(i, t) for i, (gid, _e) in enumerate(ds.samples) for t in ds.cond_tasks[gid]]
        sub = pairs if limit is None else [pairs[j] for j in np.unique(np.linspace(0, len(pairs) - 1, limit).astype(int))]
        bad_t = bad_m = 0
        for j in range(len(sub)):
            item = TaskView(ds, sub)[j]
            gid = ds.samples[sub[j][0]][0]
            r = info[(gid, json.dumps(sub[j][1][0]))]
            bad_t += int(not np.array_equal(item["cond_tgt"][0, :N].numpy(), r[key]))
            bad_m += int(not np.array_equal(item["cond_mask"][:N].sum(-1).numpy(), r["n_cand"]))
        print(f"[자료 확인: {label}] 파지 {len(ds.samples):,}개, 과제 {len(pairs):,}개 중 {len(sub):,}개 확인 — "
              f"정답 번호가 레코드와 다른 과제 {bad_t}, 후보 표시의 점 수가 다른 과제 {bad_m}", flush=True)
        if bad_t or bad_m:
            print("자료 확인 불통과 — 멈춥니다.", flush=True)
            sys.exit(2)
        return ds, pairs, info

    if args.check_train:
        for m in "ab":
            check_dir(dirs[m], "train", f"학습용 {m.upper()}", m, limit=200)
            check_dir(dirs[m], "val", f"모델 선택용 {m.upper()}", m, limit=200)
        return

    ds, pairs, info = check_dir(DATA / f"search_records_c128_eval{sfx}", "val", "평가용", "a")
    rows = [info[(ds.samples[i][0], json.dumps(t[0]))] for i, t in pairs]
    A = np.stack([r["a"] for r in rows])
    B = np.stack([r["b"] for r in rows])
    ncand = np.stack([r["n_cand"] for r in rows]).astype(np.float64)
    alpha_a = np.array([r["alpha"] for r in rows])
    R = len(rows)
    print(f"평가 레코드 {R:,}개 (파지 {len(ds.samples)}개), 팔마다 후보 수 중앙값 {np.median(ncand):.0f}", flush=True)

    # 물체 표면점 (α 계산·최소 간격 확인용)
    cfg = SearchConfig()
    samp = {}
    for i, _t in pairs:
        gid, e = ds.samples[i]
        if gid not in samp:
            d_, oid = ds.obj_of[gid]
            s = G.load_samples(ROOT / G.CACHE_ROOT, d_, oid, scale=e["total_scale"])
            pad = cfg.pad_radius_rel_multi[0]
            samp[gid] = (s, mu_tor_from_pad(pad * s.r_obj, cfg.mu_t, cfg.hertz_frac), max(pad * s.r_obj, 0.002))

    def alpha_of(combos):
        """조합 [R,3] 의 min α 와 최소 간격 충족 여부."""
        entries, feas = [], []
        for (i, _t), c in zip(pairs, combos):
            s, mu, tip = samp[ds.samples[i][0]]
            p = s.points[c]
            dmin = min(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[2]))
            feas.append(dmin >= 2.4 * tip)
            entries.append((s, c[None].astype(np.int64), mu))
        return RB._alpha_many(entries, cfg).min(1), np.array(feas)

    def summarize(name, combo, own):
        a, feas = alpha_of(combo)
        ratio = np.where(alpha_a > 0, a / alpha_a, 0.0)
        s = dict(all_own=float((combo == own).all(1).mean()), arm_own=float((combo == own).mean()),
                 all_a=float((combo == A).all(1).mean()), all_b=float((combo == B).all(1).mean()),
                 alpha_ok=float(((ratio >= ALPHA_OK) & feas).mean()), ratio_median=float(np.median(ratio)),
                 infeasible=float((~feas).mean()))
        print(f"  {name}: 세 팔 모두 일치 {pct((combo == own).all(1))} | 팔 하나 일치 {pct(combo == own)} | "
              f"새 정답(A)과 세 팔 모두 일치 {pct((combo == A).all(1))} | 옮긴 저장 정답(B)과 세 팔 모두 일치 {pct((combo == B).all(1))} | "
              f"고른 조합의 α 가 새 정답의 {ALPHA_OK}배 이상 {pct((ratio >= ALPHA_OK) & feas)} (α 비율 중앙값 {s['ratio_median']:.3f}, "
              f"최소 간격을 어긴 조합 {pct(~feas)})", flush=True)
        return s

    res = {}
    print("[기준선 — 같은 평가 레코드]")
    res["chance"] = dict(arm=float((1 / ncand).mean()), all=float((1 / ncand).prod(1).mean()))
    print(f"  우연 수준: 팔 하나 {pct(1 / ncand)} | 세 팔 모두 {pct((1 / ncand).prod(1))}")
    res["rule_near"] = summarize("단순 규칙 (접촉 중심에서 가장 가까운 후보)", np.stack([r["near"] for r in rows]), A)
    res["rule_prox"] = summarize("단순 규칙 (값싼 점수 1등 조합)", np.stack([r["prox"] for r in rows]), A)
    res["label_b"] = summarize("옮긴 저장 정답(B) 자체", B, A)

    loader = DataLoader(TaskView(ds, pairs), batch_size=96, num_workers=4)
    for m in args.models.split(","):
        if not ckpt[m].exists():
            print(f"[모델 {m.upper()}] 체크포인트가 없어 건너뜁니다: {ckpt[m]}", flush=True)
            continue
        model, ck = load_model(ckpt[m], ds.vocab, device)
        greedy, top3, tgt, mcnt, J = decode(model, loader, device)
        order = np.argsort(J)
        greedy, top3 = greedy[order][:, :N], top3[order][:, :N]
        own = A if m == "a" else B
        print(f"[모델 {m.upper()}] {ckpt[m].parent.name}, 최선 에폭 {ck['epoch']}, 검증 손실 {ck['metrics'].get('hm_main', float('nan')):.4f}")
        s = summarize("평가 레코드", greedy, own)
        s["top3_arm"] = float((top3 == own[:, :, None]).any(-1).mean())
        s["epoch"] = int(ck["epoch"])
        print(f"  팔마다 상위 3개 안에 정답 {pct((top3 == own[:, :, None]).any(-1))}")
        # 학습 파지에서의 일치율 (외우지도 못하는가 / 일반화를 못하는가)
        tr = Stage2CondDataset(split="train", records_dir=dirs[m], **dkw)
        idx = np.unique(np.linspace(0, len(tr) - 1, min(N_TRAIN_FIT, len(tr))).astype(int))
        g2, _t3, tg2, _mc, _j = decode(model, DataLoader(Subset(tr, idx.tolist()), batch_size=96, num_workers=4), device)
        s["train_all"] = float((g2[:, :N] == tg2[:, :N]).all(1).mean())
        s["train_arm"] = float((g2[:, :N] == tg2[:, :N]).mean())
        print(f"  학습 파지 {len(idx)}개(파지마다 과제 하나): 세 팔 모두 일치 {pct((g2[:, :N] == tg2[:, :N]).all(1))} | "
              f"팔 하나 일치 {pct(g2[:, :N] == tg2[:, :N])}", flush=True)
        res["model_" + m] = s
        del model

    print("\n[판정 — 계획서 고정 기준, 주 지표 = 세 팔 모두 각자의 정답과 같은 점]")
    if "model_a" in res:
        pa = res["model_a"]["all_own"]
        v = "잘 낸다" if pa >= GOOD else ("못 낸다" if pa < BAD else "그 사이 — 연구자가 정한다")
        print(f"  모델 A {100 * pa:.1f}% → {v} (기준: {100 * GOOD:.0f}% 이상 / {100 * BAD:.0f}% 미만)")
        if "model_b" in res:
            gap = pa - res["model_b"]["all_own"]
            v = ("정답의 일관성이 원인이다" if gap >= GAP_YES else
                 ("일관성은 원인이 아니다" if gap < GAP_NO else "그 사이 — 연구자가 정한다"))
            print(f"  모델 B {100 * res['model_b']['all_own']:.1f}%, A − B = {100 * gap:+.1f}%포인트 → {v} "
                  f"(기준: {100 * GAP_YES:.0f}%포인트 이상 / {100 * GAP_NO:.0f}%포인트 미만)")
        else:
            print("  모델 B 의 결과가 아직 없어 A − B 판정은 보류합니다.")
    tag = "" if args.models == "a,b" else "_" + args.models.replace(",", "")     # 일부 모델만 평가하면 따로 저장
    out = ROOT / f"scratchpad/eval_c128_learnability{tag}{sfx}.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(res=res, n_rec=R, n_grasp=len(ds.samples), models=args.models), f)
    print(f"저장 {out}, 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
