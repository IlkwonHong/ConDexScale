"""조건부 net2 학습 (2026-10-17) — train_stage2.py 와 같은 틀, 모델·데이터·손실만 조건부 판.

실행 (기존 정답 한 벌, 1단계):
  ./venv/bin/python -m contraction_net.train_stage2_cond --out checkpoints/stage2_cond_v1
여러 벌 정답 (시험 생성 데이터):
  ./venv/bin/python -m contraction_net.train_stage2_cond --records-format multi \
      --records-dir data/contraction_gt_300k/search_records_multi --far-weight 0.5 --out ...
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .stage2_cond_dataset import Stage2CondDataset
from .stage2_cond_model import Stage2CondNet, stage2_cond_loss, eval_metrics

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device, loss_kw):
    """전역(표본 가중) 집계."""
    model.eval()
    agg, wsum = defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast(device, enabled=device == "cuda"):
                out = model(batch)
                _, m = stage2_cond_loss(model, out, batch, **loss_kw)
                em, _ = eval_metrics(model, out, batch)
            for k, v in m.items():
                if k.startswith("cnt_"):
                    continue
                w = m.get("cnt_hm" if k.startswith("hm_") else f"cnt_{k[-2:]}", 1)
                agg[k] += v * w
                wsum[k] += w
            for k, (v, w) in em.items():
                agg[k] += v * w
                wsum[k] += w
    return {k: agg[k] / max(wsum[k], 1) for k in agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--com-feat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--feat-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--patch-interp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--records-dir", default="data/contraction_gt_300k/search_records")
    ap.add_argument("--records-format", default="single", choices=["single", "multi"])
    ap.add_argument("--noise-p", type=float, default=0.3, help="앞 자리별 흔들림 확률")
    ap.add_argument("--noise-weight", type=float, default=0.5)
    ap.add_argument("--far-weight", type=float, default=0.0, help="먼 첫 자리 손실 (multi 전용)")
    ap.add_argument("--wrench-feat", default="none", choices=["none", "support"],
                    help="support: 앞 자리 앵커 지지 함수 + 후보점 방향값 (28방향) 을 조건에 추가")
    ap.add_argument("--max-k", type=int, default=3, help="정답 벌 수 상한 (1 = 최선만)")
    ap.add_argument("--no-cond", action="store_true", help="앞 자리 조건 끄기 (그룹마다 독립 출력)")
    ap.add_argument("--cand-from-records", action="store_true",
                    help="후보 마스크를 레코드의 c128_cand 에서 읽는다 (EXP-004)")
    ap.add_argument("--only-n", type=int, default=None, help="그 팔 수의 레코드만 조건부 과제로 쓴다")
    ap.add_argument("--seed", type=int, default=None, help="모델 초기값·배치 순서 씨앗 (기본: 고정 안 함)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="checkpoints/stage2_cond")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    kw = dict(contact_geom=False, com_feat=args.com_feat, feat_norm=args.feat_norm,
              patch_interp=args.patch_interp, records_dir=ROOT / args.records_dir,
              records_format=args.records_format, noise_p=args.noise_p, max_k=args.max_k,
              cand_from_records=args.cand_from_records, only_n=args.only_n)
    tr = Stage2CondDataset(split="train", augment=True, **kw)
    va = Stage2CondDataset(split="val", augment=False, **kw)
    if args.smoke:
        tr = Subset(tr, range(0, len(tr), max(1, len(tr) // 2000)))
        va = Subset(va, range(0, len(va), max(1, len(va) // 400)))
        args.epochs = 2
    ds0 = tr.dataset if isinstance(tr, Subset) else tr
    print(f"train {len(tr):,} / val {len(va):,} 파지, 형식 {args.records_format}", flush=True)

    pw = args.workers > 0
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=pw)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=args.workers, persistent_workers=pw)

    model = Stage2CondNet(ds0.vocab, d_pf=11 if args.com_feat else 7,
                          d_ptf=10 if args.com_feat else 6, wrench_feat=args.wrench_feat,
                          use_cond=not args.no_cond).to(device)
    print(f"파라미터 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    loss_kw = dict(noise_weight=args.noise_weight, far_weight=args.far_weight)

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for batch in dl_tr:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast(device, enabled=device == "cuda"):
                outs = model(batch)
                loss, _ = stage2_cond_loss(model, outs, batch, **loss_kw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        m = evaluate(model, dl_va, device, loss_kw)
        rec = {"epoch": ep, "train_loss": run / max(1, len(dl_tr)),
               "sec": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in m.items()}}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                    "args": vars(args)}, out / "last.pt")
        score = m.get("hm_main", float("inf"))
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                        "args": vars(args)}, out / "best.pt")


if __name__ == "__main__":
    main()
