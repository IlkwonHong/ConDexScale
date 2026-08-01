"""Stage 2 학습 (명세 3장). 실행:
  ./venv/bin/python -m contraction_net.train_stage2 --hm-target onehot --out checkpoints/stage2_onehot
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .stage2_dataset import Stage2Dataset
from .stage2_model import Stage2Net, stage2_loss

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device):
    """전역(표본 가중) 집계 — 배치 평균의 평균 금지 (07-29 교훈)."""
    model.eval()
    agg, wsum = defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits, reg, _, heat = model(batch)
            _, m = stage2_loss(logits, reg, heat, batch)
            for k, v in m.items():
                if k.startswith("cnt_"):
                    continue
                w = m.get("cnt_hm" if k.startswith("hm_") else f"cnt_{k[-2:]}", 1)
                agg[k] += v * w
                wsum[k] += w
    return {k: agg[k] / max(wsum[k], 1) for k in agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hm-target", default="onehot", choices=["onehot", "gauss"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="checkpoints/stage2")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    kw = dict(contact_geom=False, hm_target=args.hm_target)
    tr = Stage2Dataset(split="train", augment=True, **kw)
    va = Stage2Dataset(split="val", augment=False, **kw)
    if args.smoke:
        tr = Subset(tr, range(0, len(tr), max(1, len(tr) // 2000)))
        va = Subset(va, range(0, len(va), max(1, len(va) // 400)))
        args.epochs = 2
    ds0 = tr.dataset if isinstance(tr, Subset) else tr
    print(f"train {len(tr):,} / val {len(va):,} 파지, 히트맵 과제 "
          f"{sum(len(v) for v in ds0.hm_tasks.values()):,}, target={args.hm_target}", flush=True)

    pw = args.workers > 0
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=pw)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=args.workers,
                       persistent_workers=pw)

    model = Stage2Net(ds0.vocab).to(device)
    print(f"파라미터 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for batch in dl_tr:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast(device, enabled=device == "cuda"):
                logits, reg, _, heat = model(batch)
                loss, _ = stage2_loss(logits, reg, heat, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        m = evaluate(model, dl_va, device)
        rec = {"epoch": ep, "train_loss": run / max(1, len(dl_tr)),
               "sec": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in m.items()}}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                    "args": vars(args)}, out / "last.pt")
        score = m.get("hm_top8", 0)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                        "args": vars(args)}, out / "best.pt")


if __name__ == "__main__":
    main()
