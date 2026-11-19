"""Stage 1 학습 (명세 7장).

실행: ./venv/bin/python -m contraction_net.train_stage1 [--epochs 100] [--smoke]
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .stage1_dataset import Stage1Dataset
from .stage1_model import Stage1Net, Stage1NetV2, stage1_loss

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device):
    """전역(파지 단위) 가중 집계. 주의: 배치별 평균의 평균은 검증 배치가
    taxonomy 타입별로 뭉쳐 있어 top-1 을 5%p 이상 부풀린다 (07-29 발견)."""
    model.eval()
    agg, wsum = defaultdict(float), defaultdict(float)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits, reg, pl = model(batch)
            _, m = stage1_loss(logits, reg, pl, batch)
            for k, v in m.items():
                if k.startswith("cnt_"):
                    continue
                w = m.get(f"cnt_{k[-2:]}", 1)   # top1_n3 → cnt_n3; 회귀 등은 배치 크기 근사
                agg[k] += v * w
                wsum[k] += w
    return {k: agg[k] / max(wsum[k], 1) for k in agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sigma-max", type=float, default=0.08)
    ap.add_argument("--no-contact-geom", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2000파지 2에폭 배선 확인")
    ap.add_argument("--resume", default=None, help="체크포인트(.pt)에서 모델 가중치 이어받기")
    ap.add_argument("--arch", default="v1", choices=["v1", "v2"],
                    help="v1 = 고정 사전 분류, v2 = 후보 구조 스코어러")
    ap.add_argument("--n-patch", type=int, default=128,
                    help="패치 해상도. 128 외의 값이면 patch_cache_<n>·patch_contacts_<n> 사용")
    ap.add_argument("--out", default="checkpoints/stage1")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="학습 파지 중 물체 해시 기준 고정 부분집합 비율 (데이터 규모 시험, EXP-005). "
                         "부분집합은 중첩됨: 0.25 ⊂ 0.5 ⊂ 1.0, 시드와 무관")
    ap.add_argument("--seed", type=int, default=None,
                    help="모델 초기값·배치 순서 시드 (None = 기존 동작, 고정 안 함)")
    args = ap.parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    kw = dict(contact_geom=not args.no_contact_geom, sigma_max=args.sigma_max,
              n_patch=args.n_patch)
    if args.n_patch != 128:
        base = ROOT / "data" / "contraction_net"
        kw["patch_cache_root"] = base / f"patch_cache_{args.n_patch}"
        kw["patch_contacts_dir"] = base / f"patch_contacts_{args.n_patch}"
    tr = Stage1Dataset(split="train", augment=True, **kw)
    va = Stage1Dataset(split="val", augment=False, **kw)
    if args.train_frac < 1.0:
        import hashlib
        def _u(gid):    # 물체 단위 [0,1) — 검증 분리 해시와 다른 소금
            oid = tr.obj_of[gid][1]
            return int(hashlib.md5(f"frac:{oid}".encode()).hexdigest()[:8], 16) / 16**8
        keep = [i for i, (gid, _e) in enumerate(tr.samples) if _u(gid) < args.train_frac]
        print(f"--train-frac {args.train_frac}: 학습 파지 {len(tr.samples):,} → {len(keep):,}", flush=True)
        tr = Subset(tr, keep)
    if args.smoke:
        tr = Subset(tr, range(0, len(tr), max(1, len(tr) // 2000)))
        va = Subset(va, range(0, len(va), max(1, len(va) // 400)))
        args.epochs = 2
    n_tr = len(tr)
    base_tr = tr
    while isinstance(base_tr, Subset):
        base_tr = base_tr.dataset
    vocab_n = len(base_tr.vocab)
    print(f"train {n_tr:,} / val {len(va):,} 파지, 클래스 {vocab_n}, device={device}", flush=True)

    pw = args.workers > 0    # 워커 유지 — 물체 캐시가 에폭마다 초기화되는 것 방지
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=pw)
    dl_va = DataLoader(va, batch_size=args.batch, num_workers=args.workers,
                       persistent_workers=pw)

    vocab = base_tr.vocab
    model = (Stage1NetV2(vocab, n_patch=args.n_patch) if args.arch == "v2"
             else Stage1Net(vocab_n, n_patch=args.n_patch)).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"이어받기: {args.resume} (epoch {ck.get('epoch')})", flush=True)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"파라미터 {n_param/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_total = max(1, len(dl_tr)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps_total)))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for it, batch in enumerate(dl_tr):
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast(device, enabled=device == "cuda"):
                logits, reg, pl = model(batch)
                loss, _ = stage1_loss(logits, reg, pl, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        m = evaluate(model, dl_va, device)
        score = (m.get("top1_n3", 0) + m.get("top1_n4", 0)) / 2
        rec = {"epoch": ep, "train_loss": run / max(1, len(dl_tr)),
               "sec": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in m.items()}}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                    "args": vars(args)}, out / "last.pt")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                        "args": vars(args)}, out / "best.pt")

    # 노이즈 수준별 곡선 (명세 7장). 주의: persistent worker 는 속성 변경을 못 보므로
    # σ 설정 후 새 DataLoader 를 만들어야 한다.
    va_ds = va.dataset if isinstance(va, Subset) else va
    for sig in (0.0, 0.03, 0.08):
        va_ds.augment, va_ds.sigma_fixed = sig > 0, sig
        m = evaluate(model, DataLoader(va, batch_size=args.batch, num_workers=2), device)
        va_ds.augment, va_ds.sigma_fixed = False, None
        rec = {"noise_curve_sigma": sig, **{k: round(v, 4) for k, v in m.items()}}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
