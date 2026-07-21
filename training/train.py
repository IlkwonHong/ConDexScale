#!/usr/bin/env python3
"""
Training script for GraspContractionNet (MVP).

Usage:
    cd /workspace/genhand && source venv/bin/activate
    python training/train.py --robot Robotiq --epochs 100 --batch-size 16
    python training/train.py --robot Barrett --epochs 100 --batch-size 16 --val-split 0.2
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Train GraspContractionNet MVP")
    p.add_argument("--dataset",     type=str,   default="arctic",
                   choices=["dexycb", "arctic"],
                   help="dexycb = single-hand (results_train/graph_cache); "
                        "arctic = bimanual design-B (results_arctic/arctic_cache)")
    p.add_argument("--robot",       type=str,   default="Robotiq",
                   choices=["Shadow", "Allegro", "Barrett", "Robotiq"],
                   help="Robot type — determines num_anchors (Robotiq=2, Barrett=3, else 3)")
    p.add_argument("--gate-max-dist", type=float, default=None,
                   help="ARCTIC only: drop frames where either hand's nearest-patch "
                        "distance exceeds this (metres, e.g. 0.03). Default: no gating")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val-split",   type=float, default=0.1,
                   help="Fraction of data for validation (0 = no val)")
    p.add_argument("--d-model",     type=int,   default=256)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--device",      type=str,   default="cuda:0")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Default: results_train (dexycb) / results_arctic (arctic)")
    p.add_argument("--graph-dir",   type=Path, default=None,
                   help="Default: graph_cache (dexycb) / arctic_cache/sample_graphs (arctic)")
    p.add_argument("--ckpt-dir",    type=Path,
                   default=REPO_ROOT / "checkpoints")
    p.add_argument("--resume",      type=Path,  default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--mu",          type=float, default=None,
                   help="Filter dataset by friction coefficient (e.g. 0.5, 0.9). "
                        "Default: use all mu values")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--preload",     action="store_true",
                   help="Load all JSONs into RAM at startup (faster epochs, ~1GB RAM)")
    # Loss weights
    p.add_argument("--lambda-anchor",       type=float, default=1.0)
    p.add_argument("--lambda-surface",      type=float, default=0.5)
    p.add_argument("--lambda-spread",       type=float, default=0.1)
    p.add_argument("--lambda-heatmap",      type=float, default=0.1)
    p.add_argument("--heatmap-temperature", type=float, default=0.02,
                   help="KL target bandwidth in metres (default 0.02 = 2 cm)")
    return p.parse_args()


def robot_to_num_anchors(robot: str) -> int:
    return 2 if robot == "Robotiq" else 3


def make_loaders(args):
    if args.dataset == "arctic":
        # Bimanual: num_anchors varies 2/4 per sample → model uses max (4 slots),
        # mask handles the 2-anchor case in the loss.
        from training.dataset import (ArcticContractionDataset,
                                       ARCTIC_RESULTS_DIR, ARCTIC_GRAPH_DIR)
        num_anchors = 4
        full_ds = ArcticContractionDataset(
            results_dir=args.results_dir or ARCTIC_RESULTS_DIR,
            graph_dir=args.graph_dir or ARCTIC_GRAPH_DIR,
            robot=args.robot,
            mu=args.mu,
            max_anchors=num_anchors,
            gate_max_dist=args.gate_max_dist,
            preload=args.preload,
        )
    else:
        from training.dataset import GraspContractionDataset
        num_anchors = robot_to_num_anchors(args.robot)
        full_ds = GraspContractionDataset(
            results_dir=args.results_dir or (REPO_ROOT / "data" / "results_train"),
            graph_dir=args.graph_dir or (REPO_ROOT / "data" / "graph_cache"),
            robot=args.robot,
            mu=args.mu,
            max_anchors=num_anchors,
            preload=args.preload,
        )
    N = len(full_ds)
    print(f"[dataset] {N} samples  dataset={args.dataset}  robot={args.robot}  "
          f"num_anchors={num_anchors}  gate={args.gate_max_dist}")

    if args.val_split > 0 and N > 1:
        indices = list(range(N))
        random.shuffle(indices)
        n_val = max(1, int(N * args.val_split))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
    else:
        train_ds, val_ds = full_ds, None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True)
        if val_ds is not None else None
    )
    print(f"[dataset] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")
    return train_loader, val_loader, num_anchors


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "l_anchor": 0.0, "l_surface": 0.0, "l_spread": 0.0, "l_heatmap": 0.0}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = move_batch(batch, device)
            pred  = model(batch)
            loss_dict = criterion(pred, batch)

            if train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += loss_dict[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, metrics: dict):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[resume] epoch {ckpt['epoch']}  metrics={ckpt.get('metrics', {})}")
    return ckpt["epoch"]


def fmt(metrics: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader, num_anchors = make_loaders(args)

    from training.model import GraspContractionNet
    from training.loss  import GraspContractionLoss

    model = GraspContractionNet(
        d_model=args.d_model,
        num_anchors=num_anchors,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {total_params/1e6:.2f}M params")

    criterion = GraspContractionLoss(
        lambda_anchor=args.lambda_anchor,
        lambda_surface=args.lambda_surface,
        lambda_spread=args.lambda_spread,
        lambda_heatmap=args.lambda_heatmap,
        heatmap_temperature=args.heatmap_temperature,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.ckpt_dir / "train_log.json"
    log = []

    from datetime import datetime
    mu_tag  = f"mu{args.mu}" if args.mu is not None else "muAll"
    run_name = f"{args.robot}_{mu_tag}_{datetime.now().strftime('%m%d_%H%M%S')}"
    tb_dir = args.ckpt_dir / "tb" / run_name
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[tensorboard] tensorboard --logdir {args.ckpt_dir / 'tb'}")
    print(f"[run] {run_name}")

    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics}

        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, criterion, None, device, train=False)
            row["val"] = val_metrics
            val_loss = val_metrics["loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args.ckpt_dir / "best.pt", epoch, model,
                                optimizer, scheduler, row)
        else:
            val_loss = train_metrics["loss"]

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:03d}  lr={lr_now:.2e}  "
              f"train [{fmt(train_metrics)}]"
              + (f"  val [{fmt(val_metrics)}]" if val_loader else "")
              + f"  {elapsed:.1f}s")

        # TensorBoard logging
        writer.add_scalar("lr", lr_now, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        if val_loader is not None:
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Save every 100 epochs + final
        if (epoch + 1) % 100 == 0 or epoch == args.epochs - 1:
            save_checkpoint(args.ckpt_dir / f"epoch_{epoch:03d}.pt", epoch,
                            model, optimizer, scheduler, row)

        log.append(row)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    writer.close()
    print(f"\n[done] best val loss={best_val_loss:.4f}  ckpts → {args.ckpt_dir}")


if __name__ == "__main__":
    main()
