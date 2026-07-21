#!/usr/bin/env python3
"""
Batch contracted-dataset generation from the Arctic bimanual dataset.

Processes Arctic sequences through genhand's bimanual fc_optimization pipeline
and stores result JSONs for GraspContractionNet (Module B) training.

Results are saved to:
  data/results_arctic/{subject}/{seq_name}/{robot}/mu{mu}/

Usage:
  cd /workspace/genhand && source venv/bin/activate
  python scripts/run_arctic.py --robot Robotiq --subjects all --frame-stride 30
  python scripts/run_arctic.py --robot Barrett --subjects s01 s02 --frame-stride 10 --workers 2

Options:
  --arctic-root    Path to Arctic data root (default: ../ext_proj/arctic/unpack/arctic_data)
  --robot          Barrett | Robotiq  (default: Robotiq)
  --subjects       s01..s10 or 'all'  (default: s01)
  --split          train | val | test  (default: train)
  --protocol       p1 | p2  (default: p1)
  --frame-stride   Sample every N frames  (default: 30)
  --start / --end  Index range within the dataset (default: full range)
  --workers        Parallel subprocess count  (default: 4)
  --device         CUDA device  (default: cuda:0)
  --mu             Friction coefficient  (default: 0.9)
  --contact-threshold  SDF contact threshold  (default: 0.05)
"""

import argparse
import json
import os
import sys
import threading
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_DEFAULT_ARCTIC_ROOT = str(REPO_ROOT.parent / "ext_proj" / "arctic" / "unpack" / "arctic_data")
RESULT_BASE = REPO_ROOT / "data" / "results_arctic"

VALID_SUBJECTS = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10']


def dataset_indices(arctic_root, split, protocol, subjects, frame_stride):
    """Return list of (global_idx, subject, seq_name, frame_t) tuples matching subjects."""
    data_root = Path(arctic_root) / 'data'
    split_path = data_root / 'splits_json' / f'protocol_{protocol}.json'
    with open(split_path) as f:
        sequences = json.load(f)[split]

    # Build same frame_index as ArcticFeed.__init__ to get global indices
    subject_set = set(subjects)
    entries = []
    idx = 0
    for seq_path in sequences:
        subject, seq_name = seq_path.split('/')
        obj_name = seq_name.split('_')[0]
        obj_npy = data_root / 'raw_seqs' / subject / (seq_name + '.object.npy')
        if not obj_npy.exists():
            continue
        import numpy as np
        T = np.load(str(obj_npy)).shape[0]
        for t in range(0, T, frame_stride):
            if subject in subject_set:
                entries.append((idx, subject, seq_name, t))
            idx += 1
    return entries


def result_json_path(subject, seq_name, robot, mu, idx, out_base=RESULT_BASE):
    return (
        Path(out_base) / subject / seq_name / robot / f"mu{mu:.1f}"
        / f"{idx}_{robot}_0_mu_{mu:.1f}.json"
    )


def result_done(subject, seq_name, robot, mu, idx, out_base=RESULT_BASE):
    p = result_json_path(subject, seq_name, robot, mu, idx, out_base)
    if not p.exists():
        return False
    try:
        return "error" not in json.loads(p.read_text())
    except Exception:
        return False


def format_eta(elapsed, done, total):
    if done == 0:
        return "?"
    secs = elapsed / done * (total - done)
    h, rem = divmod(int(secs), 3600)
    return f"{h}h {rem//60:02d}m"


def build_worker_code(arctic_root, indices_with_meta, robot, device, mu,
                      contact_threshold, frame_stride, split, protocol,
                      anchors_per_hand=None, out_base=str(RESULT_BASE)):
    """Build Python worker code string for a subprocess chunk."""
    return f"""
import os, sys, json, time, traceback, pathlib
sys.path.insert(0, {str(REPO_ROOT)!r})

import torch
from optimisation.Optimize import Optimization
from optimisation.dataset_opt import ArcticFeed
from optimisation.robot import Shadow, Allegro, Barrett, Robotiq

robot_cls = {{'Shadow': Shadow, 'Allegro': Allegro, 'Barrett': Barrett, 'Robotiq': Robotiq}}
robot = robot_cls[{robot!r}](batch=1, device={device!r})

dataset = ArcticFeed(
    arctic_root={arctic_root!r},
    split={split!r},
    protocol={protocol!r},
    frame_stride={frame_stride},
)

opt = Optimization(
    robot=robot,
    dataset=dataset,
    device={device!r},
    maximum_iter=[2000, 1, 1000],
    visualize=False,
    repeat=1,
    mu={mu},
    task='GH',
    source='ycb',
    contact_target_method='hdbscan',
    contact_threshold={contact_threshold},
    sdf_backend='mesh',
    anchors_per_hand={anchors_per_hand!r},
)

entries = {indices_with_meta!r}
for global_idx, subject, seq_name, frame_t in entries:
    result_dir = str(pathlib.Path({out_base!r}) / subject / seq_name / {robot!r} / f"mu{mu:.1f}")
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    opt.result_dir = result_dir
    t0 = time.time()
    try:
        opt.run_idx_json_bimanual(global_idx)
        print(f'[worker] idx={{global_idx}} {{subject}}/{{seq_name}} t={{frame_t}} done {{time.time()-t0:.1f}}s', flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f'[worker] idx={{global_idx}} ERROR {{e}}', flush=True)
"""


def run_subjects(subjects, args):
    entries = dataset_indices(
        args.arctic_root, args.split, args.protocol, subjects, args.frame_stride
    )
    if args.end is not None:
        entries = entries[args.start:args.end]
    elif args.start > 0:
        entries = entries[args.start:]

    todo = [
        (gi, sub, seq, ft) for gi, sub, seq, ft in entries
        if not result_done(sub, seq, args.robot, args.mu, gi, args.out_base)
    ]
    skipped = len(entries) - len(todo)
    total = len(todo)

    print(f"\n[run_arctic] subjects={subjects} robot={args.robot} "
          f"stride={args.frame_stride} total={len(entries)} "
          f"todo={total} skipped={skipped} workers={args.workers}")
    print(f"[run_arctic] results → {args.out_base}")

    if total == 0:
        print("[run_arctic] all done — skipping.")
        return

    n = args.workers
    chunks = [todo[i::n] for i in range(n)]
    chunks = [c for c in chunks if c]

    start_time = time.time()
    completed = [0]
    lock = threading.Lock()
    py = sys.executable

    def monitor(proc):
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"\n  {line}", flush=True)
            if "[worker] idx=" in line and (" done " in line or " ERROR " in line):
                with lock:
                    completed[0] += 1
                    done = completed[0]
                elapsed = time.time() - start_time
                eta = format_eta(elapsed, done, total)
                print(f"\r[{done:04d}/{total}] elapsed={elapsed/60:.1f}m ETA={eta}   ",
                      end="", flush=True)

    procs, monitors = [], []
    for chunk in chunks:
        code = build_worker_code(
            arctic_root=args.arctic_root,
            indices_with_meta=chunk,
            robot=args.robot,
            device=args.device,
            mu=args.mu,
            contact_threshold=args.contact_threshold,
            frame_stride=args.frame_stride,
            split=args.split,
            protocol=args.protocol,
            anchors_per_hand=args.anchors_per_hand,
            out_base=args.out_base,
        )
        proc = subprocess.Popen(
            [py, "-c", code],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(proc)
        mt = threading.Thread(target=monitor, args=(proc,), daemon=True)
        mt.start()
        monitors.append(mt)

    for proc in procs:
        proc.wait()
    for mt in monitors:
        mt.join()

    elapsed = time.time() - start_time
    print(f"\n[run_arctic] done in {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch bimanual contracted-dataset generation from Arctic.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--arctic-root", default=_DEFAULT_ARCTIC_ROOT,
                        help="Path to Arctic data root (default: %(default)s)")
    parser.add_argument("--robot", choices=["Shadow", "Allegro", "Barrett", "Robotiq"],
                        default="Robotiq")
    parser.add_argument("--subjects", nargs="+", metavar="SUBJECT", default=["s01"],
                        help="s01..s10 or 'all'")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out-base", default=str(RESULT_BASE),
                        help="output dir for result JSONs (default data/results_arctic; "
                             "use a separate dir for held-out splits, e.g. data/results_arctic_val)")
    parser.add_argument("--protocol", choices=["p1", "p2"], default="p1")
    parser.add_argument("--frame-stride", type=int, default=30,
                        help="Sample every N frames (default: 30)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--anchors-per-hand", type=int, default=None,
                        help="genhand anchors per hand for the label (human_anchor). "
                             "Default = robot.contacts_num (Robotiq -> 2/hand -> 4 total).")
    args = parser.parse_args()

    if args.subjects == ["all"]:
        subjects = VALID_SUBJECTS
    else:
        subjects = []
        for s in args.subjects:
            if s in VALID_SUBJECTS:
                subjects.append(s)
            else:
                parser.error(f"Unknown subject {s!r}. Valid: {', '.join(VALID_SUBJECTS)}")

    print(f"[run_arctic] processing subjects: {', '.join(subjects)}")
    run_subjects(subjects, args)


if __name__ == "__main__":
    main()
