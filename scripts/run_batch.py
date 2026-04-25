#!/usr/bin/env python3
"""
Batch pipeline for GT dataset generation.

Spawns N persistent worker subprocesses, each processing a chunk of samples
sequentially — amortising the dataset/robot initialisation cost over many samples.
Stage 3 (PyBullet) runs in background CPU workers after each Stage 2 completes.

gt_pv (default): Stage 1 skipped — uses DexYCB GT meshes directly.
recon_pv:        Stage 1 runs sequentially first, then feeds the Stage 2 pool.
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

from demo.run_demo import (
    resolve_pipeline,
    build_sim_code,
    result_json_path,
)

os.environ.setdefault("DEX_YCB_DIR", "/workspace/dataset")


# ---------------------------------------------------------------------------
# Worker code — executed in each subprocess via python -c
# ---------------------------------------------------------------------------

def build_worker_code(robot, indices, device, mu, pipeline, contact_threshold):
    """
    Returns Python source that initialises Optimization once, then iterates
    over `indices` calling run_idx_json / run_idx_json_nv per sample.
    """
    return f"""
import os, sys, json, time
os.environ.setdefault('DEX_YCB_DIR', '/workspace/dataset')
sys.path.insert(0, {str(REPO_ROOT)!r})

import torch
from optimisation.Optimize import Optimization
from optimisation.dataset_opt import dexycb_testfullfeed
from optimisation.robot import Shadow, Allegro, Barrett, Robotiq

robot_cls = {{'Shadow': Shadow, 'Allegro': Allegro, 'Barrett': Barrett, 'Robotiq': Robotiq}}
robot = robot_cls[{robot!r}](batch=1, device={device!r})

dataset = dexycb_testfullfeed(
    load_mesh=True,
    pc_sample=1024,
    data_sample=None,
    precept={pipeline['precept']},
    pred_mesh_mode={pipeline.get('pred_mesh_mode', 'icp')!r},
)
opt = Optimization(
    robot=robot,
    dataset=dataset,
    device={device!r},
    maximum_iter=[2000, 1, 1000],
    visualize=False,
    repeat=1,
    mu={mu},
    task={pipeline['opt_task']!r},
    source={pipeline['source']!r},
    contact_target_method={pipeline['contact_target_method']!r},
    contact_threshold={contact_threshold},
    sdf_backend={pipeline['sdf_backend']!r},
)

indices = {indices!r}
run_fn = opt.run_idx_json if {pipeline['opt_task']!r} == 'GH' else opt.run_idx_json_nv

for idx in indices:
    t0 = time.time()
    try:
        run_fn(idx)
        print(f'[worker] idx={{idx}} done {{time.time()-t0:.1f}}s', flush=True)
    except Exception as e:
        print(f'[worker] idx={{idx}} ERROR {{e}}', flush=True)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sim_result_path(repo_root, robot, mu, idx, source):
    d = "sim_result" if source == "ycb" else "sim_result_sdf"
    return repo_root / "simulation" / d / f"{robot}_{mu:.1f}_{idx}.json"


def result_done(repo_root, robot, mu, idx, source, task):
    p = result_json_path(repo_root=repo_root, robot=robot, mu=mu,
                         sample_idx=idx, source=source, task=task)
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


def launch_stage3(idx, py, repo_root, robot, mu, pipeline, sim_sem):
    """Run Stage 3 for one sample, blocking inside the semaphore."""
    def _run():
        with sim_sem:
            code = build_sim_code(
                robot_name=robot,
                sample_idx=idx,
                mu=mu,
                exp_name=pipeline["sim_exp"],
                sim_render="DIRECT",
                sim_record=False,
                test_sdf=(pipeline["source"] == "sdf"),
            )
            subprocess.run([py, "-c", code], cwd=str(repo_root),
                           capture_output=True)
    t = threading.Thread(target=_run)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch GT dataset generation.")
    parser.add_argument("--sdf-source", choices=["gt_pv", "recon_pv"], default="gt_pv")
    parser.add_argument("--method", choices=["genhand", "baseline"], default="genhand")
    parser.add_argument("--cluster-method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--robot", choices=["Shadow", "Allegro", "Barrett", "Robotiq"], default="Shadow")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5928)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opt-workers", type=int, default=8,
                        help="Parallel Stage-2 worker processes")
    parser.add_argument("--sim-workers", type=int, default=4,
                        help="Concurrent Stage-3 PyBullet processes")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--network-ckpt",
                        default=str(REPO_ROOT / "network" / "ckpt" / "snapshot_1599.pth.tar"))
    args = parser.parse_args()

    pipeline = resolve_pipeline(args.sdf_source, args.method, args.cluster_method)
    py = sys.executable
    all_indices = list(range(args.start, args.end))

    # Skip already-completed samples
    todo = [i for i in all_indices
            if not result_done(REPO_ROOT, args.robot, args.mu, i,
                               pipeline["source"], pipeline["opt_task"])]
    skipped = len(all_indices) - len(todo)
    total = len(todo)

    print(f"[run_batch] sdf={args.sdf_source} robot={args.robot} "
          f"range={args.start}..{args.end} "
          f"todo={total} skipped={skipped} "
          f"opt_workers={args.opt_workers} sim_workers={args.sim_workers}")
    print(f"[run_batch] DEX_YCB_DIR={os.environ.get('DEX_YCB_DIR', '(not set)')}")

    if total == 0:
        print("[run_batch] Stage 2 all done — checking Stage 3…")

    # Divide todo into opt_workers chunks
    n = args.opt_workers
    chunks = [todo[i::n] for i in range(n)]
    chunks = [c for c in chunks if c]

    sim_sem = threading.Semaphore(args.sim_workers)
    sim_threads = []
    start_time = time.time()
    completed = [0]
    lock = threading.Lock()

    def monitor_worker(proc, chunk):
        """Read worker stdout line-by-line; launch Stage 3 on each completion."""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"\n  {line}", flush=True)
            if "[worker] idx=" in line and " done " in line:
                idx = int(line.split("idx=")[1].split()[0])
                # Launch Stage 3 if sim result not yet present
                if not sim_result_path(REPO_ROOT, args.robot, args.mu,
                                       idx, pipeline["source"]).exists():
                    t = launch_stage3(idx, py, REPO_ROOT, args.robot,
                                      args.mu, pipeline, sim_sem)
                    sim_threads.append(t)
                with lock:
                    completed[0] += 1
                    done = completed[0]
                elapsed = time.time() - start_time
                eta = format_eta(elapsed, done, total)
                print(f"\r[{done:04d}/{total}] elapsed={elapsed/60:.1f}m ETA={eta}   ",
                      end="", flush=True)

    # Handle recon_pv: Stage 1 must run first, results feed into chunks already built
    if pipeline["run_network_stage"]:
        network_dir = REPO_ROOT / "network"
        print("[run_batch] recon_pv: running Stage 1 sequentially first…")
        for idx in todo:
            subprocess.run(
                [py, "test_single.py", "--sample-idx", str(idx),
                 "--model-path", args.network_ckpt,
                 "--recon-mode", pipeline["recon_mode"],
                 "--device", args.device],
                cwd=str(network_dir), capture_output=True,
            )

    # Launch persistent worker subprocesses
    procs = []
    monitor_threads = []
    for chunk in chunks if chunks else []:
        code = build_worker_code(
            robot=args.robot,
            indices=chunk,
            device=args.device,
            mu=args.mu,
            pipeline=pipeline,
            contact_threshold=args.contact_threshold,
        )
        proc = subprocess.Popen(
            [py, "-c", code],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(proc)
        mt = threading.Thread(target=monitor_worker, args=(proc, chunk), daemon=True)
        mt.start()
        monitor_threads.append(mt)

    for proc in procs:
        proc.wait()
    for mt in monitor_threads:
        mt.join()

    # Stage 3 catch-up: run for any sample that has Stage 2 done but no sim result yet
    # (covers skipped Stage 2 samples and any missed during worker phase)
    needs_sim = [
        i for i in all_indices
        if result_done(REPO_ROOT, args.robot, args.mu, i,
                       pipeline["source"], pipeline["opt_task"])
        and not sim_result_path(REPO_ROOT, args.robot, args.mu,
                                i, pipeline["source"]).exists()
    ]
    if needs_sim:
        print(f"\n[run_batch] Stage 3 catch-up for {len(needs_sim)} samples…")
        for i in needs_sim:
            t = launch_stage3(i, py, REPO_ROOT, args.robot, args.mu, pipeline, sim_sem)
            sim_threads.append(t)

    print("\n[run_batch] waiting for Stage 3 workers…")
    for t in sim_threads:
        t.join()

    elapsed = time.time() - start_time
    print(f"[run_batch] done in {elapsed/3600:.2f}h  ({elapsed/60:.1f}m)")

    # Collect failures
    failed = [i for i in todo
              if not result_done(REPO_ROOT, args.robot, args.mu, i,
                                 pipeline["source"], pipeline["opt_task"])]
    if failed:
        fp = REPO_ROOT / "scripts" / f"failed_{args.robot}_{args.start}_{args.end}.txt"
        fp.write_text("\n".join(map(str, sorted(failed))))
        print(f"[run_batch] {len(failed)} failures → {fp}")
    else:
        print(f"[run_batch] 0 failures")


if __name__ == "__main__":
    main()
