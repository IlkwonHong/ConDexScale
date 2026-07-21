#!/usr/bin/env python3
"""
Batch GT data generation for a single DexYCB subject.

Uses dexycb_trainfeed which generates object meshes on-the-fly from YCB
canonical models + obj_transform, so no pre-built mesh_obj files are needed.

Results are saved to:
  optimisation/results_train/<subject>/<robot>/mu<mu>/

Usage:
  cd /workspace/genhand && source venv/bin/activate
  python scripts/run_partial.py --subject 20200709-subject-01 --robot Shadow --opt-workers 4
  python scripts/run_partial.py --subject 20200709-subject-01 --robot Barrett --opt-workers 4
  python scripts/run_partial.py --subject 20200813-subject-02 --robot Robotiq --opt-workers 4
  python scripts/run_partial.py --subject 20200709-subject-01 --robot Robotiq --opt-workers 4

Options:
  --subject        DexYCB subject folder name  (default: 20200709-subject-01)
  --robot          Shadow | Allegro | Barrett | Robotiq  (default: Shadow)
  --start / --end  index range within the subject (default: 0..auto-detect)
  --opt-workers    parallel subprocess count (default: 4)
  --device         CUDA device (default: cuda:0)
  --mu             friction coefficient (default: 0.9)
  --contact-threshold  SDF contact threshold (default: 0.05)
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

_DEFAULT_DATASET = str(REPO_ROOT.parent / "dataset")
os.environ.setdefault("DEX_YCB_DIR", _DEFAULT_DATASET)

RESULT_BASE = REPO_ROOT / "data" / "results_train"

VALID_SUBJECTS = [
    '20200709-subject-01', '20200813-subject-02', '20200820-subject-03',
    '20200903-subject-04', '20200908-subject-05', '20200918-subject-06',
    '20200928-subject-07', '20201002-subject-08', '20201015-subject-09',
    '20201022-subject-10',
]

# Numeric prefix used in dexycb_train_s0.json file names
_SUBJECT_PREFIX = {s: str(i + 1) for i, s in enumerate(VALID_SUBJECTS)}

# Default camera serial — first camera in DexYCB calibration order.
# All 8 cameras capture the same 3D pose (annotations are already multi-view fused GT),
# so using one camera avoids 8x data duplication.
DEFAULT_CAMERA = '836212060125'


def subject_camera_indices(subject: str, camera: str) -> list:
    """Return dataset-space indices (within the subject's filtered list) matching one camera."""
    config = REPO_ROOT / "dataset" / "data" / "dexycb_train_s0.json"
    with open(config) as f:
        meta = json.load(f)
    prefix = _SUBJECT_PREFIX[subject] + '_'
    subject_imgs = [img for img in meta['images'] if img['file_name'].startswith(prefix)]
    return [
        k for k, img in enumerate(subject_imgs)
        if img['file_name'].split('_')[3] == camera
    ]


def result_json_path(subject, robot, mu, idx):
    return RESULT_BASE / subject / robot / f"mu{mu:.1f}" / f"{idx}_{robot}_0_mu_{mu:.1f}.json"


def result_done(subject, robot, mu, idx):
    p = result_json_path(subject, robot, mu, idx)
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


def build_worker_code(subject, robot, indices, device, mu, contact_threshold, result_base):
    dataset_dir = os.environ.get("DEX_YCB_DIR", _DEFAULT_DATASET)
    return f"""
import os, sys, json, time, traceback
os.environ.setdefault('DEX_YCB_DIR', {dataset_dir!r})
sys.path.insert(0, {str(REPO_ROOT)!r})

import torch
from optimisation.Optimize import Optimization
from optimisation.dataset_opt import dexycb_trainfeed
from optimisation.robot import Shadow, Allegro, Barrett, Robotiq

robot_cls = {{'Shadow': Shadow, 'Allegro': Allegro, 'Barrett': Barrett, 'Robotiq': Robotiq}}
robot = robot_cls[{robot!r}](batch=1, device={device!r})

dataset = dexycb_trainfeed(subject={subject!r})

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
)

import pathlib
pathlib.Path({str(result_base)!r}).mkdir(parents=True, exist_ok=True)
opt.result_dir = {str(result_base)!r}

indices = {indices!r}
for idx in indices:
    t0 = time.time()
    try:
        opt.run_idx_json(idx)
        print(f'[worker] idx={{idx}} done {{time.time()-t0:.1f}}s', flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f'[worker] idx={{idx}} ERROR {{e}}', flush=True)
"""


def run_one_subject(subject: str, args) -> None:
    """Run GT generation for a single subject."""
    all_indices = subject_camera_indices(subject, args.camera)
    if args.end is not None:
        all_indices = all_indices[args.start:args.end]
    elif args.start > 0:
        all_indices = all_indices[args.start:]

    result_base = RESULT_BASE / subject / args.robot / f"mu{args.mu:.1f}"
    result_base.mkdir(parents=True, exist_ok=True)

    todo = [i for i in all_indices if not result_done(subject, args.robot, args.mu, i)]
    skipped = len(all_indices) - len(todo)
    total = len(todo)

    print(f"\n[run_partial] subject={subject} robot={args.robot} camera={args.camera} "
          f"total_cam_indices={len(all_indices)} "
          f"todo={total} skipped={skipped} workers={args.opt_workers}")
    print(f"[run_partial] results → {result_base}")

    if total == 0:
        print("[run_partial] all done — skipping.")
        return

    n = args.opt_workers
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
            if "[worker] idx=" in line and " done " in line:
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
            subject=subject,
            robot=args.robot,
            indices=chunk,
            device=args.device,
            mu=args.mu,
            contact_threshold=args.contact_threshold,
            result_base=str(result_base),
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
    print(f"\n[run_partial] {subject} done in {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")

    failed = [i for i in todo if not result_done(subject, args.robot, args.mu, i)]
    if failed:
        fp = (REPO_ROOT / "scripts" /
              f"failed_{subject}_{args.robot}_{args.start}_{end}.txt")
        fp.write_text("\n".join(map(str, sorted(failed))))
        print(f"[run_partial] {len(failed)} failures → {fp}")
    else:
        print(f"[run_partial] 0 failures")


def main():
    parser = argparse.ArgumentParser(
        description="Batch GT generation for one or more DexYCB subjects.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--subject", default=None,
                        choices=VALID_SUBJECTS, metavar="SUBJECT",
                        help="Single subject. Ignored if --subjects is given.")
    parser.add_argument("--subjects", nargs="+", metavar="SUBJECT",
                        help=(
                            "One or more subjects, or 'all' for all 10.\n"
                            "Examples:\n"
                            "  --subjects all\n"
                            "  --subjects 20200813-subject-02 20200820-subject-03\n"
                            "  --subjects 2 3 4   (short numeric form)"
                        ))
    parser.add_argument("--robot", choices=["Shadow", "Allegro", "Barrett", "Robotiq"],
                        default="Robotiq")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive). Defaults to full subject size.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opt-workers", type=int, default=4)
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA,
                        help="Camera serial to use (default: %(default)s). "
                             "DexYCB GT is multi-view fused, so one camera is sufficient.")
    args = parser.parse_args()

    # Resolve subject list
    _NUM_TO_SUBJECT = {str(i + 1): s for i, s in enumerate(VALID_SUBJECTS)}

    if args.subjects:
        if args.subjects == ["all"]:
            subjects = VALID_SUBJECTS
        else:
            subjects = []
            for s in args.subjects:
                if s in VALID_SUBJECTS:
                    subjects.append(s)
                elif s in _NUM_TO_SUBJECT:
                    subjects.append(_NUM_TO_SUBJECT[s])
                else:
                    parser.error(
                        f"Unknown subject {s!r}.\n"
                        f"Use full name (e.g. 20200813-subject-02) or number (2..10).\n"
                        f"Valid: {', '.join(VALID_SUBJECTS)}"
                    )
    elif args.subject:
        subjects = [args.subject]
    else:
        subjects = ["20200709-subject-01"]

    print(f"[run_partial] processing {len(subjects)} subject(s): {', '.join(s.split('-')[-1] for s in subjects)}")

    for subject in subjects:
        run_one_subject(subject, args)


if __name__ == "__main__":
    main()
