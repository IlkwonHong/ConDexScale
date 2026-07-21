#!/usr/bin/env python3
"""
Standalone ARCTIC re-contraction (Module A) — design B, NO GPU.

The contracted ``human_anchor`` (the pseudo-label the network predicts) is a pure
function of the per-finger contacts, which already live in the caches.  Thanks to
design B (sample_graphs store INPUT only, anchors are read separately), we can
recompute anchors OFFLINE in seconds instead of re-running the ~37-min GPU
``run_arctic`` pipeline.  Output goes to a dedicated anchor cache, leaving the
generated result JSONs pristine and making contraction a swappable module.

Frames:
  * contact POSITIONS come from the result JSON's ``human_finger_contacts`` /
    ``human_finger_contacts_left`` — object-centric world (x_world - obj_center).
  * contact NORMALS come from the sample_graph (canonical).  Selection that needs
    normals is rotation-invariant, so we select indices in canonical frame and emit
    object-centric positions; the dataloader applies Rg.T just as for result anchors.

Anchor cache entry  data/arctic_cache/anchors/{file_name}_anchor.json:
  { "file_name", "human_anchor": [[x,y,z],...] (object-centric),
    "num_anchors", "method", "hands": ["right","left"] }

Modes (--mode):
  per_hand   one anchor per hand (right->0, left->1).  K=2.  [default]
             --rep centroid | medoid   (medoid = on-surface real fingertip)

Usage:
  cd /home/robin/a_workspace/condex && source venv/bin/activate
  python scripts/recontract_arctic.py --mode per_hand --rep medoid
  python scripts/recontract_arctic.py --mode per_hand --rep centroid --overwrite
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parents[1]
GRAPH_DIR   = REPO_ROOT / "data" / "arctic_cache" / "sample_graphs"
RESULTS_DIR = REPO_ROOT / "data" / "results_arctic"
ANCHOR_DIR  = REPO_ROOT / "data" / "arctic_cache" / "anchors"

FINGER_NAMES = ["index", "middle", "ring", "little", "thumb"]


def per_hand_anchors(contacts_oc: np.ndarray, rep: str = "centroid") -> np.ndarray:
    """contacts_oc: (10,3) object-centric, rows 0-4 = right, 5-9 = left.
    Returns (2,3): one representative per hand."""
    out = []
    for s in (slice(0, 5), slice(5, 10)):
        sub = contacts_oc[s]
        c = sub.mean(0)
        if rep == "medoid":                       # snap to nearest real fingertip
            c = sub[int(np.argmin(np.linalg.norm(sub - c, axis=1)))]
        out.append(c)
    return np.asarray(out, dtype=np.float64)       # (2,3)


def load_contacts_oc(result: dict) -> np.ndarray:
    """10 object-centric fingertip contacts: [R index..thumb, L index..thumb]."""
    r = [result["human_finger_contacts"][n]      for n in FINGER_NAMES]
    l = [result["human_finger_contacts_left"][n] for n in FINGER_NAMES]
    return np.asarray(r + l, dtype=np.float64)     # (10,3)


def contract(result: dict, mode: str, rep: str) -> dict:
    contacts_oc = load_contacts_oc(result)
    if mode == "per_hand":
        anchors = per_hand_anchors(contacts_oc, rep=rep)
        hands = ["right", "left"]
        method = f"per_hand_{rep}"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return {
        "file_name":   result["file_name"],
        "human_anchor": anchors.tolist(),
        "num_anchors": int(anchors.shape[0]),
        "method":      method,
        "hands":       hands,
    }


def main():
    p = argparse.ArgumentParser(description="Standalone ARCTIC re-contraction (design B, offline).")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--anchor-dir",  type=Path, default=ANCHOR_DIR)
    p.add_argument("--mode", type=str, default="per_hand", choices=["per_hand"])
    p.add_argument("--rep",  type=str, default="centroid", choices=["centroid", "medoid"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.anchor_dir.mkdir(parents=True, exist_ok=True)
    result_jsons = sorted(args.results_dir.rglob("*.json"))
    print(f"[recontract] {len(result_jsons)} result JSONs | mode={args.mode} rep={args.rep}")

    done = skipped = errors = 0
    seen = set()
    for jp in result_jsons:
        try:
            result = json.load(open(jp))
        except Exception:
            errors += 1; continue
        if "error" in result or result.get("file_name") is None:
            errors += 1; continue
        fn = result["file_name"]
        if fn in seen:
            skipped += 1; continue
        out_path = args.anchor_dir / f"{fn}_anchor.json"
        if out_path.exists() and not args.overwrite:
            seen.add(fn); skipped += 1; continue
        try:
            entry = contract(result, args.mode, args.rep)
        except Exception as e:
            print(f"[error] {fn}: {e}"); errors += 1; continue
        json.dump(entry, open(out_path, "w"))
        seen.add(fn); done += 1

    print(f"[done] written={done} skipped={skipped} errors={errors}")
    print(f"       anchor cache -> {args.anchor_dir}")


if __name__ == "__main__":
    main()
