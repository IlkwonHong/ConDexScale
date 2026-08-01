"""contact_cache 전체를 집계해 타입별 접촉 통계를 만든다.

출력: contact_cache/summary.json + 콘솔 마크다운 표.
집계 항목: 타입별 파지 수, 손가락별 접촉율, 접촉 손가락 개수 히스토그램,
링크별 접촉 빈도, penetration 분포.
"""

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_fk import FINGER_NAMES, LINK_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def summarize_file(npz_path: str) -> dict:
    grasp_type = Path(npz_path).relative_to(summarize_file.root).parts[0]
    z = np.load(npz_path)
    fc = z["finger_contact"]  # (N,6)
    lc = z["link_contact"]  # (N,L)
    ld = z["link_min_dist"]
    if fc.size == 0:
        return {}
    n_fingers = fc[:, :5].sum(axis=1)  # palm 제외
    return {
        "type": grasp_type,
        "n": int(fc.shape[0]),
        "finger_contact_sum": fc.sum(axis=0).tolist(),
        "n_fingers_hist": np.bincount(n_fingers, minlength=6).tolist(),
        "link_contact_sum": lc.sum(axis=0).tolist(),
        "pen_3mm": int((ld < -0.003).sum()),
        "pen_min": float(ld.min()),
        "links_total": int(ld.size),
    }


def _init(root):
    summarize_file.root = Path(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "dexonomy" / "contact_cache"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.cache)

    files = [str(p) for p in root.rglob("*.npz")]
    print(f"{len(files)} npz files")
    agg: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "finger_contact_sum": np.zeros(6, dtype=np.int64),
        "n_fingers_hist": np.zeros(6, dtype=np.int64),
        "link_contact_sum": np.zeros(len(LINK_NAMES), dtype=np.int64),
        "pen_3mm": 0, "pen_min": 0.0, "links_total": 0})
    with Pool(args.workers, initializer=_init, initargs=(str(root),)) as pool:
        for i, r in enumerate(pool.imap_unordered(summarize_file, files, chunksize=256)):
            if not r:
                continue
            a = agg[r["type"]]
            a["n"] += r["n"]
            a["finger_contact_sum"] += np.array(r["finger_contact_sum"])
            a["n_fingers_hist"] += np.array(r["n_fingers_hist"])
            a["link_contact_sum"] += np.array(r["link_contact_sum"])
            a["pen_3mm"] += r["pen_3mm"]
            a["links_total"] += r["links_total"]
            a["pen_min"] = min(a["pen_min"], r["pen_min"])
            if (i + 1) % 200000 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    def type_key(t):
        return int(t.split("_")[0])

    summary = {}
    header = f"| {'grasp type':28s} | {'grasps':>9s} | " + " | ".join(f"{f:>6s}" for f in FINGER_NAMES) + " | 평균 접촉손가락수 |"
    print("\n" + header)
    print("|" + "-" * (len(header) - 2) + "|")
    total = np.zeros(6, dtype=np.int64)
    total_n = 0
    for t in sorted(agg, key=type_key):
        a = agg[t]
        rates = a["finger_contact_sum"] / a["n"]
        mean_nf = float((a["n_fingers_hist"] * np.arange(6)).sum() / a["n"])
        print(f"| {t:28s} | {a['n']:9,d} | " + " | ".join(f"{r*100:5.1f}%" for r in rates)
              + f" | {mean_nf:.2f} |")
        total += a["finger_contact_sum"]
        total_n += a["n"]
        summary[t] = {
            "n_grasps": a["n"],
            "finger_contact_rate": {f: float(r) for f, r in zip(FINGER_NAMES, rates)},
            "n_fingers_hist": a["n_fingers_hist"].tolist(),
            "link_contact_rate": {ln: float(c / a["n"]) for ln, c in zip(LINK_NAMES, a["link_contact_sum"])},
            "penetration_below_3mm_frac": float(a["pen_3mm"] / a["links_total"]),
            "deepest_penetration_m": a["pen_min"],
        }
    print(f"| {'TOTAL':28s} | {total_n:9,d} | " + " | ".join(f"{r*100:5.1f}%" for r in total / max(1, total_n)) + " | |")
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved -> {root/'summary.json'}")


if __name__ == "__main__":
    main()
