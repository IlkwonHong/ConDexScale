#!/usr/bin/env python3
"""
Remove result/graph JSONs that do not belong to the target camera serial.

DexYCB GT is derived from multi-view fusion, so a single camera is sufficient.
This script removes the redundant 7/8 of generated files.

Usage:
  python scripts/clean_camera_data.py --dry-run          # 무엇이 삭제될지 확인
  python scripts/clean_camera_data.py                    # 실제 삭제
  python scripts/clean_camera_data.py --camera 839512060362  # 다른 카메라 선택
"""

import argparse
import json
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "results_train"
GRAPH_DIR   = REPO_ROOT / "data" / "graph_cache"
DEFAULT_CAM = "836212060125"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",  default=DEFAULT_CAM,
                   help="Camera serial to KEEP (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be deleted without deleting")
    args = p.parse_args()

    keep_cam = args.camera
    dry = args.dry_run
    prefix = "[DRY-RUN] " if dry else ""

    # ── result JSONs ──────────────────────────────────────────────────────────
    del_r, keep_r = [], []
    for path in sorted(RESULTS_DIR.glob("**/*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            fn = d.get("file_name", "")
            parts = fn.split("_") if fn else []
            cam = parts[3] if len(parts) >= 4 else ""
        except Exception:
            cam = ""

        if cam == keep_cam:
            keep_r.append(path)
        else:
            del_r.append(path)

    # ── graph JSONs ───────────────────────────────────────────────────────────
    del_g, keep_g = [], []
    for path in sorted(GRAPH_DIR.glob("*_graph.json")):
        parts = path.stem.replace("_graph", "").split("_")
        cam = parts[3] if len(parts) >= 4 else ""
        if cam == keep_cam:
            keep_g.append(path)
        else:
            del_g.append(path)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"카메라 유지: {keep_cam}")
    print(f"results_train  유지 {len(keep_r):5d}개  삭제 {len(del_r):5d}개")
    print(f"graph_cache    유지 {len(keep_g):5d}개  삭제 {len(del_g):5d}개")
    print(f"총 삭제 파일: {len(del_r) + len(del_g)}개")

    if dry:
        print("\n[DRY-RUN] 실제 삭제하려면 --dry-run 없이 실행하세요.")
        if del_r:
            print("  삭제 예시 (result):", del_r[0])
        if del_g:
            print("  삭제 예시 (graph) :", del_g[0])
        return

    print("\n삭제 중...")
    for path in del_r + del_g:
        path.unlink()
    print(f"완료. {len(del_r) + len(del_g)}개 파일 삭제됨.")


if __name__ == "__main__":
    main()
