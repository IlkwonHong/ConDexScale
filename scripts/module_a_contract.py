#!/usr/bin/env python3
"""
Module A: Contact Contraction

GenHand 결과 (5개 contact points) → M개 Contact Anchor로 압축.

각 Anchor는 물체 표면 위의 3D 위치 + 평균 force direction으로 구성.
Multi-arm 로봇에서 각 암이 담당할 접촉 영역을 의미.

Usage:
  cd /workspace/genhand && source venv/bin/activate
  python scripts/module_a_contract.py --n-arms 3 --method hdbscan
  python scripts/module_a_contract.py --n-arms 2 --method kmeans
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import hdbscan
from sklearn.cluster import KMeans


RESULT_DIR = Path(__file__).resolve().parents[1] / "optimisation" / "results_train"
CONTRACT_DIR = Path(__file__).resolve().parents[1] / "optimisation" / "contracted"

YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick',
}


def quat_to_dir(q):
    """Quaternion (x,y,z,w) → unit direction vector (z-axis of rotation)."""
    r = Rotation.from_quat(q)  # scipy: (x,y,z,w)
    return r.apply([0, 0, 1])


def contract_hdbscan(contacts, wrenches, min_cluster_size=2):
    """
    HDBSCAN 기반 contraction. 클러스터 수가 가변적.
    노이즈(-1) 포인트는 가장 가까운 클러스터에 할당.
    """
    pts = np.array(contacts)  # (5, 3)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
    labels = clusterer.fit_predict(pts)

    # 노이즈를 가장 가까운 클러스터에 병합
    valid_labels = [l for l in labels if l != -1]
    if not valid_labels:
        labels = np.zeros(len(pts), dtype=int)
    else:
        for i, l in enumerate(labels):
            if l == -1:
                dists = [np.linalg.norm(pts[i] - pts[j]) for j, lj in enumerate(labels) if lj != -1]
                nearest_valid = [j for j, lj in enumerate(labels) if lj != -1][np.argmin(dists)]
                labels[i] = labels[nearest_valid]

    return _build_anchors(pts, wrenches, labels)


def contract_kmeans(contacts, wrenches, n_clusters):
    """K-means 기반 contraction. 정확히 n_clusters개 Anchor 생성."""
    pts = np.array(contacts)  # (5, 3)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(pts)
    return _build_anchors(pts, wrenches, labels)


def _build_anchors(pts, wrenches, labels):
    """레이블 기반으로 Anchor 딕셔너리 리스트 생성."""
    anchors = []
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        cluster_pts = pts[mask]
        cluster_dirs = np.array([quat_to_dir(wrenches[i]) for i in np.where(mask)[0]])

        position = cluster_pts.mean(axis=0)
        direction = cluster_dirs.mean(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else direction

        anchors.append({
            "cluster_id": int(lbl),
            "n_contacts": int(mask.sum()),
            "position": position.tolist(),       # (3,) 물체 표면 위 위치 (hand-local)
            "force_direction": direction.tolist(),  # (3,) 이 anchor에서 가할 힘의 방향
            "contact_indices": np.where(mask)[0].tolist(),
        })
    return anchors


def process_file(path: Path, n_arms: int, method: str):
    """단일 결과 JSON → contracted anchors 딕셔너리 반환."""
    d = json.loads(path.read_text())
    if "error" in d:
        return None

    # actual_contact: (1, 5, 3) → (5, 3)
    contacts = np.array(d["actual_contact"]).squeeze(0)  # (5, 3)
    # wr (wrench quaternions): (1, 5, 4) → (5, 4)
    wrenches = np.array(d["wr"]).squeeze(0)              # (5, 4)

    if method == "hdbscan":
        anchors = contract_hdbscan(contacts, wrenches)
    elif method == "kmeans":
        anchors = contract_kmeans(contacts, wrenches, n_clusters=n_arms)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "file_name": d.get("file_name"),
        "ycb_id": d.get("ycb_id"),
        "ycb_class": YCB_CLASSES.get(d.get("ycb_id"), "unknown"),
        "method": method,
        "n_arms_requested": n_arms,
        "n_anchors": len(anchors),
        "original_contacts": contacts.tolist(),   # (5, 3)
        "anchors": anchors,                        # M개 anchor
        "qr": d.get("qr"),                        # (1, 30) 관절각 보존
        "distance_loss": d.get("distance_loss"),
        "gf_0": d.get("gf_0"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="Shadow")
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--n-arms", type=int, default=3,
                        help="Number of robot arms / contact anchors")
    parser.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--visualize", action="store_true",
                        help="Open3D로 첫 번째 샘플 시각화")
    args = parser.parse_args()

    src_dir = RESULT_DIR / args.robot / f"mu{args.mu:.1f}"
    out_dir = CONTRACT_DIR / args.robot / f"mu{args.mu:.1f}" / f"{args.method}_M{args.n_arms}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"))
    print(f"[module_a] {len(files)} result files  →  {out_dir}")
    print(f"[module_a] method={args.method}  n_arms={args.n_arms}")

    results = []
    for f in files:
        out = process_file(f, args.n_arms, args.method)
        if out is None:
            print(f"  skip (error): {f.name}")
            continue
        out_path = out_dir / f.name
        out_path.write_text(json.dumps(out, indent=2))
        results.append(out)

    # 통계 요약
    n_anchors_list = [r["n_anchors"] for r in results]
    print(f"\n[module_a] 처리 완료: {len(results)}/{len(files)} 샘플")
    print(f"  anchor 수 분포: {sorted(set(n_anchors_list))}  "
          f"(평균 {np.mean(n_anchors_list):.1f})")
    print(f"  결과 저장: {out_dir}")

    if args.visualize and results:
        _visualize(results[0])


def _visualize(result):
    """Open3D로 original contacts(빨강) vs anchors(파랑) 시각화."""
    try:
        import open3d as o3d
    except ImportError:
        print("[visualize] open3d 없음 — matplotlib fallback")
        _visualize_mpl(result)
        return

    pts_orig = np.array(result["original_contacts"])  # (5, 3)
    pts_anchor = np.array([a["position"] for a in result["anchors"]])  # (M, 3)

    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(pts_orig)
    pcd_orig.paint_uniform_color([1, 0, 0])  # 빨강: 원본 5 contacts

    pcd_anchor = o3d.geometry.PointCloud()
    pcd_anchor.points = o3d.utility.Vector3dVector(pts_anchor)
    pcd_anchor.paint_uniform_color([0, 0, 1])  # 파랑: contracted anchors

    print(f"[visualize] {result['file_name']}  "
          f"ycb={result['ycb_class']}  "
          f"contacts=5 → anchors={result['n_anchors']}")
    o3d.visualization.draw_geometries([pcd_orig, pcd_anchor],
                                      window_name="Contact Contraction")


def _visualize_mpl(result):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pts_orig = np.array(result["original_contacts"])
    pts_anchor = np.array([a["position"] for a in result["anchors"]])
    dirs = np.array([a["force_direction"] for a in result["anchors"]])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pts_orig.T, c='red', s=80, label='original (5 contacts)')
    ax.scatter(*pts_anchor.T, c='blue', s=200, marker='*', label=f'anchors ({len(pts_anchor)})')

    scale = 0.01
    for pos, d in zip(pts_anchor, dirs):
        ax.quiver(*pos, *(d * scale), color='blue', linewidth=2)

    ax.set_title(f"{result['file_name']}\n{result['ycb_class']}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/contact_contraction.png", dpi=120)
    print("[visualize] 저장: /tmp/contact_contraction.png")
    plt.show()


if __name__ == "__main__":
    main()
