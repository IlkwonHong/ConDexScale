"""Dexonomy 파지 9.5M개 전체에 대해 링크/손가락별 접촉 라벨을 계산해
data/dexonomy/contact_cache/<Type>/<objid>/floating/scaleNNN.npz 로 저장한다.

병렬화 단위 = 물체(objid): 물체당 MeshSDF를 1회만 빌드해 소속 파일 전부 처리.
resume: 출력 npz가 있으면 skip. 오류는 contact_cache/_index/errors.jsonl 에 기록.

사용:
  python scripts/dexonomy/extract_contacts.py --limit 20      # pilot
  python scripts/dexonomy/extract_contacts.py                 # full run (resume)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_fk import (  # noqa: E402
    DEFAULT_XML,
    FINGER_NAMES,
    LINK_FINGER_ID,
    LINK_NAMES,
    LINK_POINTS_CACHE,
    DexonomyShadowFK,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "pytorch_volumetric" / "src"))
import pytorch_volumetric as pv  # noqa: E402

DEXONOMY_DATA = REPO_ROOT / "data" / "dexonomy"
SUCC_ROOT = DEXONOMY_DATA / "Dexonomy_GRASP_shadow" / "succ_collect"
DATASETS = ["DGN_5k", "objaverse_5k"]

_WORKER = {}  # worker 전역 (initializer에서 채움)


def load_sdf(mesh_path: str | Path):
    """unit mesh -> pytorch_volumetric factory (closest point + normal 질의용)."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    return pv.MeshObjectFactory(
        mesh_name=None,
        preload_mesh=dict(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)),
    )


def compute_contacts_for_qpos(fk: DexonomyShadowFK, factory, qpos: np.ndarray,
                              total_scale: np.ndarray, threshold: float) -> dict:
    """qpos (N,29), total_scale (N,) -> 링크/손가락별 접촉 딕셔너리 (모두 numpy)."""
    N = qpos.shape[0]
    L = len(LINK_NAMES)
    pts_w, _ = fk.link_points_world(qpos)  # (N,P,3) torch
    pts_w = pts_w.numpy().astype(np.float64)
    P = pts_w.shape[1]

    s = total_scale.reshape(N, 1, 1)
    query = (pts_w / s).reshape(-1, 3)
    res = factory.object_frame_closest_point(torch.from_numpy(query).float(), compute_normal=True)
    dist_w = res.distance.numpy().reshape(N, P) * total_scale.reshape(N, 1)
    closest_w = res.closest.numpy().reshape(N, P, 3) * s
    normal_w = res.normal.numpy().reshape(N, P, 3)

    link_min_dist = np.empty((N, L), dtype=np.float32)
    link_pt = np.empty((N, L, 3), dtype=np.float32)
    link_nrm = np.empty((N, L, 3), dtype=np.float32)
    link_area = np.zeros((N, L), dtype=np.float32)
    link_npts = np.zeros((N, L), dtype=np.int16)
    link_centroid = np.empty((N, L, 3), dtype=np.float32)
    in_contact = dist_w < threshold  # (N,P)
    pt_w = in_contact * fk.point_areas[None, :]  # 점당 대표면적 가중치 (N,P)
    rows = np.arange(N)
    for li, (a, b) in enumerate(fk.link_slices):
        seg = dist_w[:, a:b]
        idx = seg.argmin(axis=1)
        link_min_dist[:, li] = seg[rows, idx]
        link_pt[:, li] = closest_w[:, a:b][rows, idx]
        link_nrm[:, li] = normal_w[:, a:b][rows, idx]
        wseg = pt_w[:, a:b]  # (N,K)
        area = wseg.sum(axis=1)
        link_area[:, li] = area
        link_npts[:, li] = in_contact[:, a:b].sum(axis=1)
        # 접촉 패치 centroid: 임계 내 점들의 물체표면 최근접점을 면적 가중 평균.
        # 비접촉이면 최근접점으로 폴백 (area==0으로 구분 가능).
        centroid = (closest_w[:, a:b] * wseg[:, :, None]).sum(axis=1) / np.maximum(area, 1e-12)[:, None]
        link_centroid[:, li] = np.where(area[:, None] > 0, centroid, link_pt[:, li])
    link_contact = link_min_dist < threshold

    F = len(FINGER_NAMES)
    finger_min_dist = np.empty((N, F), dtype=np.float32)
    finger_pt = np.empty((N, F, 3), dtype=np.float32)
    finger_nrm = np.empty((N, F, 3), dtype=np.float32)
    finger_link = np.empty((N, F), dtype=np.int8)
    finger_area = np.zeros((N, F), dtype=np.float32)
    finger_centroid = np.empty((N, F, 3), dtype=np.float32)
    for fi in range(F):
        lids = np.where(LINK_FINGER_ID == fi)[0]
        seg = link_min_dist[:, lids]
        idx = seg.argmin(axis=1)
        gl = lids[idx]
        finger_min_dist[:, fi] = seg[rows, idx]
        finger_pt[:, fi] = link_pt[rows, gl]
        finger_nrm[:, fi] = link_nrm[rows, gl]
        finger_link[:, fi] = gl.astype(np.int8)
        la = link_area[:, lids]  # (N,len(lids))
        fa = la.sum(axis=1)
        finger_area[:, fi] = fa
        centroid = (link_centroid[:, lids] * la[:, :, None]).sum(axis=1) / np.maximum(fa, 1e-12)[:, None]
        finger_centroid[:, fi] = np.where(fa[:, None] > 0, centroid, finger_pt[:, fi])
    finger_contact = finger_min_dist < threshold

    return dict(
        link_min_dist=link_min_dist, link_contact=link_contact,
        link_contact_point=link_pt, link_contact_normal=link_nrm,
        link_contact_area=link_area, link_n_contact_pts=link_npts,
        link_patch_centroid=link_centroid,
        finger_min_dist=finger_min_dist, finger_contact=finger_contact,
        finger_contact_point=finger_pt, finger_contact_normal=finger_nrm,
        finger_contact_link=finger_link,
        finger_contact_area=finger_area, finger_patch_centroid=finger_centroid,
        total_scale=total_scale.astype(np.float32),
    )


def resolve_dataset(objid: str) -> str | None:
    for ds in DATASETS:
        if (DEXONOMY_DATA / ds / "processed_data" / objid / "mesh" / "simplified.obj").exists():
            return ds
    return None


def mesh_path_from_scene_path(scene_path: str) -> Path:
    """npy 내부 scene_path(권위 소스) -> unit mesh 경로."""
    parts = Path(scene_path).parts  # .../<dataset>/scene_cfg/<objid>/floating/scaleNNN.npy
    dataset = parts[parts.index("scene_cfg") - 1]
    objid = parts[parts.index("scene_cfg") + 1]
    return DEXONOMY_DATA / dataset / "processed_data" / objid / "mesh" / "simplified.obj"


def build_file_index(index_path: Path, force: bool = False) -> dict:
    """objid -> {dataset, files: [상대경로...]}; JSON 캐시.

    폴더명이 실제 물체와 다른 업스트림 노이즈(예: 10_Power_Disk/ls)는
    npy 내부 scene_path로 해석해 검증한다.
    """
    if index_path.exists() and not force:
        return json.loads(index_path.read_text())
    index: dict[str, dict] = {}
    unresolved = []
    for tdir in sorted(SUCC_ROOT.iterdir()):
        if not tdir.is_dir():
            continue
        for odir in tdir.iterdir():
            objid = odir.name
            files = [str(p.relative_to(SUCC_ROOT)) for p in odir.glob("floating/*.npy")]
            entry = index.get(objid)
            if entry is None:
                ds = resolve_dataset(objid)
                if ds is None and files:  # 폴더명이 물체가 아님 -> scene_path로 해석
                    d = np.load(SUCC_ROOT / files[0], allow_pickle=True).item()
                    mesh = mesh_path_from_scene_path(str(d["scene_path"]))
                    if not mesh.exists():
                        unresolved.append(objid)
                        continue
                    print(f"  [index] '{objid}' -> {mesh.parts[-3]} (scene_path로 해석)")
                    ds = "scene_path"
                entry = index[objid] = {"dataset": ds, "files": []}
            entry["files"].extend(files)
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} objids unresolved (첫 10개): {unresolved[:10]}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    return index


def _init_worker(cfg: dict):
    torch.set_num_threads(1)
    _WORKER["fk"] = DexonomyShadowFK(xml_path=cfg["xml"], points_cache=cfg["points_cache"])
    _WORKER["cfg"] = cfg


def process_object(task: tuple) -> dict:
    objid, dataset, files = task
    cfg = _WORKER["cfg"]
    fk = _WORKER["fk"]
    out_root = Path(cfg["out_root"])
    n_files = n_grasps = n_skip = 0
    errors = []
    t0 = time.time()
    factory, factory_mesh = None, None
    for rel in files:
        out_path = out_root / rel.replace(".npy", ".npz")
        if out_path.exists() and not cfg["overwrite"]:
            n_skip += 1
            continue
        try:
            d = np.load(SUCC_ROOT / rel, allow_pickle=True).item()
            mesh = mesh_path_from_scene_path(str(d["scene_path"]))  # 권위 소스
            if factory is None or factory_mesh != mesh:
                factory = load_sdf(mesh)
                factory_mesh = mesh
            qpos = np.asarray(d[cfg["qpos_key"]], dtype=np.float64)
            bucket = float(Path(rel).stem.replace("scale", "")) / 100.0
            total_scale = bucket * np.asarray(d["scene_scale"], dtype=np.float64)
            # 결정성: sdf의 ray-noise가 np.random을 쓰므로 파일 경로 기반 시드 고정
            np.random.seed(zlib.crc32(rel.encode()) & 0xFFFFFFFF)
            if qpos.shape[0] == 0:
                result = {}
            else:
                result = compute_contacts_for_qpos(fk, factory, qpos, total_scale, cfg["threshold"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez(f, **result)
            os.replace(tmp, out_path)
            n_files += 1
            n_grasps += qpos.shape[0]
        except Exception:
            errors.append({"file": rel, "traceback": traceback.format_exc()})
    return dict(objid=objid, n_files=n_files, n_grasps=n_grasps, n_skip=n_skip,
                errors=errors, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="처리할 물체 수 제한 (pilot)")
    ap.add_argument("--threshold", type=float, default=0.003)
    ap.add_argument("--qpos-key", default="grasp_qpos",
                    choices=["grasp_qpos", "pregrasp_qpos", "squeeze_qpos"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    suffix = "" if args.qpos_key == "grasp_qpos" else "_" + args.qpos_key.split("_")[0]
    out_root = Path(args.out) if args.out else DEXONOMY_DATA / f"contact_cache{suffix}"
    index_dir = out_root / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    index = build_file_index(index_dir / "file_index.json", force=args.reindex)
    tasks = sorted(
        ((objid, e["dataset"], e["files"]) for objid, e in index.items()),
        key=lambda t: -len(t[2]))
    if args.limit:
        tasks = tasks[:args.limit]
    total_files = sum(len(t[2]) for t in tasks)
    print(f"objects: {len(tasks)}, files: {total_files}, workers: {args.workers}, "
          f"qpos_key: {args.qpos_key}, threshold: {args.threshold*1000:.1f}mm")

    meta = dict(link_names=LINK_NAMES, finger_names=FINGER_NAMES,
                link_finger_id=LINK_FINGER_ID.tolist(), threshold=args.threshold,
                qpos_key=args.qpos_key, xml=str(DEFAULT_XML), version=2)
    (out_root / "meta.json").write_text(json.dumps(meta, indent=1))

    cfg = dict(out_root=str(out_root), threshold=args.threshold, qpos_key=args.qpos_key,
               overwrite=args.overwrite, xml=str(DEFAULT_XML), points_cache=str(LINK_POINTS_CACHE))
    err_f = open(index_dir / "errors.jsonl", "a")
    t0 = time.time()
    done_f = done_g = done_o = n_err = 0
    last_print = 0.0
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,), maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(process_object, tasks, chunksize=1):
            done_o += 1
            done_f += r["n_files"] + r["n_skip"]
            done_g += r["n_grasps"]
            for e in r["errors"]:
                n_err += 1
                err_f.write(json.dumps(e) + "\n")
            err_f.flush()
            now = time.time()
            if now - last_print > 30 or done_o == len(tasks):
                rate = done_f / max(1e-9, now - t0)
                eta = (total_files - done_f) / max(1e-9, rate)
                print(f"[{now-t0:7.0f}s] objects {done_o}/{len(tasks)}  files {done_f}/{total_files}"
                      f"  grasps {done_g}  errors {n_err}  ({rate:.0f} files/s, ETA {eta/60:.0f}min)",
                      flush=True)
                last_print = now
    err_f.close()
    print(f"DONE in {(time.time()-t0)/60:.1f} min; errors: {n_err}")


if __name__ == "__main__":
    main()
