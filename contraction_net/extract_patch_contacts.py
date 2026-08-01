"""실측 패치 접촉 추출 — 1차분 10만 파지의 "접촉 부위 → 닿은 패치(+면적)" 테이블.

그릴링 질문 5 확정안: 그래프의 접촉부위↔패치 엣지를 근사(원반 반경)가 아니라
실측으로 만든다. 접촉 테이블과 같은 판정(손 표면점 896개, SDF 거리 < 3mm)을
재실행하되, 닿은 점마다 (최근접 패치, 소속 접촉 부위, 점 면적)을 남긴다.

출력: data/contraction_net/patch_contacts/part_*.npz — 평평한 행 구조
  grasp_id (str) / unit (int8, 0~5 = index..palm) / patch (int16, 0~127) / area (f32)
검증: 접촉 테이블의 부위별 접촉 플래그·면적과 대조 (같은 코드·같은 임계라 일치해야 함).

실행: ./venv/bin/python -m contraction_net.extract_patch_contacts [workers] [limit_groups]
"""

import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dexonomy"))

MANIFEST = ROOT / "data" / "contraction_gt_300k" / "pilot_manifest.tsv"
CONTACT = ROOT / "data" / "dexonomy" / "contact_cache"
SUCC = ROOT / "data" / "dexonomy" / "Dexonomy_GRASP_shadow" / "succ_collect"
PATCH = ROOT / "data" / "contraction_net" / "patch_cache"
OUT = ROOT / "data" / "contraction_net" / "patch_contacts"
N_GRASP = 100_000          # 1차분과 동일: 목록 앞 10만 행
THRESHOLD = 0.003          # 접촉 테이블 v2 와 동일 (3mm)
_W = {}


def _init(patch_dir=None):
    import torch
    torch.set_num_threads(1)
    from shadow_fk import DexonomyShadowFK, LINK_FINGER_ID
    from extract_contacts import load_sdf  # noqa: F401  (경로 부작용 포함)
    fk = DexonomyShadowFK()
    point_unit = np.empty(fk.point_areas.shape[0], dtype=np.int8)
    for li, (a, b) in enumerate(fk.link_slices):
        point_unit[a:b] = LINK_FINGER_ID[li]
    _W.update(fk=fk, point_unit=point_unit, load_sdf=load_sdf,
              patch=Path(patch_dir) if patch_dir else PATCH)   # spawn 워커용 경로 전달


def _one_group(job):
    """(ds, oid, [(rel, [ridx...]), ...]) -> 행 배열 + 검증 통계"""
    import torch
    ds, oid, rels = job
    fk, point_unit, load_sdf = _W["fk"], _W["point_unit"], _W["load_sdf"]
    mesh = ROOT / "data" / "dexonomy" / ds / "processed_data" / oid / "mesh" / "simplified.obj"
    factory = load_sdf(mesh)
    with np.load(_W["patch"] / ds / f"{oid}.npz") as z:
        centers = z["centers"].astype(np.float64)         # 단위 스케일
    G, U, P_, A = [], [], [], []
    n_flag_mismatch = n_units_checked = 0
    for rel, ridxs in rels:
        d = np.load(SUCC / rel, allow_pickle=True).item()
        qpos = d["grasp_qpos"][ridxs]
        with np.load(CONTACT / rel.replace(".npy", ".npz")) as c:
            ts = c["total_scale"][ridxs].astype(np.float64)
            fc = c["finger_contact"][ridxs]
        pts_w, _ = fk.link_points_world(qpos)
        pts_w = pts_w.numpy().astype(np.float64)
        N, P = pts_w.shape[:2]
        q = (pts_w / ts[:, None, None]).reshape(-1, 3)
        res = factory.object_frame_closest_point(torch.from_numpy(q).float(),
                                                 compute_normal=False)
        dist = res.distance.numpy().reshape(N, P) * ts[:, None]
        closest_u = res.closest.numpy().reshape(N, P, 3)   # 단위 스케일 좌표
        contact = dist < THRESHOLD
        for r in range(N):
            m = contact[r]
            if not m.any():
                continue
            cd = ((closest_u[r][m][:, None] - centers[None]) ** 2).sum(-1)
            pid = cd.argmin(1)
            acc = np.zeros((6, len(centers)))
            np.add.at(acc, (point_unit[m], pid),
                      fk.point_areas[m] * ts[r] * ts[r])   # 면적은 실스케일 (m²)
            uu, pp = np.nonzero(acc)
            gid = f"{rel}#{ridxs[r]}"
            G += [gid] * len(uu)
            U.append(uu.astype(np.int8)); P_.append(pp.astype(np.int16))
            A.append(acc[uu, pp].astype(np.float32))
            # 검증: 부위별 접촉 유무가 접촉 테이블 플래그와 일치하는가
            has = acc.sum(1) > 0
            n_units_checked += 6
            n_flag_mismatch += int((has != fc[r]).sum())
    if U:
        return (np.array(G), np.concatenate(U), np.concatenate(P_),
                np.concatenate(A), n_flag_mismatch, n_units_checked)
    return (np.array([], dtype=str), np.array([], np.int8),
            np.array([], np.int16), np.array([], np.float32),
            n_flag_mismatch, n_units_checked)


def main(workers=15, limit_groups=None, patch_name=None, out_name=None):
    global OUT
    patch_dir = (ROOT / "data" / "contraction_net" / patch_name) if patch_name else PATCH
    if out_name:
        OUT = ROOT / "data" / "contraction_net" / out_name
    rows = MANIFEST.read_text().splitlines()[1:][:N_GRASP]
    by_obj = defaultdict(lambda: defaultdict(list))
    for line in rows:
        _t, ds, oid, rel, ridx, _s, _n = line.split("\t")
        by_obj[(ds, oid)][rel].append(int(ridx))
    jobs = [(ds, oid, sorted(rels.items())) for (ds, oid), rels in sorted(by_obj.items())]
    if limit_groups:
        jobs = jobs[:limit_groups]
    n_grasp = sum(len(r) for j in jobs for _, r in j[2])
    print(f"물체 그룹 {len(jobs):,}, 파지 {n_grasp:,}, workers={workers}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    t0 = time.time()
    buf, part, done_grasp = [], 0, 0
    mism = checked = 0
    with ctx.Pool(workers, initializer=_init, initargs=(str(patch_dir),)) as pool:
        for k, out in enumerate(pool.imap_unordered(_one_group, jobs, chunksize=4)):
            buf.append(out[:4])
            mism += out[4]; checked += out[5]
            done_grasp += len(set(out[0].tolist()))
            if (k + 1) % 200 == 0:
                r = (k + 1) / (time.time() - t0)
                print(f"[{time.strftime('%H:%M:%S')}] 그룹 {k+1:,}/{len(jobs):,} "
                      f"({r:.1f}그룹/s, 남은 예상 {(len(jobs)-k-1)/r/60:.0f}분, "
                      f"플래그 불일치 {mism}/{checked})", flush=True)
            if sum(len(b[0]) for b in buf) > 2_000_000:
                _flush(buf, part); buf, part = [], part + 1
    _flush(buf, part)
    print(f"완료: 파지 {done_grasp:,}, 플래그 불일치 {mism}/{checked} "
          f"({mism/max(checked,1):.2%}), {(time.time()-t0)/60:.0f}분", flush=True)


def _flush(buf, part):
    if not buf:
        return
    np.savez_compressed(OUT / f"part_{part}.npz",
                        grasp_id=np.concatenate([b[0] for b in buf]),
                        unit=np.concatenate([b[1] for b in buf]),
                        patch=np.concatenate([b[2] for b in buf]),
                        area=np.concatenate([b[3] for b in buf]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15,
         int(sys.argv[2]) if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None,
         sys.argv[4] if len(sys.argv) > 4 else None)
