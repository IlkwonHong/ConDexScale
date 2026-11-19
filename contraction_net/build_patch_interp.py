"""패치 토큰 섞음 가중치 캐시 생성 (2026-10-17) — Stage2Dataset(patch_interp=True) 용.

물체마다 점 2048개의 가까운 패치 중심 3개 인덱스·가중치를 data/contraction_net/patch_interp3/ 에 저장.
실행: ./venv/bin/python -m contraction_net.build_patch_interp
"""

from multiprocessing import Pool

import numpy as np

from .stage2_dataset import Stage2Dataset, patch_interp_weights, INTERP_ROOT

_DS = None


def _one(key):
    out = INTERP_ROOT / key[0] / f"{key[1]}.npz"
    if out.exists():
        return 0
    obj = _DS._objects.loader_fn(key)
    idx, w = patch_interp_weights(obj["pts"], obj["centers"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, idx=idx.astype(np.int16), w=w)
    return 1


def main():
    global _DS
    keys = set()
    for split in ("train", "val"):
        ds = Stage2Dataset(split=split, augment=False, contact_geom=False, com_feat=True)
        keys |= {ds.obj_of[g] for g, _ in ds.samples}
        _DS = ds
    keys = sorted(keys)
    print(f"물체 {len(keys)}개", flush=True)
    with Pool(12) as pool:
        n = sum(pool.imap_unordered(_one, keys, chunksize=32))
    print(f"새로 저장 {n}개 → {INTERP_ROOT}")


if __name__ == "__main__":
    main()
