"""Dexonomy 접촉 추출 파이프라인의 asset 준비 상태 점검 (idempotent).

- data/dexonomy/{DGN_5k,objaverse_5k}/processed_data 존재 확인 (없으면 tar 추출 안내)
- extlib/Dexonomy clone + mujoco_menagerie shadow_hand mesh 존재 확인 (없으면 명령 안내)
- --inspect: MJCF 모델 구조 리포트 (nq, joint 순서, collision geom)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEXONOMY_DATA = REPO_ROOT / "data" / "dexonomy"
CLONE = REPO_ROOT / "extlib" / "Dexonomy"
XML = CLONE / "assets" / "hand" / "shadow" / "right.xml"
MESH_DIR = CLONE / "third_party" / "mujoco_menagerie" / "shadow_hand" / "assets"


def check() -> bool:
    ok = True
    for ds, tarball in [("DGN_5k", "DGN_5k_processed.tar.gz"),
                        ("objaverse_5k", "objaverse_5k_processed.tar.gz")]:
        p = DEXONOMY_DATA / ds / "processed_data"
        if p.is_dir():
            print(f"[ok] {p}")
        else:
            ok = False
            print(f"[MISSING] {p}\n  -> cd {DEXONOMY_DATA} && tar -xzf {tarball}")
    if XML.exists():
        print(f"[ok] {XML}")
    else:
        ok = False
        print(f"[MISSING] {XML}\n  -> git clone --depth 1 https://github.com/JYChen18/Dexonomy {CLONE}")
    if (MESH_DIR / "palm.obj").exists():
        print(f"[ok] {MESH_DIR}")
    else:
        ok = False
        print(f"[MISSING] {MESH_DIR}\n  -> git clone --depth 1 --filter=blob:none --sparse "
              f"https://github.com/google-deepmind/mujoco_menagerie {CLONE/'third_party'/'mujoco_menagerie'}"
              f" && cd {CLONE/'third_party'/'mujoco_menagerie'} && git sparse-checkout set shadow_hand")
    return ok


def inspect():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(XML))
    print(f"mujoco {mujoco.__version__}; nq={m.nq} njnt={m.njnt} nbody={m.nbody} ngeom={m.ngeom}")
    print("joints (qposadr 순):")
    order = [None] * m.nq
    for j in range(m.njnt):
        order[int(m.jnt_qposadr[j])] = m.joint(j).name
    print("  " + ", ".join(order))
    n_coll = sum(1 for g in range(m.ngeom)
                 if int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
    print(f"collision geoms: {n_coll}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    ok = check()
    if args.inspect and ok:
        inspect()
    sys.exit(0 if ok else 1)
