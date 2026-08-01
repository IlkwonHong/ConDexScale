"""P0-lite: 파일럿 grasp 층화 추출 → 매니페스트 TSV + 물체 목록 생성.

층화 규칙 (계획서 D9/파일럿 설계):
- 31개 taxonomy 타입에 균등 배분.
- 타입 안에서는 물체를 셔플해 물체당 파일 1개, 파일당 grasp 1개(첫 행 —
  행 순서가 스케일 비정렬임을 확인했으므로 편향 없음).
- per-grasp 스케일은 scene_scale[행]에서 읽는다 (파일명 아님).

실행:
  ./venv/bin/python -m contraction_gt.pilot_select <succ_collect_dir> <out_dir> [총개수]
출력:
  <out_dir>/pilot_manifest.tsv   (grasp 1건 = 1행)
  <out_dir>/pilot_objects.txt    (캐시 빌드용 dataset object_id 목록)
"""

import random
import sys
from pathlib import Path

import numpy as np

from .object_cache import parse_scene_path


def select_pilot(succ_dir, out_dir, total=10_000, seed=0):
    succ_dir, out_dir = Path(succ_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    type_dirs = sorted(d for d in succ_dir.iterdir() if d.is_dir())
    per_type = total // len(type_dirs)
    rows, objects = [], set()

    for tdir in type_dirs:
        obj_dirs = [d for d in tdir.iterdir() if d.is_dir()]
        rng.shuffle(obj_dirs)
        count = 0
        for odir in obj_dirs:
            if count >= per_type:
                break
            files = sorted((odir / "floating").glob("*.npy"))
            if not files:
                continue
            f = rng.choice(files)
            try:
                d = np.load(f, allow_pickle=True).item()
                ds, oid = parse_scene_path(str(d["scene_path"]))
                scale = float(np.asarray(d["scene_scale"]).ravel()[0])
                n_in_file = int(d["grasp_qpos"].shape[0])
            except Exception as e:
                print(f"건너뜀 {f}: {e!r}", flush=True)
                continue
            rows.append((tdir.name, ds, oid, str(f.relative_to(succ_dir)), 0, scale, n_in_file))
            objects.add((ds, oid))
            count += 1
        print(f"{tdir.name}: {count}개", flush=True)

    with open(out_dir / "pilot_manifest.tsv", "w") as fh:
        fh.write("type\tdataset\tobject_id\tnpy_rel_path\trow_index\tscene_scale\tn_in_file\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    with open(out_dir / "pilot_objects.txt", "w") as fh:
        for ds, oid in sorted(objects):
            fh.write(f"{ds} {oid}\n")
    print(f"총 grasp {len(rows)}개, 물체 {len(objects)}개")
    return len(rows), len(objects)


if __name__ == "__main__":
    succ = sys.argv[1]
    out = sys.argv[2]
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    select_pilot(succ, out, total)
