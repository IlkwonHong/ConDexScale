"""전 레코드 시뮬 판정 러너 — 우선순위 정렬 전량판.

gate_runner(대표 1개/그룹)와 같은 판정 프로토콜을 전 레코드(N≥3, α>floor)로 확장한다.
차이점:
  1. 대상 = 전 레코드에서 기존 gate_records.npz 에 있는 idx 제외.
  2. 순서 = 파지×N 그룹 내 α 내림차순 순위의 라운드 로빈 (모든 그룹의 2순위
     → 3순위 → …). 중간에 끊겨도 "상위 K개까지 판정 완료" 상태가 되도록.
  3. 체크포인트 = CHECKPOINT_EVERY 개마다 누적 결과를 원자적으로 저장
     (gate_records_all.partial.npz → 완주 시 gate_records_all.npz 로 개명).
  4. 준비(잡 생성)는 블록 단위 이중 버퍼 — 시뮬 풀이 도는 동안 다음 블록 준비.

실행: ./venv/bin/python -m contraction_gt.gate_runner_all <manifest_dir> <workers> [floor] [limit]
출력 필드는 gate_records.npz 와 동일 (idx, verdict, grasp_id, n_anchors, min_alpha).
"""

import sys
import time
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .gate_runner import _load_records, _prepare, _worker

BLOCK = 20_000
CHECKPOINT_EVERY = 20_000


def _priority_order(m, floor, done_idx):
    """전 레코드(N≥3, α>floor, 미판정)를 그룹 내 α 순위 라운드 로빈으로 정렬."""
    groups = defaultdict(list)
    for i in range(len(m["min_alpha"])):
        if int(m["n_anchors"][i]) < 3 or m["min_alpha"][i] <= floor:
            continue
        if i in done_idx:
            continue
        groups[(str(m["grasp_id"][i]), int(m["n_anchors"][i]))].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: -m["min_alpha"][i])
    order = []
    rank = 0
    while True:
        layer = [g[rank] for g in groups.values() if len(g) > rank]
        if not layer:
            break
        order.extend(layer)
        rank += 1
    return order


def _save(path_dir, verdicts, m, final):
    idx_arr = np.array(sorted(verdicts))
    v_arr = np.array([verdicts[i] for i in idx_arr])
    tmp = path_dir / "gate_records_all.tmp.npz"
    np.savez_compressed(tmp, idx=idx_arr, verdict=v_arr,
                        grasp_id=m["grasp_id"][idx_arr],
                        n_anchors=m["n_anchors"][idx_arr],
                        min_alpha=m["min_alpha"][idx_arr])
    name = "gate_records_all.npz" if final else "gate_records_all.partial.npz"
    tmp.replace(path_dir / name)


def run(manifest_dir, workers, floor=0.02, limit=None):
    from multiprocessing import get_context
    mdir = Path(manifest_dir)
    m = _load_records(manifest_dir)

    done_idx = set()
    gate_path = mdir / "gate_records.npz"
    if gate_path.exists():
        with np.load(gate_path, allow_pickle=True) as z:
            done_idx = set(z["idx"].astype(int).tolist())
    # 이전 전량판 중단분 재개
    partial_path = mdir / "gate_records_all.partial.npz"
    verdicts = {}
    if partial_path.exists():
        with np.load(partial_path, allow_pickle=True) as z:
            verdicts = dict(zip(z["idx"].astype(int).tolist(), z["verdict"].tolist()))
        done_idx |= set(verdicts)

    order = _priority_order(m, floor, done_idx)
    if limit:
        order = order[:limit]
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] 대상 {len(order):,}개 "
          f"(기왕 판정 {len(done_idx):,} 제외), workers={workers}", flush=True)

    blocks = [order[i:i + BLOCK] for i in range(0, len(order), BLOCK)]
    next_jobs = {}

    def prep(bi):
        next_jobs[bi] = _prepare(np.array(blocks[bi]), m, manifest_dir)

    prep(0)
    ctx = get_context("spawn")
    t0 = time.time()
    n_done = 0
    with ctx.Pool(workers) as pool:
        for bi, block in enumerate(blocks):
            jobs = next_jobs.pop(bi)
            th = None
            if bi + 1 < len(blocks):
                th = threading.Thread(target=prep, args=(bi + 1,))
                th.start()
            for idx, verdict, _nu in pool.imap_unordered(_worker, jobs, chunksize=8):
                verdicts[idx] = verdict
                n_done += 1
                if n_done % 2000 == 0:
                    r = n_done / (time.time() - t0)
                    eta = (len(order) - n_done) / r / 3600
                    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {n_done:,}/{len(order):,} "
                          f"({r:.1f}개/s, 남은 예상 {eta:.1f}h)", flush=True)
                if n_done % CHECKPOINT_EVERY == 0:
                    _save(mdir, verdicts, m, final=False)
            if th is not None:
                th.join()

    _save(mdir, verdicts, m, final=(limit is None))
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] 완료 {n_done:,}개, "
          f"판정 분포: {dict(Counter(verdicts.values()))}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]),
        float(sys.argv[3]) if len(sys.argv) > 3 else 0.02,
        int(sys.argv[4]) if len(sys.argv) > 4 else None)
