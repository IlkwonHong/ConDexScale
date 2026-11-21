"""키프레임 추출기 — 동적 v1 구현 1번 (설계: md_claude/plans/dynamic_v1_design.md §3).

규칙 (7월 타당성 측정과 동일 — scripts/measure_arctic_contact_events.py 에서 포팅):
  손가락별 거리 → 이력 문턱(1.5cm 붙음/2.2cm 떨어짐) + 3프레임 디바운스
  → 손 수준 접촉 상태 {무접촉, 오른손, 왼손, 양손}
  키프레임 = 손 상태 변화 프레임 ∪ 관절 모션 구간 경계, 0.5초(15프레임) 병합
추가 산출: hold 구간(키프레임 사이 구간)별 손 상태·관절 통계 — contraction 입력 단위.

입력: data/arctic_cache/contact_dist/{subject}__{seq}.npz  (dist (T,10), theta (T,))
출력: data/contraction_dyn/keyframes/{subject}__{seq}.json
검증: 인자 없이 실행하면 301 시퀀스 전체를 7월 통계
  (md_claude/keyframes/contact_event_stats.json)와 대조한다.

실행: ./venv/bin/python -m contraction_dyn.keyframes            # 전체 + 검증
      ./venv/bin/python -m contraction_dyn.keyframes s01__box_grab_01   # 일부만
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/arctic_cache/contact_dist"
OUT = ROOT / "data/contraction_dyn/keyframes"
STATS = ROOT / "md_claude/keyframes/contact_event_stats.json"

FPS = 30.0
THR_ON, THR_OFF, MIN_LEN = 0.015, 0.022, 3     # 7월 확정 파라미터
ARTI_OMEGA_THR, SMOOTH_WIN, MERGE_F = 0.10, 5, 15
HAND_STATE = ["none", "right", "left", "both"]


def hysteresis_contact(dist, thr_on=THR_ON, thr_off=THR_OFF):
    state = np.zeros(len(dist), dtype=bool)
    cur = dist[0] < thr_on
    for t in range(len(dist)):
        if cur and dist[t] > thr_off:
            cur = False
        elif not cur and dist[t] < thr_on:
            cur = True
        state[t] = cur
    return state


def debounce(state, min_len=MIN_LEN):
    state = state.copy()
    changed = True
    while changed:
        changed = False
        idx = np.flatnonzero(np.diff(state.astype(np.int8))) + 1
        bounds = np.concatenate([[0], idx, [len(state)]])
        for i in range(1, len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_len:
                state[s:e] = state[s - 1]
                changed = True
                break
    return state


def count_segments(state):
    idx = np.flatnonzero(np.diff(state.astype(np.int8)))
    bounds = np.concatenate([[0], idx + 1, [len(state)]])
    return [(int(bounds[i]), int(bounds[i + 1]))
            for i in range(len(bounds) - 1) if state[bounds[i]]]


def extract(dist, theta):
    """키프레임 + hold 구간. dist (T,10) [오른손5, 왼손5], theta (T,)."""
    T = dist.shape[0]
    finger = np.stack([debounce(hysteresis_contact(dist[:, i]))
                       for i in range(10)], axis=1)
    right = debounce(finger[:, :5].any(axis=1))
    left = debounce(finger[:, 5:].any(axis=1))
    hand_set = right.astype(np.int8) + 2 * left.astype(np.int8)
    set_changes = np.flatnonzero(np.diff(hand_set)) + 1

    omega = np.gradient(theta) * FPS
    if SMOOTH_WIN > 1:
        omega = np.convolve(omega, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
    moving = debounce(np.abs(omega) > ARTI_OMEGA_THR, MIN_LEN * 2)
    arti_segs = count_segments(moving)

    kf = sorted(set(list(set_changes)
                    + [s for s, _ in arti_segs] + [e - 1 for _, e in arti_segs]))
    merged = []
    for f in kf:
        if not merged or f - merged[-1] > MERGE_F:
            merged.append(int(f))

    # hold 구간 = 키프레임 사이 (앞뒤 포함). contraction 은 rest 아닌 구간에서 수행.
    bounds = [0] + merged + [T]
    segments = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s <= 0:
            continue
        hs = int(np.bincount(hand_set[s:e], minlength=4).argmax())
        segments.append(dict(
            start=int(s), end=int(e), n_frames=int(e - s),
            hand_state=HAND_STATE[hs],
            arti_mean=round(float(theta[s:e].mean()), 4),
            arti_range=round(float(theta[s:e].max() - theta[s:e].min()), 4),
            arti_moving_frac=round(float(moving[s:e].mean()), 3),
            rep_frame=int((s + e) // 2),
        ))
    return dict(n_frames=T, keyframes=merged,
                arti_segments=[[int(a), int(b)] for a, b in arti_segs],
                segments=segments)


def run(names=None, verify=True):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {f"{r['subject']}__{r['seq']}": r
             for r in json.load(open(STATS))} if verify and STATS.exists() else {}
    files = sorted(CACHE.glob("*.npz"))
    if names:
        files = [f for f in files if f.stem in names]
    n_ok = n_bad = 0
    for f in files:
        with np.load(f) as z:
            dist, theta = z["dist"], z["theta"]
        r = extract(dist, theta)
        json.dump(r, open(OUT / f"{f.stem}.json", "w"), ensure_ascii=False)
        ref = stats.get(f.stem)
        if ref is not None:
            same = (r["keyframes"] == ref["keyframe_frames"]
                    and len(r["keyframes"]) == ref["keyframes_est"])
            n_ok += same
            n_bad += not same
            if not same:
                print(f"  불일치 {f.stem}: 내 {len(r['keyframes'])}개 vs 7월 "
                      f"{ref['keyframes_est']}개")
        if names:
            holds = [s for s in r["segments"] if s["hand_state"] != "none"]
            print(f"{f.stem}: 키프레임 {len(r['keyframes'])}개, hold 구간 "
                  f"{len(holds)}개 (무접촉 제외), 관절 구간 {len(r['arti_segments'])}개")
            for s in r["segments"]:
                print(f"   [{s['start']:4d}-{s['end']:4d}] {s['hand_state']:5s} "
                      f"관절 {s['arti_mean']:+.2f}±{s['arti_range']:.2f}"
                      f"{' (관절 모션)' if s['arti_moving_frac'] > 0.5 else ''}")
    if stats:
        print(f"\n7월 통계 대조: 일치 {n_ok} / 불일치 {n_bad}")
    print(f"저장: {OUT} ({len(files)}개)")


if __name__ == "__main__":
    run(sys.argv[1:] or None)
