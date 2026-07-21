"""Knot 스플라인: 채널당 K개 조절점 → 전체 T프레임 선형보간.

CEM 파라미터화의 핵심 압축 수단 (dexscale_mvp.md §1 Q5). 검증은 항상 전 스텝
롤아웃에서 하고, 파라미터만 knot에 둔다. coarse-to-fine: K를 늘려 재최적화할 때
refine_knots()로 이전 해를 새 K의 초기 μ로 승격한다.
"""
import numpy as np


def knot_times(n_knots, n_frames):
    """knot들의 프레임 위치 (양 끝 포함 균등 배치)."""
    return np.linspace(0.0, float(n_frames - 1), n_knots)


def interp_knots(knot_vals, n_frames):
    """(..., K) knot 값 → (..., T) 프레임별 값. 선형보간."""
    knot_vals = np.asarray(knot_vals, dtype=np.float64)
    K = knot_vals.shape[-1]
    tk = knot_times(K, n_frames)
    tq = np.arange(n_frames, dtype=np.float64)
    flat = knot_vals.reshape(-1, K)
    out = np.empty((flat.shape[0], n_frames))
    for i in range(flat.shape[0]):
        out[i] = np.interp(tq, tk, flat[i])
    return out.reshape(*knot_vals.shape[:-1], n_frames)


def refine_knots(knot_vals, new_K, n_frames):
    """coarse-to-fine: 기존 knot 곡선을 유지한 채 knot 수만 new_K로 늘림."""
    curve = interp_knots(knot_vals, n_frames)          # (..., T)
    tk_new = knot_times(new_K, n_frames)
    idx = np.clip(np.round(tk_new).astype(int), 0, n_frames - 1)
    return curve[..., idx]
