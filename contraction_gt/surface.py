"""P2 핵심 유틸: 물체 표면 후보점 샘플링과 ρ 영역 조회.

물체당 한 번 샘플링해 두고(grasp 150만 개에 상각), grasp 마다는 조회만 한다.

주의(파일럿 체크리스트 7번 관련): ρ 영역은 측지 거리 대신 유클리드 거리 +
법선 반구 필터로 근사한다. 얇은 물체에서 반대면으로 새는 것을 법선 필터가
막는다. 곡률 기반 μ_tor 보정은 v1 에서 미구현(0 채움) — 파일럿에서 보정
필요성을 판단한다.
"""

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class SurfaceSamples:
    points: np.ndarray    # [M, 3]
    normals: np.ndarray   # [M, 3] — 안쪽(inward) 법선. 파이프라인 전체 규약:
                          # 법선 = 접촉힘이 물체를 미는 방향(물체 내부로 향함)
    r_obj: float          # 외접반경 (질량중심 기준 최대 정점 거리)
    com: np.ndarray       # [3] 질량중심(부피 기준, 실패 시 정점 평균)


def sample_surface(mesh: trimesh.Trimesh, count: int = 2048, seed: int = 0) -> SurfaceSamples:
    """면적 가중 균일 표면 샘플 + 안쪽 법선(trimesh 바깥 법선을 뒤집음)."""
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(2**31)))
    normals = -mesh.face_normals[face_idx]
    try:
        com = np.asarray(mesh.center_mass, dtype=np.float64)
        if not np.isfinite(com).all():
            raise ValueError
    except Exception:
        com = mesh.vertices.mean(axis=0)
    r_obj = float(np.linalg.norm(mesh.vertices - com, axis=1).max())
    return SurfaceSamples(points=np.asarray(points, dtype=np.float64),
                          normals=np.asarray(normals, dtype=np.float64),
                          r_obj=r_obj, com=com)


def _farthest_point_subsample(points, k, seed=0):
    """커버리지 보존 서브샘플 (farthest point sampling)."""
    n = len(points)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        idx = int(d.argmax())
        chosen.append(idx)
        d = np.minimum(d, np.linalg.norm(points - points[idx], axis=1))
    return np.asarray(chosen)


def candidates_in_region(samples: SurfaceSamples, center, rho, ref_normal=None,
                         k_max=32, normal_dot_min=0.1, seed=0):
    """center 주위 ρ 이내 표면 후보 인덱스 (최대 k_max 개, 커버리지 보존).

    ref_normal 이 주어지면 내적 > normal_dot_min 인 점만 남겨 얇은 물체의
    반대면 누출을 차단한다. 필터로 후보가 사라지면 필터 없이 재시도한다.
    """
    center = np.asarray(center, dtype=np.float64)
    d = np.linalg.norm(samples.points - center, axis=1)
    mask = d <= rho
    if ref_normal is not None:
        nmask = samples.normals @ np.asarray(ref_normal, dtype=np.float64) > normal_dot_min
        if (mask & nmask).any():
            mask = mask & nmask
    if not mask.any():
        mask = d <= d.min() + 1e-9   # 최소한 최근접점 1개
    idx = np.nonzero(mask)[0]
    sub = _farthest_point_subsample(samples.points[idx], k_max, seed=seed)
    return idx[sub]
