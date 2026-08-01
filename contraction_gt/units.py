"""P4-1: 접촉 유닛 병합과 연결 파티션 열거.

계획서 D2: 링크 단위 접촉 → 손가락/손바닥 유닛으로 병합 → 엄지(또는 대립
폴백 유닛)는 고정 앵커, 나머지를 인접 그래프 위의 연결 그룹 N-1개로 분할.

인접 규칙:
- 손가락은 손 순서(index-middle-ring-little)에서 "접촉 중인 손가락끼리" 연속
  (중간 손가락이 비접촉이면 그 틈은 닫힌 것으로 본다 — 계획서 D2의 카운팅과 동일).
- 손바닥(palm)은 접촉 중인 모든 손가락과 인접 (물리적 허브).
- 엄지는 파티션에 참여하지 않는다.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

THUMB = "thumb"
PALM = "palm"
FINGER_ORDER = ["index", "middle", "ring", "little"]
ALL_UNITS = [THUMB] + FINGER_ORDER + [PALM]


@dataclass
class UnitPatch:
    """한 유닛(손가락/손바닥)의 병합된 접촉 패치."""
    unit: str
    points: np.ndarray        # [n, 3]
    normals: np.ndarray       # [n, 3]
    areas: np.ndarray         # [n]
    centroid: np.ndarray      # [3] 면적 가중 (표면 투영 전)
    mean_normal: np.ndarray   # [3] 면적 가중 단위 법선
    total_area: float
    patch_radius: float       # centroid 에서 멤버 접촉점까지 최대 거리


def merge_contacts(points, normals, areas, unit_labels):
    """접촉점 배열 -> {unit: UnitPatch}. unit_labels 는 ALL_UNITS 의 원소."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    patches = {}
    for unit in ALL_UNITS:
        mask = np.asarray([u == unit for u in unit_labels])
        if not mask.any():
            continue
        p, n, a = points[mask], normals[mask], areas[mask]
        w = a / a.sum() if a.sum() > 0 else np.full(len(a), 1.0 / len(a))
        centroid = (w[:, None] * p).sum(axis=0)
        mean_n = (w[:, None] * n).sum(axis=0)
        norm = np.linalg.norm(mean_n)
        mean_n = mean_n / norm if norm > 1e-12 else n[0]
        patches[unit] = UnitPatch(
            unit=unit, points=p, normals=n, areas=a,
            centroid=centroid, mean_normal=mean_n,
            total_area=float(a.sum()),
            patch_radius=float(np.linalg.norm(p - centroid, axis=1).max()),
        )
    return patches


def opposition_unit(patches):
    """엄지 비접촉 폴백(D2): 나머지 유닛들의 면적 가중 평균 법선과 가장
    마주보는(내적 최소) 유닛을 대립 앵커 역할로 지정한다."""
    units = list(patches)
    if len(units) < 2:
        raise ValueError("대립 유닛을 정하려면 유닛이 2개 이상 필요")
    best_unit, best_score = None, np.inf
    for u in units:
        others = [patches[v] for v in units if v != u]
        w = np.array([o.total_area for o in others])
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        rest_n = (w[:, None] * np.stack([o.mean_normal for o in others])).sum(axis=0)
        nrm = np.linalg.norm(rest_n)
        if nrm < 1e-12:
            continue
        score = float(patches[u].mean_normal @ (rest_n / nrm))
        if score < best_score:
            best_unit, best_score = u, score
    return best_unit if best_unit is not None else units[0]


def _adjacency(rest_units):
    """rest(비엄지) 유닛들의 인접 그래프 dict."""
    fingers = [f for f in FINGER_ORDER if f in rest_units]
    adj = {u: set() for u in rest_units}
    for a, b in zip(fingers[:-1], fingers[1:]):   # 접촉 손가락끼리 연속(틈 닫힘)
        adj[a].add(b)
        adj[b].add(a)
    if PALM in rest_units:
        for f in fingers:
            adj[PALM].add(f)
            adj[f].add(PALM)
    return adj


def _connected(block, adj):
    block = set(block)
    if len(block) <= 1:
        return True
    seen, stack = set(), [next(iter(block))]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] & block - seen)
    return seen == block


def _set_partitions(items, k):
    """items 를 k개 비공백 블록으로 나누는 모든 집합 분할 (순서 무시)."""
    if k == 1:
        yield [list(items)]
        return
    if len(items) < k:
        return
    first, rest = items[0], items[1:]
    # first 가 단독 블록
    for sub in _set_partitions(rest, k - 1):
        yield [[first]] + sub
    # first 를 기존 블록에 합류
    for sub in _set_partitions(rest, k):
        for i in range(len(sub)):
            yield sub[:i] + [[first] + sub[i]] + sub[i + 1 :]


def enumerate_partitions(patches, n_anchors):
    """{unit: UnitPatch} -> 파티션 목록. 각 파티션 = 유닛 이름 그룹의 리스트,
    첫 그룹이 항상 엄지(또는 대립 폴백) 고정 앵커.

    반환 예 (N=3): [["thumb"], ["palm", "index"], ["middle", "ring"]]
    """
    units = list(patches)
    if len(units) < n_anchors:
        return []
    anchor0 = THUMB if THUMB in patches else opposition_unit(patches)
    rest = [u for u in units if u != anchor0]
    if n_anchors == 1:
        return [[[anchor0] + rest]] if not rest else []
    adj = _adjacency(rest)
    out = []
    seen = set()
    for part in _set_partitions(rest, n_anchors - 1):
        if all(_connected(block, adj) for block in part):
            key = frozenset(frozenset(b) for b in part)
            if key not in seen:
                seen.add(key)
                out.append([[anchor0]] + [sorted(b) for b in part])
    return out
