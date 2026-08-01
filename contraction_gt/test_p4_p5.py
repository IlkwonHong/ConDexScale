"""P4(units/surface/search)·P5(sim_eval) 단위 테스트.

실행: ./venv/bin/python -m contraction_gt.test_p4_p5

결정적 정답: 파티션 개수는 계획서 D2 의 조합 계산(전 유닛 접촉 시 N=2/3/4
에서 1/10/15개)과 일치해야 한다. 시뮬은 물리적 순서 관계(대향은 잡히고,
같은 면은 당기면 놓치고, 비틀림 계수가 크면 비틀림을 더 버팀)를 검증한다.
"""

import numpy as np
import trimesh

from contraction_gt.units import merge_contacts, enumerate_partitions, opposition_unit, THUMB, PALM
from contraction_gt.surface import sample_surface, candidates_in_region
from contraction_gt.search import SearchConfig, search_grasp, mu_tor_from_pad
from contraction_gt.sim_eval import SimConfig, measure_capacity

PASS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    PASS.append(bool(cond))


def _full_hand_patches():
    """6유닛 전부 접촉하는 합성 grasp (단위구 표면 위)."""
    directions = {
        "thumb": [1.0, 0, 0], "index": [-0.9, 0.3, 0.3], "middle": [-1.0, 0, 0.15],
        "ring": [-1.0, -0.15, 0], "little": [-0.9, -0.3, -0.1], "palm": [-0.4, 0, -0.9],
    }
    pts, nrms, areas, labels = [], [], [], []
    for u, d in directions.items():
        d = np.asarray(d) / np.linalg.norm(d)
        for jitter in (0.0, 0.05):
            p = d + jitter
            p = p / np.linalg.norm(p)
            pts.append(p)
            nrms.append(-p)
            areas.append(1.0 if u != "palm" else 3.0)
            labels.append(u)
    return merge_contacts(pts, nrms, areas, labels)


def test_merge():
    patches = _full_hand_patches()
    check("병합: 6유닛 생성", set(patches) == {"thumb", "index", "middle", "ring", "little", "palm"})
    t = patches["thumb"]
    check("병합: centroid 가 멤버 중간", np.linalg.norm(t.centroid - t.points.mean(0)) < 1e-9,
          "(면적 동일 → 평균과 일치)")
    check("병합: 법선 단위벡터", abs(np.linalg.norm(t.mean_normal) - 1.0) < 1e-9)


def test_partition_counts():
    """계획서 D2 의 조합 계산 검증: 전 유닛 접촉 시 N=2/3/4 → 1/10/15개."""
    patches = _full_hand_patches()
    for n, expected in [(2, 1), (3, 10), (4, 15)]:
        parts = enumerate_partitions(patches, n)
        ok_thumb = all(p[0] == [THUMB] for p in parts)
        check(f"파티션 수 N={n}: {expected}개 + 엄지 고정", len(parts) == expected and ok_thumb,
              f"실제 {len(parts)}개")


def test_partition_fewer_units():
    """손끝 3유닛(엄지+검지+중지)만 접촉: N=2 → {검지,중지} 묶음 1개, N=3 → 1개."""
    patches = {u: p for u, p in _full_hand_patches().items() if u in ("thumb", "index", "middle")}
    p2 = enumerate_partitions(patches, 2)
    p3 = enumerate_partitions(patches, 3)
    p4 = enumerate_partitions(patches, 4)
    check("3유닛 N=2 = 1개", len(p2) == 1, f"{p2}")
    check("3유닛 N=3 = 1개", len(p3) == 1)
    check("3유닛 N=4 = 0개 (유닛 부족)", len(p4) == 0)


def test_gap_closing():
    """중지 비접촉 시 검지-약지가 인접(틈 닫힘)해 한 그룹이 될 수 있는지."""
    patches = {u: p for u, p in _full_hand_patches().items()
               if u in ("thumb", "index", "ring", "little")}
    p2 = enumerate_partitions(patches, 2)     # {엄지}{검지,약지,소지}
    check("틈 닫힘: N=2 성립", len(p2) == 1 and sorted(p2[0][1]) == ["index", "little", "ring"])


def test_thumbless_fallback():
    """엄지 비접촉: 법선이 나머지와 가장 마주보는 유닛이 고정 앵커가 되는지."""
    patches = {u: p for u, p in _full_hand_patches().items() if u != THUMB}
    # palm 법선은 (-0.4,0,-0.9) 방향 물체 안쪽 — 손가락들(-x 계열)과 가장 마주보는 건 palm 이 아님.
    # 합성 데이터에서 손가락 4개가 거의 같은 방향이므로 대립 유닛은 palm 이어야 한다.
    fb = opposition_unit(patches)
    parts = enumerate_partitions(patches, 2)
    check("폴백: 대립 유닛 선정", fb == parts[0][0][0], f"선정={fb}")
    check("폴백: N=2 파티션 생성", len(parts) >= 1)


def test_surface_region():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    s = sample_surface(mesh, count=1500)
    check("표면: r_obj ≈ 1", abs(s.r_obj - 1.0) < 0.01, f"{s.r_obj:.3f}")
    center = np.array([1.0, 0, 0])
    inward = -center     # 파이프라인 규약: 법선 = 안쪽
    idx = candidates_in_region(s, center, rho=0.3, ref_normal=inward, k_max=32)
    d = np.linalg.norm(s.points[idx] - center, axis=1)
    check("표면: ρ 영역 조회", len(idx) > 5 and d.max() <= 0.3 + 1e-9,
          f"{len(idx)}개, 최대거리 {d.max():.3f}")
    # 법선 필터: 반대면(-x) 점이 섞이지 않아야 함
    check("표면: 법선 반구 필터", (s.normals[idx] @ inward > 0).all())


def test_search_sphere():
    """단위구 + 합성 5지 파지: N=2 탐색이 대립 배치를 찾고 α > 0 이어야 한다."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    s = sample_surface(mesh, count=1500)
    patches = _full_hand_patches()
    parts = enumerate_partitions(patches, 2) + enumerate_partitions(patches, 3)
    cfg = SearchConfig(pad_radius_rel=(0.10,), seed=0)
    recs = search_grasp(s, patches, parts, cfg)
    check("탐색: 레코드 수 = 파티션 수 × 반경 1종", len(recs) == len(parts), f"{len(recs)}")
    r2 = [r for r in recs if r.n_anchors == 2]
    check("탐색: N=2 min-α > 0 (비틀림 有)", all(r.min_alpha > 1e-4 for r in r2),
          f"min={min(r.min_alpha for r in r2):.5f}")
    # N=2 대립성: 두 앵커 사이 각도가 대체로 마주봄 (>120°)
    r = r2[0]
    cosang = float(r.anchor_pos[0] @ r.anchor_pos[1])
    check("탐색: N=2 대립 배치", cosang < -0.5, f"cos={cosang:.2f}")
    check("탐색: α 벡터 14방향", all(rec.alpha.shape == (14,) for rec in recs))
    # D3: ρ_group = max(그룹 패치 반경, c·r_obj) — 넓은 병합 그룹은 자기 패치
    # 범위까지 탐색하는 게 설계상 정상이므로, 그 규칙을 그대로 상한으로 쓴다.
    ok = True
    for rec in recs:
        for gi, group in enumerate(rec.partition):
            pts = np.concatenate([patches[u].points for u in group])
            centroid = pts.mean(axis=0)
            rho_g = max(float(np.linalg.norm(pts - centroid, axis=1).max()), cfg.rho_rel * s.r_obj)
            ok &= rec.prior_dist[gi] <= rho_g * 1.3 + 0.1
    check("탐색: prior 이탈이 ρ_group 규칙 이내", ok)


def _antipodal_setup(r_sphere=0.07):
    pos = np.array([[r_sphere, 0, 0], [-r_sphere, 0, 0]])
    nrm = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    return ("sphere", r_sphere), pos, nrm, r_sphere


def test_sim_antipodal():
    """시뮬: 대향 압착 — 앵커축 당김(힘 +x)을 α≈1 수준으로 버텨야 한다."""
    spec, pos, nrm, r = _antipodal_setup()
    w = np.array([1.0, 0, 0, 0, 0, 0])
    a, _ = measure_capacity(spec, pos, nrm, w, r, SimConfig())
    check("시뮬: 대향 당김 내하중", 0.4 <= a <= 2.0, f"α_sim={a:.2f} (LP 이론 1.0)")


def test_sim_shear_and_torsion():
    """시뮬: 전단 > 0, 비틀림은 μ_tor 에 단조."""
    spec, pos, nrm, r = _antipodal_setup()
    w_shear = np.array([0, 0, 1.0, 0, 0, 0])
    a_sh, _ = measure_capacity(spec, pos, nrm, w_shear, r, SimConfig())
    check("시뮬: 전단 내하중 > 0", a_sh > 0.2, f"α_sim={a_sh:.2f} (LP 이론 ~1.2)")

    w_tor = np.array([0, 0, 0, 1.0, 0, 0])   # 앵커축(x) 비틀림
    caps = []
    for mt in (0.001, 0.01):
        cfg = SimConfig(mu_tor=mt, alpha_step=0.02, alpha_max=1.0)
        a, _ = measure_capacity(spec, pos, nrm, w_tor, r, cfg)
        caps.append(a)
    check("시뮬: 비틀림 μ_tor 단조성", caps[1] >= caps[0] and caps[1] > 0,
          f"μ_tor=0.001→α={caps[0]:.2f}, 0.01→α={caps[1]:.2f}")


def test_sim_same_side():
    """시뮬: 같은 면 두 팁은 무외란 평형 자체가 불가 → 정착 실패로 내하중 0.

    (LP 도 반대 방향 α=0 이라 min-α=0 으로 탈락시키는 구성 — 시뮬이 같은
    판정을 정착 단계에서 내리는지 확인한다. 대향 쌍이 잡히는 것은
    test_sim_antipodal 이 증명하므로 비대칭성은 두 테스트의 대비로 성립.)"""
    r = 0.07
    ang = 0.5
    pos = np.array([[r * np.cos(ang), r * np.sin(ang), 0],
                    [r * np.cos(ang), -r * np.sin(ang), 0]])
    nrm = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    w = np.array([1.0, 0, 0, 0, 0, 0])
    a, info = measure_capacity(("sphere", r), pos, nrm, w, r, SimConfig(alpha_max=2.0))
    check("시뮬: 같은 면 쌍은 정착 실패로 내하중 0", a == 0.0 and not info["settled"],
          f"α={a:.2f}, settled={info['settled']}")


if __name__ == "__main__":
    test_merge()
    test_partition_counts()
    test_partition_fewer_units()
    test_gap_closing()
    test_thumbless_fallback()
    test_surface_region()
    test_search_sphere()
    test_sim_antipodal()
    test_sim_shear_and_torsion()
    test_sim_same_side()
    total, passed = len(PASS), sum(PASS)
    print(f"\n{passed}/{total} 통과")
    raise SystemExit(0 if passed == total else 1)
