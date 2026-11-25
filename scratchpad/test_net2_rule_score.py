"""손 규칙 항별 검산 — 정답을 손으로 아는 정육면체에서 (2026-11-07).

정육면체 반변 1, 무게중심 0, r_obj = √3. 면마다 18×18 격자점, 안쪽 법선.
③ 은 우리 α 선형계획(_batched_alpha, 권위 있는 잣대)과 순서가 맞는지 대조한다.
실행: ./venv/bin/python scratchpad/test_net2_rule_score.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scratchpad"))

from net2_rule_score import (geometry, support_table, flatness, rule_terms, select_diverse,  # noqa: E402
                             combine, W_DEFAULT)
from contraction_gt.search import _batched_alpha, mu_tor_from_pad                          # noqa: E402
from contraction_gt.surface import SurfaceSamples                                           # noqa: E402

R_OBJ = np.sqrt(3.0)
MU_TOR_N = mu_tor_from_pad(0.10 * R_OBJ, 0.6) / R_OBJ


def cube(nper=18):
    g = (np.arange(nper) + 0.5) / nper * 2 - 1
    u, v = np.meshgrid(g, g, indexing="ij")
    u, v = u.ravel(), v.ravel()
    one = np.ones_like(u)
    pts, nrm = [], []
    for ax in range(3):
        for s in (1.0, -1.0):
            p = np.zeros((len(u), 3))
            p[:, ax] = s * one
            o = [a for a in range(3) if a != ax]
            p[:, o[0]], p[:, o[1]] = u, v
            nn = np.zeros_like(p)
            nn[:, ax] = -s
            pts.append(p)
            nrm.append(nn)
    return np.concatenate(pts), np.concatenate(nrm)


def nearest(pts, q):
    return int(np.argmin(np.linalg.norm(pts - np.asarray(q, float), axis=1)))


def patch(pts, nrm, center, face_normal, rad):
    c = np.asarray(center, float)
    return (np.linalg.norm(pts - c, axis=1) <= rad) & (nrm @ np.asarray(face_normal, float) > 0.99)


def main():
    pts, nrm = cube()
    x, n, dist = geometry(pts, nrm, np.zeros(3), R_OBJ)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'통과' if cond else '실패'}] {name} {detail}")

    # 0. 단일 점 지지량 h — 손 계산: 윗면 중심(안쪽 법선 −z)
    H = support_table(x, n, MU_TOR_N)
    i = nearest(pts, [0, 0, 1])
    # wrench_dirs 순서: 힘 +x,+y,+z,−x,−y,−z (0..5)
    check("h: 윗면 점이 +z 외란을 막는 양 = 1 (아래로 누름)", abs(H[i, 2] - 1.0) < 1e-6, f"{H[i, 2]:.4f}")
    check("h: 윗면 점이 −z 외란(당김)을 막는 양 = 0", abs(H[i, 5]) < 1e-9, f"{H[i, 5]:.4f}")
    check("h: 윗면 점이 +x 외란을 막는 양 = 마찰 피라미드 0.6/√2~0.6",
          0.6 / np.sqrt(2) - 1e-6 <= H[i, 0] <= 0.6 + 1e-6, f"{H[i, 0]:.4f}")

    # ④ 평탄도
    f = flatness(n, dist)
    i_c, i_e = nearest(pts, [0, 0, 1]), nearest(pts, [0.95, 0, 1])
    check("④ 면 가운데 평탄도 = 1", abs(f[i_c] - 1.0) < 1e-9, f"{f[i_c]:.3f}")
    check("④ 모서리 옆 평탄도 = 0 (반경 안에 90° 다른 면)", f[i_e] < 1e-9, f"{f[i_e]:.3f}")

    # ①② — 그룹0: +x 면 조각 ∪ +z 면 조각,  그룹1: −x 면 가운데,  그룹2: +y 면 가운데
    m0 = patch(pts, nrm, [1, 0, 0], [-1, 0, 0], 0.5) | patch(pts, nrm, [0, 0, 1], [0, 0, -1], 0.5)
    m1 = patch(pts, nrm, [-1, 0, 0], [1, 0, 0], 0.5)
    m2 = patch(pts, nrm, [0, 1, 0], [0, -1, 0], 0.5)
    masks = np.stack([m0, m1, m2])
    T, info = rule_terms(x, n, dist, masks, MU_TOR_N)
    check("① 대표점의 가까움 = 1", abs(T["prox"][1, info["reps"][1]] - 1.0) < 1e-9)
    ix, iz = nearest(pts, [1, 0, 0]), nearest(pts, [0, 0, 1])
    # 다른 팔 방향 평균 s = (+x + −y)/2 → +x 면 점(n=−x): ½(1+½)=0.75, +z 면 점(n=−z): 0.5
    check("② +x 면 점 마주 누름 = 0.75", abs(T["oppose"][0, ix] - 0.75) < 1e-6, f"{T['oppose'][0, ix]:.4f}")
    check("② +z 면 점 마주 누름 = 0.5", abs(T["oppose"][0, iz] - 0.5) < 1e-6, f"{T['oppose'][0, iz]:.4f}")

    # ③ — 그룹1·2 를 ±x 면 가운데(서로 마주 봄)에 두고, 그룹0 후보 = +z 면 ∪ +y 면 ∪ +x 면 조각
    g0 = (patch(pts, nrm, [0, 0, 1], [0, 0, -1], 0.6) | patch(pts, nrm, [0, 1, 0], [0, -1, 0], 0.6)
          | patch(pts, nrm, [1, 0, 0], [-1, 0, 0], 0.6))
    g1 = patch(pts, nrm, [-1, 0, 0], [1, 0, 0], 0.3)
    g2 = patch(pts, nrm, [1, 0, 0], [-1, 0, 0], 0.3)
    masks3 = np.stack([g0, g1, g2])
    T3, info3 = rule_terms(x, n, dist, masks3, MU_TOR_N)
    samples = SurfaceSamples(points=pts, normals=nrm, r_obj=R_OBJ, com=np.zeros(3))
    cand = np.nonzero(g0)[0]
    combos = np.stack([cand, np.full_like(cand, info3["reps"][1]), np.full_like(cand, info3["reps"][2])], 1)
    a = _batched_alpha(samples, combos.astype(np.int64), 0.6,
                       np.full(len(combos), MU_TOR_N * R_OBJ), "cuda").min(1)
    raw = info3["cover_raw"][0, cand]
    i_dup, i_top = nearest(pts[cand], [1, 0.3, 0.3]), nearest(pts[cand], [0, 0.3, 1])
    print(f"    +x 면(이미 있는 팔과 같은 면) 원값 {raw[i_dup]:.3f} α {a[i_dup]:.3f} / "
          f"+z 면 원값 {raw[i_top]:.3f} α {a[i_top]:.3f}")
    check("③ 같은 면 중복보다 새 면이 원값·α 모두 높음",
          raw[i_dup] < raw[i_top] and a[i_dup] < a[i_top])
    rk = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rk(raw), rk(a))[0, 1]
    print(f"    ③ 원값과 선형계획 α 의 순위 상관 (후보 {len(cand)}점): {rho:.2f}")
    check("③ 원값이 α 와 같은 방향 (순위 상관 > 0.5)", rho > 0.5)

    # ⑤ 다양한 고르기
    top = np.where(patch(pts, nrm, [0, 0, 1], [0, 0, -1], 2.0), 1.0, -np.inf)
    top[nearest(pts, [0, 0, 1])] = 2.0                  # 한 점 근처에 점수 봉우리
    for j in np.nonzero(np.linalg.norm(pts - [0, 0, 1], axis=1) < 0.3)[0]:
        top[j] = max(top[j], 1.5)
    sel = select_diverse(top, dist)
    dmin = min(dist[a_, b_] for a_ in sel for b_ in sel if a_ != b_)
    check("⑤ 8점, 서로 0.10·r 이상 떨어짐", len(sel) == 8 and dmin >= 0.10 - 1e-9, f"최소 거리 {dmin:.3f}")
    small = np.full(len(pts), -np.inf)
    small[np.argsort(np.linalg.norm(pts - [0, 0, 1], axis=1))[:5]] = 1.0
    check("⑤ 범위가 5점뿐이면 5점", len(select_diverse(small, dist)) == 5)

    sc = combine(T, masks, W_DEFAULT)
    check("점수: 범위 밖은 −inf, 안은 유한", np.isinf(sc[~masks]).all() and np.isfinite(sc[masks]).all())
    print("\n전체:", "통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
