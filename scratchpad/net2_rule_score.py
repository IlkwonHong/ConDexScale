"""손으로 짠 net2 대체 채점 규칙 (2026-11-07, 블로그 'Weight-Space Programming' (가) 실험).

net2 가 하던 일 = 팔(그룹)마다 후보 범위 안 표면점에 점수를 매기고 상위 8점을 고르는 것.
이 파일은 그 일을 학습 없이 물리 규칙으로 한다. 뒤 과정(8^3 조합 α, 다듬기, 시뮬)은 그대로.

좌표: x = (점 − 무게중심) / r_obj,  n = 안쪽 법선(접촉힘이 물체를 미는 방향, 파이프라인 규약).
그룹 g 의 대표점 rep_g = 후보 범위 점들의 평균 위치에 가장 가까운 범위 안 점 (사람 접촉 중심의 대용).
그룹 g 의 대표 방향 n̄_g = rep_g 둘레 FLAT_RADIUS 안 법선 평균 (정규화).

항 네 개 (모두 0~1, 클수록 좋음):
  ① prox   사람 손에서 너무 멀지 않기   = 1 − |x_p − x_rep_g| / (범위 안 최대 거리)
  ② oppose 다른 팔들과 마주 누르기      = ½ (1 − n_p · mean_{j≠g} n̄_j)
            (다른 팔들이 미는 방향의 반대로 밀수록 1, 다른 팔들이 서로 상쇄하면 모두 ½)
  ③ cover  못 막는 방향 채우기
            점마다 마찰 각뿔 극선 6개의 렌치(힘 3 + 돌림힘 3)를 만들고, 방향 표본 d (28개) 마다
            "외란 d 를 막으려면 −d 를 내야 한다" → 그 점이 낼 수 있는 양 h_p(d) = max(0, max_k (−d)·극선_k).
            팔마다 힘 상한 1 이므로 여러 팔의 합 능력은 h 의 합 (민코프스키 합의 지지 함수).
            raw(p) = min_d [ h_p(d) + Σ_{j≠g} h_rep_j(d) ]  → 범위 안 최소·최대로 0~1 정규화.
  ④ flat   몇 mm 어긋나도 버티기       = 1 − (FLAT_RADIUS 안 이웃 법선과의 최대 각도) / 90°
점수 = Σ w·항.  ⑤ 고르기 = 1등부터, 이미 고른 점에서 EXCL_RADIUS 안은 건너뛰며 8점 (모자라면 반경 절반).

가중치는 검증 결과를 보기 전에 물리 판단으로 고정했다 (W_DEFAULT). 검증 파지로 맞추지 않는다.
"""
import numpy as np
import torch

from contraction_gt.metric import ray_wrench_matrix
from contraction_net.stage2_cond_model import wrench_dirs

MU_T = 0.6
W_DEFAULT = dict(prox=0.25, oppose=0.5, cover=1.0, flat=0.5)
TERMS = ("prox", "oppose", "cover", "flat")
FLAT_RADIUS = 0.15     # r_obj 배 — 팁 반지름(0.10) 규모, 이웃이 몇 개는 들어오도록 한 칸(0.078) 의 2배
EXCL_RADIUS = 0.10     # r_obj 배 — 팁 반지름. 이보다 가까운 두 점은 사실상 같은 자리
K_PICK = 8


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def geometry(pts, nrm, com, r_obj):
    """정규화 위치 x [P,3], 단위 안쪽 법선 n [P,3], 점끼리 거리 dist [P,P]."""
    x = (np.asarray(pts, np.float64) - np.asarray(com, np.float64)) / r_obj
    n = unit(np.asarray(nrm, np.float64))
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return x, n, dist


def support_table(x, n, mu_tor_n):
    """H [P, 28]: 점 p 하나(힘 상한 1)가 외란 방향 d 를 막는 데 낼 수 있는 양 h_p(d)."""
    W = ray_wrench_matrix(torch.from_numpy(x)[None], torch.from_numpy(n)[None],
                          torch.zeros(1, 3, dtype=torch.float64), 1.0, MU_T, mu_tor_n)[0].numpy()  # [P,6w,6k]
    D = wrench_dirs(torch.float64, "cpu").numpy()                                                  # [28,6]
    proj = np.einsum("dw,pwk->pdk", -D, W)                                                         # [P,28,6]
    return np.maximum(proj.max(-1), 0.0)


def flatness(n, dist, radius=FLAT_RADIUS):
    """f [P]: 1 − (반경 안 이웃 법선과의 최대 각도)/90°, 0~1."""
    cos = np.clip(n @ n.T, -1.0, 1.0)
    ang = np.where(dist <= radius, np.arccos(cos), 0.0)
    return np.clip(1.0 - ang.max(1) / (np.pi / 2), 0.0, 1.0)


def group_reps(x, n, dist, masks):
    """그룹마다 대표점 번호와 대표 방향."""
    reps, nbar = [], []
    for mk in masks:
        idx = np.nonzero(mk)[0]
        c = x[idx].mean(0)
        r = int(idx[np.argmin(np.linalg.norm(x[idx] - c, axis=1))])
        reps.append(r)
        nbar.append(unit(n[dist[r] <= FLAT_RADIUS].mean(0)))
    return np.array(reps), np.array(nbar)


def _minmax_in_mask(v, mk):
    out = np.full_like(v, np.nan)
    a, b = v[mk].min(), v[mk].max()
    out[mk] = 0.5 if b - a < 1e-9 else (v[mk] - a) / (b - a)
    return out


def rule_terms(x, n, dist, masks, mu_tor_n, cover_raw_override=None):
    """항 네 개 {이름: [G, P]} (범위 밖은 nan) + 진단용 원값.

    cover_raw_override: [G, P] 가 주어지면 ③ 의 원값을 그것으로 바꾼다 (선형계획 α 판 비교용).
    """
    G, P = masks.shape
    reps, nbar = group_reps(x, n, dist, masks)
    H = support_table(x, n, mu_tor_n)
    flat = flatness(n, dist)
    out = {k: np.full((G, P), np.nan) for k in TERMS}
    cover_raw = np.full((G, P), np.nan)
    for g in range(G):
        mk = masks[g]
        others = [j for j in range(G) if j != g]
        # ① 사람 손에서 너무 멀지 않기
        d = np.linalg.norm(x - x[reps[g]], axis=1)
        out["prox"][g, mk] = 1.0 - d[mk] / max(d[mk].max(), 1e-9)
        # ② 다른 팔들과 마주 누르기
        s = nbar[others].mean(0)
        out["oppose"][g, mk] = 0.5 * (1.0 - n[mk] @ s)
        # ③ 못 막는 방향 채우기
        if cover_raw_override is None:
            h_oth = H[reps[others]].sum(0)                                   # [28]
            cover_raw[g, mk] = (H[mk] + h_oth[None]).min(1)
        else:
            cover_raw[g, mk] = cover_raw_override[g, mk]
        out["cover"][g] = _minmax_in_mask(cover_raw[g], mk)
        # ④ 몇 mm 어긋나도 버티기
        out["flat"][g, mk] = flat[mk]
    return out, dict(reps=reps, nbar=nbar, cover_raw=cover_raw, H=H, flat=flat)


def combine(terms, masks, weights=W_DEFAULT):
    """점수 [G, P], 범위 밖 −inf."""
    sc = sum(weights[k] * np.nan_to_num(terms[k]) for k in TERMS if weights.get(k, 0.0) != 0.0)
    if np.isscalar(sc):
        sc = np.zeros(masks.shape)
    return np.where(masks, sc, -np.inf)


def select_topk(score_g, k=K_PICK):
    m = int(np.isfinite(score_g).sum())
    return np.argsort(-score_g, kind="stable")[:min(k, m)]


def select_diverse(score_g, dist, k=K_PICK, excl=EXCL_RADIUS):
    """⑤ 1등부터, 이미 고른 점에서 excl 안은 건너뛰며 k점. 모자라면 반경을 절반씩 줄여 채운다."""
    m = int(np.isfinite(score_g).sum())
    k = min(k, m)
    order = np.argsort(-score_g, kind="stable")[:m]
    chosen = []
    r = excl
    while len(chosen) < k and r > 1e-4:
        for i in order:
            if len(chosen) >= k:
                break
            if i in chosen:
                continue
            if all(dist[i, c] >= r for c in chosen):
                chosen.append(int(i))
        r *= 0.5
    for i in order:                      # 마지막 안전망
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(int(i))
    return np.array(chosen, np.int64)
