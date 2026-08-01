"""P4-2: ρ 영역 coarse-to-fine 탐색 — proxy 필터 → 배치 α(IPM) → 재샘플 정제.

계획서 D3 의 절차를 구현한다:
  그룹별 ρ 내 후보 K개 → 전 조합 proxy 채점 → 상위 C 조합만 14방향 α →
  최고점 주변 재샘플 1회 (최고 후보를 포함시켜 coarse 결과보다 나빠지지 않게).

배치 구조: 한 grasp 의 (파티션 × 패드 반경) 전 작업을 앵커 수 N 별로 묶어
alpha_batched 한 번에 넣는다. grasp 간 배칭은 파이프라인 러너(P6)의 몫.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .metric import ray_wrench_matrix, direction_set, alpha_batched
from .surface import SurfaceSamples, candidates_in_region


@dataclass
class SearchConfig:
    rho_rel: float = 0.15           # 탐색 반경 c (파일럿 검사 1로 확정: N=2 손해 0,
                                    # N≥3 은 충실성-품질 절충의 사용자 결정)
    pad_radius_rel: tuple = (0.05, 0.10, 0.20)   # N=2 전용 라벨 차원 (반경이 성패를 가름)
    pad_radius_rel_multi: tuple = (0.10,)        # N≥3 은 반경 무의미(파일럿 항목 2) — 1종
    mu_t: float = 0.6
    hertz_frac: float = 0.5         # 유효 접촉반경 = hertz_frac × 패드 반경 (실기 캘리브레이션 항목)
    k_candidates: int = 32          # 그룹당 후보 수 (N=4 는 아래 값)
    k_candidates_n4: int = 12
    top_combos: int = 128           # N≥3 proxy 통과 폭 (파일럿 검사 2: 128에서 손실 중앙값 0)
    top_combos_n2: int = 32         # N=2 는 32로 손실 0
    random_combos: int = 64         # proxy 가 헛짚는 꼬리를 위한 무작위 보험
    refine_k: int = 10              # 재샘플 후보 수
    refine_shrink: float = 3.0      # 재샘플 반경 = ρ / refine_shrink
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


def mu_tor_from_pad(pad_radius, mu_t, hertz_frac=0.5):
    """구형 소프트 팁의 비틀림 모멘트 팔(길이 단위) 근사 (D4).

    명목 그립력에서의 헤르츠 접촉반경 a ≈ hertz_frac × 패드 반경으로 두고
    균일 압력 원판 공식 τ_max = (2/3)·μ_t·f_n·a 를 쓴다. hertz_frac 은
    파일럿에서 시뮬과 대조해 캘리브레이션한다. 곡률 보정은 v1 미구현.
    """
    return (2.0 / 3.0) * mu_t * hertz_frac * pad_radius


@dataclass
class SearchRecord:
    partition: list               # 유닛 그룹 리스트, [0]이 엄지(또는 폴백) 그룹
    pad_radius_rel: float
    n_anchors: int
    anchor_idx: np.ndarray        # 표면 샘플 인덱스 [N]
    anchor_pos: np.ndarray        # [N, 3] (물리 좌표)
    anchor_normal: np.ndarray     # [N, 3]
    alpha: np.ndarray             # [14]
    min_alpha: float
    weak_dir: int                 # argmin 방향 인덱스 (시뮬 검증 대상 d*)
    prior_dist: np.ndarray        # [N] 그룹 centroid(표면 투영)로부터의 거리
    mu_tor: float
    stage: str = "refine"         # coarse/refine 중 최종 채택 단계


def _proxy_wrench(samples: SurfaceSamples, idx):
    """후보점들의 단위 법선 wrench [K, 6] (proxy 용, 정규화 좌표)."""
    p = (samples.points[idx] - samples.com) / samples.r_obj
    n = samples.normals[idx]
    return np.concatenate([n, np.cross(p, n)], axis=1)


def _feasible_candidates(samples: SurfaceSamples, idx, tip_r, gap=0.002):
    """팁 구가 물체의 다른 부위와 간섭하지 않는 후보만 남긴다 (오목 물체 대비).

    팁 중심 = 접촉점 - 안쪽법선 × (tip_r + gap). 표면 샘플 중 팁 반경의 95%
    이내로 파고드는 점이 있으면 그 후보는 물리적으로 놓을 수 없다.
    """
    centers = samples.points[idx] - samples.normals[idx] * (tip_r + gap)
    d = np.linalg.norm(samples.points[None, :, :] - centers[:, None, :], axis=-1)
    return idx[d.min(axis=1) >= tip_r * 0.95]


def _top_combos(cand_sets, samples, top_c, min_sep=0.0, n_random=0, seed=0):
    """전 조합 proxy 채점 → 상위 top_c 조합의 인덱스 배열 [C, N].

    proxy = ‖Σ_i (단위 법선 wrench)‖ — 대립(opposition)이 잘 될수록 낮다.
    min_sep > 0 이면 앵커 쌍 간격이 그 미만인 조합을 제외한다 (팁 겹침 방지 —
    탐색은 앵커를 점으로 다루므로 팁의 물리적 크기는 여기서 강제해야 한다).
    """
    ws = [_proxy_wrench(samples, idx) for idx in cand_sets]
    acc = ws[0]                                   # [K0, 6]
    for w in ws[1:]:
        acc = acc[..., None, :] + w               # [..., Ki, 6] 누적 브로드캐스트
    score = np.linalg.norm(acc, axis=-1)          # [K0, K1, ...]

    if min_sep > 0.0:
        n = len(cand_sets)
        for i in range(n):
            for j in range(i + 1, n):
                pi = samples.points[cand_sets[i]]
                pj = samples.points[cand_sets[j]]
                dij = np.linalg.norm(pi[:, None, :] - pj[None, :, :], axis=-1)
                shape = [1] * n
                shape[i], shape[j] = len(cand_sets[i]), len(cand_sets[j])
                score = np.where(dij.reshape(shape) < min_sep, np.inf, score)

    flat = score.reshape(-1)
    finite = np.isfinite(flat)
    if not finite.any():
        return np.zeros((0, len(cand_sets)), dtype=np.int64)
    order = np.argsort(flat)
    top = order[finite[order]][: min(top_c, int(finite.sum()))]
    if n_random > 0 and int(finite.sum()) > len(top):
        rng = np.random.default_rng(seed)
        pool = np.setdiff1d(np.nonzero(finite)[0], top)
        if len(pool):
            top = np.concatenate([top, rng.choice(pool, min(n_random, len(pool)),
                                                  replace=False)])
    combos = np.stack(np.unravel_index(top, score.shape), axis=1)  # [C, N] (후보셋 내 위치)
    return np.stack([cand_sets[i][combos[:, i]] for i in range(len(cand_sets))], axis=1)


def _batched_alpha(samples, combo_idx, mu_t, mu_tor_per_job, device):
    """combo_idx [J, N] (표면 샘플 인덱스) -> alpha [J, 14] (numpy)."""
    dtype = torch.float64
    pos = torch.as_tensor(
        (samples.points[combo_idx] - samples.com) / samples.r_obj, dtype=dtype, device=device)
    nrm = torch.as_tensor(samples.normals[combo_idx], dtype=dtype, device=device)
    J, N = combo_idx.shape
    com = torch.zeros(J, 3, dtype=dtype, device=device)
    mu_tor = torch.as_tensor(mu_tor_per_job, dtype=dtype, device=device)
    mu_tor = mu_tor.unsqueeze(-1).expand(J, N) / samples.r_obj   # 길이 → 정규화 좌표
    W = ray_wrench_matrix(pos, nrm, com, 1.0, mu_t, mu_tor)
    dirs = direction_set(pos)
    return alpha_batched(W, dirs).cpu().numpy()


def search_grasp(samples: SurfaceSamples, patches, partitions, cfg: SearchConfig):
    """한 grasp 의 전 (파티션 × 패드 반경) 탐색 -> SearchRecord 리스트.

    partitions: units.enumerate_partitions 출력들의 리스트 (전 N 합쳐서).
    """
    records = []
    # ---- coarse 단계 작업 수집 (N 별로 묶어 LP 배칭) ----
    jobs = []   # (partition, pad_rel, cand_sets, centers, combos[C,N])
    for part in partitions:
        n_anchors = len(part)
        k_cand = cfg.k_candidates_n4 if n_anchors >= 4 else cfg.k_candidates
        cand_sets, centers = [], []
        for group in part:
            pts = np.concatenate([patches[u].points for u in group])
            areas = np.concatenate([patches[u].areas for u in group])
            w = areas / areas.sum() if areas.sum() > 0 else None
            centroid = np.average(pts, axis=0, weights=w)
            nrm = np.average(np.concatenate([patches[u].normals for u in group]),
                             axis=0, weights=w)
            patch_r = float(np.linalg.norm(pts - centroid, axis=1).max())
            rho = max(patch_r, cfg.rho_rel * samples.r_obj)
            idx = candidates_in_region(samples, centroid, rho, ref_normal=nrm,
                                       k_max=k_cand, seed=cfg.seed)
            cand_sets.append(idx)
            centers.append(samples.points[idx[np.linalg.norm(
                samples.points[idx] - centroid, axis=1).argmin()]])
        # 패드 반경마다 실현 가능성이 다르므로 (간섭·간격) 조합도 반경별로 만든다
        pads = cfg.pad_radius_rel if n_anchors == 2 else cfg.pad_radius_rel_multi
        width = cfg.top_combos_n2 if n_anchors == 2 else cfg.top_combos
        for pad_rel in pads:
            tip_r = max(pad_rel * samples.r_obj, 0.002)
            fsets = [_feasible_candidates(samples, idx, tip_r) for idx in cand_sets]
            if any(len(s) == 0 for s in fsets):
                continue          # 이 반경에서는 배치 불가 — 정직한 탈락
            combos = _top_combos(fsets, samples, width, min_sep=2.4 * tip_r,
                                 n_random=cfg.random_combos, seed=cfg.seed)
            if len(combos) == 0:
                continue
            jobs.append((part, pad_rel, fsets, np.asarray(centers), combos))

    for n_anchors in sorted({len(j[0]) for j in jobs}):
        group_jobs = [j for j in jobs if len(j[0]) == n_anchors]
        combos_all = np.concatenate([j[4] for j in group_jobs])          # [ΣC, N]
        mu_tor_all = np.concatenate([
            np.full(len(j[4]), mu_tor_from_pad(j[1] * samples.r_obj, cfg.mu_t, cfg.hertz_frac))
            for j in group_jobs])
        alpha = _batched_alpha(samples, combos_all, cfg.mu_t, mu_tor_all, cfg.device)
        min_a = alpha.min(axis=1)

        # ---- 작업별 best 선택 후 refine ----
        ofs = 0
        for part, pad_rel, cand_sets, centers, combos in group_jobs:
            C = len(combos)
            a_job, ma_job = alpha[ofs : ofs + C], min_a[ofs : ofs + C]
            ofs += C
            best = int(ma_job.argmax())
            best_combo, best_alpha = combos[best], a_job[best]
            mu_tor = mu_tor_from_pad(pad_rel * samples.r_obj, cfg.mu_t, cfg.hertz_frac)

            # refine: best 지점 주변 미세 재샘플 (best 후보를 포함시켜 비퇴행 보장)
            tip_r = max(pad_rel * samples.r_obj, 0.002)
            ref_sets = []
            for gi in range(n_anchors):
                p_best = samples.points[best_combo[gi]]
                rho_ref = max(cfg.rho_rel * samples.r_obj / cfg.refine_shrink, 1e-6)
                idx = candidates_in_region(samples, p_best, rho_ref,
                                           ref_normal=samples.normals[best_combo[gi]],
                                           k_max=cfg.refine_k, seed=cfg.seed + 1)
                idx = _feasible_candidates(samples, idx, tip_r)
                if best_combo[gi] not in idx:
                    idx = np.concatenate([[best_combo[gi]], idx])
                ref_sets.append(idx)
            ref_combos = _top_combos(ref_sets, samples, cfg.top_combos,
                                     min_sep=2.4 * tip_r)
            if len(ref_combos) == 0:
                ref_combos = best_combo[None, :]
            ref_alpha = _batched_alpha(samples, ref_combos, cfg.mu_t,
                                       np.full(len(ref_combos), mu_tor), cfg.device)
            ref_min = ref_alpha.min(axis=1)
            rbest = int(ref_min.argmax())

            if ref_min[rbest] >= ma_job[best]:
                final_combo, final_alpha, stage = ref_combos[rbest], ref_alpha[rbest], "refine"
            else:
                final_combo, final_alpha, stage = best_combo, best_alpha, "coarse"

            records.append(SearchRecord(
                partition=part, pad_radius_rel=pad_rel, n_anchors=n_anchors,
                anchor_idx=final_combo,
                anchor_pos=samples.points[final_combo].copy(),
                anchor_normal=samples.normals[final_combo].copy(),
                alpha=final_alpha, min_alpha=float(final_alpha.min()),
                weak_dir=int(final_alpha.argmin()),
                prior_dist=np.linalg.norm(samples.points[final_combo] - centers, axis=1),
                mu_tor=mu_tor, stage=stage,
            ))
    return records
