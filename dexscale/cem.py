"""Cross-Entropy Method (대각 가우시안, ask/tell 인터페이스).

분포 = 차원별 (μ, σ). ask()로 후보 pop개 샘플, tell()로 elite 통계로 평활 갱신.
평가기(rollout)와 완전 분리 — CMA-ES 교체 시 이 파일만 바꾸면 됨 (설계문서 §1 부속).
"""
import numpy as np


class CEM:
    def __init__(self, mu0, sigma0, pop=64, n_elite=16, alpha_mu=0.7,
                 alpha_sigma=0.3, momentum=1.0, sigma_min=1e-3, lo=None, hi=None):
        """
        mu0, sigma0 : (D,) 초기 분포. mu0=0 이면 "사람 guidance 그대로"가 출발점.
        alpha_mu    : elite 평균 반영 비율
        alpha_sigma : elite 분산 반영 비율 — μ보다 느리게 (σ 조기붕괴 방지)
        momentum    : σ² 에 (μ 이동량)²을 더해 이동 중 탐색 유지
                      (60D toy: 기본값 없이 0.574, 있으면 0.027 — 2026-07-15 튜닝)
        lo, hi      : (D,) 물리적 범위 — 샘플 후 클립 (δ≥0, 접선 보정 한계 등)
        """
        self.mu = np.asarray(mu0, dtype=np.float64).copy()
        self.sigma = np.asarray(sigma0, dtype=np.float64).copy()
        self.pop, self.n_elite = pop, n_elite
        self.alpha_mu, self.alpha_sigma, self.momentum = alpha_mu, alpha_sigma, momentum
        self.sigma_min = sigma_min
        self.lo = None if lo is None else np.asarray(lo, dtype=np.float64)
        self.hi = None if hi is None else np.asarray(hi, dtype=np.float64)
        self.best_theta = self.mu.copy()
        self.best_cost = np.inf
        self.generation = 0

    def ask(self, rng):
        thetas = self.mu[None] + self.sigma[None] * rng.standard_normal((self.pop, self.mu.size))
        if self.lo is not None:
            thetas = np.clip(thetas, self.lo[None], self.hi[None])
        thetas[0] = self.mu  # 현재 평균도 항상 평가 (elitism·모니터링)
        return thetas

    def tell(self, thetas, costs):
        costs = np.asarray(costs, dtype=np.float64)
        # NaN 가드: 물리 발산 롤아웃은 큰 비용으로 치환 (2026-07-16 gen5 pop=nan 재발 방지)
        costs = np.where(np.isfinite(costs), costs, 1e6)
        order = np.argsort(costs)
        elite = thetas[order[: self.n_elite]]
        mu_new = (1 - self.alpha_mu) * self.mu + self.alpha_mu * elite.mean(0)
        var_e = elite.var(0) + self.momentum * (mu_new - self.mu) ** 2
        self.sigma = np.sqrt((1 - self.alpha_sigma) * self.sigma ** 2
                             + self.alpha_sigma * var_e)
        self.sigma = np.maximum(self.sigma, self.sigma_min)
        self.mu = mu_new
        if costs[order[0]] < self.best_cost:
            self.best_cost = float(costs[order[0]])
            self.best_theta = thetas[order[0]].copy()
        self.generation += 1
        return {"gen": self.generation, "best": self.best_cost,
                "elite_mean": float(costs[order[: self.n_elite]].mean()),
                "pop_mean": float(costs.mean()), "sigma_mean": float(self.sigma.mean())}
