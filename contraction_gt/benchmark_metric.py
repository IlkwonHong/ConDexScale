"""메트릭 모듈 처리량 벤치마크 — 계획서 4장의 시간 추정을 실측으로 교체하기 위한 도구.

실행: ./venv/bin/python -m contraction_gt.benchmark_metric
출력: N별 LP/s (GPU float64), 파일럿·전량 소요 외삽.
"""

import time

import torch

from contraction_gt.metric import ray_wrench_matrix, direction_set, alpha_batched


def bench(N, n_problems, device, reps=3):
    torch.manual_seed(0)
    dtype = torch.float64
    pos = torch.nn.functional.normalize(
        torch.randn(n_problems, N, 3, dtype=dtype, device=device), dim=-1)
    normal = torch.nn.functional.normalize(
        -pos + 0.3 * torch.randn_like(pos), dim=-1)
    com = torch.zeros(n_problems, 3, dtype=dtype, device=device)
    W = ray_wrench_matrix(pos, normal, com, 1.0, 0.6, 0.02)
    dirs = direction_set(pos)

    alpha_batched(W, dirs)  # warmup (컴파일·캐시)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.time()
        alpha_batched(W, dirs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    n_lp = n_problems * dirs.shape[1]
    best = min(times)
    return n_lp / best, best, n_lp


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    total_rate = {}
    for N in (2, 3, 4):
        n_problems = 30000
        rate, t, n_lp = bench(N, n_problems, device)
        total_rate[N] = rate
        print(f"N={N}: {n_lp:,}개 LP / {t:.2f}s = {rate:,.0f} LP/s")

    # 외삽: 파일럿(1만 grasp)과 1차 전량(150만 grasp, 파일당 1개)
    # grasp당 LP 수 = 파티션 ~10 × 반경 3종 × (조합 32+재샘플 16) × 14방향
    lp_per_grasp = 10 * 3 * 48 * 14
    mean_rate = sum(total_rate.values()) / 3
    for name, n_grasp in [("파일럿(1만, 스윕 포함 ~10배)", 10_000 * 10),
                          ("1차 전량(150만)", 1_500_000)]:
        sec = n_grasp * lp_per_grasp / mean_rate
        print(f"{name}: {n_grasp * lp_per_grasp / 1e9:.1f}G LP → {sec / 3600:.1f} 시간")
