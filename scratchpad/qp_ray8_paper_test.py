"""8모서리 내접 원뿔 배분 종이 시험.

검사: ① 순수 들기 — 합력 z가 무게 10.7N 을 채우는가
      ② 들기 + 요(yaw) 0.6N·m — 팁별 접선/법선 비 ≤ μ(0.5) 인가
        (4모서리판 결함: 대각선에서 0.67 까지 나왔음)
실행: ./venv/bin/python scratchpad/qp_ray8_paper_test.py
"""
import numpy as np

MU_QP, LEVER, N_RAY = 0.5, 0.1, 8
MU_RAY = MU_QP * np.cos(np.pi / N_RAY)


def qp_forces(P, Nin, w_des):
    cols = []
    for p, n in zip(P, Nin):
        t1 = np.cross(n, [0, 0, 1.0])
        nt1 = np.linalg.norm(t1)
        t1 = t1 / nt1 if nt1 > 1e-6 else np.array([1.0, 0, 0])
        t2 = np.cross(n, t1)
        for k in range(N_RAY):
            a = 2 * np.pi * k / N_RAY
            s = np.cos(a) * t1 + np.sin(a) * t2
            e = (n + MU_RAY * s) / np.linalg.norm(n + MU_RAY * s)
            cols.append(np.concatenate([e, np.cross(p, e) / LEVER]))
    A = np.array(cols).T
    w = np.concatenate([w_des[:3], w_des[3:] / LEVER])
    x = np.zeros(len(cols))
    L = np.linalg.norm(A, 2) ** 2
    for _ in range(300):
        x = np.clip(x - (A.T @ (A @ x - w)) / L, 0, 15.0)
    return np.array([np.array(cols[N_RAY * i:N_RAY * i + N_RAY])[:, :3].T
                     @ x[N_RAY * i:N_RAY * i + N_RAY] for i in range(len(P))])


P = np.array([[+0.10, 0.00, 0.00], [-0.10, 0.00, 0.00],
              [0.00, +0.04, +0.05], [0.00, -0.04, +0.05]])
Nin = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, -1]], float)
names = ["left_A(옆)", "right_B(옆)", "left_B(윗)", "right_A(윗)"]

for w_des, tag in [(np.array([0, 0, 10.7, 0, 0, 0]), "순수 들기"),
                   (np.array([0, 0, 10.7, 0, 0, 0.6]), "들기 + 요 0.6N·m"),
                   (np.array([0, 0, 10.7, 0.6, 0, 0]), "들기 + 롤 0.6N·m")]:
    f = qp_forces(P, Nin, w_des)
    F = f.sum(0)
    T = sum(np.cross(p, fi) for p, fi in zip(P, f))
    print(f"\n=== {tag} ===  합력z {F[2]:+.2f}N  "
          f"실현토크 ({T[0]:+.2f},{T[1]:+.2f},{T[2]:+.2f})N·m")
    for k in range(4):
        fn = float(f[k] @ Nin[k])
        ft = np.linalg.norm(f[k] - fn * Nin[k])
        r = ft / fn if fn > 0.5 else 0.0
        print(f"  {names[k]}: 법선 {fn:5.2f}N 접선 {ft:5.2f}N "
              f"비 {r:.3f} {'✗원뿔 초과' if r > MU_QP + 1e-6 else '✓'}")
