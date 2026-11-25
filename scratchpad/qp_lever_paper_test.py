"""회전력 행 단위 통일(지렛대 환산) 종이 단위시험 — 수정 전/후 비교.

상황 재현: 박스(무게 10.7N)를 옆면 팁 2 + 윗면 팁 2로 쥔 채,
기울어짐이 시작되어 복원 회전력 0.6N·m(roll)이 함께 요구될 때
배분이 실제로 그 회전력을 만들어내는가.
실행: python3 scratchpad/qp_lever_paper_test.py
"""
import numpy as np

MU_QP = 0.5


def qp_forces(P, Nin, w_des, lever=None):
    cols = []
    for p, n in zip(P, Nin):
        t1 = np.cross(n, [0, 0, 1.0])
        nt1 = np.linalg.norm(t1)
        t1 = t1 / nt1 if nt1 > 1e-6 else np.array([1.0, 0, 0])
        t2 = np.cross(n, t1)
        for s in (t1, -t1, t2, -t2):
            e = (n + MU_QP * s) / np.linalg.norm(n + MU_QP * s)
            tq = np.cross(p, e) / (lever if lever else 1.0)
            cols.append(np.concatenate([e, tq]))
    A = np.array(cols).T
    w = w_des.copy()
    if lever:
        w = np.concatenate([w_des[:3], w_des[3:] / lever])
    x = np.zeros(len(cols))
    L = np.linalg.norm(A, 2) ** 2
    for _ in range(300):
        x = np.clip(x - (A.T @ (A @ x - w)) / L, 0, 10.0)
    f = np.array([np.array(cols[4 * i:4 * i + 4])[:, :3].T @ x[4 * i:4 * i + 4]
                  for i in range(len(P))])
    return f


# 무게중심 기준 앵커 위치·안쪽 법선 (박스 20×15×10cm 근사)
P = np.array([[+0.10, 0.00, 0.00],    # left_A  옆면(+x면, 안쪽 = −x)
              [-0.10, 0.00, 0.00],    # right_B 옆면(−x면, 안쪽 = +x)
              [0.00, +0.04, +0.05],   # left_B  윗면(안쪽 = −z)
              [0.00, -0.04, +0.05]])  # right_A 윗면(안쪽 = −z)
Nin = np.array([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, -1]], float)
names = ["left_A(옆)", "right_B(옆)", "left_B(윗)", "right_A(윗)"]

for w_des, tag in [
        (np.array([0, 0, 10.7, 0, 0, 0]), "순수 들기 (기존 검증 상황)"),
        (np.array([0, 0, 10.7, 0.6, 0, 0]), "들기 + 복원 roll 0.6N·m"),
]:
    print(f"\n=== 요구: {tag} ===")
    for lever, nm in [(None, "수정 전 (단위 불일치)"), (0.1, "수정 후 (지렛대 환산)")]:
        f = qp_forces(P, Nin, w_des, lever)
        F = f.sum(0)
        T = sum(np.cross(p, fi) for p, fi in zip(P, f))
        print(f"[{nm}] 합력 z={F[2]:+.2f}N  실현 roll={T[0]:+.3f}N·m "
              f"(요구 {w_des[3]:+.1f})")
        for k in range(4):
            print(f"    {names[k]}: ({f[k][0]:+5.2f},{f[k][1]:+5.2f},"
                  f"{f[k][2]:+5.2f})N")
