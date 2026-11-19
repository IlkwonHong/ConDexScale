"""ℓ∞ 배분 여유 힘 배분기 — 영공간 분해 + 스프링 반응 (설계: md_claude/plans/rho_continuous_control_v1.md §3).

문제 (매 틱):
    팁 힘 f = f₀ + N·λ          f₀ = G⁺ w_req (최소 노름 특수해), N = G 영공간의 직교 정규 기저
    min ‖λ‖²                    (맞누르는 힘 전체 크기 최소 = 총 힘 최소, f₀ ⊥ 영공간)
    s.t. 각 팁 i, 각뿔 면 k:   h_ik · f_i  ≥  r · ‖D M_iᵀ h_ik‖₁          (배분 여유)
         각 팁 i:              n_i · f_i   ≤  cap_i − r · ‖D M_iᵀ n_i‖₁     (상한, 외란 포함)
         r = r_req (1단계) / 0 ≤ r ≤ r_req 최대화 (2단계) / 잔차 최소화 (3단계 폴백)

배분 여유 조건의 유래: 외란 박스의 꼭짓점 v ∈ {−1,+1}^6 마다 "f_i + M_i(r·D·v) 가 각뿔 안"
을 요구하면, 면 k 하나에 대해 h·f_i + r·(h·M_i D v) ≥ 0 이 모든 v 에서 성립해야 하고,
{±1}^6 위에서 선형식의 최솟값은 −‖계수‖₁ 이므로 64개 조건이 h·f_i ≥ r·‖D M_iᵀ h‖₁ 하나로
정확히(근사 없이) 합쳐진다. 따라서 제약 수는 64×A×9 가 아니라 9A 개다.

스프링 반응 M_i = −G_iᵀ (Σ_j G_j G_jᵀ)⁻¹ : 외란 δw 가 들어와 배분기가 반응하기 전에, 팁들이
같은 강성의 등방 스프링으로 물체와 함께 밀리며 겪는 힘 변화 Δf_i = M_i δw (강성 값은 상쇄).
전제 = 외란 동안 미끄러짐 없음. 접촉 상실(누르는 힘 < 0)은 각뿔 면 조건이 잡는다.

단위: r 은 ρ 와 같은 정규화(힘 f_cap, 토크 f_cap·r_obj). D = diag(f_cap×3, f_cap·r_obj×3).

부가 기능:
    margin_of(f)  — 임의의 힘 상태(예: 시뮬 실측 팁 힘)의 배분 여유 r 을 닫힌식으로 계산.
    reference_*   — scipy(HiGHS/SLSQP) 참조 풀이 (단위시험용).
"""
import numpy as np
import scipy.sparse as sp

try:
    import osqp
except ImportError:                      # 참조 풀이만 쓰는 환경
    osqp = None


# ------------------------------------------------------------------ 기하
def tangent_frame(n):
    h = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    t1 = np.cross(h, n)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def pyramid(n, mu, n_edge=8):
    """안쪽 법선 n 둘레의 내접 각뿔. 반환 (E [n_edge,3] 모서리 방향, H [n_edge,3] 면 안쪽 법선).

    f 가 각뿔 안 ⇔ H @ f ≥ 0 (모든 면). 모서리 방향과 접선 프레임은 static_hold_test /
    play_contact_mj 의 8광선 규약과 동일 (μ_r = μ cos(π/n_edge) 내접).
    """
    n = np.asarray(n, float) / np.linalg.norm(n)
    mu_r = mu * np.cos(np.pi / n_edge)
    t1, t2 = tangent_frame(n)
    E = []
    for k in range(n_edge):
        a = 2 * np.pi * k / n_edge
        e = n + mu_r * (np.cos(a) * t1 + np.sin(a) * t2)
        E.append(e / np.linalg.norm(e))
    E = np.array(E)
    H = []
    for k in range(n_edge):
        h = np.cross(E[k], E[(k + 1) % n_edge])
        h /= np.linalg.norm(h)
        if h @ n < 0:
            h = -h
        H.append(h)
    return E, np.array(H)


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def grasp_blocks(P_rel):
    """G_i = [I; [r_i]×] (6×3) 목록 — 힘 → (힘, 질량중심 기준 토크)."""
    return [np.vstack([np.eye(3), skew(r)]) for r in P_rel]


# ------------------------------------------------------------------ 배분기
class MarginAllocator:
    """한 접촉 집합에 대한 배분기. setup(기하) 후 solve(요구 렌치) 를 매 틱 호출."""

    def __init__(self, mu=0.5, f_cap=30.0, r_req=0.15, n_edge=8,
                 eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, lam_reg=1e-6, cont_w=0.0):
        """cont_w: 연속성 가중치 — 목적에 cont_w·‖f − f_prev‖² 추가 (힘 공간, 기저 무관). 0 이면 없음."""
        self.mu, self.f_cap, self.r_req, self.n_edge = mu, f_cap, r_req, n_edge
        self.cont_w = float(cont_w)
        self.f_prev = None                      # 직전 틱 총 팁 힘 [3A] (접촉 수 바뀌면 리셋)
        self.osqp_settings = dict(verbose=False, eps_abs=eps_abs, eps_rel=eps_rel,
                                  max_iter=max_iter, polishing=True)
        self.lam_reg = lam_reg           # 2단계(최대 r) LP 의 λ 정규화
        self.lam_prev = None
        self.ready = False

    # ---- 기하 조립 ----
    def setup(self, P, N_in, com, r_obj, cap=None, d_diag=None):
        """P [A,3] 팁 위치, N_in [A,3] 안쪽 법선 (세계 좌표), com [3], r_obj 물체 반경,
        cap [A] 팁별 상한 (기본 f_cap; 새로 붙은 팁의 램프에 사용),
        d_diag [6] 외란 박스 축 배율 (기본 = ρ 정규화; 단위시험용 덮어쓰기)."""
        P, N_in = np.asarray(P, float), np.asarray(N_in, float)
        A = len(P)
        self.A = A
        self.N_in = N_in / np.linalg.norm(N_in, axis=1, keepdims=True)
        self.cap = np.full(A, self.f_cap) if cap is None else np.asarray(cap, float)
        rel = P - np.asarray(com, float)
        self.Gs = grasp_blocks(rel)
        self.G = np.hstack(self.Gs)                          # [6, 3A]
        # 영공간 (직교 정규) + 특수해용 의사역행렬
        U, S, Vt = np.linalg.svd(self.G)
        tol = max(S[0], 1e-12) * 1e-9
        rank = int((S > tol).sum())
        self.rank = rank
        self.Nmat = Vt[rank:].T                              # [3A, 3A−rank]
        self.Gpinv = np.linalg.pinv(self.G, rcond=1e-9)
        # 스프링 반응 (강성 상쇄)
        Kg = sum(Gi @ Gi.T for Gi in self.Gs)                # [6,6]
        Kg_inv = np.linalg.pinv(Kg)
        self.Ms = [-Gi.T @ Kg_inv for Gi in self.Gs]         # [3,6] 각각
        # 외란 박스 축 배율 (ρ 정규화와 동일)
        self.d_diag = (np.array([self.f_cap] * 3 + [self.f_cap * r_obj] * 3)
                       if d_diag is None else np.asarray(d_diag, float))
        self.r_obj = r_obj
        # 각뿔 면 / 상한 행 — f 에 대한 선형 계수 C (행 × 3A) 와 r 계수 g (‖D M_iᵀ h‖₁)
        rows_face, g_face = [], []
        rows_cap, g_cap = [], []
        self.H = []
        for i in range(A):
            _E, H = pyramid(self.N_in[i], self.mu, self.n_edge)
            self.H.append(H)
            for h in H:
                row = np.zeros(3 * A)
                row[3 * i:3 * i + 3] = h
                rows_face.append(row)
                g_face.append(np.abs(self.d_diag * (self.Ms[i].T @ h)).sum())
            row = np.zeros(3 * A)
            row[3 * i:3 * i + 3] = self.N_in[i]
            rows_cap.append(row)
            g_cap.append(np.abs(self.d_diag * (self.Ms[i].T @ self.N_in[i])).sum())
        self.C_face, self.g_face = np.array(rows_face), np.array(g_face)   # [8A,3A],[8A]
        self.C_cap, self.g_cap = np.array(rows_cap), np.array(g_cap)       # [A,3A],[A]
        # λ 좌표에서의 계수 (한 번만)
        self.CN_face = self.C_face @ self.Nmat
        self.CN_cap = self.C_cap @ self.Nmat
        self.n_lam = self.Nmat.shape[1]
        if self.lam_prev is not None and len(self.lam_prev) != self.n_lam:
            self.lam_prev = None
        if self.f_prev is not None and len(self.f_prev) != 3 * A:
            self.f_prev = None
        self.ready = True

    # ---- 보조 ----
    def particular(self, w_req):
        return self.Gpinv @ np.asarray(w_req, float)

    def forces_from_lam(self, f0, lam):
        return (f0 + self.Nmat @ lam).reshape(self.A, 3)

    def margin_of(self, f, cap=None):
        """임의 힘 상태 f [A,3] 의 배분 여유 r (닫힌식). 음수 = 이미 각뿔 밖/상한 초과."""
        f = np.asarray(f, float).reshape(-1)
        cap = self.cap if cap is None else np.asarray(cap, float)
        s_face = self.C_face @ f                      # 면 여유 [8A]
        s_cap = cap - self.C_cap @ f                  # 상한 여유 [A]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_face = np.where(self.g_face > 1e-12, s_face / self.g_face,
                              np.where(s_face >= 0, np.inf, -np.inf))
            r_cap = np.where(self.g_cap > 1e-12, s_cap / self.g_cap,
                             np.where(s_cap >= 0, np.inf, -np.inf))
        return float(min(r_face.min(), r_cap.min()))

    def margin_detail(self, f, cap=None):
        """배분 여유의 최약 조건: (r, 팁 i, 면 k 또는 -1(상한), 최악 꼭짓점 v [6])."""
        f = np.asarray(f, float).reshape(-1)
        cap = self.cap if cap is None else np.asarray(cap, float)
        best = (np.inf, -1, -1, np.zeros(6))
        for i in range(self.A):
            for k in range(self.n_edge):
                row = i * self.n_edge + k
                if self.g_face[row] > 1e-12:
                    r = (self.C_face[row] @ f) / self.g_face[row]
                    if r < best[0]:
                        c = self.d_diag * (self.Ms[i].T @ self.H[i][k])
                        best = (float(r), i, k, -np.sign(c))
            if self.g_cap[i] > 1e-12:
                r = (cap[i] - self.C_cap[i] @ f) / self.g_cap[i]
                if r < best[0]:
                    c = self.d_diag * (self.Ms[i].T @ self.N_in[i])
                    best = (float(r), i, -1, np.sign(c))
        return best

    def margin_dir(self, f, v, cap=None):
        """고정 방향 v [6] (외란 = s·D·v)에 대한 예측 임계 s: 그 방향에서 여유를 잃는 조건들의 최솟값."""
        f = np.asarray(f, float).reshape(-1)
        cap = self.cap if cap is None else np.asarray(cap, float)
        Dv = self.d_diag * np.asarray(v, float)
        s_min = np.inf
        for i in range(self.A):
            df = self.Ms[i] @ Dv                          # 단위 s 당 팁 i 힘 변화
            for k in range(self.n_edge):
                loss = -(self.H[i][k] @ df)               # 면 여유 감소율
                if loss > 1e-12:
                    s_min = min(s_min, (self.H[i][k] @ f[3 * i:3 * i + 3]) / loss)
            gain = self.N_in[i] @ df
            if gain > 1e-12:
                s_min = min(s_min, (cap[i] - self.N_in[i] @ f[3 * i:3 * i + 3]) / gain)
        return float(s_min)

    def cone_slack(self, f):
        """팁별 최소 면 여유 (N 단위 힘 성분, 0 이상이면 각뿔 안) [A]."""
        f = np.asarray(f, float).reshape(-1)
        return (self.C_face @ f).reshape(self.A, self.n_edge).min(1)

    def _osqp(self, P, q, Aeq, l, u, x0=None):
        prob = osqp.OSQP()
        prob.setup(P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(Aeq), l=l, u=u,
                   **self.osqp_settings)
        if x0 is not None:
            prob.warm_start(x=x0)
        res = prob.solve()
        return res.x, res.info.status

    def _pq(self, f0):
        """1단계 목적 ‖λ‖² + cont_w‖f0 + Nλ − f_prev‖² 의 (P, q). N 직교정규 → NᵀN = I."""
        nl = self.n_lam
        if self.cont_w > 0 and self.f_prev is not None and len(self.f_prev) == len(f0):
            P = 2 * (1 + self.cont_w) * np.eye(nl)
            q = -2 * self.cont_w * (self.Nmat.T @ (self.f_prev - f0))
        else:
            P, q = 2 * np.eye(nl), np.zeros(nl)
        return P, q

    def _remember(self, f_total):
        self.f_prev = np.asarray(f_total, float).reshape(-1).copy()

    # ---- 본체 ----
    def solve(self, w_req, cap=None, r_req=None, f_fixed=None, w_base=None):
        """요구 렌치 w_req [6] (N, N·m, 질량중심 기준) → dict.

        f_fixed [A,3]: 배분기 밖에서 이미 정해진 팁 힘(예: 운반 당김 1500×Δ). 총 힘 = f_fixed + f₀' + Nλ 에
                       각뿔·상한·여유 제약을 걸고, 등식은 나머지가 (w_req − G f_fixed) 를 채움 ("당김을 알려 주기").
        w_base [6]:    용량 인식 다듬기. 반드시 만들 몫(중력)이고, 나머지 w_req − w_base 는 배율 s∈[0,1] 로 축소
                       가능. 1단계가 안 되면 **여유 r_req 를 지킨 채 s 를 최대화**(stage 4)하고, 그래도 안 되면
                       옛 2·3단계(여유 축소 → 잔차)로 물러난다.

        반환: f [A,3] 총 팁 힘, lam, r(확보한 여유), s(요구 배율, 다듬기 없으면 1), stage(1/4/2/3), status,
              resid(3단계 잔차), r_max(2단계를 거쳤을 때의 최대 가능 여유, 아니면 None).
        """
        assert self.ready
        if osqp is None:
            raise RuntimeError("osqp 미설치 — reference_* 를 쓰거나 pip install osqp")
        cap = self.cap if cap is None else np.asarray(cap, float)
        r_req = self.r_req if r_req is None else float(r_req)
        w_req = np.asarray(w_req, float)
        ff = (np.zeros(3 * self.A) if f_fixed is None
              else np.asarray(f_fixed, float).reshape(-1))
        # 고정 힘(당김)이 만드는 렌치는 **추가 운동 요구**로 그대로 둔다 — 나머지는 w_req 를 채우고,
        # 총 힘 = ff + f₀ + Nλ 에 각뿔·상한·여유 제약. (초기 구현은 w_req − G·ff 로 두어 당김의 순 렌치를
        # 배분기가 상쇄해 버렸고, 그 판은 낙하했다 — 2026-09-06 정정)
        w_eff = w_req
        f0 = ff + self.particular(w_eff)              # 총 힘의 특수해 (고정 힘 포함)
        b_face = self.C_face @ f0                     # [8A]
        b_cap = cap - self.C_cap @ f0                 # [A]
        nl = self.n_lam
        out = dict(stage=1, status="", r=r_req, r_max=None, resid=0.0, s=1.0)

        # ---- 1단계: r = r_req, min ‖λ‖² ----
        if nl > 0:
            Aeq = np.vstack([self.CN_face, self.CN_cap])
            l = np.concatenate([r_req * self.g_face - b_face, np.full(self.A, -np.inf)])
            u = np.concatenate([np.full(len(b_face), np.inf), b_cap - r_req * self.g_cap])
            P1, q1 = self._pq(f0)
            x, st = self._osqp(P1, q1, Aeq, l, u, self.lam_prev)
            if st in ("solved", "solved inaccurate"):
                lam = x
                self.lam_prev = lam.copy()
                f_out = self.forces_from_lam(f0, lam)
                self._remember(f_out)
                out.update(status=st, lam=lam, f=f_out)
                return out
        else:
            # 영공간 없음 (A ≤ 2 이면서 최대 랭크) — f = f₀ 의 여유를 그대로 평가
            r_here = self.margin_of(f0.reshape(self.A, 3), cap)
            if r_here >= r_req - 1e-9:
                out.update(status="unique", lam=np.zeros(0),
                           f=f0.reshape(self.A, 3))
                return out

        # ---- 4단계(용량 인식 다듬기): r = r_req 고정, 가변 렌치 배율 s ∈ [0,1] 최대화 ----
        if w_base is not None and nl > 0:
            wb = np.asarray(w_base, float)
            f0b = ff + self.particular(wb)                        # 기본 몫만의 특수해 (고정 힘 포함)
            f0v = self.particular(w_req - wb)                       # 가변 몫 단위 특수해
            bF_b, bF_v = self.C_face @ f0b, self.C_face @ f0v
            bC_b, bC_v = self.C_cap @ f0b, self.C_cap @ f0v
            # 변수 (λ, s):  CN λ + bF_v s ≥ r g − bF_b ;  CN λ + bC_v s ≤ cap − r g_cap − bC_b ; 0 ≤ s ≤ 1
            # max s 는 LP — OSQP 는 이 스케일(λ N 단위 vs s 무차원)에서 미수렴하므로 HiGHS 로 정확히 푼다 (~1ms)
            from scipy.optimize import linprog
            A_ub = np.vstack([np.hstack([-self.CN_face, -bF_v[:, None]]),
                              np.hstack([self.CN_cap, bC_v[:, None]])])
            b_ub = np.concatenate([-(r_req * self.g_face - bF_b), cap - r_req * self.g_cap - bC_b])
            c = np.zeros(nl + 1)
            c[-1] = -1.0
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nl + [(0.0, 1.0)], method="highs")
            x, st = (res.x, "solved") if res.status == 0 else (None, f"lp:{res.status}")
            if st == "solved" and x is not None:
                s_use = 0.98 * float(np.clip(x[-1], 0.0, 1.0))
                f0s = f0b + s_use * f0v
                bF, bC = self.C_face @ f0s, cap - self.C_cap @ f0s
                Aeq1 = np.vstack([self.CN_face, self.CN_cap])
                l1 = np.concatenate([r_req * self.g_face - bF, np.full(self.A, -np.inf)])
                u1 = np.concatenate([np.full(len(bF), np.inf), bC - r_req * self.g_cap])
                P4, q4 = self._pq(f0s)
                x1, st1 = self._osqp(P4, q4, Aeq1, l1, u1, x[:nl])
                if st1 in ("solved", "solved inaccurate"):
                    lam = x1
                    self.lam_prev = lam.copy()
                    f_out = (f0s + self.Nmat @ lam).reshape(self.A, 3)
                    self._remember(f_out)
                    out.update(stage=4, status=st1, lam=lam, s=s_use, f=f_out)
                    return out

        # ---- 2단계: 가능한 최대 r ∈ [0, r_req] ----
        out["stage"] = 2
        # 변수 (λ, r): 면  CN λ − g r ≥ −b ; 상한 CN λ + g r ≤ b_cap ; 0 ≤ r ≤ r_req
        Aeq = np.block([[self.CN_face, -self.g_face[:, None]],
                        [self.CN_cap, self.g_cap[:, None]],
                        [np.zeros((1, nl)), np.ones((1, 1))]])
        l = np.concatenate([-b_face, np.full(self.A, -np.inf), [0.0]])
        u = np.concatenate([np.full(len(b_face), np.inf), b_cap, [r_req]])
        P = np.diag(np.concatenate([np.full(nl, 2 * self.lam_reg), [0.0]]))
        q = np.concatenate([np.zeros(nl), [-1.0]])
        x, st = self._osqp(P, q, Aeq, l, u)
        if st in ("solved", "solved inaccurate") and x is not None:
            r_max = float(max(0.0, min(x[-1], r_req)))
            out["r_max"] = r_max
            r_use = 0.98 * r_max                    # 경계에서 살짝 물러난 값으로 재풀이
            if nl > 0:
                Aeq1 = np.vstack([self.CN_face, self.CN_cap])
                l1 = np.concatenate([r_use * self.g_face - b_face, np.full(self.A, -np.inf)])
                u1 = np.concatenate([np.full(len(b_face), np.inf), b_cap - r_use * self.g_cap])
                P2, q2 = self._pq(f0)
                x1, st1 = self._osqp(P2, q2, Aeq1, l1, u1, x[:nl])
                if st1 in ("solved", "solved inaccurate"):
                    lam = x1
                    self.lam_prev = lam.copy()
                    f_out = self.forces_from_lam(f0, lam)
                    self._remember(f_out)
                    out.update(status=st1, lam=lam, r=r_use, f=f_out)
                    return out
            else:
                out.update(status="unique", lam=np.zeros(0), r=r_max,
                           f=f0.reshape(self.A, 3))
                return out

        # ---- 3단계: 요구 렌치 자체가 불가능 → 잔차 최소화 (힘·토크 저울 = r_obj) ----
        out["stage"] = 3
        A3 = 3 * self.A
        Sw = np.array([1.0] * 3 + [1.0 / self.r_obj] * 3)
        GS = Sw[:, None] * self.G
        P = 2 * (GS.T @ GS + 1e-6 * np.eye(A3))
        q = -2 * GS.T @ (Sw * np.asarray(w_req, float))
        Aeq = np.vstack([self.C_face, self.C_cap])
        l = np.concatenate([np.zeros(len(self.g_face)), np.full(self.A, -np.inf)])
        u = np.concatenate([np.full(len(self.g_face), np.inf), cap])
        x, st = self._osqp(P, q, Aeq, l, u)
        if x is None or st not in ("solved", "solved inaccurate"):
            x = np.zeros(A3)
            st = f"stage3 fail: {st}"
        f = x.reshape(self.A, 3)
        self._remember(f)
        out.update(status=st, lam=None, r=self.margin_of(f, cap),
                   f=f, resid=float(np.linalg.norm(Sw * (self.G @ x - w_req))))
        return out


# ------------------------------------------------------------------ 참조 풀이 (검증용)
def reference_stage1(alloc, w_req, r_req=None, cap=None):
    """scipy SLSQP 로 1단계(min ‖λ‖²)를 독립적으로 풂. 반환 (lam, f) 또는 None."""
    from scipy.optimize import minimize
    cap = alloc.cap if cap is None else np.asarray(cap, float)
    r_req = alloc.r_req if r_req is None else r_req
    f0 = alloc.particular(w_req)
    b_face = alloc.C_face @ f0
    b_cap = cap - alloc.C_cap @ f0
    nl = alloc.n_lam
    cons = [dict(type="ineq", fun=lambda lam: alloc.CN_face @ lam + b_face - r_req * alloc.g_face,
                 jac=lambda lam: alloc.CN_face),
            dict(type="ineq", fun=lambda lam: b_cap - r_req * alloc.g_cap - alloc.CN_cap @ lam,
                 jac=lambda lam: -alloc.CN_cap)]
    best = None
    for x0 in (np.zeros(nl), np.ones(nl), -np.ones(nl)):
        res = minimize(lambda lam: lam @ lam, x0, jac=lambda lam: 2 * lam,
                       constraints=cons, method="SLSQP",
                       options=dict(maxiter=2000, ftol=1e-14))
        viol = max(0.0, -(cons[0]["fun"](res.x)).min(), -(cons[1]["fun"](res.x)).min())
        if viol < 1e-6 and (best is None or res.fun < best[0]):
            best = (res.fun, res.x)
    if best is None:
        return None
    lam = best[1]
    return lam, alloc.forces_from_lam(f0, lam)


def reference_rmax(alloc, w_req, r_req=None, cap=None):
    """scipy HiGHS LP 로 2단계(최대 r)를 독립적으로 풂. 반환 r_max 또는 None(불가능)."""
    from scipy.optimize import linprog
    cap = alloc.cap if cap is None else np.asarray(cap, float)
    r_req = alloc.r_req if r_req is None else r_req
    f0 = alloc.particular(w_req)
    b_face = alloc.C_face @ f0
    b_cap = cap - alloc.C_cap @ f0
    nl = alloc.n_lam
    # 변수 (λ, r). 부등식 A_ub x ≤ b_ub:  −CN λ + g r ≤ b ;  CN λ + g r ≤ b_cap
    A_ub = np.vstack([np.hstack([-alloc.CN_face, alloc.g_face[:, None]]),
                      np.hstack([alloc.CN_cap, alloc.g_cap[:, None]])])
    b_ub = np.concatenate([b_face, b_cap])
    c = np.zeros(nl + 1)
    c[-1] = -1.0
    bounds = [(None, None)] * nl + [(0.0, r_req)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    return float(res.x[-1])
