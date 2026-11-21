"""조건부 net2 (2026-10-17) — Stage2Net 을 상속, 기존 파일은 그대로.

몸통·net1·점 임베딩 e_p·그룹 토큰 T 는 Stage2Net 그대로. 히트맵만 그룹 순서대로 앞 자리를 보고 낸다:

  s_k(p) = sc_mlp( e_p + tok_proj(T_k) + C_k(p) ),   C_k(p) = Σ_{j<k} φ([ e_{g_j},  rel(p, g_j) ])
  rel(p, q) = [ (pos_p − pos_q) 3,  |pos_p − pos_q| 1,  n_p·n_q 1 ]   (pos = 무게중심 기준 / r_obj)
  P_k(p | 앞 자리) = 그룹 k 마스크 안 softmax(s_k)

k = 0 (첫 그룹) 은 C = 0 이라 Stage2Net 의 히트맵과 같은 식이다.

렌치 특징 (2026-10-24, wrench_feat="support"):
  점 p 의 법선 렌치 w_p = [n_p ; pos_p × n_p] (6).  방향 표본 D (28 = 힘 ±축 6 + 힘 대각 8 + 토크 ±축 6 + 토크 대각 8).
  지지 함수  h_k(d) = max_{j<k} d·w_{g_j}  (앞 자리 앵커들이 d 방향으로 밀 수 있는 정도, k=0 이면 0)
  후보점 방향값  d·w_p
  s_k(p) = sc_mlp( e_p + tok_proj(T_k) + C_k(p) + ψ([h_k(d) ; d·w_p]) ),  ψ: 2|D| → ds → ds
  (앵커별 합으로 만들 수 없는 최댓값 형태의 요약을 별도 경로로 보탠다)

손실 (stage2_cond_loss):
  L_기본  = −log Σ_m P(G_m) / n,  log P(G) = Σ_k log P_k(g_k | g_<k)    (정답 벌 m = 1..K, n = 그룹 수)
  L_흔들림 = −Σ_{k≥1} log P_k(g_k^(m) | 흔들린 앞 자리) / (n−1)          (흔들린 샘플만)
  L_먼곳  = −Σ_{k≥1} log P_k(f_k | f_<k) / (n−1)                         (먼 첫 자리 조합이 있는 샘플만)
  (그룹 수로 나눔 = 기존 net2 의 그룹 평균 교차 엔트로피와 비중을 맞춤)
  전체 = L_net1 + hm_weight × (L_기본 + noise_weight × L_흔들림 + far_weight × L_먼곳)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_model import stage1_loss
from .stage1_dataset import N_UNIT
from .stage2_model import Stage2Net

N_REL = 5


def wrench_dirs(dtype, device):
    """고정 방향 표본 [28, 6]: 힘 ±축·대각, 토크 ±축·대각 (단위 벡터)."""
    ax = torch.eye(3, dtype=dtype, device=device)
    axes = torch.cat([ax, -ax], 0)                                        # [6,3]
    sg = torch.tensor([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], dtype=dtype, device=device)
    diag = sg / sg.norm(dim=-1, keepdim=True)                             # [8,3]
    f = torch.cat([axes, diag], 0)                                        # [14,3]
    z = torch.zeros_like(f)
    return torch.cat([torch.cat([f, z], -1), torch.cat([z, f], -1)], 0)   # [28,6]


def normal_wrench(pos, nrm):
    """점마다 법선 렌치 [B,P,6] = [n ; pos × n]  (pos: 무게중심 기준 / r_obj)."""
    return torch.cat([nrm, torch.cross(pos, nrm, dim=-1)], -1)


class Stage2CondNet(Stage2Net):
    def __init__(self, vocab, d_score=128, wrench_feat="none", use_cond=True, **kw):
        super().__init__(vocab, d_score=d_score, **kw)
        self.use_cond = use_cond              # False = 앞 자리 조건 없이 그룹마다 독립 (2×2 비교의 기준선)
        self.cond_mlp = nn.Sequential(nn.Linear(d_score + N_REL, d_score), nn.GELU(),
                                      nn.Linear(d_score, d_score))
        self.wrench_feat = wrench_feat
        if wrench_feat == "support":
            nd = wrench_dirs(torch.float32, "cpu").shape[0]
            self.wr_mlp = nn.Sequential(nn.Linear(2 * nd, d_score), nn.GELU(),
                                        nn.Linear(d_score, d_score))

    # ── 공통 부분: 몸통 → net1 출력, 점 임베딩, 그룹 토큰 (규칙 순서) ──
    def embed(self, batch):
        x, h = self.encode(batch)
        B = x.shape[0]
        units = x[:, self.n_patch:self.n_patch + N_UNIT]
        fu = self.phi_u(units)
        T = torch.einsum("cgu,bud->bcgd", self.member.to(fu.dtype), fu)
        C = T.shape[1]
        ctx = self.ctx_proj(h)[:, None, None, :].expand(-1, C, 4, -1)
        size = self.gsize.to(fu.dtype)[None, :, :, None].expand(B, -1, -1, -1)
        g = self.phi_g(torch.cat([T, ctx, size], -1))
        g = g * self.gmask.to(fu.dtype)[None, :, :, None]
        logits = self.psi(g.sum(2)).squeeze(-1)
        reg = self.reg_head(h)

        if "point_patch3" in batch:
            xs = x[:, :self.n_patch]
            i3, w3 = batch["point_patch3"], batch["point_patch3_w"].to(x.dtype)
            xp = sum(torch.gather(xs, 1, i3[..., j, None].expand(-1, -1, x.shape[-1]))
                     * w3[..., j, None] for j in range(i3.shape[-1]))
        else:
            xp = torch.gather(x[:, :self.n_patch], 1,
                              batch["point_patch"][..., None].expand(-1, -1, x.shape[-1]))
        e_p = self.pt_mlp(torch.cat([xp, batch["point_feat"]], -1))       # [B,P,ds]
        ug = batch["cond_ug"]
        G = F.one_hot(ug.clamp(min=0), 4).to(fu.dtype) * (ug >= 0)[..., None].to(fu.dtype)
        Tg = self.tok_proj(torch.einsum("bug,buh->bgh", G, fu))           # [B,4,ds]
        return logits, reg, e_p, Tg

    def phi(self, e_p, pos, nrm, q):
        """앞 자리 점 q [B] 하나가 모든 점 p 에 주는 조건 φ([e_q, rel(p,q)]) → [B,P,ds]."""
        ar = torch.arange(e_p.shape[0], device=e_p.device)
        q = q.clamp(min=0)
        dpos = pos - pos[ar, q][:, None]
        rel = torch.cat([dpos, dpos.norm(dim=-1, keepdim=True),
                         (nrm * nrm[ar, q][:, None]).sum(-1, keepdim=True)], -1)
        eq = e_p[ar, q][:, None].expand(-1, e_p.shape[1], -1)
        return self.cond_mlp(torch.cat([eq, rel.to(eq.dtype)], -1))

    def wrench_proj(self, batch):
        """후보점 방향값 d·w_p [B,P,|D|] (파지당 1회)."""
        pos, nrm = batch["cond_pos"], batch["cond_nrm"]
        D = wrench_dirs(pos.dtype, pos.device)
        return normal_wrench(pos, nrm) @ D.T

    def support_term(self, proj, h, e_p):
        """ψ([h_k(d) ; d·w_p]) [B,P,ds].  proj [B,P,|D|], h [B,|D|] (앞 자리 지지 함수)."""
        hb = h[:, None].expand(-1, proj.shape[1], -1)
        return self.wr_mlp(torch.cat([hb, proj], -1).to(e_p.dtype))

    def support_update(self, h, proj, q):
        """앞 자리에 점 q [B] 가 추가될 때 지지 함수 갱신: h ← max(h, d·w_q)."""
        ar = torch.arange(proj.shape[0], device=proj.device)
        return torch.maximum(h, proj[ar, q.clamp(min=0)])

    def chain_logits(self, e_p, Tg, batch, prefix, k_from=0, k_to=4):
        """앞 자리 prefix [B,4] 를 차례로 조건으로 넣으며 그룹 k 의 log P_k [B,P] 를 낸다 (k_from ≤ k < k_to).

        그룹 수를 넘는 행은 마스크를 전부 열어 NaN 기울기를 막는다 (손실에서는 0 으로 뺀다).
        """
        pos, nrm, mask, n_g = batch["cond_pos"], batch["cond_nrm"], batch["cond_mask"], batch["cond_n"]
        cum = torch.zeros_like(e_p)
        sup = self.wrench_feat == "support"
        if sup:
            proj = self.wrench_proj(batch)
            h = torch.zeros(proj.shape[0], proj.shape[2], dtype=proj.dtype, device=proj.device)
        for k in range(k_to):
            if k > 0 and self.use_cond:
                cum = cum + self.phi(e_p, pos, nrm, prefix[:, k - 1])
                if sup:
                    h = self.support_update(h, proj, prefix[:, k - 1])
            if k < k_from:
                continue
            rows = n_g > k
            if not rows.any():
                break
            z = e_p + Tg[:, k, None] + cum
            if sup:
                z = z + self.support_term(proj, h, e_p)
            lg = self.sc_mlp(z).squeeze(-1).float()
            mk = mask[:, k] | (~rows)[:, None]
            yield k, rows, lg.masked_fill(~mk, float("-inf")).log_softmax(-1)

    def chain_logp(self, e_p, Tg, batch, prefix, target, k_from=0):
        """Σ_k log P_k(target_k | prefix_<k) [B] (그룹 수까지)."""
        total = e_p.new_zeros(e_p.shape[0], dtype=torch.float32)
        for k, rows, logp in self.chain_logits(e_p, Tg, batch, prefix, k_from):
            v = logp.gather(1, target[:, k:k + 1].clamp(min=0))[:, 0]
            total = total + torch.where(rows, v, torch.zeros_like(v))
        return total

    def forward(self, batch):
        logits, reg, e_p, Tg = self.embed(batch)
        return logits, reg, e_p, Tg


def stage2_cond_loss(model, out, batch, hm_weight=1.0, noise_weight=0.5, far_weight=0.0):
    logits, reg, e_p, Tg = out
    total, metrics = stage1_loss(logits, reg, None, batch)
    n_g, K = batch["cond_n"], batch["cond_k"]
    tgt = batch["cond_tgt"]                                               # [B,MAX_K,4]
    B = tgt.shape[0]
    ar = torch.arange(B, device=tgt.device)

    # ① 기본: 정답 벌들의 확률 합  −log Σ_m P(G_m)
    lps = []
    for m in range(tgt.shape[1]):
        if not (K > m).any():
            break
        lp = model.chain_logp(e_p, Tg, batch, tgt[:, m], tgt[:, m])
        lps.append(torch.where(K > m, lp, torch.full_like(lp, float("-inf"))))
    # 그룹 수로 나눠 기존 net2(그룹 평균 교차 엔트로피)와 net1 손실 대비 비중을 맞춘다
    l_main = -torch.logsumexp(torch.stack(lps, 1), 1) / n_g.float()       # [B]
    loss_hm = l_main.mean()
    metrics["hm_main"] = float(l_main.mean())
    metrics["cnt_hm"] = B

    # ② 작은 흔들림: 흔들린 앞 자리 + 원래 짝 (첫 자리 항 없음)
    on = batch["cond_noise_on"].any(-1)
    if noise_weight > 0 and on.any():
        orig = tgt[ar, batch["cond_noise_m"]]
        lp = model.chain_logp(e_p, Tg, batch, batch["cond_noise"], orig, k_from=1)
        l_noise = -(lp / (n_g - 1).clamp(min=1).float())[on].mean()
        loss_hm = loss_hm + noise_weight * l_noise
        metrics["hm_noise"] = float(l_noise)

    # ③ 먼 첫 자리: 그 조합 자신의 앞 자리 + 짝 (첫 자리 항 없음)
    fon = batch["cond_far_on"]
    if far_weight > 0 and fon.any():
        far = batch["cond_far"]
        lp = model.chain_logp(e_p, Tg, batch, far, far, k_from=1)
        l_far = -(lp / (n_g - 1).clamp(min=1).float())[fon].mean()
        loss_hm = loss_hm + far_weight * l_far
        metrics["hm_far"] = float(l_far)

    total = total + hm_weight * loss_hm
    return total, metrics


@torch.no_grad()
def eval_metrics(model, out, batch):
    """검증 지표 (정답 첫 벌 기준):
      tf_top8_g{k} : 앞 자리에 정답을 넣었을 때 그룹 k 의 정답이 상위 8 안
      gr_hit_g{k}  : 탐욕 디코딩(앞 자리 = 모델 1등)에서 그룹 k 1등이 어느 정답 벌의 그 자리와 같음
    반환: {이름: (값, 표본 수)}, 탐욕 조합 [B,4]
    """
    _, _, e_p, Tg = out
    tgt_all, K = batch["cond_tgt"], batch["cond_k"]
    tgt = tgt_all[:, 0]
    res = {}
    for k, rows, logp in model.chain_logits(e_p, Tg, batch, tgt):
        top8 = logp.topk(8, -1).indices
        res[f"tf_top8_g{k}"] = (float((top8 == tgt[:, k:k + 1]).any(-1)[rows].float().mean()), int(rows.sum()))
    greedy = torch.full_like(tgt, -1)
    pos, nrm, mask, n_g = batch["cond_pos"], batch["cond_nrm"], batch["cond_mask"], batch["cond_n"]
    cum = torch.zeros_like(e_p)
    sup = model.wrench_feat == "support"
    if sup:
        proj = model.wrench_proj(batch)
        h = torch.zeros(proj.shape[0], proj.shape[2], dtype=proj.dtype, device=proj.device)
    for k in range(4):
        if k > 0 and model.use_cond:
            cum = cum + model.phi(e_p, pos, nrm, greedy[:, k - 1])
            if sup:
                h = model.support_update(h, proj, greedy[:, k - 1])
        rows = n_g > k
        if not rows.any():
            break
        z = e_p + Tg[:, k, None] + cum
        if sup:
            z = z + model.support_term(proj, h, e_p)
        lg = model.sc_mlp(z).squeeze(-1).float()
        greedy[:, k] = lg.masked_fill(~(mask[:, k] | (~rows)[:, None]), float("-inf")).argmax(-1)
        valid_m = torch.arange(tgt_all.shape[1], device=K.device)[None] < K[:, None]
        hit = ((tgt_all[:, :, k] == greedy[:, k:k + 1]) & valid_m).any(-1)
        res[f"gr_hit_g{k}"] = (float(hit[rows].float().mean()), int(rows.sum()))
    return res, greedy
