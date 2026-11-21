"""Stage 1 그래프 트랜스포머 (명세 5장).

전역 어텐션 + 관계 유형 7종의 헤드별 학습 바이어스 (그릴링 질문 6 확정).
노드 135 = 패치 128 + 접촉 부위 6 + CLS. 헤드: 클래스 logits [C] + N별 품질 회귀 [3].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_dataset import N_PATCH, N_UNIT, N_NODE

N_REL = 7
# 초기화 사다리 (명세): 인접·접촉·self = 0~+1, 비인접·비접촉 = -2, 부위-부위·CLS = 0
_BIAS_INIT = [0.5, -2.0, 1.0, -2.0, 0.0, 0.0, 0.5]


class _Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.bias = nn.Parameter(
            torch.tensor(_BIAS_INIT).repeat(heads, 1) + 0.01 * torch.randn(heads, N_REL))
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rel_1h):
        B, N, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / self.dh ** 0.5          # [B,H,N,N]
        # 관계 유형 바이어스: one-hot 행렬곱 (advanced indexing 대비 ~4배 빠름)
        att = att + torch.einsum("bijr,hr->bhij", rel_1h, self.bias.to(att.dtype))
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ffn(self.ln2(x)))


class Stage1Net(nn.Module):
    def __init__(self, n_classes, d=256, layers=4, heads=8, d_local=64,
                 d_id=32, dropout=0.1, n_patch=N_PATCH, d_pf=7):
        super().__init__()
        self.n_patch = n_patch
        self.d_pf = d_pf
        self.pointnet = nn.Sequential(nn.Linear(6, d_local), nn.GELU(),
                                      nn.Linear(d_local, d_local))
        self.patch_enc = nn.Linear(d_pf + d_local, d)
        self.unit_id = nn.Embedding(N_UNIT, d_id)
        self.unit_enc = nn.Linear(10 + d_id, d)
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList(_Block(d, heads, dropout) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.cls_head = nn.Sequential(nn.Linear(d + 1, 256), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(256, n_classes))
        self.reg_head = nn.Sequential(nn.Linear(d + 1, 128), nn.GELU(),
                                      nn.Linear(128, 3))
        # v1.1: 클래스별 물리 합격 확률 (라벨 = 전량 시뮬 판정) — 불합격 꼬리 필터용
        self.pass_head = nn.Sequential(nn.Linear(d + 1, 256), nn.GELU(),
                                       nn.Linear(256, n_classes))

    def encode(self, batch):
        """트렁크: 노드 출력 [B,135,d] 와 전역 문맥 h [B,d+1] 반환 (v1·v2 공유)."""
        pf, loc, lm = batch["patch_feat"], batch["patch_local"], batch["patch_local_mask"]
        B = pf.shape[0]
        z = self.pointnet(loc)                                   # [B,128,K,64]
        z = z.masked_fill(~lm[..., None], torch.finfo(z.dtype).min).amax(2)
        z = torch.where(lm.any(-1, keepdim=True), z, torch.zeros_like(z))
        patches = self.patch_enc(torch.cat([pf, z], -1))         # [B,128,d]
        uid = self.unit_id(torch.arange(N_UNIT, device=pf.device)).expand(B, -1, -1)
        units = self.unit_enc(torch.cat([batch["unit_feat"], uid], -1))
        x = torch.cat([patches, units, self.cls_tok.expand(B, -1, -1)], 1)
        rel_1h = F.one_hot(batch["rel_type"].long(), 7).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, rel_1h)
        h = torch.cat([self.ln_out(x[:, -1]), batch["log_robj"]], -1)
        return x, h

    def forward(self, batch):
        _, h = self.encode(batch)
        return self.cls_head(h), self.reg_head(h), self.pass_head(h)


class Stage1NetV2(Stage1Net):
    """v2: ① 고정 사전 분류 → 후보 구조 스코어러, ③ 합격 확률 헤드 삭제.

    유효 후보의 그룹 구성(클래스 사전)을 소속 행렬로 만들어, 트랜스포머의 접촉
    부위 노드 출력을 그룹별로 "부위별 변환 후 합"(DeepSets)해 후보를 채점한다.
    트렁크·② 회귀는 부모 그대로.
    """

    def __init__(self, vocab, d=256, layers=4, heads=8, d_local=64, d_id=32,
                 dropout=0.1, d_score=128, n_patch=N_PATCH, d_pf=7):
        super().__init__(len(vocab), d, layers, heads, d_local, d_id, dropout, n_patch, d_pf)
        del self.cls_head, self.pass_head
        C = len(vocab)
        member = torch.zeros(C, 4, N_UNIT)
        from .stage1_dataset import UNITS
        for c, blocks in enumerate(vocab.classes):
            for g, blk in enumerate(blocks):
                for u in blk:
                    member[c, g, UNITS.index(u)] = 1.0
        self.register_buffer("member", member)                    # [C,4,6]
        self.register_buffer("gsize", member.sum(-1))             # [C,4]
        self.register_buffer("gmask", (member.sum(-1) > 0).float())
        self.phi_u = nn.Sequential(nn.Linear(d, d_score), nn.GELU(),
                                   nn.Linear(d_score, d_score))
        self.ctx_proj = nn.Linear(d + 1, d_score)
        self.phi_g = nn.Sequential(nn.Linear(2 * d_score + 1, 256), nn.GELU(),
                                   nn.Linear(256, d_score))
        self.psi = nn.Sequential(nn.Linear(d_score, d_score), nn.GELU(),
                                 nn.Linear(d_score, 1))

    def forward(self, batch):
        x, h = self.encode(batch)
        B = x.shape[0]
        units = x[:, self.n_patch:self.n_patch + N_UNIT]          # [B,6,d]
        fu = self.phi_u(units)                                    # [B,6,ds]
        T = torch.einsum("cgu,bud->bcgd", self.member.to(fu.dtype), fu)  # [B,C,4,ds]
        C = T.shape[1]
        ctx = self.ctx_proj(h)[:, None, None, :].expand(-1, C, 4, -1)
        size = self.gsize.to(fu.dtype)[None, :, :, None].expand(B, -1, -1, -1)
        g = self.phi_g(torch.cat([T, ctx, size], -1))             # [B,C,4,ds]
        g = g * self.gmask.to(fu.dtype)[None, :, :, None]
        logits = self.psi(g.sum(2)).squeeze(-1)                   # [B,C]
        return logits, self.reg_head(h), None


def stage1_loss(logits, reg, pass_logits, batch, reg_weight=0.5, pass_weight=0.5):
    """CE(soft target) + Huber 회귀 + BCE(클래스별 합격 확률). 지표도 함께 반환."""
    total = logits.new_zeros(())
    metrics = {}
    for n in (3, 4):
        t, v = batch[f"target_n{n}"], batch[f"valid_n{n}"]
        rows = t.sum(-1) > 0
        if rows.any():
            lg = logits[rows].masked_fill(~v[rows], float("-inf"))
            logp = lg.log_softmax(-1)
            ce = -(t[rows] * logp.masked_fill(~v[rows], 0.0)).sum(-1)
            total = total + ce.mean()
            pred = lg.argmax(-1)
            metrics[f"top1_n{n}"] = (pred == t[rows].argmax(-1)).float().mean().item()
            metrics[f"pass_n{n}"] = (t[rows].gather(1, pred[:, None])[:, 0] > 0
                                     ).float().mean().item()
            metrics[f"ce_n{n}"] = ce.mean().item()
            metrics[f"cnt_n{n}"] = int(rows.sum())   # 전역 가중 집계용 (배치 편향 방지)
        if pass_logits is not None:                     # v2 는 합격 헤드 없음
            pt = batch[f"pass_n{n}"]
            known = pt >= 0
            if known.any():
                bce = F.binary_cross_entropy_with_logits(pass_logits[known], pt[known])
                total = total + pass_weight * bce
                with torch.no_grad():
                    acc = ((pass_logits[known] > 0) == (pt[known] > 0.5)).float().mean()
                metrics[f"passhead_acc_n{n}"] = acc.item()
    rm = batch["reg_mask"]
    if rm.any():
        rl = F.huber_loss(reg[rm], batch["reg_target"][rm])
        total = total + reg_weight * rl
        metrics["reg_mae"] = (reg[rm] - batch["reg_target"][rm]).abs().mean().item()
    return total, metrics
