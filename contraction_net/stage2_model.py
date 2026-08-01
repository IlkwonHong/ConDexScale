"""Stage 2 모델 — Stage1NetV2 + 파티션 조건 히트맵 헤드 (명세 2장).

분해형 브리지: 점 임베딩(그룹 무관, 1회) + 그룹 토큰 가산 → 점수.
φᵤ 는 net1 스코어러와 공유.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_model import Stage1NetV2, stage1_loss
from .stage1_dataset import N_UNIT


class Stage2Net(Stage1NetV2):
    def __init__(self, vocab, d=256, d_score=128, n_patch=128, **kw):
        super().__init__(vocab, d=d, d_score=d_score, n_patch=n_patch, **kw)
        self.pt_mlp = nn.Sequential(nn.Linear(d + 6, 256), nn.GELU(),
                                    nn.Linear(256, d_score))
        self.tok_proj = nn.Linear(d_score, d_score)
        self.sc_mlp = nn.Sequential(nn.GELU(), nn.Linear(d_score, d_score),
                                    nn.GELU(), nn.Linear(d_score, 1))

    def forward(self, batch):
        x, h = self.encode(batch)
        B = x.shape[0]
        units = x[:, self.n_patch:self.n_patch + N_UNIT]           # [B,6,d]
        fu = self.phi_u(units)                                     # [B,6,ds]

        # ── net1: 파티션 스코어러 + 회귀 (Stage1NetV2 와 동일 로직) ──
        T = torch.einsum("cgu,bud->bcgd", self.member.to(fu.dtype), fu)
        C = T.shape[1]
        ctx = self.ctx_proj(h)[:, None, None, :].expand(-1, C, 4, -1)
        size = self.gsize.to(fu.dtype)[None, :, :, None].expand(B, -1, -1, -1)
        g = self.phi_g(torch.cat([T, ctx, size], -1))
        g = g * self.gmask.to(fu.dtype)[None, :, :, None]
        logits = self.psi(g.sum(2)).squeeze(-1)
        reg = self.reg_head(h)

        # ── net2: 히트맵 (추첨된 파티션 조건) ──
        # 점 임베딩: 소속 패치 노드 출력 gather + 점 로컬 특징
        xp = torch.gather(x[:, :self.n_patch], 1,
                          batch["point_patch"][..., None].expand(-1, -1, x.shape[-1]))
        e_p = self.pt_mlp(torch.cat([xp, batch["point_feat"]], -1))   # [B,P,ds]
        # 그룹 토큰: 추첨 파티션의 부위→그룹 one-hot 합
        ug = batch["hm_unit_group"]                                   # [B,6], -1 = 없음
        G = F.one_hot(ug.clamp(min=0), 4).to(fu.dtype) * (ug >= 0)[..., None].to(fu.dtype)
        Tg = torch.einsum("bug,buh->bgh", G, fu)                      # [B,4,ds]
        heat = self.sc_mlp(e_p[:, None] + self.tok_proj(Tg)[:, :, None]).squeeze(-1)
        return logits, reg, None, heat                                # heat [B,4,P]


def stage2_loss(logits, reg, heat, batch, hm_weight=1.0):
    total, metrics = stage1_loss(logits, reg, None, batch)
    ug, tgt, mask = batch["hm_unit_group"], batch["hm_target"], batch["hm_mask"]
    n_g = batch["hm_n_groups"]
    soft = batch.get("hm_soft")
    hm_ce, hits, hits_n = [], 0, 0
    for g in range(4):
        rows = n_g > g
        if not rows.any():
            continue
        lg = heat[rows, g].masked_fill(~mask[rows, g], float("-inf"))
        t = tgt[rows, g]
        if soft is None:
            hm_ce.append(F.cross_entropy(lg, t, reduction="mean"))
        else:
            logp = lg.log_softmax(-1).masked_fill(~mask[rows, g], 0.0)
            hm_ce.append(-(soft[rows, g] * logp).sum(-1).mean())
        top8 = lg.topk(min(8, lg.shape[-1]), -1).indices
        hits += int((top8 == t[:, None]).any(-1).sum())
        hits_n += int(rows.sum())
    if hm_ce:
        total = total + hm_weight * torch.stack(hm_ce).mean()
        metrics["hm_ce"] = float(torch.stack(hm_ce).mean())
        metrics["hm_top8"] = hits / max(hits_n, 1)
        metrics["cnt_hm"] = hits_n
    return total, metrics
