"""
GraspContractionNet — MVP

Architecture (plan_for_network_1.md §8):
  ObjectTokenEncoder      patch positions + normals → [B, P, d]
  ContactTokenEncoder     contact positions + normals + finger_id → [B, N, d]
  InteractionTransformer  joint transformer over [object | contact] tokens
  AnchorQueryDecoder      M learnable queries × cross-attention → [B, M, d]
  PatchHeatmapHead        dot-product logits → softmax weighted patch positions
  AssignmentHead          soft assignment anchor → contact  [B, M, N]

Output dict (plan §12):
  anchor_patch_logits         [B, M, P]
  anchor_positions            [B, M, 3]
  human_to_anchor_assignment  [B, M, N]
"""

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ────────────────────────────────────────────────────────────────────────────

class ObjectTokenEncoder(nn.Module):
    """Encode object patch (position + normal + curvature) into a feature token."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 64),  nn.LayerNorm(64),  nn.ReLU(),   # pos(3)+normal(3)+curv(1)
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, d_model),
        )

    def forward(self, patch_positions: torch.Tensor, patch_normals: torch.Tensor,
                patch_curvature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_positions: [B, P, 3]
            patch_normals:   [B, P, 3]
            patch_curvature: [B, P]    normal-variance curvature proxy
        Returns:
            [B, P, d_model]
        """
        x = torch.cat([patch_positions, patch_normals,
                        patch_curvature.unsqueeze(-1)], dim=-1)  # [B, P, 7]
        return self.mlp(x)


class ContactTokenEncoder(nn.Module):
    """Encode finger contact (position + normal + finger identity) into a token.

    hand_joints ([B, N_fingers, joints_per_finger, 3]) can be added in V2
    as a per-finger MANO pose feature. Omitted in MVP.
    """

    def __init__(self, d_model: int = 256, num_fingers: int = 5,
                 d_finger: int = 16):
        super().__init__()
        self.finger_embed = nn.Embedding(num_fingers, d_finger)
        self.mlp = nn.Sequential(
            nn.Linear(6 + d_finger, 64),  nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, d_model),
        )

    def forward(self, contact_positions: torch.Tensor,
                contact_normals: torch.Tensor,
                finger_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            contact_positions: [B, N, 3]
            contact_normals:   [B, N, 3]
            finger_ids:        [B, N]  int64
        Returns:
            [B, N, d_model]
        """
        f = self.finger_embed(finger_ids)                              # [B, N, d_finger]
        x = torch.cat([contact_positions, contact_normals, f], dim=-1)
        return self.mlp(x)


class InteractionTransformer(nn.Module):
    """Joint transformer over concatenated [object | contact] tokens.

    Lets each token attend to both object geometry and human contacts,
    producing contact-aware object tokens and object-aware contact tokens.
    """

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, obj_tokens: torch.Tensor,
                contact_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_tokens:     [B, P, d]
            contact_tokens: [B, N, d]
        Returns:
            obj_tokens:     [B, P, d]  (now contact-aware)
            contact_tokens: [B, N, d]  (now object-aware)
        """
        P = obj_tokens.shape[1]
        tokens = torch.cat([obj_tokens, contact_tokens], dim=1)  # [B, P+N, d]
        tokens = self.transformer(tokens)
        return tokens[:, :P], tokens[:, P:]


class AnchorQueryDecoder(nn.Module):
    """DETR-style cross-attention decoder with M learnable anchor queries.

    Each query slot learns to find one contracted robot contact anchor.
    """

    def __init__(self, d_model: int = 256, num_anchors: int = 3,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_anchors, d_model) * 0.02)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4), nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, obj_tokens: torch.Tensor,
                contact_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_tokens:     [B, P, d]
            contact_tokens: [B, N, d]
        Returns:
            anchor_tokens:  [B, M, d]
        """
        B = obj_tokens.shape[0]
        kv = torch.cat([obj_tokens, contact_tokens], dim=1)       # [B, P+N, d]
        q = self.queries.unsqueeze(0).expand(B, -1, -1).clone()   # [B, M, d]

        for attn, ffn, norm in zip(self.cross_attn_layers, self.ffn_layers, self.norms):
            q_attn, _ = attn(q, kv, kv)
            q = norm(q + q_attn)
            q = q + ffn(q)

        return q  # [B, M, d]


class PatchHeatmapHead(nn.Module):
    """Predict heatmap over object patches → aggregate anchor positions.

    Uses dot-product between anchor tokens and object tokens so the head
    is resolution-agnostic (works for any P without re-training).
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.anchor_proj = nn.Linear(d_model, d_model)
        self.patch_proj  = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, anchor_tokens: torch.Tensor,
                obj_tokens: torch.Tensor,
                patch_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            anchor_tokens:   [B, M, d]
            obj_tokens:      [B, P, d]
            patch_positions: [B, P, 3]
        Returns:
            logits:    [B, M, P]  raw scores
            positions: [B, M, 3]  softmax-weighted patch positions
        """
        a = self.anchor_proj(anchor_tokens)  # [B, M, d]
        p = self.patch_proj(obj_tokens)      # [B, P, d]
        logits = torch.bmm(a, p.transpose(1, 2)) * self.scale  # [B, M, P]
        weights = torch.softmax(logits, dim=-1)                 # [B, M, P]
        positions = torch.einsum('bmp,bpd->bmd', weights, patch_positions)
        return logits, positions


class AssignmentHead(nn.Module):
    """Soft assignment: how much each anchor represents each human contact.

    S[m, j] = softmax over j of  (anchor_m · contact_j)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.anchor_proj  = nn.Linear(d_model, d_model)
        self.contact_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, anchor_tokens: torch.Tensor,
                contact_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor_tokens:  [B, M, d]
            contact_tokens: [B, N, d]
        Returns:
            [B, M, N]  soft assignment weights (sum-to-1 over N)
        """
        a = self.anchor_proj(anchor_tokens)          # [B, M, d]
        c = self.contact_proj(contact_tokens)        # [B, N, d]
        logits = torch.bmm(a, c.transpose(1, 2)) * self.scale  # [B, M, N]
        return torch.softmax(logits, dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# Full network
# ────────────────────────────────────────────────────────────────────────────

class GraspContractionNet(nn.Module):
    """MVP Grasp Contraction Network.

    Maps dense human finger contacts + object surface patches →
    M sparse contracted robot contact anchors.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_anchors: int = 3,
        num_fingers: int = 5,
        d_finger: int = 16,
        nhead: int = 4,
        interaction_layers: int = 3,
        decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_anchors = num_anchors

        self.obj_encoder     = ObjectTokenEncoder(d_model)
        self.contact_encoder = ContactTokenEncoder(d_model, num_fingers, d_finger)
        self.interaction     = InteractionTransformer(d_model, nhead, interaction_layers, dropout)
        self.decoder         = AnchorQueryDecoder(d_model, num_anchors, nhead, decoder_layers)
        self.heatmap_head    = PatchHeatmapHead(d_model)
        self.assignment_head = AssignmentHead(d_model)

    def forward(self, batch: dict) -> dict:
        """
        Args (all tensors on the same device):
            patch_positions:   [B, P, 3]
            patch_normals:     [B, P, 3]
            patch_curvature:   [B, P]
            contact_positions: [B, N, 3]
            contact_normals:   [B, N, 3]
            finger_ids:        [B, N]  int64

        Returns:
            anchor_patch_logits:        [B, M, P]
            anchor_positions:           [B, M, 3]
            human_to_anchor_assignment: [B, M, N]
        """
        patch_pos  = batch["patch_positions"]    # [B, P, 3]
        patch_nor  = batch["patch_normals"]      # [B, P, 3]
        patch_curv = batch["patch_curvature"]    # [B, P]
        contact_pos = batch["contact_positions"] # [B, N, 3]
        contact_nor = batch["contact_normals"]   # [B, N, 3]
        finger_ids  = batch["finger_ids"]        # [B, N]

        obj_tokens     = self.obj_encoder(patch_pos, patch_nor, patch_curv)   # [B, P, d]
        contact_tokens = self.contact_encoder(contact_pos, contact_nor, finger_ids)  # [B, N, d]

        obj_tokens, contact_tokens = self.interaction(obj_tokens, contact_tokens)

        anchor_tokens = self.decoder(obj_tokens, contact_tokens)              # [B, M, d]

        logits, positions = self.heatmap_head(anchor_tokens, obj_tokens, patch_pos)
        assignment = self.assignment_head(anchor_tokens, contact_tokens)

        return {
            "anchor_patch_logits":        logits,      # [B, M, P]
            "anchor_positions":           positions,   # [B, M, 3]
            "human_to_anchor_assignment": assignment,  # [B, M, N]
        }
