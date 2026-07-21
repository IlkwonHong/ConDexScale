"""
MVP losses for GraspContractionNet (plan_for_network_1.md §4, §8).

MVP losses:
  L_anchor   Hungarian-matched L2 to pseudo-label anchors
  L_surface  nearest patch distance (anchors should stay on object surface)
  L_spread   pairwise repulsion to prevent anchor collapse
  L_heatmap  KL divergence — supervise patch heatmap directly

Added in V2: L_intent_contact (OT), L_role
Added in V3: L_fc (differentiable force-existence QP)
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _hungarian_matching(pred_positions: torch.Tensor,
                        gt_positions: torch.Tensor,
                        anchor_mask: torch.Tensor) -> list[tuple]:
    """
    Compute Hungarian matching per batch item.

    Returns:
        list of (row_ind, col_ind) — pred slot indices and GT indices.
        Empty arrays when no valid GT anchors for that item.
    """
    B = pred_positions.shape[0]
    matchings = []
    for b in range(B):
        valid = anchor_mask[b]
        gt_valid = gt_positions[b][valid]   # [K, 3]
        K = gt_valid.shape[0]
        if K == 0:
            matchings.append(([], []))
            continue
        cost = torch.cdist(pred_positions[b], gt_valid, p=2)   # [M, K]
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        matchings.append((row_ind, col_ind))
    return matchings


def hungarian_anchor_loss(pred_positions: torch.Tensor,
                          gt_positions: torch.Tensor,
                          anchor_mask: torch.Tensor,
                          matchings: list[tuple] | None = None) -> torch.Tensor:
    """
    Hungarian-matched L2 loss between predicted and GT anchors.

    Args:
        pred_positions: [B, M, 3]  predicted anchor positions
        gt_positions:   [B, M, 3]  padded GT anchors (pseudo-labels)
        anchor_mask:    [B, M]     bool, True for valid GT anchors
        matchings:      pre-computed from _hungarian_matching (optional)

    Returns:
        scalar loss
    """
    if matchings is None:
        matchings = _hungarian_matching(pred_positions, gt_positions, anchor_mask)

    B = pred_positions.shape[0]
    total_loss = pred_positions.new_zeros(1)

    for b, (row_ind, col_ind) in enumerate(matchings):
        if len(row_ind) == 0:
            continue
        valid = anchor_mask[b]
        gt_valid = gt_positions[b][valid]   # [K, 3]
        K = gt_valid.shape[0]
        cost = torch.cdist(pred_positions[b], gt_valid, p=2)   # [M, K]
        matched_cost = cost[row_ind, col_ind].sum()
        total_loss = total_loss + matched_cost / K

    return total_loss / B


def heatmap_kl_loss(logits: torch.Tensor,
                    gt_positions: torch.Tensor,
                    patch_positions: torch.Tensor,
                    anchor_mask: torch.Tensor,
                    matchings: list[tuple],
                    temperature: float = 0.02) -> torch.Tensor:
    """
    KL divergence between predicted patch heatmap and GT-derived target distribution.

    For each Hungarian-matched (pred_slot, gt_anchor) pair:
      target = softmax(-||gt_anchor - patch_pos|| / temperature)   # peaked at nearest patch
      L      = KL(target || softmax(logits[slot]))

    Args:
        logits:          [B, M, P]  raw heatmap scores from PatchHeatmapHead
        gt_positions:    [B, M, 3]  padded GT anchors
        patch_positions: [B, P, 3]  object patch positions (canonical)
        anchor_mask:     [B, M]     bool
        matchings:       from _hungarian_matching — shared with l_anchor
        temperature:     bandwidth in metres; ~patch spacing (~0.01–0.02m)

    Returns:
        scalar loss
    """
    total_loss = logits.new_zeros(1)
    n_pairs = 0

    for b, (row_ind, col_ind) in enumerate(matchings):
        if len(row_ind) == 0:
            continue
        valid = anchor_mask[b]
        gt_valid = gt_positions[b][valid]   # [K, 3]
        patches = patch_positions[b]        # [P, 3]

        for pred_slot, gt_idx in zip(row_ind, col_ind):
            gt_anchor = gt_valid[gt_idx]                                            # [3]
            dists = torch.norm(patches - gt_anchor.unsqueeze(0), dim=-1)            # [P]
            target = torch.softmax(-dists / temperature, dim=-1)                    # [P]
            log_pred = torch.log_softmax(logits[b, pred_slot], dim=-1)              # [P]
            # KL(target || pred) = sum(target * (log target - log_pred))
            total_loss = total_loss + F.kl_div(log_pred, target, reduction="sum")
            n_pairs += 1

    return total_loss / max(n_pairs, 1)


def surface_loss(pred_positions: torch.Tensor,
                 patch_positions: torch.Tensor,
                 anchor_mask: torch.Tensor) -> torch.Tensor:
    """
    Encourage predicted anchors to lie on the object surface.
    Uses nearest-patch Euclidean distance as a surface proxy.

    Args:
        pred_positions:  [B, M, 3]
        patch_positions: [B, P, 3]
        anchor_mask:     [B, M] bool

    Returns:
        scalar loss
    """
    # Pairwise distances: [B, M, P]
    dists = torch.cdist(pred_positions, patch_positions, p=2)
    min_dists, _ = dists.min(dim=-1)          # [B, M]
    min_dists = min_dists * anchor_mask.float()
    n_valid = anchor_mask.float().sum().clamp(min=1)
    return min_dists.sum() / n_valid


def spread_loss(pred_positions: torch.Tensor,
                anchor_mask: torch.Tensor,
                sigma: float = 0.05) -> torch.Tensor:
    """
    Repulsion loss to prevent anchors from collapsing to the same region.

    L_spread = sum_{i≠j} exp(-d(i,j)^2 / sigma^2)

    Args:
        pred_positions: [B, M, 3]
        anchor_mask:    [B, M] bool
        sigma:          bandwidth (metres), default 5 cm

    Returns:
        scalar loss
    """
    B, M, _ = pred_positions.shape
    if M < 2:
        return pred_positions.new_zeros(1)

    # [B, M, M] pairwise squared distances
    diff = pred_positions.unsqueeze(2) - pred_positions.unsqueeze(1)
    sq_dist = (diff ** 2).sum(dim=-1)  # [B, M, M]

    # Mask diagonal and invalid anchors
    eye = torch.eye(M, device=pred_positions.device, dtype=torch.bool)
    sq_dist = sq_dist.masked_fill(eye.unsqueeze(0), float('inf'))

    # Valid pair mask: both anchors must be valid
    pair_mask = anchor_mask.unsqueeze(2) & anchor_mask.unsqueeze(1)  # [B, M, M]
    pair_mask = pair_mask & ~eye.unsqueeze(0)

    repulsion = torch.exp(-sq_dist / (sigma ** 2))
    repulsion = repulsion * pair_mask.float()

    n_pairs = pair_mask.float().sum().clamp(min=1)
    return repulsion.sum() / n_pairs


class GraspContractionLoss(torch.nn.Module):
    """Combined loss.

    L = λ_anchor * L_anchor + λ_surface * L_surface + λ_spread * L_spread
      + λ_heatmap * L_heatmap
    """

    def __init__(
        self,
        lambda_anchor:      float = 1.0,
        lambda_surface:     float = 0.5,
        lambda_spread:      float = 0.1,
        lambda_heatmap:     float = 0.1,
        surface_sigma:      float = 0.05,
        heatmap_temperature: float = 0.02,
    ):
        super().__init__()
        self.lambda_anchor       = lambda_anchor
        self.lambda_surface      = lambda_surface
        self.lambda_spread       = lambda_spread
        self.lambda_heatmap      = lambda_heatmap
        self.surface_sigma       = surface_sigma
        self.heatmap_temperature = heatmap_temperature

    def forward(self, pred: dict, batch: dict) -> dict:
        """
        Args:
            pred:  output of GraspContractionNet.forward()
            batch: dataloader batch (contains human_anchor, anchor_mask, patch_positions)

        Returns:
            dict with 'loss' (total) and individual terms for logging
        """
        pred_pos  = pred["anchor_positions"]          # [B, M, 3]
        logits    = pred["anchor_patch_logits"]       # [B, M, P]
        gt_pos    = batch["human_anchor"]             # [B, max_M, 3]
        mask      = batch["anchor_mask"]              # [B, max_M]
        patch_pos = batch["patch_positions"]          # [B, P, 3]

        # Compute Hungarian matching once — shared by l_anchor and l_heatmap
        matchings = _hungarian_matching(pred_pos, gt_pos, mask)

        l_anchor  = hungarian_anchor_loss(pred_pos, gt_pos, mask, matchings)
        l_surface = surface_loss(pred_pos, patch_pos, mask)
        l_spread  = spread_loss(pred_pos, mask, sigma=self.surface_sigma)
        l_heatmap = heatmap_kl_loss(logits, gt_pos, patch_pos, mask,
                                    matchings, self.heatmap_temperature)

        total = (self.lambda_anchor  * l_anchor
               + self.lambda_surface * l_surface
               + self.lambda_spread  * l_spread
               + self.lambda_heatmap * l_heatmap)

        return {
            "loss":      total,
            "l_anchor":  l_anchor.detach(),
            "l_surface": l_surface.detach(),
            "l_spread":  l_spread.detach(),
            "l_heatmap": l_heatmap.detach(),
        }
