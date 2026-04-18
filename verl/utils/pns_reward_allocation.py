"""PNS-Based Baseline + Normalized Surplus Reward Redistribution.

Implements budget-preserving step-level reward redistribution for reasoning RL / GRPO.
Given a trajectory-level final reward R and step-level PNS scores, redistributes R
into per-step rewards such that:
  1. Reward conservation holds: sum(r_t) = R
  2. Steps with higher PNS receive more reward
  3. Redistribution follows a normalized credit-sharing rule (no hand-tuned multipliers)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch

__all__ = [
    "expected_pns_score",
    "compute_normalized_surplus_shares",
    "allocate_reward",
    "pns_baseline_normalized_surplus_allocation",
    "PNSVariant",
    "compute_pns_diagnostics",
]


class PNSVariant(str, Enum):
    UNIFORM = "uniform"
    DIRECT_NORMALIZED = "direct_normalized"
    SURPLUS = "surplus"


# ---------------------------------------------------------------------------
# Core functions (spec §9)
# ---------------------------------------------------------------------------


def expected_pns_score(
    step_logits_or_probs: torch.Tensor,
    pns_values: Optional[torch.Tensor] = None,
    mode: str = "classification",
) -> torch.Tensor:
    """Convert PNS model outputs into scalar step scores (spec §9.1).

    Args:
        step_logits_or_probs: For regression mode, shape ``[T]`` (raw scores).
            For classification mode, shape ``[T, K]`` (probabilities or logits).
        pns_values: Numeric mapping for each class, shape ``[K]``.
            Required when ``mode="classification"``.
        mode: ``"regression"`` or ``"classification"``.

    Returns:
        Scalar step scores, shape ``[T]``.
    """
    if mode == "regression":
        return step_logits_or_probs.float()

    if mode != "classification":
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'regression' or 'classification'.")

    probs = step_logits_or_probs.float()
    if probs.dim() == 2 and probs.shape[-1] > 1:
        if not torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0], device=probs.device), atol=1e-3):
            probs = torch.softmax(probs, dim=-1)

    if pns_values is None:
        raise ValueError("pns_values is required for classification mode")
    v = pns_values.float().to(probs.device)
    return (probs * v.unsqueeze(0)).sum(dim=-1)


def compute_normalized_surplus_shares(
    scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute normalized surplus shares φ_t from scalar step scores (spec §9.2).

    Args:
        scores: Scalar PNS scores, shape ``[T]``.
        eps: Small constant for numerical stability.

    Returns:
        Normalized surplus shares, shape ``[T]``, summing to 1.
    """
    T = scores.shape[0]
    c_min = scores.min()
    surplus = (scores - c_min).clamp(min=0.0)
    total_surplus = surplus.sum()

    if total_surplus <= eps:
        return torch.full((T,), 1.0 / T, device=scores.device, dtype=scores.dtype)

    return surplus / total_surplus


def compute_direct_normalized_shares(
    scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ablation Variant B: direct normalized score shares (spec §14 Variant B).

    φ_t = c_t / sum(c_j).  Falls back to uniform if all scores are non-positive.
    """
    T = scores.shape[0]
    total = scores.sum()
    if total <= eps:
        return torch.full((T,), 1.0 / T, device=scores.device, dtype=scores.dtype)
    return scores / total


def allocate_reward(
    final_reward: torch.Tensor,
    shares: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Produce final per-step allocation from shares (spec §9.3).

    r_t = (1-α) * R/T + α * R * φ_t

    Args:
        final_reward: Scalar trajectory reward ``R`` (shape ``[]`` or ``[1]``).
        shares: Normalized shares, shape ``[T]``.
        alpha: Interpolation between uniform (0) and PNS-aware (1).

    Returns:
        Per-step rewards, shape ``[T]``.
    """
    R = final_reward.float().squeeze()
    T = shares.shape[0]
    uniform = 1.0 / T
    return (1.0 - alpha) * R * uniform + alpha * R * shares


def allocate_reward_uniform(
    final_reward: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Ablation Variant A: uniform allocation (spec §14 Variant A).

    r_t = R / T
    """
    R = final_reward.float().squeeze()
    return torch.full((T,), R / T, device=final_reward.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# End-to-end wrapper (spec §9.4)
# ---------------------------------------------------------------------------


def pns_baseline_normalized_surplus_allocation(
    final_reward: torch.Tensor,
    pns_output: torch.Tensor,
    mode: str = "classification",
    pns_values: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    eps: float = 1e-8,
    variant: str = "surplus",
) -> dict[str, torch.Tensor]:
    """End-to-end PNS reward redistribution (spec §9.4 + §14).

    Args:
        final_reward: Scalar trajectory reward ``R``.
        pns_output: Raw PNS model output.  Shape ``[T]`` (regression)
            or ``[T, K]`` (classification).
        mode: ``"regression"`` or ``"classification"``.
        pns_values: Class-to-value mapping for classification, shape ``[K]``.
        alpha: Interpolation coefficient in ``[0, 1]``.
        eps: Numerical stability constant.
        variant: One of ``"uniform"``, ``"direct_normalized"``, ``"surplus"``.

    Returns:
        Dict with ``"scores"``, ``"shares"``, ``"step_rewards"`` (all shape ``[T]``).
    """
    scores = expected_pns_score(pns_output, pns_values=pns_values, mode=mode)

    variant_enum = PNSVariant(variant)
    T = scores.shape[0]

    if variant_enum == PNSVariant.UNIFORM:
        shares = torch.full((T,), 1.0 / T, device=scores.device, dtype=scores.dtype)
        step_rewards = allocate_reward_uniform(final_reward, T)
    elif variant_enum == PNSVariant.DIRECT_NORMALIZED:
        shares = compute_direct_normalized_shares(scores, eps=eps)
        step_rewards = allocate_reward(final_reward, shares, alpha=alpha)
    elif variant_enum == PNSVariant.SURPLUS:
        shares = compute_normalized_surplus_shares(scores, eps=eps)
        step_rewards = allocate_reward(final_reward, shares, alpha=alpha)
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return {
        "scores": scores,
        "shares": shares,
        "step_rewards": step_rewards,
    }


# ---------------------------------------------------------------------------
# Batch version (operates on a list of trajectories with varying step counts)
# ---------------------------------------------------------------------------


def pns_allocation_batch(
    final_rewards: torch.Tensor,
    pns_outputs: list[torch.Tensor],
    mode: str = "classification",
    pns_values: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    eps: float = 1e-8,
    variant: str = "surplus",
) -> list[dict[str, torch.Tensor]]:
    """Batch version: apply PNS allocation per trajectory.

    Args:
        final_rewards: Shape ``[B]``, one scalar reward per trajectory.
        pns_outputs: List of length ``B``, each element is a tensor of shape
            ``[T_i]`` (regression) or ``[T_i, K]`` (classification).
        Others: same as :func:`pns_baseline_normalized_surplus_allocation`.

    Returns:
        List of dicts, one per trajectory.
    """
    B = final_rewards.shape[0]
    assert len(pns_outputs) == B
    results = []
    for i in range(B):
        results.append(
            pns_baseline_normalized_surplus_allocation(
                final_reward=final_rewards[i],
                pns_output=pns_outputs[i],
                mode=mode,
                pns_values=pns_values,
                alpha=alpha,
                eps=eps,
                variant=variant,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Diagnostics / logging (spec §13)
# ---------------------------------------------------------------------------


def compute_pns_diagnostics(
    all_scores: list[torch.Tensor],
    all_shares: list[torch.Tensor],
    all_step_rewards: list[torch.Tensor],
    all_final_rewards: torch.Tensor,
) -> dict[str, float]:
    """Compute per-batch diagnostics for PNS redistribution (spec §13).

    Returns a dict of scalar metrics suitable for logging.
    """
    scores_flat = torch.cat(all_scores)
    shares_flat = torch.cat(all_shares)
    step_rewards_flat = torch.cat(all_step_rewards)

    metrics = {
        "pns/score_mean": scores_flat.mean().item(),
        "pns/score_std": scores_flat.std().item() if scores_flat.numel() > 1 else 0.0,
    }

    surplus_values = []
    for s in all_scores:
        surplus_values.append((s - s.min()).clamp(min=0.0))
    surplus_flat = torch.cat(surplus_values)
    metrics["pns/surplus_mean"] = surplus_flat.mean().item()
    metrics["pns/surplus_std"] = surplus_flat.std().item() if surplus_flat.numel() > 1 else 0.0

    entropy_list = []
    max_share_list = []
    uniform_fallback_count = 0
    for shares in all_shares:
        log_shares = torch.log(shares.clamp(min=1e-12))
        entropy_list.append(-(shares * log_shares).sum().item())
        max_share_list.append(shares.max().item())
        T = shares.shape[0]
        if torch.allclose(shares, torch.full_like(shares, 1.0 / T), atol=1e-6):
            uniform_fallback_count += 1

    metrics["pns/share_entropy_mean"] = sum(entropy_list) / max(len(entropy_list), 1)
    metrics["pns/max_share_mean"] = sum(max_share_list) / max(len(max_share_list), 1)
    metrics["pns/uniform_fallback_pct"] = uniform_fallback_count / max(len(all_shares), 1)

    if scores_flat.numel() == step_rewards_flat.numel() and scores_flat.numel() > 1:
        vx = scores_flat - scores_flat.mean()
        vy = step_rewards_flat - step_rewards_flat.mean()
        denom = vx.norm() * vy.norm()
        corr = (vx * vy).sum() / denom.clamp(min=1e-12)
        metrics["pns/score_reward_correlation"] = corr.item()

    return metrics
