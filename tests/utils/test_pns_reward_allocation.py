"""Tests for PNS reward allocation, step segmentation, and token mapping.

Covers the four sanity checks from spec §12, ablation variants (§14),
step segmentation strategies, and token-level mapping.
"""

import pytest
import torch

from verl.utils.pns_reward_allocation import (
    PNSVariant,
    allocate_reward,
    allocate_reward_uniform,
    compute_direct_normalized_shares,
    compute_normalized_surplus_shares,
    compute_pns_diagnostics,
    expected_pns_score,
    pns_baseline_normalized_surplus_allocation,
)
from verl.utils.pns_step_segmenter import StepSegment, segment_steps
from verl.utils.pns_token_mapping import (
    broadcast_step_rewards_to_tokens,
)


# ═══════════════════════════════════════════════════════════════════════
# §12 Sanity Checks
# ═══════════════════════════════════════════════════════════════════════


class TestSanityChecks:
    """Spec §12: four required sanity checks."""

    def test_reward_conservation(self):
        """Check 1: sum(r_t) ≈ R for every trajectory."""
        R = torch.tensor(5.0)
        scores = torch.tensor([0.2, 0.5, 0.8, 0.3])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=0.5)
        assert torch.allclose(step_rewards.sum(), R, atol=1e-6)

    def test_reward_conservation_negative(self):
        """Conservation also holds for negative rewards."""
        R = torch.tensor(-3.0)
        scores = torch.tensor([0.1, 0.9, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=0.7)
        assert torch.allclose(step_rewards.sum(), R, atol=1e-6)

    def test_uniform_fallback(self):
        """Check 2: identical scores → r_t = R/T for all t."""
        R = torch.tensor(4.0)
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=0.5)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.allclose(step_rewards, expected, atol=1e-6)

    def test_monotonicity(self):
        """Check 3: higher PNS score → higher reward share."""
        R = torch.tensor(10.0)
        scores = torch.tensor([0.1, 0.5, 0.9])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=0.8)
        assert step_rewards[2] > step_rewards[1] > step_rewards[0]

    def test_alpha_zero_is_uniform(self):
        """Check 4a: alpha=0 → fully uniform."""
        R = torch.tensor(6.0)
        scores = torch.tensor([0.1, 0.9, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=0.0)
        expected = torch.tensor([2.0, 2.0, 2.0])
        assert torch.allclose(step_rewards, expected, atol=1e-6)

    def test_alpha_one_is_fully_pns(self):
        """Check 4b: alpha=1 → fully determined by surplus shares."""
        R = torch.tensor(6.0)
        scores = torch.tensor([0.1, 0.9, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        step_rewards = allocate_reward(R, shares, alpha=1.0)
        expected = R * shares
        assert torch.allclose(step_rewards, expected, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Expected PNS Score
# ═══════════════════════════════════════════════════════════════════════


class TestExpectedPNSScore:
    def test_regression_passthrough(self):
        raw = torch.tensor([0.3, 0.7, 0.1])
        out = expected_pns_score(raw, mode="regression")
        assert torch.allclose(out, raw.float())

    def test_classification_expected_value(self):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        pns_values = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        out = expected_pns_score(probs, pns_values=pns_values, mode="classification")
        assert out.shape == (2,)
        expected_0 = 0.1 * 0.0 + 0.2 / 3 + 0.3 * 2 / 3 + 0.4 * 1.0
        assert abs(out[0].item() - expected_0) < 1e-5

    def test_classification_requires_pns_values(self):
        probs = torch.tensor([[0.5, 0.5]])
        with pytest.raises(ValueError, match="pns_values"):
            expected_pns_score(probs, mode="classification")


# ═══════════════════════════════════════════════════════════════════════
# Surplus Shares
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizedSurplusShares:
    def test_sums_to_one(self):
        scores = torch.tensor([0.2, 0.5, 0.8, 0.3])
        shares = compute_normalized_surplus_shares(scores)
        assert abs(shares.sum().item() - 1.0) < 1e-6

    def test_all_equal_gives_uniform(self):
        scores = torch.tensor([0.5, 0.5, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        assert torch.allclose(shares, torch.tensor([1 / 3, 1 / 3, 1 / 3]), atol=1e-6)

    def test_minimum_step_gets_zero_surplus(self):
        scores = torch.tensor([0.1, 0.5, 0.9])
        shares = compute_normalized_surplus_shares(scores)
        assert shares[0].item() < 1e-6

    def test_non_negative(self):
        scores = torch.tensor([-0.5, 0.0, 0.5])
        shares = compute_normalized_surplus_shares(scores)
        assert (shares >= 0).all()


# ═══════════════════════════════════════════════════════════════════════
# Ablation Variants (§14)
# ═══════════════════════════════════════════════════════════════════════


class TestAblationVariants:
    def test_variant_a_uniform(self):
        R = torch.tensor(9.0)
        result = pns_baseline_normalized_surplus_allocation(
            final_reward=R,
            pns_output=torch.tensor([0.1, 0.5, 0.9]),
            mode="regression",
            variant="uniform",
        )
        expected = torch.tensor([3.0, 3.0, 3.0])
        assert torch.allclose(result["step_rewards"], expected, atol=1e-6)
        assert torch.allclose(result["step_rewards"].sum(), R, atol=1e-6)

    def test_variant_b_direct_normalized(self):
        R = torch.tensor(6.0)
        scores = torch.tensor([1.0, 2.0, 3.0])
        result = pns_baseline_normalized_surplus_allocation(
            final_reward=R,
            pns_output=scores,
            mode="regression",
            variant="direct_normalized",
            alpha=1.0,
        )
        assert torch.allclose(result["step_rewards"].sum(), R, atol=1e-6)

    def test_variant_c_surplus_default(self):
        R = torch.tensor(10.0)
        scores = torch.tensor([0.2, 0.5, 0.8])
        result = pns_baseline_normalized_surplus_allocation(
            final_reward=R,
            pns_output=scores,
            mode="regression",
            variant="surplus",
            alpha=0.5,
        )
        assert torch.allclose(result["step_rewards"].sum(), R, atol=1e-6)
        assert result["step_rewards"][2] > result["step_rewards"][0]


# ═══════════════════════════════════════════════════════════════════════
# Direct Normalized Shares
# ═══════════════════════════════════════════════════════════════════════


class TestDirectNormalizedShares:
    def test_sums_to_one(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        shares = compute_direct_normalized_shares(scores)
        assert abs(shares.sum().item() - 1.0) < 1e-6

    def test_zero_scores_fallback(self):
        scores = torch.tensor([0.0, 0.0, 0.0])
        shares = compute_direct_normalized_shares(scores)
        assert torch.allclose(shares, torch.tensor([1 / 3, 1 / 3, 1 / 3]), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Step Segmentation
# ═══════════════════════════════════════════════════════════════════════


class TestStepSegmentation:
    def test_double_newline_basic(self):
        text = "Step one content\n\nStep two content\n\nStep three"
        segs = segment_steps(text, strategy="double_newline")
        assert len(segs) == 3
        assert segs[0].text == "Step one content"
        assert segs[1].text == "Step two content"
        assert segs[2].text == "Step three"

    def test_double_newline_no_delimiter(self):
        text = "Single block of text with no double newlines"
        segs = segment_steps(text, strategy="double_newline")
        assert len(segs) == 1

    def test_double_newline_preserves_offsets(self):
        text = "AAA\n\nBBB\n\nCCC"
        segs = segment_steps(text, strategy="double_newline")
        for seg in segs:
            assert text[seg.char_start : seg.char_end] == seg.text

    def test_step_marker(self):
        text = "Step 1: Do something\nStep 2: Do more\nStep 3: Finish"
        segs = segment_steps(text, strategy="step_marker")
        assert len(segs) == 3

    def test_think_tag(self):
        text = "<think>Let me think about this carefully</think>The answer is 42."
        segs = segment_steps(text, strategy="think_tag")
        assert len(segs) == 2
        assert "think" in segs[0].text.lower()
        assert "42" in segs[1].text

    def test_sentence_segmenter(self):
        text = "First sentence. Second sentence. Third sentence."
        segs = segment_steps(text, strategy="sentence")
        assert len(segs) == 3


# ═══════════════════════════════════════════════════════════════════════
# Token Mapping
# ═══════════════════════════════════════════════════════════════════════


class TestBroadcastStepRewards:
    def test_basic_broadcast(self):
        step_rewards = torch.tensor([1.0, 2.0, 3.0])
        spans = [(0, 3), (3, 7), (7, 10)]
        out = broadcast_step_rewards_to_tokens(step_rewards, spans, response_length=10)
        assert out.shape == (10,)
        assert (out[:3] == 1.0).all()
        assert (out[3:7] == 2.0).all()
        assert (out[7:10] == 3.0).all()

    def test_with_mask(self):
        step_rewards = torch.tensor([1.0, 2.0])
        spans = [(0, 3), (3, 6)]
        mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
        out = broadcast_step_rewards_to_tokens(step_rewards, spans, response_length=8, response_mask=mask)
        assert out[5] == 0.0
        assert out[0] == 1.0

    def test_uncovered_tokens_are_zero(self):
        step_rewards = torch.tensor([5.0])
        spans = [(2, 5)]
        out = broadcast_step_rewards_to_tokens(step_rewards, spans, response_length=10)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] == 5.0
        assert out[5] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════


class TestDiagnostics:
    def test_returns_expected_keys(self):
        scores = [torch.tensor([0.2, 0.5, 0.8])]
        shares = [compute_normalized_surplus_shares(scores[0])]
        step_rewards = [allocate_reward(torch.tensor(6.0), shares[0], alpha=0.5)]
        final_rewards = torch.tensor([6.0])

        diag = compute_pns_diagnostics(scores, shares, step_rewards, final_rewards)
        assert "pns/score_mean" in diag
        assert "pns/surplus_mean" in diag
        assert "pns/share_entropy_mean" in diag
        assert "pns/max_share_mean" in diag
        assert "pns/uniform_fallback_pct" in diag
        assert "pns/score_reward_correlation" in diag

    def test_uniform_fallback_detected(self):
        scores = [torch.tensor([0.5, 0.5, 0.5])]
        shares = [compute_normalized_surplus_shares(scores[0])]
        step_rewards = [allocate_reward(torch.tensor(3.0), shares[0], alpha=0.5)]
        final_rewards = torch.tensor([3.0])

        diag = compute_pns_diagnostics(scores, shares, step_rewards, final_rewards)
        assert diag["pns/uniform_fallback_pct"] == 1.0
