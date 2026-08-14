"""Contrast strategies for ONLY decoding.

Extracted verbatim from the if/elif chain inside only_sample.sample() and
greedy_search(). Each strategy owns one way of combining the model's normal
logits with a second stream.

WHY THIS IS SAFE TO EXTRACT
    The arithmetic is pure tensor computation over two logit tensors — no
    model, no weights, no GPU. Only the decoding loop that calls it needs
    those. BenchyBench/tests/test_contrast_strategies.py asserts every
    strategy reproduces the original expression bitwise on random tensors.

HOW ONLY DIFFERS FROM DeGF
    DeGF obtains its second stream from a separate forward pass over a
    generated reference image. ONLY obtains it from the SAME forward pass, by
    intervening on one transformer layer:

        outputs, logits_cd = self(**model_inputs, ...)

    So the second stream costs nothing extra. That is the paper's efficiency
    claim.

    The gate also differs. DeGF switches on Jensen-Shannon divergence; ONLY
    switches on TOTAL VARIATION DISTANCE. The threshold is nevertheless named
    `js_gamma` throughout, inherited from an earlier JS-based formulation
    whose computation is still present, commented out, in only_sample.py. The
    name does not describe the metric — a real trap when tuning it.

CAUTION — this file determines published numbers. Restructuring around these
expressions is safe because it is tested; changing the arithmetic inside them
is not routine refactoring. See BenchyBench/PIPELINES.md first.
"""

import torch
from torch import nn


class ContrastStrategy:
    """Base class: combine two logit streams into corrected logits."""

    def combine(self, logits, logits_ref):
        raise NotImplementedError

    @staticmethod
    def plausibility_cutoff(logits, beta):
        """Threshold below which a token is excluded.

        Computed from the ORIGINAL logits, never the corrected ones, so the
        constraint reflects what the model found plausible BEFORE correction
        and the correction cannot promote a token it never considered.
        """
        return torch.log(torch.tensor(beta)) + logits.max(dim=-1, keepdim=True).values

    @staticmethod
    def apply_cutoff(corrected, logits, cutoff):
        """Mask tokens whose ORIGINAL logit falls below `cutoff`."""
        return corrected.masked_fill(logits < cutoff, -float("inf"))


class RitualContrast(ContrastStrategy):
    """Additive: amplify agreement with a positive reference view."""

    def __init__(self, alpha_pos):
        self.alpha_pos = alpha_pos

    def combine(self, logits, logits_ref):
        return (logits + self.alpha_pos * logits_ref)


class VCDContrast(ContrastStrategy):
    """Visual contrastive decoding: push away from a distorted view."""

    def __init__(self, alpha_neg):
        self.alpha_neg = alpha_neg

    def combine(self, logits, logits_ref):
        return (1 + self.alpha_neg) * logits - self.alpha_neg * logits_ref


class M3IDContrast(ContrastStrategy):
    """Contrastive with a schedule that STRENGTHENS over token position.

    The direction is the opposite of what "decay schedule" suggests. gamma_t
    decays, but appears as (1 - gamma_t)/gamma_t, which grows: ~0.02 at t=1,
    ~53.6 at t=200. A VLM's conditioning on the image fades as the generated
    text lengthens, so the visual correction is amplified to counteract that
    rather than backed off.

    Stateful — `t` advances on every call.
    """

    DECAY = -0.02

    def __init__(self, t=0):
        self.t = t

    def combine(self, logits, logits_ref):
        gamma_t = torch.exp(torch.tensor(self.DECAY * self.t))
        result = logits + (logits - logits_ref) * (1 - gamma_t) / gamma_t
        self.t += 1
        return result


class ONLYContrast(ContrastStrategy):
    """ONLY proper: switch direction per token on total variation distance.

    Structurally the same per-token sign switch as DeGF, but the distance is
    TVD rather than JS, and the second stream comes from a layer intervention
    in the same forward pass rather than a second pass:

        tvd <  gamma   streams agree    -> ADD the intervened stream
        tvd >= gamma   streams disagree -> SUBTRACT it

    Unlike DeGF's threshold, this one IS configurable (js_gamma in
    config.json), so it is a constructor argument rather than a class
    constant.

    Stateful: tvd_list accumulates across the generation for the run log.
    """

    def __init__(self, gamma, alpha_pos, alpha_neg):
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.tvd_list = []
        self.contrastive_count = 0
        self.token_count = 0

    @staticmethod
    def total_variation_distance(logits, logits_ref):
        """Sum of absolute differences between the two distributions.

        Note this is the L1 distance, i.e. twice the standard TVD, which is
        conventionally half the L1 norm. The threshold is calibrated against
        this scale, so do not "correct" it without recalibrating js_gamma.
        """
        return torch.sum(torch.abs(nn.functional.softmax(logits, dim=-1) - nn.functional.softmax(logits_ref, dim=-1)))

    def combine(self, logits, logits_ref):
        tvd = self.total_variation_distance(logits, logits_ref)
        self.tvd_list.append(tvd.item())
        self.token_count += 1

        if tvd < self.gamma:
            return logits + self.alpha_pos * logits_ref

        self.contrastive_count += 1
        return (1 + self.alpha_neg) * logits - self.alpha_neg * logits_ref
