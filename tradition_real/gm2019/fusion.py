"""Sless et al. (2019), section 3.1, equations (3)-(4), classical baseline.

This is a paper-based implementation, NOT the authors' neural Occupancy Net.
Hit/free probabilities and finite temporal window are project parameters;
the paper does not specify values sufficient for an exact baseline replay.
"""
import numpy as np
from tradition_real.autoware.costmap import FREE_SPACE, LETHAL_OBSTACLE


def logit(probability):
    p = np.asarray(probability, dtype=np.float64)
    if not np.all((p > 0) & (p < 1)):
        raise ValueError("Probability must be strictly between zero and one")
    return np.log(p) - np.log1p(-p)


class LogOddsFusion:
    def __init__(self, shape, p_hit=.7, p_free=.35, prior=.5):
        if not 0 < p_free < prior < p_hit < 1:
            raise ValueError("Expected 0 < p_free < prior < p_hit < 1")
        self.prior_log_odds = float(logit(prior))
        self.hit_increment = float(logit(p_hit)) - self.prior_log_odds
        self.free_increment = float(logit(p_free)) - self.prior_log_odds
        self.log_odds = np.full(shape, self.prior_log_odds, np.float64)
        self.observed = np.zeros(shape, bool)

    def update(self, costs):
        costs = np.asarray(costs)
        if costs.shape != self.log_odds.shape:
            raise ValueError("Evidence shape does not match fusion grid")
        if not np.isin(costs, [0, 128, 255]).all():
            raise ValueError("Expected ternary Autoware evidence")
        hit, free = costs == LETHAL_OBSTACLE, costs == FREE_SPACE
        self.log_odds[hit] += self.hit_increment
        self.log_odds[free] += self.free_increment
        self.observed |= hit | free
        # Unknown contributes no evidence and does not decay the posterior.

    @property
    def probability(self):
        # Numerically stable sigmoid without altering/clipping the stored odds.
        positive = self.log_odds >= 0
        result = np.empty_like(self.log_odds)
        result[positive] = 1 / (1 + np.exp(-self.log_odds[positive]))
        exp = np.exp(self.log_odds[~positive])
        result[~positive] = exp / (1 + exp)
        return result
