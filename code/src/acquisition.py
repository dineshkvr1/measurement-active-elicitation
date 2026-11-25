"""Acquisition policies for the streaming label budget.

For each subject--benchmark pair the predictor may purchase ground-truth labels
on acquisition-pool items, under the nested budget schedule {0,1,3,7,15,31}, to
sharpen its ability posterior and improve predictions on a disjoint, frozen
evaluation pool.  Each policy scores the remaining candidate items; the harness
buys the top ones up to the budget.

Four policies, from the pre-analysis plan:

* ``random``                -- the competition default (no information used).
* ``uncertainty``           -- our prior heuristic: prefer |p - 0.5| small.
* ``fisher``                -- p(1-p), the Fisher information one label carries
                               about ability (maximised at p = 0.5).
* ``expected_brier``        -- novel: expected reduction in Brier over the frozen
                               evaluation pool, the quantity the ALC metric
                               actually rewards.

All scores are "higher = acquire first".
"""

from __future__ import annotations

import numpy as np

from irt import ability_posterior, predict_prob, sigmoid


def score_random(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform random priority (competition default policy)."""
    return rng.random(n)


def score_uncertainty(p_cand: np.ndarray) -> np.ndarray:
    """Prior heuristic: most uncertain items (prediction nearest 0.5) first.

    Returned as 0.5 - |p - 0.5| so that larger = closer to 0.5 = acquire first.
    """
    return 0.5 - np.abs(p_cand - 0.5)


def score_fisher(p_cand: np.ndarray) -> np.ndarray:
    """Fisher information p(1-p) that one logistic label carries about ability.

    Maximised at p = 0.5.  This is the principled form the uncertainty heuristic
    only approximates.
    """
    return p_cand * (1.0 - p_cand)


def score_expected_brier(
    theta_map: float,
    post_var: float,
    betas_obs: np.ndarray,
    labels_obs: np.ndarray,
    beta_cand: np.ndarray,
    beta_eval: np.ndarray,
    prior_mean: float,
    prior_var: float,
) -> np.ndarray:
    """Expected reduction in evaluation-pool Brier from buying each candidate.

    A candidate item's label is unknown, so we average over its two possible
    outcomes (weighted by the current predicted probability).  For each outcome
    we refit the ability posterior, re-predict the frozen evaluation pool, and
    measure how much the pool's *expected* Brier score falls.  Because every
    evaluation item shares the subject's ability, a label that sharpens the
    ability posterior improves all of them at once --- this is exactly what the
    area-under-the-Brier-curve metric rewards.

    Returns an array of expected Brier *reductions* (higher = acquire first).
    """
    beta_cand = np.asarray(beta_cand, dtype=float)
    beta_eval = np.asarray(beta_eval, dtype=float)

    # current expected Brier on the eval pool under the present posterior
    p_eval = predict_prob(theta_map, post_var, beta_eval)
    # E[(p - Y)^2] = p(1-p) when Y ~ Bernoulli(p_true); using self-consistent
    # plug-in p_true = p_eval gives the posterior-expected Brier p(1-p).
    base = np.mean(p_eval * (1.0 - p_eval))

    p_cand = predict_prob(theta_map, post_var, beta_cand)
    gains = np.empty(beta_cand.size)

    for k, bc in enumerate(beta_cand):
        exp_brier = 0.0
        for outcome, w in ((1.0, p_cand[k]), (0.0, 1.0 - p_cand[k])):
            betas_new = np.append(betas_obs, bc)
            labels_new = np.append(labels_obs, outcome)
            th, pv = ability_posterior(betas_new, labels_new,
                                       prior_mean, prior_var)
            p_eval_new = predict_prob(th, pv, beta_eval)
            exp_brier += w * np.mean(p_eval_new * (1.0 - p_eval_new))
        gains[k] = base - exp_brier

    return gains


POLICIES = ("random", "uncertainty", "fisher", "expected_brier")
