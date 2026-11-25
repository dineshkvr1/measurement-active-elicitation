"""Offline replay harness: Brier learning curve and ALC under a label budget.

For each subject we split that subject's items into a 50/50 acquisition pool and
a frozen evaluation pool (the competition's design).  Difficulties are estimated
once from *other* subjects (no leakage).  For each policy we replay the nested
budget schedule {0,1,3,7,15,31}: at each budget we have purchased that many
acquisition labels (chosen by the policy), refit the ability posterior, and
score the frozen evaluation pool with the Brier score.  The Brier ALC is the
budget-weighted average of the curve.
"""

from __future__ import annotations

import numpy as np

from irt import ability_posterior, predict_prob, fit_difficulties
import acquisition as acq

BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with ten fixed bins."""
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / p.size) * abs(p[m].mean() - y[m].mean())
    return float(total)


def _acquire_order(policy, p_cand, theta, pv, betas_obs, labels_obs,
                   beta_cand, beta_eval, prior_mean, prior_var, rng):
    """Return candidate indices in acquisition priority order for one policy."""
    if policy == "random":
        s = acq.score_random(rng, beta_cand.size)
    elif policy == "uncertainty":
        s = acq.score_uncertainty(p_cand)
    elif policy == "fisher":
        s = acq.score_fisher(p_cand)
    elif policy == "expected_brier":
        s = acq.score_expected_brier(theta, pv, betas_obs, labels_obs,
                                     beta_cand, beta_eval,
                                     prior_mean, prior_var)
    else:
        raise ValueError(policy)
    return np.argsort(-s)  # highest score first


def replay_subject(R_row, betas, prior_mean, prior_var, policy, seed,
                   true_theta=None):
    """Brier (and optional ability-error) at each budget for one subject.

    ``R_row`` is the subject's 0/1 outcomes over all items; ``betas`` the
    item difficulties.  Items split 50/50 into acquisition / evaluation.
    If ``true_theta`` is given, also return |theta_hat - true_theta| per budget.
    """
    rng = np.random.default_rng(seed)
    n = R_row.size
    perm = rng.permutation(n)
    acq_idx, eval_idx = perm[: n // 2], perm[n // 2:]

    beta_cand = betas[acq_idx]
    y_cand = R_row[acq_idx]
    beta_eval = betas[eval_idx]
    y_eval = R_row[eval_idx]

    curve = []
    theta_err = []
    for budget in BUDGETS:
        if budget == 0:
            betas_obs = np.array([]); labels_obs = np.array([])
        else:
            # greedy nested acquisition: re-score and pick up to `budget` items
            theta, pv = ability_posterior(np.array([]), np.array([]),
                                          prior_mean, prior_var)
            chosen: list[int] = []
            remaining = list(range(beta_cand.size))
            while len(chosen) < min(budget, beta_cand.size):
                bo = beta_cand[chosen]; lo = y_cand[chosen]
                theta, pv = ability_posterior(bo, lo, prior_mean, prior_var)
                p_rem = predict_prob(theta, pv, beta_cand[remaining])
                order = _acquire_order(policy, p_rem, theta, pv, bo, lo,
                                       beta_cand[remaining], beta_eval,
                                       prior_mean, prior_var, rng)
                pick = remaining[order[0]]
                chosen.append(pick)
                remaining.remove(pick)
            betas_obs = beta_cand[chosen]; labels_obs = y_cand[chosen]

        theta, pv = ability_posterior(betas_obs, labels_obs,
                                      prior_mean, prior_var)
        p_eval = predict_prob(theta, pv, beta_eval)
        curve.append(brier(p_eval, y_eval))
        if true_theta is not None:
            theta_err.append(abs(theta - true_theta))

    if true_theta is not None:
        return np.array(curve), np.array(theta_err), y_eval, beta_eval
    return np.array(curve), y_eval, beta_eval


def replay_dataset(data, policy, n_fit_subjects=None, seed=0):
    """Replay every subject under one policy; return per-subject Brier curves.

    Difficulties are fit from the first ``n_fit_subjects`` subjects (held out
    from evaluation) so the budget-0 prior never sees evaluation labels.
    """
    R = data["R"]
    n_sub = R.shape[0]
    if n_fit_subjects is None:
        n_fit_subjects = n_sub // 3
    fit_rows = R[:n_fit_subjects]
    _, betas = fit_difficulties(fit_rows)

    eval_rows = R[n_fit_subjects:]
    curves = []
    for s, row in enumerate(eval_rows):
        c, _, _ = replay_subject(row, betas, data["prior_mean"],
                                 data["prior_var"], policy, seed + s)
        curves.append(c)
    return np.array(curves)  # (n_eval_subjects, 6)


def replay_dataset_ability(data, policy, n_fit_subjects=None, seed=0):
    """Mean |theta_hat - true_theta| at each budget under one policy.

    Requires ``data['true_theta']`` (synthetic worlds only).  Shows how fast
    each acquisition policy pins down the latent ability --- the quantity the
    policy actually controls, before it is filtered through the predictor.
    """
    R = data["R"]
    n_sub = R.shape[0]
    if n_fit_subjects is None:
        n_fit_subjects = n_sub // 3
    _, betas = fit_difficulties(R[:n_fit_subjects])
    eval_rows = R[n_fit_subjects:]
    true_theta = data["true_theta"][n_fit_subjects:]

    errs = []
    for s, row in enumerate(eval_rows):
        _, te, _, _ = replay_subject(row, betas, data["prior_mean"],
                                     data["prior_var"], policy, seed + s,
                                     true_theta=true_theta[s])
        errs.append(te)
    return np.array(errs)  # (n_eval_subjects, 6)


def replay_dataset_real(data, policy, n_fit_subjects=None, min_items=40, seed=0):
    """Replay on a genuine sparse response matrix (NaN = unobserved).

    Item difficulties are fit from held-out fit-subjects only; each evaluation
    subject is then replayed on its own observed items that have a fitted
    difficulty (leakage-free), split 50/50 into acquisition and evaluation.
    Subjects with too few usable items are skipped.  Returns per-subject Brier
    curves over the budget schedule, using the real 0/1 outcomes.
    """
    R = data["R"]
    n_sub = R.shape[0]
    if n_fit_subjects is None:
        n_fit_subjects = n_sub // 3

    # difficulties from fit-subjects only; items they never answered get NaN beta
    _, betas = fit_difficulties(R[:n_fit_subjects])
    fit_mask = (~np.isnan(R[:n_fit_subjects])).any(axis=0)
    betas = np.where(fit_mask, betas, np.nan)

    curves = []
    for s in range(n_fit_subjects, n_sub):
        row = R[s]
        usable = np.where((~np.isnan(row)) & (~np.isnan(betas)))[0]
        if usable.size < min_items:
            continue
        sub_betas = betas[usable]
        sub_y = row[usable]
        c, _, _ = replay_subject(sub_y, sub_betas, data["prior_mean"],
                                 data["prior_var"], policy, seed + s)
        curves.append(c)
    return np.array(curves)


def alc(curves: np.ndarray) -> np.ndarray:
    """Budget-weighted ALC per subject from Brier curves (n_subjects, 6)."""
    return curves @ ALC_WEIGHTS


def bootstrap_ci(values: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap mean and (1-alpha) CI over subjects."""
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.array([values[rng.integers(0, n, n)].mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
