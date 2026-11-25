"""Rasch item-response model with a Bayesian (MAP / Laplace) posterior over
subject ability.

The competition task --- predict whether a subject (an AI system) answers an
item correctly --- is modelled as a Rasch interaction

    P(correct | subject s, item i) = sigmoid(theta_s - beta_i),

where theta_s is the subject's latent ability and beta_i the item's difficulty.
Difficulties are estimated once (cold start); ability is updated online as
ground-truth labels are purchased under the acquisition budget.

Given labels for a subject, we place a Gaussian prior theta ~ N(mu, sigma^2) and
compute the MAP estimate by Newton's method, plus the Laplace (second-order)
posterior variance.  The posterior *mean* drives the prediction; the posterior
*variance* drives acquisition.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def ability_posterior(
    betas: np.ndarray,
    labels: np.ndarray,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    n_newton: int = 50,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """MAP estimate and Laplace posterior variance of a subject's ability.

    Maximises the log-posterior

        sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ] - (theta - mu)^2 / (2 var)

    with p_i = sigmoid(theta - beta_i), by Newton's method.

    Parameters
    ----------
    betas   : difficulties of the items whose labels were observed, shape (n,).
    labels  : 0/1 outcomes for those items, shape (n,).
    prior_mean, prior_var : Gaussian prior on ability.

    Returns
    -------
    (theta_map, post_var) where post_var is the inverse negative Hessian at the
    mode (the Laplace approximation to the posterior variance).

    With no labels the prior is returned unchanged, so this also handles the
    budget-0 cold-start case.
    """
    betas = np.asarray(betas, dtype=float)
    labels = np.asarray(labels, dtype=float)

    if betas.size == 0:
        return float(prior_mean), float(prior_var)

    theta = float(prior_mean)
    for _ in range(n_newton):
        p = sigmoid(theta - betas)
        # gradient of the log-posterior w.r.t. theta
        grad = np.sum(labels - p) - (theta - prior_mean) / prior_var
        # negative second derivative (observed information + prior precision)
        info = np.sum(p * (1.0 - p)) + 1.0 / prior_var
        step = grad / info
        theta += step
        if abs(step) < tol:
            break

    p = sigmoid(theta - betas)
    info = np.sum(p * (1.0 - p)) + 1.0 / prior_var
    post_var = 1.0 / info
    return float(theta), float(post_var)


def predict_prob(
    theta_map: float,
    post_var: float,
    beta_eval: np.ndarray | float,
    marginalise: bool = True,
) -> np.ndarray | float:
    """Predicted P(correct) for evaluation items of difficulty ``beta_eval``.

    If ``marginalise`` is True, integrate the logistic over the Gaussian ability
    posterior with the standard probit approximation

        E[sigmoid(theta - beta)] ~= sigmoid( (mu - beta) / sqrt(1 + pi/8 * var) ),

    which pulls predictions toward 0.5 when the posterior is wide (low budget).
    Otherwise use the plug-in posterior mean.
    """
    z = theta_map - np.asarray(beta_eval, dtype=float)
    if marginalise:
        z = z / np.sqrt(1.0 + (np.pi / 8.0) * post_var)
    return sigmoid(z)


def fit_difficulties(
    response_matrix: np.ndarray,
    n_iter: int = 200,
    lr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint MLE of Rasch abilities and difficulties from a response matrix.

    ``response_matrix`` is (n_subjects, n_items) of 0/1 (NaN = unobserved).
    Returns (thetas, betas).  Used to estimate item difficulties once, offline,
    before the online acquisition phase; identifiability is fixed by centring
    the difficulties at zero.
    """
    R = np.asarray(response_matrix, dtype=float)
    n_sub, n_item = R.shape
    mask = ~np.isnan(R)
    thetas = np.zeros(n_sub)
    betas = np.zeros(n_item)

    for _ in range(n_iter):
        # update abilities
        P = sigmoid(thetas[:, None] - betas[None, :])
        g_t = np.nansum(np.where(mask, R - P, 0.0), axis=1)
        h_t = np.nansum(np.where(mask, P * (1 - P), 0.0), axis=1) + 1e-6
        thetas += lr * g_t / h_t
        # update difficulties (note sign: d/dbeta = -(R - P))
        P = sigmoid(thetas[:, None] - betas[None, :])
        g_b = np.nansum(np.where(mask, -(R - P), 0.0), axis=0)
        h_b = np.nansum(np.where(mask, P * (1 - P), 0.0), axis=0) + 1e-6
        betas += lr * g_b / h_b
        betas -= betas.mean()  # centring constraint for identifiability

    return thetas, betas
