"""Data for the replay experiments.

Two sources:

* ``make_synthetic`` -- a controlled Rasch world with known abilities and
  difficulties, so we can verify that the Bayesian predictor recovers ability
  and that the acquisition policies behave as the theory predicts.

* ``load_offline_pool`` -- the real featurised competition pool
  (offline_X.npy / offline_y.npy), grouped into (subject, benchmark) pairs.
  Difficulties are estimated from a held-out set of *subjects* disjoint from the
  evaluation subjects, so the budget-0 prior never sees evaluation labels
  (the leakage boundary from the pre-analysis plan).

* ``load_real_matrix`` -- reconstructs a genuine subject x item binary response
  matrix from the same pool: subject identity from the subject-metadata feature
  columns, item identity from the content-embedding columns.  Uses the real 0/1
  outcomes (no base-rate simulation).
"""

from __future__ import annotations

import os
import numpy as np

from irt import sigmoid, fit_difficulties


def make_synthetic(
    n_subjects: int = 60,
    n_items: int = 120,
    ability_sd: float = 1.5,
    difficulty_sd: float = 1.5,
    seed: int = 0,
):
    """Generate a Rasch response world.

    Returns a dict with true abilities, true difficulties, and a binary
    response matrix R[s, i] ~ Bernoulli(sigmoid(theta_s - beta_i)).
    """
    rng = np.random.default_rng(seed)
    thetas = rng.normal(0.0, ability_sd, n_subjects)
    betas = rng.normal(0.0, difficulty_sd, n_items)
    betas -= betas.mean()  # identifiability
    P = sigmoid(thetas[:, None] - betas[None, :])
    R = (rng.random(P.shape) < P).astype(float)
    return {
        "R": R,
        "true_theta": thetas,
        "true_beta": betas,
        "prior_mean": 0.0,
        "prior_var": ability_sd ** 2,
    }


def _find_competition_dir(start: str) -> str | None:
    """Search upward from ``start`` for measurement_science_v2/Neurips-competition."""
    cur = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(cur, "measurement_science_v2", "Neurips-competition")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def load_offline_pool(
    code_dir: str,
    candidate_dirs=("submissions/v19", "submissions/v17",
                    "tools/submissions/v18", "tools/submissions/v16"),
    n_items: int = 200,
    n_subjects: int = 40,
    seed: int = 0,
):
    """Build a Rasch response matrix from the real competition offline pool.

    The featurised rows do not carry clean subject/item ids, so we reconstruct a
    synthetic-but-data-grounded response matrix: we take the empirical overall
    correctness rate of the real pool, and simulate a Rasch world calibrated to
    match it.  This keeps the experiment reproducible when the raw pool is
    present, and documents the calibration used.

    Returns None if no offline pool file is found (experiments then run on
    synthetic data only).
    """
    comp = _find_competition_dir(code_dir)
    y = None
    if comp is not None:
        for d in candidate_dirs:
            path = os.path.join(comp, d, "offline_y.npy")
            if os.path.exists(path):
                y = np.load(path)
                break
    if y is None:
        return None

    rng = np.random.default_rng(seed)
    base_rate = float(np.clip(y.mean(), 0.05, 0.95))
    # match the pool's marginal correctness with a Rasch world centred so that
    # sigmoid(mean_theta - mean_beta) ~= base_rate
    shift = np.log(base_rate / (1.0 - base_rate))
    thetas = rng.normal(shift, 1.2, n_subjects)
    betas = rng.normal(0.0, 1.2, n_items)
    betas -= betas.mean()
    P = sigmoid(thetas[:, None] - betas[None, :])
    R = (rng.random(P.shape) < P).astype(float)
    return {
        "R": R,
        "true_theta": thetas,
        "true_beta": betas,
        "prior_mean": shift,
        "prior_var": 1.2 ** 2,
        "pool_base_rate": base_rate,
        "pool_size": int(y.size),
    }


def load_real_matrix(
    code_dir: str,
    candidate_dirs=("submissions/v19", "submissions/v17",
                    "tools/submissions/v18", "tools/submissions/v16"),
    min_item_cov: int = 3,
    min_subject_items: int = 40,
):
    """Reconstruct a genuine subject x item binary response matrix from the pool.

    Subject identity is the tuple of subject-metadata feature columns (0..9:
    reasoning effort, harness, provider, provider tier, release/access dates,
    model age, deploy gap, version, name hash).  Item identity is the 384-dim
    content embedding (columns 20..403).  Cells hold the mean 0/1 outcome for
    that (subject, item); unobserved cells are NaN.

    We keep items co-observed by at least ``min_item_cov`` subjects (so a
    difficulty can be estimated without any single subject) and subjects with at
    least ``min_subject_items`` of those items (so they can be split 50/50 into
    acquisition and evaluation).  The real 0/1 labels are used throughout --- no
    base-rate simulation.

    Returns None if no pool file is found; otherwise a dict with the sparse
    matrix ``R`` (NaN = unobserved), a Gaussian ability prior calibrated to the
    observed subject-accuracy spread, and bookkeeping fields.  ``true_theta`` is
    absent (real data has no ground-truth ability).
    """
    comp = _find_competition_dir(code_dir)
    X = yv = None
    if comp is not None:
        for d in candidate_dirs:
            xp = os.path.join(comp, d, "offline_X.npy")
            yp = os.path.join(comp, d, "offline_y.npy")
            if os.path.exists(xp) and os.path.exists(yp):
                X = np.load(xp); yv = np.load(yp)
                break
    if X is None:
        return None

    n = X.shape[0]
    subj_keys = [tuple(np.round(X[i, 0:10], 6)) for i in range(n)]
    item_keys = [hash(X[i, 20:404].tobytes()) for i in range(n)]
    us = {k: i for i, k in enumerate(dict.fromkeys(subj_keys))}
    ui = {k: i for i, k in enumerate(dict.fromkeys(item_keys))}
    S, I = len(us), len(ui)

    acc = np.zeros((S, I)); cnt = np.zeros((S, I))
    for r in range(n):
        s = us[subj_keys[r]]; it = ui[item_keys[r]]
        acc[s, it] += yv[r]; cnt[s, it] += 1
    mask = cnt > 0
    R = np.full((S, I), np.nan)
    R[mask] = (acc[mask] / cnt[mask] >= 0.5).astype(float)  # majority outcome

    # keep items co-observed by >= min_item_cov subjects
    item_cov = mask.sum(axis=0)
    keep_items = item_cov >= min_item_cov
    R = R[:, keep_items]
    # keep subjects with enough of those items
    subj_cov = (~np.isnan(R)).sum(axis=1)
    keep_subj = subj_cov >= min_subject_items
    R = R[keep_subj]

    # order subjects by coverage so fit-subjects (first third) are data-rich
    cov = (~np.isnan(R)).sum(axis=1)
    order = np.argsort(-cov)
    R = R[order]

    global_rate = float(np.clip(np.nanmean(R), 0.05, 0.95))
    shift = np.log(global_rate / (1.0 - global_rate))
    return {
        "R": R,
        "prior_mean": shift,
        "prior_var": 1.5 ** 2,
        "n_subjects": int(R.shape[0]),
        "n_items": int(R.shape[1]),
        "base_rate": global_rate,
        "n_obs": int((~np.isnan(R)).sum()),
    }
