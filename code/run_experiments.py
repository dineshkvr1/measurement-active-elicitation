"""Run all experiments for the final project and save figures + tables.

Produces, under ``results/``:
  * learning_curves.png  -- Brier vs budget, one line per acquisition policy
  * calibration.png      -- reliability diagram at a low budget
  * ability_recovery.png -- estimated vs true ability (synthetic)
  * alc_table.csv        -- ALC with bootstrap 95% CIs per policy
  * results.json         -- all numbers used in the manuscript

Run:  python run_experiments.py
Everything is CPU-only and deterministic (fixed seeds); a full run is a few
minutes.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from irt import ability_posterior, predict_prob, fit_difficulties  # noqa: E402
import replay as rp  # noqa: E402
import acquisition as acq  # noqa: E402
import data as dat  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
SEED = 0


def run_policies(dataset, tag):
    out = {}
    for policy in acq.POLICIES:
        curves = rp.replay_dataset(dataset, policy, seed=SEED)
        a = rp.alc(curves)
        mean, lo, hi = rp.bootstrap_ci(a, seed=SEED)
        out[policy] = {
            "curve_mean": curves.mean(axis=0).tolist(),
            "curve_sd": curves.std(axis=0).tolist(),
            "alc_mean": mean,
            "alc_lo": lo,
            "alc_hi": hi,
        }
        print(f"[{tag}] {policy:15s} ALC={mean:.4f}  "
              f"95% CI=[{lo:.4f}, {hi:.4f}]")
    return out


def plot_learning_curves(res, tag):
    plt.figure(figsize=(5, 3.5))
    labels = {"random": "Random (default)",
              "uncertainty": "Uncertainty |p-0.5|",
              "fisher": "Fisher p(1-p)",
              "expected_brier": "Expected-Brier-reduction"}
    for policy in acq.POLICIES:
        y = res[policy]["curve_mean"]
        plt.plot(rp.BUDGETS, y, marker="o", label=labels[policy])
    plt.xlabel("Label budget")
    plt.ylabel("Brier score (lower is better)")
    plt.title(f"Brier learning curve ({tag})")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(RESULTS, f"learning_curves_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ability_recovery(dataset):
    """Estimated vs true ability at budget 15 (synthetic only)."""
    _, betas = fit_difficulties(dataset["R"][: dataset["R"].shape[0] // 3])
    eval_rows = dataset["R"][dataset["R"].shape[0] // 3:]
    est, true = [], dataset["true_theta"][dataset["R"].shape[0] // 3:]
    rng = np.random.default_rng(SEED)
    for row in eval_rows:
        n = row.size
        idx = rng.permutation(n)[:15]
        th, _ = ability_posterior(betas[idx], row[idx],
                                  dataset["prior_mean"], dataset["prior_var"])
        est.append(th)
    plt.figure(figsize=(3.6, 3.6))
    plt.scatter(true, est, s=18, alpha=0.7)
    lim = [min(min(true), min(est)), max(max(true), max(est))]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("True ability"); plt.ylabel("Estimated ability (budget 15)")
    plt.title("Ability recovery")
    plt.tight_layout()
    path = os.path.join(RESULTS, "ability_recovery.png")
    plt.savefig(path, dpi=150)
    plt.close()
    corr = float(np.corrcoef(true, est)[0, 1])
    return path, corr


def calibration_experiment(dataset):
    """ECE of the marginalised vs plug-in predictor at budget 3."""
    _, betas = fit_difficulties(dataset["R"][: dataset["R"].shape[0] // 3])
    eval_rows = dataset["R"][dataset["R"].shape[0] // 3:]
    rng = np.random.default_rng(SEED)
    res = {}
    for name, marg in (("marginalised", True), ("plug_in", False)):
        ps, ys = [], []
        for row in eval_rows:
            n = row.size
            perm = rng.permutation(n)
            a_idx, e_idx = perm[: n // 2][:3], perm[n // 2:]
            th, pv = ability_posterior(betas[a_idx], row[a_idx],
                                       dataset["prior_mean"],
                                       dataset["prior_var"])
            p = predict_prob(th, pv, betas[e_idx], marginalise=marg)
            ps.append(p); ys.append(row[e_idx])
        ps = np.concatenate(ps); ys = np.concatenate(ys)
        res[name] = {"ece": rp.ece(ps, ys), "brier": rp.brier(ps, ys)}
        print(f"[calibration] {name:12s} ECE={res[name]['ece']:.4f} "
              f"Brier={res[name]['brier']:.4f}")
    return res


def ability_error_experiment(dataset, tag):
    """Mean ability error vs budget per policy; shows the acquisition effect
    in the quantity the policy directly controls."""
    plt.figure(figsize=(5, 3.5))
    labels = {"random": "Random", "uncertainty": "Uncertainty",
              "fisher": "Fisher p(1-p)",
              "expected_brier": "Expected-Brier"}
    out = {}
    for policy in acq.POLICIES:
        errs = rp.replay_dataset_ability(dataset, policy, seed=SEED)
        mean_curve = errs.mean(axis=0)
        out[policy] = mean_curve.tolist()
        plt.plot(rp.BUDGETS, mean_curve, marker="o", label=labels[policy])
        print(f"[ability-err {tag}] {policy:15s} "
              f"budget1={mean_curve[1]:.3f} budget3={mean_curve[2]:.3f} "
              f"budget7={mean_curve[3]:.3f}")
    plt.xlabel("Label budget")
    plt.ylabel("Mean |ability error|")
    plt.title(f"Ability estimation error ({tag})")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"ability_error_{tag}.png"), dpi=150)
    plt.close()
    return out


def main():
    all_res = {}

    # Two synthetic regimes:
    #  * "easy": difficulty spread ~ ability spread; ability is easy to estimate
    #    from any labels, so acquisition has little room to help (pre-registered
    #    a~=b~=random expectation).
    #  * "hetero": wide difficulty spread, so many acquisition items are extreme
    #    (p near 0/1, uninformative); informative-item selection should matter.
    print("=== Synthetic Rasch world: easy regime ===")
    syn_easy = dat.make_synthetic(difficulty_sd=1.0, ability_sd=1.0, seed=SEED)
    all_res["synthetic_easy"] = run_policies(syn_easy, "easy")
    plot_learning_curves(all_res["synthetic_easy"], "easy")

    print("\n=== Synthetic Rasch world: heterogeneous regime ===")
    syn_het = dat.make_synthetic(n_items=160, difficulty_sd=3.0,
                                 ability_sd=1.0, seed=SEED)
    all_res["synthetic_hetero"] = run_policies(syn_het, "hetero")
    plot_learning_curves(all_res["synthetic_hetero"], "hetero")

    _, corr = plot_ability_recovery(syn_het)
    all_res["ability_recovery_corr"] = corr
    print(f"[synthetic] ability recovery corr (budget 15) = {corr:.3f}")
    all_res["ability_error"] = ability_error_experiment(syn_het, "hetero")
    all_res["calibration"] = calibration_experiment(syn_het)

    print("\n=== Real offline pool: reconstructed response matrix ===")
    real = dat.load_real_matrix(os.path.dirname(__file__),
                                min_item_cov=3, min_subject_items=25)
    if real is not None:
        print(f"[offline-real] {real['n_subjects']} subjects x "
              f"{real['n_items']} items, {real['n_obs']} observations, "
              f"base rate {real['base_rate']:.3f}")
        off = {}
        for policy in acq.POLICIES:
            curves = rp.replay_dataset_real(real, policy, min_items=25,
                                            seed=SEED)
            a = rp.alc(curves)
            mean, lo, hi = rp.bootstrap_ci(a, seed=SEED)
            off[policy] = {"curve_mean": curves.mean(axis=0).tolist(),
                           "alc_mean": mean, "alc_lo": lo, "alc_hi": hi,
                           "n_eval_subjects": int(curves.shape[0])}
            print(f"[offline-real] {policy:15s} ALC={mean:.4f}  "
                  f"95% CI=[{lo:.4f}, {hi:.4f}]  "
                  f"(n={curves.shape[0]} subjects)")
        all_res["offline"] = off
        plot_learning_curves(off, "offline")
        all_res["offline_meta"] = {"n_subjects": real["n_subjects"],
                                   "n_items": real["n_items"],
                                   "n_obs": real["n_obs"],
                                   "base_rate": real["base_rate"]}
    else:
        print("[offline-real] no offline pool found; synthetic only")

    # ALC table
    lines = ["dataset,policy,alc_mean,alc_lo,alc_hi"]
    for ds in ("synthetic_easy", "synthetic_hetero", "offline"):
        if ds not in all_res:
            continue
        for policy in acq.POLICIES:
            r = all_res[ds][policy]
            lines.append(f"{ds},{policy},{r['alc_mean']:.4f},"
                         f"{r['alc_lo']:.4f},{r['alc_hi']:.4f}")
    with open(os.path.join(RESULTS, "alc_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved results to {RESULTS}/")


if __name__ == "__main__":
    main()
