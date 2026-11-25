# Measurement-Theoretic Active Elicitation for Predicting AI Evaluation Outcomes

Code for the CS329H final project. We recast the NeurIPS 2026 Predictive AI
Evaluation Competition as a measurement problem: predicting whether a subject
answers an item correctly is a Rasch interaction `sigmoid(theta_s - beta_i)`
between subject ability and item difficulty, and deciding which labels to buy
under the nested budget `{0,1,3,7,15,31}` is efficient Fisher-information
elicitation. We compare a Bayesian IRT predictor against our black-box baseline
and four acquisition policies via an offline replay harness scored by Brier ALC.

## Setup

```bash
cd final_project/code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. CPU only; a full run takes a few minutes.

## Reproduce

```bash
python run_experiments.py
```

Writes to `results/`:

| File | Content | Paper artifact |
|------|---------|----------------|
| `learning_curves_synthetic.png` | Brier vs budget per policy | Fig. 1 |
| `learning_curves_offline.png`   | same, real offline pool     | Fig. 2 |
| `ability_recovery.png`          | estimated vs true ability   | Fig. 3 |
| `alc_table.csv`                 | ALC + bootstrap 95% CIs     | Table 1 |
| `results.json`                  | every number in the paper   | — |

Deterministic: all randomness is seeded (`SEED = 0` in `run_experiments.py`).

## Repository structure

```
code/
  src/
    irt.py           Rasch model; MAP/Laplace ability posterior; difficulty MLE
    acquisition.py   four policies: random, uncertainty, Fisher, expected-Brier
    replay.py        offline replay harness; Brier, ECE, ALC, bootstrap CIs
    data.py          synthetic Rasch generator; real offline-pool loader
  run_experiments.py orchestrates all experiments and figures
  requirements.txt
  results/           generated outputs (created on first run)
```

## Data

* **Synthetic Rasch world** — controlled ground truth (known abilities and
  difficulties); verifies ability recovery and policy behaviour.
* **Real reconstructed response matrix** — a genuine subject x item binary
  matrix rebuilt from `offline_X.npy`/`offline_y.npy` of our competition
  submissions (`measurement_science_v2/Neurips-competition/submissions/v19` or
  `v17`). Subject identity comes from the subject-metadata feature columns, item
  identity from the 384-dim content embedding; cells hold the real 0/1 outcome.
  After keeping items co-observed by >= 3 subjects and subjects with >= 25 such
  items, this yields ~119 subjects over ~1,531 items (~9,255 real outcomes). If
  the pool is absent, the experiments run on synthetic data only and say so.

Item difficulties are estimated from a held-out third of subjects disjoint from
the evaluation subjects, and only items those fit-subjects answered are scored,
so no evaluation label trains its own subject's ability.

## Notes / attribution

The competition infrastructure (feature extraction, submission harness, offline
pool) is the author's own prior work from the NeurIPS 2026 competition. The
measurement-theoretic contribution here --- the Bayesian IRT predictor and the
Fisher / expected-Brier-reduction acquisition policies --- is new for this
project.
