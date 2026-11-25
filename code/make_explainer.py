"""Generate a one-figure plain-language explainer of the project.

Saves results/project_explainer.png -- a schematic of the predict/acquire loop
plus the headline finding, for slides or the paper.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

BLUE = "#3B6EA5"
GREEN = "#4C9A6B"
AMBER = "#C08A2E"
GREY = "#5B6670"
LIGHT = "#EEF2F6"
RED = "#B3503C"

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 96, "Predicting AI Evaluation Outcomes under a Label Budget",
        ha="center", va="top", fontsize=15, fontweight="bold")
ax.text(50, 90.5,
        "Guess whether each AI gets each test question right; buy a few real "
        "answers to improve the guesses.",
        ha="center", va="top", fontsize=9.5, color=GREY)


def box(x, y, w, h, text, fc, tc="white", fs=9.2, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.6,rounding_size=2",
                 fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=16, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}"))


# ---- The loop (top band) ----
y = 66; h = 13
box(3, y, 19, h, "1. PREDICT\noutcomes from\nability vs difficulty", BLUE, fs=8.6)
box(27, y, 19, h, "2. ACQUIRE\nmost-informative\nlabel (p ≈ 0.5)", AMBER, fs=8.6)
box(51, y, 19, h, "3. UPDATE\nthe ability\nposterior", GREEN, fs=8.6)
box(75, y, 22, h, "4. RE-PREDICT\n+ SCORE\n(Brier ALC)", BLUE, fs=8.6)

arrow(22, y + h / 2, 27, y + h / 2)
arrow(46, y + h / 2, 51, y + h / 2)
arrow(70, y + h / 2, 75, y + h / 2)
# repeat loop arrow underneath
arrow(86, y, 34, y - 9, color=RED, rad=-0.25, lw=1.8)
ax.text(60, y - 11.5, "repeat over budgets  1 -> 3 -> 7 -> 15 -> 31",
        ha="center", va="center", fontsize=8.5, color=RED, style="italic")

# ---- Left: the exam idea ----
ax.add_patch(FancyBboxPatch((3, 20), 44, 26,
             boxstyle="round,pad=0.6,rounding_size=2", fc=LIGHT, ec=GREY, lw=1))
ax.text(25, 43.5, "Predictor: a Rasch / IRT model", ha="center",
        fontsize=10.5, fontweight="bold", color=GREY)
ax.text(25, 38.5,
        "Each subject has an ABILITY; each item a DIFFICULTY.",
        ha="center", fontsize=9, color="#333")
ax.text(25, 32.5,
        "strong subj. vs easy item  ->  likely RIGHT\n"
        "weak subj.  vs hard item   ->  likely WRONG\n"
        "ability ≈ difficulty        ->  P ≈ 0.5",
        ha="center", va="center", fontsize=8.6, color="#333", family="monospace")
ax.text(25, 23.5,
        "Calibrated: predicts ~0.5 when few labels are seen.",
        ha="center", fontsize=8.3, color=BLUE, style="italic")

# ---- Right: the finding ----
ax.add_patch(FancyBboxPatch((51, 20), 46, 26,
             boxstyle="round,pad=0.6,rounding_size=2",
             fc="#FBF0EE", ec=RED, lw=1.2))
ax.text(74, 43.5, "What we find", ha="center", fontsize=10.5,
        fontweight="bold", color=RED)
ax.text(53.5, 38.5, "Information-directed acquisition learns",
        ha="left", fontsize=9, color="#2E7D4F")
ax.text(53.5, 35.0, "ability ~35% faster",
        ha="left", fontsize=9, color="#2E7D4F")
ax.text(95.5, 36.7, "✓", ha="right", fontsize=12, color="#2E7D4F",
        fontweight="bold")
ax.text(53.5, 30.5, "...but Brier ALC is essentially unchanged",
        ha="left", fontsize=9, color=RED)
ax.text(95.5, 30.5, "✗", ha="right", fontsize=12, color=RED,
        fontweight="bold")
ax.text(74, 24.5,
        "A calibrated predictor already hedges,\n"
        "so the acquisition policy adds little.",
        ha="center", va="center", fontsize=8.6, color="#333")

# ---- Bottom takeaway banner ----
box(3, 6, 94, 9,
    "TAKEAWAY:  acquisition and calibration are substitutes — "
    "the predictor, not the policy, drives the score.",
    GREY, fs=9.8)

plt.tight_layout()
out = os.path.join(RESULTS, "project_explainer.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print("saved", out)
