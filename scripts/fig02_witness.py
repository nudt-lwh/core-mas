"""fig02: TW-Hyper witness structure example (AND/OR + provenance + granularity).

Concrete example from the synthetic domain: a budget-check task has two
alternative sufficient paths (raw price record, or verified predicate); a
cheapest-compliant task needs an AND bundle of four requirements.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"

fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, text, color=BLUE, fs=8, tc="white", rounded=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1" if rounded else "square",
                       facecolor=color, edgecolor="none")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(x1, y1, x2, y2, color=INK, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                        color=color, lw=1.4, linestyle=style)
    ax.add_patch(a)


# left: budget-check task with OR paths
ax.text(0.4, 4.35, "Task q1: \"is S4's price below 120?\"", fontsize=10, weight="bold", color=INK)
box(0.4, 3.7, 4.3, 0.45, "W(q1): W1 = {price_S4, exact}   OR   W2 = {price_S4, <120}", BLUE, 7)
box(0.6, 2.2, 2.0, 0.85, "raw: S4 price = 62\nsource quote-S4-07", AQUA, 7.5)
box(3.1, 2.2, 2.0, 0.85, "predicate: S4 price < 120\nrule budget_v1\nsupports: budget_check\nnot: exact_price", AQUA, 7)
arrow(1.6, 3.66, 1.6, 3.1, style="--")
arrow(4.1, 3.66, 4.1, 3.1, style="--")
ax.text(0.4, 1.9, "granularity = capability:\npredicates cannot answer exact-price questions",
        fontsize=7.5, color=MUTED)

# right: cheapest-compliant AND bundle
ax.text(5.6, 4.35, "Task q2: \"cheapest compliant of S1, S5?\"", fontsize=10, weight="bold", color=INK)
box(5.6, 3.7, 4.1, 0.45, "W(q2): AND {price_S1, comp_S1, price_S5, comp_S5}", ORANGE, 7)
box(5.8, 2.4, 1.55, 0.8, "price_S1", AQUA, 7.5)
box(7.55, 2.4, 1.55, 0.8, "comp_S1", AQUA, 7.5)
box(5.8, 1.15, 1.55, 0.8, "price_S5", AQUA, 7.5)
box(7.55, 1.15, 1.55, 0.8, "comp_S5", AQUA, 7.5)
for xx in (6.575, 8.325):
    arrow(xx, 3.66, xx, 3.25)
ax.text(5.8, 0.55, "all four must survive on accessible agents — single-item greedy stalls on\nthis bundle (zero marginal gain for any one fact alone)",
        fontsize=7.5, color=MUTED)

out = Path(__file__).resolve().parents[2] / "figures"
fig.savefig(out / "fig02_witness.png", dpi=300)
fig.savefig(out / "fig02_witness.svg")
print("saved figures/fig02_witness.png/.svg")
