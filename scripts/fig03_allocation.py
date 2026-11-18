"""fig03: CWRA allocation example — the complementarity trap and bundle actions.

Panel A: a task needs f2 AND f3 jointly; single-item greedy adds either one
with zero immediate coverage gain (stalls). Panel B: the bundle action places
{f2, f3} together; losing agent B still leaves one complete path via A.
Validated palette; task witness bundle shown as an AND gate.
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
GRAY = "#8a93a6"

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for ax in axes:
    ax.set_xlim(0, 5); ax.set_ylim(0, 4.2); ax.axis("off")


def box(ax, x, y, w, h, text, color=BLUE, fs=8, tc="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor="none")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                        color=color, lw=1.4, linestyle=style)
    ax.add_patch(a)


def agent(ax, x, name, units, failed=False):
    color = ORANGE if failed else GRAY
    box(ax, x, 1.15, 1.2, 0.55, name, color, 8)
    ax.text(x + 0.6, 0.55, units, ha="center", va="top", fontsize=7.5, color=MUTED)


# ---- panel A: greedy stalls ----
ax = axes[0]
ax.text(0.25, 3.85, "(a) single-item greedy: zero marginal gain", fontsize=9.5,
        weight="bold", color=INK)
box(ax, 0.4, 2.9, 1.7, 0.55, "task q: needs\nf2 AND f3", BLUE, 8)
agent(ax, 0.5, "agent A", "f2")
agent(ax, 2.0, "agent B", "f3")
agent(ax, 3.5, "agent C", "—")
arrow(ax, 1.25, 2.9, 1.1, 2.25)
arrow(ax, 1.25, 2.9, 2.6, 2.25)
ax.text(0.25, 2.62, "adding f2 alone: C(q,s,x) unchanged (f3 missing)", fontsize=7.5, color=INK)
ax.text(0.25, 2.37, "adding f3 alone: C(q,s,x) unchanged (f2 missing)", fontsize=7.5, color=INK)
ax.text(0.25, 2.12, "greedy sees no positive gain and stops", fontsize=8, color=ORANGE, weight="bold")

# ---- panel B: bundle action ----
ax = axes[1]
ax.text(0.25, 3.85, "(b) CWRA bundle action {f2, f3} → agent A", fontsize=9.5,
        weight="bold", color=INK)
box(ax, 0.4, 2.9, 1.7, 0.55, "task q: needs\nf2 AND f3", BLUE, 8)
agent(ax, 0.5, "agent A", "f2, f3  ✓")
agent(ax, 2.0, "agent B", "f3", failed=True)
agent(ax, 3.5, "agent C", "f3")
arrow(ax, 1.25, 2.9, 1.1, 2.25)
ax.text(0.25, 2.62, "bundle gain: task covered even if B fails", fontsize=7.5, color=AQUA, weight="bold")
ax.text(0.25, 2.37, "fault-weighted score = gain / token cost", fontsize=7.5, color=INK)
ax.text(0.25, 2.12, "replication only where the\ncounterfactual gain pays for it", fontsize=7.5, color=MUTED)

fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "figures"
fig.savefig(out / "fig03_allocation.png", dpi=300)
fig.savefig(out / "fig03_allocation.svg")
print("saved figures/fig03_allocation.png/.svg")
