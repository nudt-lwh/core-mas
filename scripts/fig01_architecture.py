"""fig01: CORE-MAS architecture (matplotlib boxes/arrows, editable SVG+PNG).

Three phases: write-time planning (TW-Hyper + CWRA) -> agents with allocated
multi-granularity memory -> failure -> ECCR recovery loop.
Validated palette: blue structural, aqua actions, orange failure path.
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
SURFACE = "#f2f4f8"

fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis("off")


def box(x, y, w, h, text, color=BLUE, fs=9, text_color="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=color, edgecolor="none")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, weight="bold")


def arrow(x1, y1, x2, y2, color=INK, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, color=color, lw=1.6, linestyle=style)
    ax.add_patch(a)


# phase 1: planning
box(0.3, 4.6, 3.2, 0.9, "TW-Hyper\nwitness structures\n(AND/OR + provenance)", BLUE)
box(0.3, 3.4, 3.2, 0.9, "CWRA allocation\nbundle greedy + local search", BLUE)
arrow(1.9, 4.6, 1.9, 4.35)
ax.text(4.1, 4.0, "dev tasks", fontsize=8, color=MUTED)
arrow(3.55, 4.1, 1.9, 4.05, MUTED, style="--")

# phase 2: agents
agents = [("agent a", "finance", 5.0), ("agent b", "compliance", 6.6),
          ("agent c", "logistics", 8.2)]
for label, dom, x in agents:
    box(x, 3.4, 1.3, 0.9, f"{label}\n{dom}", AQUA)
    ax.text(x + 0.65, 2.75, "raw facts\nverified predicates\nprovenance",
            ha="center", va="top", fontsize=7, color=MUTED)
box(5.8, 4.8, 2.9, 0.7, "coordinator (shared memory)", "#8a93a6")
arrow(4.3, 3.85, 5.9, 3.85)

# budget line
ax.plot([0.3, 9.6], [2.45, 2.45], color=INK, lw=0.8, ls=":")
ax.text(0.3, 2.3, "per-agent budget b_i  ·  team budget B  ·  access permissions",
        fontsize=7.5, color=MUTED)

# phase 3: failure + recovery
box(2.6, 0.9, 1.6, 0.9, "failure s:\nagent b lost", ORANGE)
arrow(7.25, 2.45, 3.9, 1.35, ORANGE)
box(5.4, 0.9, 3.6, 0.9, "ECCR recovery\nquery survivors → verify sources\nanswer or refuse", BLUE)
arrow(4.25, 1.35, 5.35, 1.35)
ax.text(7.2, 0.35, "output: answer with verified evidence\nor insufficient_evidence",
        ha="center", fontsize=8, color=INK)

ax.text(5, 5.95, "CORE-MAS: write-time planning → fault-aware placement → evidence-constrained recovery",
        ha="center", fontsize=11, color=INK, weight="bold")
out = Path(__file__).resolve().parents[2] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "fig01_architecture.png", dpi=300)
fig.savefig(out / "fig01_architecture.svg")
print("saved figures/fig01_architecture.png/.svg")
