"""fig04: witness coverage vs capacity (E1 main structural result).

Small multiples: one panel per test seed. Three series (CORE-MAS,
task-aware item replication, frequency replication), fault-weighted
coverage ((s1+s2)/2), bootstrap 95% CI bands of the mean.
Validated palette (light): blue/orange/aqua slots.
Output: figures/fig04_coverage.png (300dpi) + .svg + source data CSV.
"""

import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_mas.allocators import STRATEGIES
from core_mas.datagen import generate
from core_mas.e1_stats import per_task_coverage
from core_mas.eval_structural import AGENT_SET, make_budgets
from core_mas.failure_scenarios import FailureSpec

SERIES = [
    ("core_mas", "CORE-MAS", "#2a78d6"),
    ("task_aware_item", "Task-aware item", "#eb6834"),
    ("frequency_replication", "Frequency", "#1baf7a"),
]
CAPS = [0.4, 0.6, 0.8, 1.0]
SEEDS = [200, 201, 202]
INK = "#0b0b0b"
MUTED = "#52514e"


def mean_ci(vec, rng, R=1000):
    n = len(vec)
    means = []
    for _ in range(R):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(sum(vec[i] for i in idx) / n)
    means.sort()
    return means[int(0.025 * R)], means[int(0.975 * R)]


def main():
    from pathlib import Path as _P
    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(exist_ok=True)
    data_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)

    for ax, seed in zip(axes, SEEDS):
        world = generate(seed)
        for sid, label, color in SERIES:
            ys, lo, hi = [], [], []
            for cap in CAPS:
                budgets = make_budgets(world, cap)
                alloc = STRATEGIES[sid](world.units, AGENT_SET, budgets,
                                        world.dev_tasks, world)
                v1 = per_task_coverage(world.test_tasks, alloc,
                                       FailureSpec("s1"), AGENT_SET, 42)
                v2 = per_task_coverage(world.test_tasks, alloc,
                                       FailureSpec("s2"), AGENT_SET, 42)
                vw = [(a + b) / 2 for a, b in zip(v1, v2)]
                m = sum(vw) / len(vw)
                l, h = mean_ci(vw, random.Random(seed * 100 + int(cap * 100)))
                ys.append(m); lo.append(l); hi.append(h)
                data_rows.append({"seed": seed, "capacity": cap,
                                  "strategy": sid, "mean": round(m, 4),
                                  "ci_lo": round(l, 4), "ci_hi": round(h, 4)})
            ax.plot(CAPS, ys, color=color, lw=2, marker="o", ms=6,
                    markeredgewidth=0, zorder=3, label=label)
            ax.fill_between(CAPS, lo, hi, color=color, alpha=0.12, lw=0, zorder=1)
            if sid == "core_mas":
                # direct label at the last point (text ink, not series color)
                ax.annotate("CORE-MAS", (CAPS[-1], ys[-1]), xytext=(4, -2),
                            textcoords="offset points", color=INK,
                            fontsize=8, va="center")
        ax.set_title(f"test seed {seed}", fontsize=9, color=INK, pad=6)
        ax.set_xticks(CAPS)
        ax.set_xticklabels([f"{c:.0%}" for c in CAPS], fontsize=8)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", color="#e0ded9", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b7b1")
        ax.tick_params(colors=MUTED, labelsize=8)
        if seed == SEEDS[0]:
            ax.set_ylabel("fault-weighted witness coverage\n(mean of s1, s2)",
                          fontsize=8, color=INK)
        if seed == SEEDS[1]:
            ax.set_xlabel("persistent memory budget (share of dedup raw cost)",
                          fontsize=8, color=INK)
            ax.legend(fontsize=8, frameon=False, loc="upper left",
                      bbox_to_anchor=(0.05, 0.97))
    fig.tight_layout()
    fig.savefig(out_dir / "fig04_coverage.png", dpi=300)
    fig.savefig(out_dir / "fig04_coverage.svg")
    data_out = out_dir.parent / "core-mas-code" / "experiment_data" / "runs" / "run_e1"
    data_out.mkdir(parents=True, exist_ok=True)
    (data_out / "fig04_data.csv").write_text(
        "\n".join(f"{r['seed']},{r['capacity']},{r['strategy']},{r['mean']},"
                  f"{r['ci_lo']},{r['ci_hi']}" for r in data_rows),
        encoding="utf-8")
    print("saved figures/fig04_coverage.png/.svg + fig04_data.csv")


if __name__ == "__main__":
    main()
