"""fig05: LLM-layer accuracy (E2) — dual-model panels: (a) Qwen3-4B, (b) deepseek-chat.

2x2 grid: rows = models, columns = budget 80%/100%. Grouped bars by condition,
±1.96·SE of per-cell binomial mean. Validated palette: CORE-MAS blue, frequency
aqua, task-aware orange, uniform gray (aqua contrast WARN -> direct labels +
table fallback, per fig04).
"""

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
GRAY = "#8a93a6"

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "core-mas-code" / "experiment_data" / "runs"

DATASETS = [
    ("Qwen3-4B", [str(RUNS / "run_e2_qwen4b_repair_cloud.csv")]),
    ("deepseek-chat", [str(RUNS / "run_e2_ds_repair_cloud.csv"),
                       str(RUNS / "run_e2_ds_trad_repair_cloud.csv")]),
]

conds = ["s0", "s1_finance", "s1_logistics"]
labels = ["no-fault", "finance lost", "logistics lost"]
strategies = [("core_mas", "CORE-MAS", BLUE),
              ("frequency_replication", "frequency", AQUA),
              ("task_aware_item", "task-aware", ORANGE),
              ("uniform_sharding", "uniform", GRAY),
              ("traditional_placement", "traditional", "#b8b8b2")]

fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4), sharey=True, sharex=True)

for row, (model_name, paths) in enumerate(DATASETS):
    rows = []
    for path in paths:
        rows.extend(csv.DictReader(open(path, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        # ECCR-v2 repair branch is the main protocol; fall back to V1 field
        key = (r["strategy"], r["capacity"], r["condition"])
        val = r.get("llm_correct_repair") or r.get("llm_correct")
        if val not in (None, ""):
            by[key].append(int(val))
    model_strategies = [s for s in strategies
                        if by.get((s[0], "0.8", "s0"))]  # only series present in data
    for col, cap in enumerate(["0.8", "1.0"]):
        ax = axes[row][col]
        x = list(range(len(conds)))
        w = 0.2
        for k, (sid, label, color) in enumerate(model_strategies):
            means, errs = [], []
            for cond in conds:
                vals = by[(sid, cap, cond)]
                m = sum(vals) / len(vals)
                se = sqrt(m * (1 - m) / len(vals)) * 1.96
                means.append(m)
                errs.append(se)
            xs = [xi + (k - 1.5) * w for xi in x]
            ax.bar(xs, means, width=w, color=color, yerr=errs, capsize=2.5,
                   error_kw=dict(lw=0.8, ecolor=INK), label=label, zorder=3)
            # direct value labels for CORE-MAS bars (selective, not every point);
            # white backing so labels never sit on the error bars
            if sid == "core_mas":
                for xi, m, e in zip(xs, means, errs):
                    ax.text(xi, m + e + 0.03, f"{m:.2f}", ha="center",
                            va="bottom", fontsize=6.5, color=INK,
                            bbox=dict(facecolor="white", alpha=0.8,
                                      edgecolor="none", pad=0.8))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"budget {int(float(cap)*100)}%", fontsize=9, color=INK, pad=6)
        ax.set_ylim(0, 0.92)
        ax.grid(axis="y", color="#e0ded9", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        if col == 0:
            ax.set_ylabel(f"({chr(97 + row)}) {model_name}\nLLM-layer task accuracy",
                          fontsize=8, color=INK)
        if row == 0 and col == 1:
            from matplotlib.patches import Patch
            handles = [Patch(color=c, label=lb) for _, lb, c in strategies]
            ax.legend(handles=handles, fontsize=7, frameon=False,
                      loc="upper right", ncol=1, handlelength=1.4,
                      borderaxespad=0.2)

fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "figures"
fig.savefig(out / "fig05_llm.png", dpi=300)
fig.savefig(out / "fig05_llm.svg")
fig.savefig(out / "fig05_llm.pdf")
print("saved figures/fig05_llm.png/.svg (dual-model panels)")
