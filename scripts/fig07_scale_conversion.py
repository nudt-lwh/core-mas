"""Generate two separate figures:
fig07_scale.png  — scalability of the structural advantage (M x N).
fig07_convert.png — evidence-to-answer conversion with the refusal/wrong split.
"""

import csv
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
GRIDC = "#e0ded9"

R = Path(__file__).resolve().parents[2] / "core-mas-code" / "experiment_data" / "runs"
OUT = Path(__file__).resolve().parents[2] / "figures"

# ---------- Figure: scalability ----------
rows = list(csv.DictReader(open(R / "run_e1_scale" / "e1_scale.csv", encoding="utf-8")))
adv = {}
for m in ["8", "16", "32", "64", "128"]:
    for n in ["3", "6", "9"]:
        cm = next((r for r in rows if r["n_suppliers"] == m and r["n_agents"] == n
                   and r["strategy"] == "core_mas"), None)
        fr = next((r for r in rows if r["n_suppliers"] == m and r["n_agents"] == n
                   and r["strategy"] == "frequency_replication"), None)
        if cm and fr:
            adv[(m, n)] = (float(cm["fw"]) - float(fr["fw"])) * 100

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
ms = [8, 16, 32, 64, 128]
for n, color, ls in [("3", BLUE, "-"), ("6", AQUA, "-"), ("9", ORANGE, "-")]:
    xs, ys = [], []
    for m in ms:
        if (str(m), n) in adv:
            xs.append(m)
            ys.append(adv[(str(m), n)])
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, ls=ls, zorder=3)
    if n == "9":
        ax.annotate(f"$N={n}$", xy=(xs[-1], ys[-1]), xytext=(-6, -8),
                    textcoords="offset points", fontsize=10.5, color=color,
                    va="top", ha="right")
    else:
        ax.annotate(f"$N={n}$", xy=(xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=10.5, color=color, va="center")
ax.axhline(0, color=INK, lw=0.9, zorder=2)
ax.axhspan(-25, 0, color="#eb6834", alpha=0.06, zorder=0)
ax.set_xscale("log", base=2)
ax.set_xticks(ms)
ax.set_xticklabels([str(m) for m in ms], fontsize=10.0)
ax.set_ylim(-25, 22)
ax.set_xlabel("suppliers $M$ (80% budget, seed 200)", fontsize=10.0, color=INK)
ax.set_ylabel("fw-coverage advantage\nCORE-MAS − frequency (pp)", fontsize=10.0, color=INK)
ax.grid(color=GRIDC, lw=0.8, zorder=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig07_scale.png", dpi=300)
fig.savefig(OUT / "fig07_scale.pdf")
plt.close(fig)

# ---------- Figure: conversion + error split ----------
def conversion_stats(path, strat="core_mas"):
    rs = [r for r in csv.DictReader(open(path, encoding="utf-8"))
          if r["strategy"] == strat]
    c1 = [r for r in rs if r["structural_correct"] == "1"]
    a1 = sum(1 for r in c1 if r.get("llm_correct_repair", r.get("llm_correct")) == "1")
    a0 = [r for r in c1 if r.get("llm_correct_repair", r.get("llm_correct")) == "0"]
    refused = sum(1 for r in a0 if r.get("insufficient_repair",
                                         r.get("insufficient")) == "1")
    wrong = len(a0) - refused
    n = len(c1)
    return a1 / n, refused / n, wrong / n, n

models = [
    ("Qwen3-4B", conversion_stats(R / "run_e2_qwen4b_repair_cloud.csv")),
    ("Qwen3-8B", conversion_stats(R / "run_md4_8b_repair.csv")),
    ("Qwen3-14B", conversion_stats(R / "run_md4_14b_repair.csv")),
    ("deepseek-chat", conversion_stats(R / "run_e2_ds_repair_cloud.csv")),
]

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
x = range(len(models))
for i, (name, (conv, ref, wr, n)) in enumerate(models):
    ax.bar(i, conv, width=0.52, color=BLUE, zorder=3,
           label="converted (A=1)" if i == 0 else None)
    ax.bar(i, ref, bottom=conv, width=0.52, color=GRAY, zorder=3,
           label="conservative refusal" if i == 0 else None)
    ax.bar(i, wr, bottom=conv + ref, width=0.52, color=ORANGE, zorder=3,
           label="wrong answer" if i == 0 else None)
    ax.text(i, conv / 2, f"{conv:.2f}", ha="center", va="center",
            fontsize=10.0, color="white", fontweight="bold")
    if ref >= 0.10:
        ax.text(i, conv + ref / 2, f"{ref:.2f}", ha="center", va="center",
                fontsize=9.0, color="white")
    if wr >= 0.08:
        ax.text(i, conv + ref + wr / 2, f"{wr:.2f}", ha="center", va="center",
                fontsize=9.0, color="white")
    ax.text(i, 1.02, f"$C{{=}}1$: {n}", ha="center", va="bottom",
            fontsize=8.5, color=MUTED)
ax.set_xticks(list(x))
ax.set_xticklabels([m[0] for m in models], fontsize=10.0)
ax.set_ylim(0, 1.12)
ax.set_ylabel("share of structurally-sufficient tasks", fontsize=10.0, color=INK)
ax.grid(axis="y", color=GRIDC, lw=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3,
          handlelength=1.1, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(OUT / "fig07_convert.png", dpi=300)
fig.savefig(OUT / "fig07_convert.pdf")
plt.close(fig)
print("saved figures/fig07_scale.png and figures/fig07_convert.png")
