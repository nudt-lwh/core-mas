"""fig06: E4 ablation contribution (fault-weighted coverage loss vs full).

Horizontal bars: coverage loss when each component is removed, at 80% and
100% capacity (the discriminating region). Negative = component contributes.
Validated palette (light): blue fill; text ink; recessive grid.
Output: figures/fig06_ablations.png (300dpi) + .svg + source data CSV.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0b0b0b"
MUTED = "#52514e"
BLUE = "#2a78d6"
CAPS = ["0.8", "1.0"]

VARIANTS = [
    ("w_o_bundles", "w/o bundle search"),
    ("w_o_recovery", "w/o cross-agent recovery"),
    ("w_o_failure", "w/o failure modeling"),
    ("w_o_witness", "w/o witness modeling"),
    ("w_o_granularity", "w/o multi-granularity"),
]


def main():
    rows = list(csv.DictReader(open("experiment_data/runs/run_e1/e1_extra.csv",
                                    encoding="utf-8")))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["exp"] == "E4" and r["capacity"] in CAPS:
            by[r["variant"]][r["capacity"]].append(float(r["fw"]))
    full = {c: sum(by["full"][c]) / len(by["full"][c]) for c in CAPS}

    data_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharex=True)
    for ax, cap in zip(axes, CAPS):
        losses = []
        labels = []
        for vid, label in VARIANTS:
            m = sum(by[vid][cap]) / len(by[vid][cap])
            loss = m - full[cap]
            losses.append(loss)
            labels.append(label)
            data_rows.append({"capacity": cap, "variant": vid,
                              "coverage_loss_pp": round(loss * 100, 2)})
        y = list(range(len(losses)))[::-1]
        ax.barh(y, [l * 100 for l in losses], color=BLUE, height=0.55)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"budget {int(float(cap)*100)}%", fontsize=9, color=INK, pad=6)
        ax.grid(axis="x", color="#e0ded9", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b7b1")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.invert_yaxis()
        if cap == "0.8":
            ax.set_xlabel("fault-weighted coverage change vs full CORE-MAS (pp)\n"
                          "left = component contributes", fontsize=8, color=INK)
        else:
            ax.set_xlabel("coverage change vs full (pp)", fontsize=8, color=INK)
        fig.tight_layout()
    out = Path(__file__).resolve().parents[2] / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig06_ablations.png", dpi=300)
    fig.savefig(out / "fig06_ablations.svg")
    data_out = out.parent / "core-mas-code" / "experiment_data" / "runs" / "run_e1"
    data_out.mkdir(parents=True, exist_ok=True)
    (data_out / "fig06_data.csv").write_text(
        "\n".join(f"{r['capacity']},{r['variant']},{r['coverage_loss_pp']}"
                  for r in data_rows), encoding="utf-8")
    print("saved figures/fig06_ablations.png/.svg + fig06_data.csv")


if __name__ == "__main__":
    main()
