import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "glucoimg_representation_main_results.csv")
OUT_BASE = os.path.join(HERE, "glucoimg_representation_horizon_comparison")
OUT_REL_BASE = os.path.join(HERE, "glucoimg_representation_relative_improvement")


def setup_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def setup_relative_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
    })


def plot_metric(ax, df, metric, ylabel, panel=None):
    order = ["SeqOnly", "RP", "GAF", "MTF", "GlucoImg"]
    colors = {
        "SeqOnly": "#7A7A7A",
        "RP": "#59A14F",
        "GAF": "#E15759",
        "MTF": "#F28E2B",
        "GlucoImg": "#3366CC",
    }
    labels = {
        "SeqOnly": "SeqOnly",
        "RP": "GlucoImg-RP",
        "GAF": "GlucoImg-GAF",
        "MTF": "GlucoImg-MTF",
        "GlucoImg": "GlucoImg",
    }
    for model in order:
        sub = df[df["model"] == model].sort_values("horizon_min")
        is_main = model == "GlucoImg"
        is_base = model == "SeqOnly"
        ax.plot(
            sub["horizon_min"],
            sub[metric],
            label=labels[model],
            color=colors[model],
            linewidth=3.0 if is_main else 2.0,
            linestyle="--" if is_base else "-",
            marker="o",
            markersize=5.2 if is_main else 4.0,
            markeredgewidth=1.2 if is_main else 0.8,
            markerfacecolor="white" if is_base else colors[model],
            markeredgecolor=colors[model],
            zorder=4 if is_main else 3,
        )

    ax.set_xlabel("Prediction horizon (min)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([15, 30, 45, 60, 75, 90])
    ax.grid(True, color="#D0D0D0", alpha=0.32, linewidth=0.8)
    ax.set_facecolor("white")
    if panel:
        ax.text(
            0.015, 0.965, panel,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=14, fontweight="bold",
        )
    ymin = df[metric].min()
    ymax = df[metric].max()
    margin = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)


def compute_relative_improvement(df):
    rows = []
    base = df[df["model"] == "SeqOnly"].set_index("horizon_min")
    for _, row in df[df["model"] != "SeqOnly"].iterrows():
        horizon = int(row["horizon_min"])
        ref = base.loc[horizon]
        rows.append({
            "model": row["model"],
            "horizon_min": horizon,
            "MAE_improvement": (float(ref["MAE"]) - float(row["MAE"])) / float(ref["MAE"]) * 100.0,
            "RMSE_improvement": (float(ref["RMSE"]) - float(row["RMSE"])) / float(ref["RMSE"]) * 100.0,
        })
    return pd.DataFrame(rows)


def plot_relative_metric(ax, df, metric, ylabel):
    order = ["RP", "GAF", "MTF", "GlucoImg"]
    styles = {
        "RP": {"color": "#5B8E5A", "linestyle": "--", "linewidth": 1.75},
        "GAF": {"color": "#7F7F7F", "linestyle": ":", "linewidth": 1.85},
        "MTF": {"color": "#7B5EA7", "linestyle": "-.", "linewidth": 1.75},
        "GlucoImg": {"color": "#3366CC", "linestyle": "-", "linewidth": 2.65},
    }
    labels = {
        "RP": "GlucoImg-RP",
        "GAF": "GlucoImg-GAF",
        "MTF": "GlucoImg-MTF",
        "GlucoImg": "GlucoImg",
    }
    for model in order:
        sub = df[df["model"] == model].sort_values("horizon_min")
        is_main = model == "GlucoImg"
        x = sub["horizon_min"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        dense_x = np.linspace(x.min(), x.max(), 180)
        if PchipInterpolator is not None:
            dense_y = PchipInterpolator(x, y)(dense_x)
        else:
            dense_y = np.interp(dense_x, x, y)
        style = styles[model]
        ax.plot(
            dense_x,
            dense_y,
            label=labels[model],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            zorder=4 if is_main else 3,
        )
        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=3.7 if is_main else 2.8,
            markeredgewidth=0.9 if is_main else 0.7,
            markerfacecolor=style["color"] if is_main else "white",
            markeredgecolor=style["color"],
            zorder=5 if is_main else 4,
        )
    ax.axhline(0, color="#8A8A8A", linewidth=1.0, linestyle="--", alpha=0.45, zorder=1)
    ax.set_xlabel("Prediction horizon (min)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([15, 30, 45, 60, 75, 90])
    ax.grid(True, color="#D0D0D0", alpha=0.32, linewidth=0.8)
    ax.set_facecolor("white")
    ymin = min(0.0, float(df[metric].min()))
    ymax = float(df[metric].max())
    margin = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - 0.2, ymax + margin)


def main():
    setup_style()
    df = pd.read_csv(DATA_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), sharex=True)
    plot_metric(axes[0], df, "MAE", "MAE (mg/dL)", "A")
    plot_metric(axes[1], df, "RMSE", "RMSE (mg/dL)", "B")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        handlelength=2.2,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.965, bottom=0.27, wspace=0.22)

    fig.savefig(f"{OUT_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASE}.png", dpi=600, bbox_inches="tight")
    print(f"Saved {OUT_BASE}.pdf")
    print(f"Saved {OUT_BASE}.png")

    setup_relative_style()
    rel = compute_relative_improvement(df)
    rel.to_csv(os.path.join(HERE, "glucoimg_representation_relative_improvement.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), sharex=True)
    plot_relative_metric(axes[0], rel, "MAE_improvement", "Relative MAE improvement (%)")
    plot_relative_metric(axes[1], rel, "RMSE_improvement", "Relative RMSE improvement (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.86, bottom=0.18, wspace=0.26)
    fig.savefig(f"{OUT_REL_BASE}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_REL_BASE}.png", dpi=600, bbox_inches="tight")
    print(f"Saved {OUT_REL_BASE}.pdf")
    print(f"Saved {OUT_REL_BASE}.png")


if __name__ == "__main__":
    main()
