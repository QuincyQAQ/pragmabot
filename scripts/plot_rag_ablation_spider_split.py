#!/usr/bin/env python3
"""Radar charts and bar charts for RAG retrieval ablation (GPT-4o vs GPT-4o-mini)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as PltPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# --- Configuration ---
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_style("darkgrid")

plt.rc("font", size=14)
plt.rc("axes", titlesize=16)
plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("legend", fontsize=14)
plt.rc("figure", titlesize=16)


# --- 1. The Radar Factory Class ---
def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` Axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return PltPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                spine = Spine(axes=self, spine_type="circle", path=PltPath.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# --- 2. Data Definition ---
tasks = [
    "Put apple",
    "Move candy",
    "Move egg",
    "Pick plate",
    "Put ball",
    "Put orange",
    "Move paper",
    "Move screw",
    "Move sushi",
    "Move grape",
    "Pick carton",
    "Pick towel",
]

accuracy_data = {
    "rag-4o": [18, 18, 18, 18, 14, 16, 15, 14, 18, 18, 18, 15],
    "all-4o": [18, 18, 18, 18, 3, 9, 15, 17, 15, 18, 17, 12],
    "rand-4o": [1, 6, 3, 2, 1, 0, 3, 5, 5, 3, 4, 4],
    "rag-mini": [18, 14, 18, 18, 10, 13, 11, 11, 14, 17, 18, 15],
    "all-mini": [18, 15, 18, 18, 2, 7, 9, 8, 3, 9, 15, 6],
    "rand-mini": [3, 8, 5, 1, 1, 2, 4, 8, 4, 0, 15, 13],
}

count_tokens = {
    "rag-4o": {"mean": 1204.277778, "std": 50.39236878},
    "all-4o": {"mean": 8997.8125, "std": 1.950322727},
    "rand-4o": {"mean": 1139.458333, "std": 36.54690922},
    "rag-mini": {"mean": 1211.145833, "std": 46.71008168},
    "all-mini": {"mean": 8997.763889, "std": 1.858588498},
    "rand-mini": {"mean": 1138.173611, "std": 36.71080134},
}

query_times = {
    "rag-4o": {"mean": 9.844881634, "std": 1.880686734},
    "all-4o": {"mean": 10.60049179, "std": 2.339222007},
    "rand-4o": {"mean": 10.19366781, "std": 2.031552455},
    "rag-mini": {"mean": 6.762942289, "std": 1.901865884},
    "all-mini": {"mean": 7.195822703, "std": 3.925338118},
    "rand-mini": {"mean": 7.230206359, "std": 2.34854612},
}

# Normalize Accuracy Data
for key in accuracy_data:
    accuracy_data[key] = [val / 18 * 100 for val in accuracy_data[key]]


# --- 3. Plotting Logic ---
if __name__ == "__main__":
    # Setup
    N = len(tasks)
    theta = radar_factory(N, frame="polygon")
    colors = sns.color_palette()

    all_methods = ["rag-4o", "all-4o", "rand-4o", "rag-mini", "all-mini", "rand-mini"]
    methods_4o = ["rag-4o", "all-4o", "rand-4o"]
    methods_mini = ["rag-mini", "all-mini", "rand-mini"]

    method_colors = {m: c for m, c in zip(all_methods, colors)}

    def format_radar_ax(ax, title):
        ax.set_varlabels(tasks)
        for label, angle in zip(ax.get_xticklabels(), theta):
            angle_deg = np.degrees(angle) % 360
            label.set_horizontalalignment(
                "right" if 0 < angle_deg < 180 else "left" if 180 < angle_deg < 360 else "center"
            )
            label.set_verticalalignment(
                "bottom"
                if (0 <= angle_deg < 90 or 270 < angle_deg <= 360)
                else "top" if (90 < angle_deg < 270) else "center"
            )

        ax.tick_params(axis="x", which="major", pad=0)
        ax.set_rgrids([25, 50, 75], labels=["25", "50", "75"], angle=0, verticalalignment="center")
        ax.set_ylim(0, 101)
        if title:
            ax.set_title(title, weight="bold", size="large", pad=25)

    # --- LAYOUT CHANGES HERE ---
    # figsize increased width (24) to support 3 columns nicely
    fig = plt.figure(figsize=(11, 9))

    # 2 rows, 3 columns
    # width_ratios=[1, 1, 0.8] ensures the radar columns are wide, and the bar chart column is slightly narrower
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.6], hspace=0.4, wspace=0.5)

    # --- Subplot 1: Radar GPT-4o (Row 0, Spans Cols 0 & 1) ---
    ax_radar_4o = fig.add_subplot(gs[0, 0], projection="radar")
    for method in methods_4o:
        ax_radar_4o.plot(theta, accuracy_data[method], color=method_colors[method], label=method, linewidth=2.5)
        ax_radar_4o.fill(theta, accuracy_data[method], facecolor=method_colors[method], alpha=0.2)

    format_radar_ax(ax_radar_4o, None)
    # Legend placed slightly differently since the plot is wider now
    ax_radar_4o.legend(loc="upper right", bbox_to_anchor=(1.72, 1.25), framealpha=0.4)

    # --- Subplot 2: Radar GPT-4o-mini (Row 1, Spans Cols 0 & 1) ---
    ax_radar_mini = fig.add_subplot(gs[1, 0], projection="radar")
    for method in methods_mini:
        ax_radar_mini.plot(theta, accuracy_data[method], color=method_colors[method], label=method, linewidth=2.5)
        ax_radar_mini.fill(theta, accuracy_data[method], facecolor=method_colors[method], alpha=0.2)

    format_radar_ax(ax_radar_mini, None)
    ax_radar_mini.legend(loc="upper right", bbox_to_anchor=(1.72, 1.25), framealpha=0.4)

    # --- Subplot 3: Token Count (Row 0, Col 2) ---
    ax_tokens = fig.add_subplot(gs[0, 1])
    token_means = [count_tokens[m]["mean"] for m in all_methods]
    token_stds = [count_tokens[m]["std"] for m in all_methods]
    bar_colors = [method_colors[m] for m in all_methods]

    ax_tokens.bar(all_methods, token_means, color=bar_colors, capsize=5, alpha=0.8)
    ax_tokens.set_title(r"Prompt Tokens ($\downarrow$)")
    ax_tokens.set_ylabel("Token Count")
    ax_tokens.set_xticklabels(all_methods, rotation=35, ha="right")

    # --- Subplot 4: Response Time (Row 1, Col 2) ---
    ax_time = fig.add_subplot(gs[1, 1])
    time_means = [query_times[m]["mean"] for m in all_methods]
    time_stds = [query_times[m]["std"] for m in all_methods]

    ax_time.bar(all_methods, time_means, color=bar_colors, capsize=5, alpha=0.8)
    ax_time.set_title(r"Response Time ($\downarrow$)")
    ax_time.set_ylabel("Time [s]")
    ax_time.set_xticklabels(all_methods, rotation=35, ha="right")

    plt.tight_layout()

    save_path = Path(__file__).parent / "output" / "rag_ablation_layout.pdf"
    plt.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")
    print(f"Saving figure to {save_path}")

    plt.show()
