#!/usr/bin/env python3
"""Bar charts comparing picking success and pushing distance with/without image annotation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Set Seaborn dark grid style
sns.set_style("darkgrid")

plt.rc("font", size=14)  # default text sizes
plt.rc("axes", titlesize=16)  # axes title size
plt.rc("axes", labelsize=14)  # x and y label size
plt.rc("xtick", labelsize=14)  # x tick label size
plt.rc("ytick", labelsize=14)  # y tick label size
plt.rc("legend", fontsize=14)  # legend font size
plt.rc("figure", titlesize=14)  # figure title size

# Data for picking
picking_objects = ["box", "mug", "banana", "drumstick", "skewer", "ice cream", "brush"]
picking_w_som = [100, 100, 80, 80, 100, 80, 80]
picking_wo_som = [100, 100, 80, 40, 20, 40, 20]

# Data for pushing
pushing_actions = [
    "egg to sushi",
    "sushi to plate",
    "cherry to banana",
    "grape to banana",
    "screw to toolbox",
    "candy to banana",
    "paper to box",
]
pushing_actions = [action.replace(" to ", " →\n") for action in pushing_actions]
pushing_w_som = [4.45, 4.80, 7.26, 8.35, 3.90, 1.46, 1.37]
pushing_wo_som = [9.08, 12.00, 8.13, 14.86, 10.50, 4.37, 7.25]

# Common bar width
width = 0.35

# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(2, 1, figsize=(8, 5))

# ---- Left Plot: Picking ----
x1 = np.arange(len(picking_objects))
axes[0].bar(x1 - width / 2, picking_w_som, width, label="w/ annotation", alpha=0.8)
axes[0].bar(x1 + width / 2, picking_wo_som, width, label="w/o annotation", alpha=0.8)
axes[0].set_title(r"Success rate for picking objects ($\uparrow$)")
axes[0].set_ylabel("Success rate [%]")
axes[0].set_xticks(x1)
axes[0].set_xticklabels(picking_objects, rotation=0, ha="center")
axes[0].set_ylim(0, 105)
axes[0].legend(loc="lower left")

# ---- Right Plot: Pushing ----
x2 = np.arange(len(pushing_actions))
axes[1].bar(x2 - width / 2, pushing_w_som, width, label="w/ annotation", alpha=0.8)
axes[1].bar(x2 + width / 2, pushing_wo_som, width, label="w/o annotation", alpha=0.8)
axes[1].set_title(r"Distance to target for pushing objects ($\downarrow$)")
axes[1].set_ylabel("Distance error [cm]")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(pushing_actions, rotation=0, ha="center")
axes[1].set_ylim(0, max(pushing_wo_som + pushing_w_som) + 1)
axes[1].set_yticks([6, 12])
axes[1].legend(loc="upper right")

# Adjust layout
plt.tight_layout()

save_path = Path(__file__).parent / "output" / "annotation_ablation.pdf"
plt.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")
print(f"Saved figure to {save_path}")

plt.show()
