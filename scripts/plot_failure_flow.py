#!/usr/bin/env python3
"""Generate a Sankey diagram of failure categories across evaluation tasks."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import webcolors


# Helper function to wrap long labels
def wrap_label(text, max_chars=23):
    """
    Insert <br> tags into the text every max_chars characters,
    trying to break at spaces.
    """
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "<br>".join(lines)


# Define the data
data = {
    "Task (failures)": [
        "Put apple on plate (container obstructs) (0/7)",
        "Move tiny candy (towel nearby) (2/9)",
        "Move egg (open view) (0/5)",
        "Pick up bowl (apple inside) (1/6)",
        "Put tennis ball in box (mug obstructs) (2/7)",
        "Put orange/ball on plate (fan blocks) (2/10)",
        "Move crumpled paper (brush nearby) (3/8)",
        "Move screw (towel nearby) (1/7)",
        "Move sushi (open view) (2/7)",
        "Move grape/cherry (open view) (3/10)",
        "Pick up box (apple on top) (1/7)",
        "Pick up towel (orange on top) (2/8)",
    ],
    "Retrieval Failed": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    "Overrides Experience": [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    "False Detection": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Wrong Mask Selection": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Poor Grasp Generation": [0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1],
    "Inaccurate Depth Value": [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    "Success": [7, 7, 5, 5, 5, 8, 5, 6, 5, 7, 6, 6],
    "Total": [7, 9, 5, 6, 7, 10, 8, 7, 7, 10, 7, 8],
}

df = pd.DataFrame(data)

# Total number of failures
total_success = df["Success"].sum()
total_failures = (
    df[
        [
            "Retrieval Failed",
            "Overrides Experience",
            "False Detection",
            "Wrong Mask Selection",
            "Poor Grasp Generation",
            "Inaccurate Depth Value",
        ]
    ]
    .sum()
    .sum()
)

# Define nodes (correct order and hierarchy)
nodes = [
    "All Trials",  # 0
    "Success",  # 1
    "Failure",  # 2
    "RAG",  # 3
    "Retrieval Failed",  # 4
    "VLM",  # 5
    "Overrides Experience",  # 6
    "False Detection",  # 7
    "Wrong Mask Selection",  # 8
    "Execution",  # 9
    "Poor Grasp Generation",  # 10
    "Inaccurate Depth Value",  # 11
]

# Define colors for nodes
node_colors = [
    "#888888",  # All Trials (Neutral Gray)
    "#4CAF50",  # Success (Vibrant Green)
    "#F44336",  # Failure (Vibrant Red)
    "#2196F3",  # RAG (Blue)
    "#9C27B0",  # Not retrieved (Purple)
    "#FF9800",  # VLM (Orange)
    "#009688",  # Do not follow (Teal)
    "#795548",  # False negative (Brown)
    "#607D8B",  # Wrong number (Blue Gray)
    "#E91E63",  # Execution (Pink)
    "#8B0000",  # Poor Grasp Generation (Dark Red/Burgundy) - Changed from green
    "#FF5722",  # Bad depth (Deep Orange)
]

# Create links
links = []

# All Trials → Success and Failure
links.append({"source": 0, "target": 1, "value": total_success})
links.append({"source": 0, "target": 2, "value": total_failures})

# Failure → High-level categories
mem_retrieval_value = df["Retrieval Failed"].sum()
vlm_reasoning_value = df["Overrides Experience"].sum() + df["False Detection"].sum() + df["Wrong Mask Selection"].sum()
low_level_value = df["Poor Grasp Generation"].sum() + df["Inaccurate Depth Value"].sum()

links.append({"source": 2, "target": 3, "value": mem_retrieval_value})  # Failure → RAG
links.append({"source": 2, "target": 5, "value": vlm_reasoning_value})  # Failure → VLM
links.append({"source": 2, "target": 9, "value": low_level_value})  # Failure → Execution

# RAG → Only one child
links.append({"source": 3, "target": 4, "value": mem_retrieval_value})

# VLM → its three children
links.append({"source": 5, "target": 6, "value": df["Overrides Experience"].sum()})
links.append({"source": 5, "target": 7, "value": df["False Detection"].sum()})
links.append({"source": 5, "target": 8, "value": df["Wrong Mask Selection"].sum()})

# Execution → children
links.append({"source": 9, "target": 10, "value": df["Poor Grasp Generation"].sum()})
links.append({"source": 9, "target": 11, "value": df["Inaccurate Depth Value"].sum()})


# Function to convert color names to RGBA with transparency
def color_to_rgba(color_name, alpha=0.5):
    try:
        # Handle hex colors
        if color_name.startswith("#"):
            rgb = webcolors.hex_to_rgb(color_name)
        # Handle rgb() colors
        elif color_name.startswith("rgb("):
            rgb_vals = color_name[4:-1].split(",")
            rgb = tuple(int(val.strip()) for val in rgb_vals)
        # Handle named colors
        else:
            rgb = webcolors.name_to_rgb(color_name)
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    except (ValueError, AttributeError):
        # Fallback color if conversion fails
        return f"rgba(128, 128, 128, {alpha})"


# Create semi-transparent link colors based on target node color
link_colors = []
for link in links:
    target_color = node_colors[link["target"]]
    rgba_color = color_to_rgba(target_color, alpha=0.3)  # 50% transparency
    link_colors.append(rgba_color)

# Format node labels with text (number) format
formatted_labels = [
    f"<b>{wrap_label(nodes[0])} ({total_success + total_failures})</b>",  # All Trials
    f"<b>{wrap_label(nodes[1])} ({total_success})</b>",  # Success
    f"<b>{wrap_label(nodes[2])} ({total_failures})</b>",  # Failure
    f"<b>{wrap_label(nodes[3])} ({mem_retrieval_value})</b>",  # RAG
    f"<b>{wrap_label(nodes[4])} ({df['Retrieval Failed'].sum()})</b>",  # Not retrieved
    f"<b>{wrap_label(nodes[5])} ({vlm_reasoning_value})</b>",  # VLM
    f"<b>{wrap_label(nodes[6])} ({df['Overrides Experience'].sum()})</b>",  # Do not follow
    f"<b>{wrap_label(nodes[7])} ({df['False Detection'].sum()})</b>",  # False negative
    f"<b>{wrap_label(nodes[8])} ({df['Wrong Mask Selection'].sum()})</b>",  # Wrong number
    f"<b>{wrap_label(nodes[9])} ({low_level_value})</b>",  # Execution
    f"<b>{wrap_label(nodes[10])} ({df['Poor Grasp Generation'].sum()})</b>",  # Bad grasp
    f"<b>{wrap_label(nodes[11])} ({df['Inaccurate Depth Value'].sum()})</b>",  # Bad depth
]

# Define x positions for columns
x_positions = {
    "All Trials": 0.0,
    "Success": 0.2,
    "Failure": 0.2,
    "RAG": 0.4,
    "Retrieval Failed": 0.75,
    "VLM": 0.4,
    "Overrides Experience": 0.75,
    "False Detection": 0.75,
    "Wrong Mask Selection": 0.75,
    "Execution": 0.4,
    "Poor Grasp Generation": 0.75,
    "Inaccurate Depth Value": 0.75,
}

# Define y positions to align nodes vertically
y_positions = {
    "All Trials": 0.5,
    "Success": 0.75,
    "Failure": 0.25,
    "RAG": 0.9,
    "Retrieval Failed": 1.0,
    "VLM": 0.7,
    "Overrides Experience": 0.8,
    "False Detection": 0.6,
    "Wrong Mask Selection": 0.7,
    "Execution": 0.5,
    "Poor Grasp Generation": 0.3,
    "Inaccurate Depth Value": 0.4,
}

# Map node names to x and y positions
node_x = [x_positions[node] for node in nodes]
node_y = [y_positions[node] for node in nodes]

# Create Sankey diagram with fixed positions
fig = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",  # Helps with alignment
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=formatted_labels,
                color=node_colors,
                x=node_x,  # Add x positions
                y=node_y,  # Add y positions
            ),
            link=dict(
                source=[link["source"] for link in links],
                target=[link["target"] for link in links],
                value=[link["value"] for link in links],
                label=[f"{link['value']}" for link in links],
                color=link_colors,  # Edge colors with transparency
            ),
        )
    ]
)

# Update layout with larger font and centered title
fig.update_layout(
    font_size=35,  # Larger font
    title_font_size=35,
    height=600,
    width=2200,
    title_x=0.5,
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
)

# Save to HTML
save_path = Path(__file__).parent / "output" / "sankey_failure_analysis.html"
fig.write_html(save_path)

print(f"Sankey diagram saved to '{save_path}'")
