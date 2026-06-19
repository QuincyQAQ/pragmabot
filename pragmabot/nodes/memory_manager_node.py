#!/usr/bin/env python3
"""ROS node with Gradio UI for managing Long-Term Memory embeddings."""

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
import numpy as np
import rospy
from openai import OpenAI

from pragmabot.memory_manager import MemoryManager
from pragmabot.simple_config import get_config
from pragmabot.vlm_client import VLMClient
from pragmabot.utils import get_scenario_key

logger = logging.getLogger(__name__)


class MemoryManagerRos:
    """ROS node providing a Gradio UI for LTM inspection and embedding management."""

    def __init__(self) -> None:
        """Initialize the ROS node, VLM client, and memory manager."""
        rospy.init_node("memory_manager_node")

        logging.basicConfig(
            level=logging.INFO, format="[%(levelname)s] [%(name)s]: %(message)s", stream=sys.stdout, force=True
        )

        self.config = get_config()

        # Initialize the OpenAI client based on configured model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        rospy.loginfo("Using %s with OpenAI", self.config.vlm.text_embedding_model)

        self.vlm_client = VLMClient(self.client, self.config.vlm)
        self.memory_manager = MemoryManager(self.vlm_client)

        self.start_gradio_interface()

    def run(self) -> None:
        """Block on rospy.spin() until shutdown."""
        rospy.spin()

    def start_gradio_interface(self) -> None:
        """Build and launch the Gradio web UI for LTM inspection and embedding management."""

        def on_refresh():
            """Reload LTM data from disk and refresh the table and heatmap.

            Returns:
                A tuple of (status_markdown, table_rows, heatmap_figure).
            """
            self.memory_manager.load()
            rows, fig = self._render_memory_table()
            return self._build_status_text(), rows, fig

        def on_build_missing(progress=gr.Progress(track_tqdm=True)):
            """Build embeddings for entries that don't have one yet.

            Args:
                progress: Gradio progress tracker for tqdm integration.

            Returns:
                A tuple of (status_markdown, table_rows, heatmap_figure).
            """
            try:
                count = self.memory_manager.build_missing_embeddings()
                rows, fig = self._render_memory_table()
                status_text = self._build_status_text()
                if count > 0:
                    return f"{status_text}\n\nBuilt **{count}** missing embeddings.", rows, fig
                else:
                    return f"{status_text}\n\nAll entries already have embeddings.", rows, fig
            except Exception as e:
                rospy.logerr("Failed to build embeddings: %s", e)
                return f"Error: {e}", [], None

        def on_query_similarity(instruction_text, scene_text):
            """Embed the query scenario and return the table sorted by similarity.

            Args:
                instruction_text: The instruction string.
                scene_text: The scene description string.

            Returns:
                A tuple of (status_markdown, table_rows, heatmap_figure).
            """
            try:
                scenario_text = get_scenario_key(instruction_text, scene_text)
                return self._query_similarity(scenario_text)
            except Exception as e:
                rospy.logerr("Failed to query similarity: %s", e)
                return f"Error: {e}", [], None

        custom_css = """
        .status-box {
            padding: 8px 12px;
            border-radius: 6px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            min-height: 40px;
        }

        .right-sidebar {
            border-left: 1px solid #e5e7eb;
            padding-left: 20px !important;
        }
        """

        # Pre-render initial table and heatmap from already-loaded data
        initial_rows, initial_fig = self._render_memory_table()
        initial_status = self._build_status_text()

        with gr.Blocks(css=custom_css, title="LTM Manager") as demo:
            with gr.Row():
                with gr.Column(scale=4):
                    memory_table = gr.Dataframe(
                        headers=["#", "Time", "Scenario", "Experience", "Embedding", "Similarity"],
                        datatype=["number", "str", "str", "str", "str", "number"],
                        col_count=(6, "fixed"),
                        wrap=True,
                        interactive=False,
                        value=initial_rows,
                    )
                    heatmap = gr.Plot(label="Cosine Similarity Heatmap", value=initial_fig)

                with gr.Column(scale=1, elem_classes="right-sidebar"):
                    gr.Markdown("### 📔 Similarity")
                    instruction_input = gr.Textbox(
                        label="Instruction",
                        placeholder="Enter the task instruction...",
                    )
                    scene_input = gr.Textbox(
                        label="Scene",
                        placeholder="Enter the scene description...",
                    )
                    query_button = gr.Button("🔍 Search", variant="secondary")

                    gr.HTML("<hr style='margin-top: 30px; margin-bottom: 30px;'>")

                    gr.Markdown("### 🤖 Memory Management")
                    with gr.Row():
                        refresh_button = gr.Button("🔄 Refresh from Disk")
                        build_button = gr.Button("🔨 Build Missing Embeddings", variant="primary")

            status = gr.Markdown(
                value=initial_status,
                elem_classes=["status-box"],
            )

            refresh_button.click(
                on_refresh,
                inputs=[],
                outputs=[status, memory_table, heatmap],
            )
            build_button.click(
                on_build_missing,
                inputs=[],
                outputs=[status, memory_table, heatmap],
            )
            query_button.click(
                on_query_similarity,
                inputs=[instruction_input, scene_input],
                outputs=[status, memory_table, heatmap],
            )

        demo.launch(share=self.config.gradio_share, inline=False, prevent_thread_lock=True, server_name="0.0.0.0")

    def _render_memory_table(self, query_embedding: np.ndarray = None) -> tuple:
        """Build a unified table and cosine-similarity heatmap from in-memory LTM data.

        Args:
            query_embedding: Optional query embedding vector. When provided, a
                "Similarity" column is added and rows are sorted descending by score.

        Returns:
            A tuple of (table_rows, heatmap_figure).
        """
        if self.memory_manager.df.empty:
            return [], None

        table_rows = []
        embedding_matrix = []

        for i, row in self.memory_manager.df.iterrows():
            emb = row.get("embedding", None)
            emb_str = ""
            sim_score = None
            if self.memory_manager.is_embedding_valid(emb):
                emb_str = self._truncate_embedding(emb)
                embedding_matrix.append(emb)
                if query_embedding is not None:
                    sim_score = self.memory_manager.cosine_similarity(query_embedding, emb)
                    sim_score = round(sim_score, 4) if sim_score is not None else None

            table_rows.append(
                [
                    i + 1,
                    row.get("time", ""),
                    row.get("scenario", ""),
                    row.get("experience", ""),
                    emb_str,
                    sim_score,
                ]
            )

        # Sort by similarity descending when a query is provided
        if query_embedding is not None:
            table_rows.sort(key=lambda r: r[5] if r[5] is not None else -2.0, reverse=True)

        # Build cosine similarity heatmap
        fig = None
        if len(embedding_matrix) > 0:
            mat = np.array(embedding_matrix)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            mat_normalized = mat / norms
            similarity = mat_normalized @ mat_normalized.T

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(similarity, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_title("Cosine Similarity Between LTM Embeddings", fontsize=14)
            ax.set_xlabel("Experience Index")
            ax.set_ylabel("Experience Index")
            fig.colorbar(im, ax=ax, label="Cosine Similarity")
            fig.tight_layout()

        return table_rows, fig

    def _query_similarity(self, scenario_text: str) -> tuple:
        """Embed a query scenario and return the LTM table sorted by cosine similarity.

        Args:
            scenario_text: The scenario string to embed and compare against LTM.

        Returns:
            A tuple of (status_text, table_rows, heatmap_figure).
        """
        if not scenario_text or not scenario_text.strip():
            rows, fig = self._render_memory_table()
            return self._build_status_text() + "\n\nPlease enter a scenario to query.", rows, fig

        embeddings, _, _ = self.vlm_client.get_text_embedding(scenario_text.strip())
        query_emb = np.array(embeddings[0], dtype=float)

        rows, fig = self._render_memory_table(query_embedding=query_emb)
        status = self._build_status_text() + f'\n\nShowing similarity to: *"{scenario_text.strip()}"*'
        return status, rows, fig

    def _build_status_text(self) -> str:
        """Build the status text showing loaded entry counts.

        Returns:
            A markdown string with LTM and embedding file entry counts.
        """
        mm = self.memory_manager
        return (
            f"LTM: `{mm.ltm_path.resolve()}` — **{mm.n_ltm_entries}** entries\n\n"
            f"Embeddings: `{mm.embeddings_path.resolve()}` — **{mm.n_embedding_entries}** entries"
        )

    @staticmethod
    def _truncate_embedding(emb: np.ndarray, max_elements: int = 4) -> str:
        """Return a truncated string preview of an embedding vector.

        Args:
            emb: The embedding array.
            max_elements: Number of leading elements to show.

        Returns:
            A string like "[0.012, -0.034, 0.056, 0.078, ...]  (1536-d, norm=1.00)".
        """
        preview = ", ".join(f"{v:.3f}" for v in emb[:max_elements])
        norm = float(np.linalg.norm(emb))
        return f"[{preview}, ...]  ({len(emb)}-d, norm={norm:.2f})"


if __name__ == "__main__":
    manager = MemoryManagerRos()

    try:
        manager.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down...")
    except Exception as e:
        rospy.logerr("Unexpected error: %s", e)
    finally:
        rospy.loginfo("LTM Embeddings node terminated.")
