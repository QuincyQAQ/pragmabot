"""Long-term memory management with text embedding-based retrieval."""

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pragmabot.vlm_client import VLMClient
from pragmabot.conversation_builder import ConversationBuilder
from pragmabot.utils import get_package_path, get_scenario_key

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage long-term memory using text embeddings for similarity-based retrieval.

    LTM is stored across two CSV files:
    - ``ltm.csv``: human-readable entries (time, scenario, experience).
    - ``ltm_<embedding_model>.csv``: embeddings keyed by scenario.

    The two files are merged on the ``scenario`` column at load time.
    """

    def __init__(self, vlm_client: VLMClient, conversation_log: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize the memory manager with a VLM client and load LTM from disk.

        Args:
            vlm_client: Client for querying the VLM API (embeddings).
            conversation_log: Shared mutable list for the Gradio UI log.
        """
        self.vlm_client = vlm_client
        self.conversation_log = conversation_log

        self.ltm_folder_path = get_package_path() / "data" / "ltm"
        self.ltm_path = self.ltm_folder_path / "ltm.csv"

        model_name = vlm_client.config.text_embedding_model
        self.embeddings_path = self.ltm_folder_path / f"ltm_{model_name}.csv"

        self.load()

    @property
    def n_ltm_entries(self) -> int:
        """Return the number of LTM entries."""
        return len(self.df)

    @property
    def n_embedding_entries(self) -> int:
        """Return the number of entries that have a valid embedding."""
        return int(self.df["embedding"].apply(self.is_embedding_valid).sum())

    def load(self) -> None:
        """Load LTM data and embeddings from disk, merging on scenario.

        Reads ``ltm.csv`` and the model-specific embeddings CSV, then merges
        them on the ``scenario`` column via a left join.  Entries without a
        matching embedding will have ``None`` in the ``embedding`` column.
        """
        ltm_df = self._read_csv_safe(self.ltm_path, ["time", "scenario", "experience"])
        emb_df = self._read_csv_safe(self.embeddings_path, ["scenario", "embedding"])

        if not emb_df.empty:
            emb_df["embedding"] = emb_df["embedding"].apply(self._b64_to_embedding)

        if not ltm_df.empty and not emb_df.empty:
            self.df = ltm_df.merge(emb_df, on="scenario", how="left")
        elif not ltm_df.empty:
            logger.warning("Found LTM entries but no embeddings. Initializing with empty embeddings.")
            self.df = ltm_df.copy()
            self.df["embedding"] = None
        else:
            logger.warning("No LTM entries found. Starting with an empty memory.")
            self.df = pd.DataFrame(columns=["time", "scenario", "experience", "embedding"])

        logger.info("Loaded %d LTM entries (%s).", self.n_ltm_entries, self.ltm_path)
        logger.info("Loaded %d embedding entries (%s).", self.n_embedding_entries, self.embeddings_path)

    def retrieve_relevant_experiences(
        self, instruction: str, initial_scene_description: str, top_k: int = 5, use_random_retrieval: bool = False
    ) -> Tuple[List[str], List[float], float, int, str]:
        """Retrieve top-k most similar experiences from long-term memory.

        Args:
            instruction: The user's instruction.
            initial_scene_description: The initial scene description.
            top_k: Number of top similar experiences to return.
            use_random_retrieval: If True, return randomly instead of by similarity.

        Returns:
            A tuple of (experience_list, similarities, embedding_time, embedding_tokens, sorted_str).
        """

        if top_k > 0 and not use_random_retrieval:
            # Generate embedding for the query
            builder = ConversationBuilder(self.conversation_log)
            query = get_scenario_key(instruction, initial_scene_description)
            query_embeddings, embedding_response_time, embedding_prompt_tokens = self.vlm_client.get_text_embedding(
                query, builder
            )
            query_embedding = np.array(query_embeddings[0], dtype=float)

            # Compute similarities on a copy to avoid mutating self.df
            similarities = self.df["embedding"].apply(lambda x: self.cosine_similarity(x, query_embedding))
            sorted_experiences = self.df.assign(similarities=similarities).sort_values("similarities", ascending=False)
            top_k_experiences = sorted_experiences.head(top_k)
        else:
            embedding_response_time, embedding_prompt_tokens = 0.0, 0

            # Random shuffle with zero similarities
            sorted_experiences = self.df.assign(similarities=0.0).sample(frac=1).reset_index(drop=True)
            top_k_experiences = sorted_experiences.head(top_k) if top_k > 0 else sorted_experiences

        top_k_experiences_list = []
        for _, row in top_k_experiences.iterrows():
            exp_dict = {"scenario": row["scenario"], "experience": row["experience"]}
            exp_json = json.dumps(exp_dict, indent=2)
            top_k_experiences_list.append(f"{exp_json}\n")

        top_k_similarities = top_k_experiences["similarities"].tolist()

        sorted_experiences_list = []
        for _, row in sorted_experiences.iterrows():
            sorted_experiences_list.append(f"{row['similarities']:.4f}: {row['scenario']!r}")
        sorted_experiences_str = "\n".join(sorted_experiences_list)

        return (
            top_k_experiences_list,
            top_k_similarities,
            embedding_response_time,
            embedding_prompt_tokens,
            sorted_experiences_str,
        )

    def save_experience(self, instruction: str, initial_scene_description: str, experience: str) -> None:
        """Save a new experience to LTM with its embedding.

        Appends to both ``ltm.csv`` and the embeddings CSV, and updates the
        in-memory DataFrame.

        Args:
            instruction: The task instruction.
            initial_scene_description: The initial scene description.
            experience: The summarized experience text.
        """
        scenario = get_scenario_key(instruction, initial_scene_description)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experience_text = f"Experience: {experience}"

        # Compute embedding
        builder = ConversationBuilder(self.conversation_log)
        embeddings, _, _ = self.vlm_client.get_text_embedding(scenario, builder)
        embedding = embeddings[0]

        # Append to ltm.csv
        ltm_row = pd.DataFrame([{"time": timestamp, "scenario": scenario, "experience": experience_text}])
        needs_header = not self.ltm_path.exists() or self.ltm_path.stat().st_size == 0
        ltm_row.to_csv(self.ltm_path, mode="a", header=needs_header, index=False)

        # Append to embeddings CSV
        emb_row = pd.DataFrame(
            [{"scenario": scenario, "embedding": self._embedding_to_b64(np.array(embedding, dtype=np.float32))}]
        )
        needs_header = not self.embeddings_path.exists() or self.embeddings_path.stat().st_size == 0
        emb_row.to_csv(self.embeddings_path, mode="a", header=needs_header, index=False)

        # Update in-memory dataframe
        new_row = {
            "time": timestamp,
            "scenario": scenario,
            "experience": experience_text,
            "embedding": np.array(embedding, dtype=float),
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        builder.log_user_message(f"New experience saved to LTM ({self.ltm_path.name}).")
        logger.info("New experience saved to LTM: %s", self.ltm_path)

    def build_missing_embeddings(self) -> int:
        """Compute embeddings for LTM entries that don't have one yet.

        Returns:
            The number of new embeddings computed.
        """
        mask = self.df["embedding"].apply(lambda x: not self.is_embedding_valid(x))
        missing = self.df[mask]

        if missing.empty:
            logger.info("All entries already have embeddings.")
            return 0

        scenarios = missing["scenario"].tolist()
        logger.info("Computing embeddings for %d entries...", len(scenarios))

        builder = ConversationBuilder(self.conversation_log)
        embeddings, _, _ = self.vlm_client.get_text_embedding(scenarios, builder)

        for idx, emb in zip(missing.index, embeddings):
            self.df.at[idx, "embedding"] = np.array(emb, dtype=float)

        self._save_embeddings()
        logger.info("Built %d missing embeddings.", len(scenarios))
        return len(scenarios)

    def _save_embeddings(self) -> None:
        """Write the current in-memory embeddings to the embeddings CSV on disk.

        Overwrites the file at ``self.embeddings_path`` with all rows from the
        in-memory DataFrame, serialising each embedding as a Python list string.
        """
        rows = []
        for _, row in self.df.iterrows():
            emb = row["embedding"]
            if self.is_embedding_valid(emb):
                emb_arr = np.array(emb, dtype=np.float32) if not isinstance(emb, np.ndarray) else emb.astype(np.float32)
                emb_str = self._embedding_to_b64(emb_arr)
            else:
                emb_str = ""
            rows.append({"scenario": row["scenario"], "embedding": emb_str})
        emb_df = pd.DataFrame(rows)
        emb_df.to_csv(self.embeddings_path, index=False)

    @staticmethod
    def is_embedding_valid(emb) -> bool:
        """Check whether an embedding value is present (not None/NaN).

        Args:
            emb: The embedding value to check. Can be a NumPy array, None, or NaN.

        Returns:
            True if the embedding is a valid value, False if None or NaN.
        """
        if emb is None:
            return False
        if isinstance(emb, float) and np.isnan(emb):
            return False
        return True

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First input vector
            b: Second input vector.

        Returns:
            The cosine similarity in the range [-1, 1]
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @staticmethod
    def _read_csv_safe(path: Path, columns: list) -> pd.DataFrame:
        """Read a CSV file safely, returning an empty DataFrame if missing or empty.

        Args:
            path: The path to the CSV file.
            columns: Column names for the fallback empty DataFrame.

        Returns:
            A DataFrame with the CSV data, or an empty DataFrame with the
            given columns if the file is missing or empty.
        """

        if path.exists() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                pass
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _embedding_to_b64(emb: np.ndarray) -> str:
        """Encode a NumPy embedding as a base64 string (float32 binary).

        Args:
            emb: 1-D NumPy array to encode.

        Returns:
            Base64-encoded string of the raw float32 bytes.
        """
        return base64.b64encode(emb.astype(np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _b64_to_embedding(s: str) -> Optional[np.ndarray]:
        """Decode a base64 string back to a NumPy float32 array.

        Args:
            s: Base64-encoded string, or empty/None.

        Returns:
            A 1-D NumPy float32 array, or None if the input is missing/empty.
        """
        if not s or (isinstance(s, float) and np.isnan(s)):
            return None
        s = s.strip()
        if not s:
            return None
        return np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()
