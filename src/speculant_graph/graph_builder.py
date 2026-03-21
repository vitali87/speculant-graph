import pickle
from pathlib import Path
from datetime import datetime
from typing import Literal

import networkx as nx
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from speculant_graph.download_utils import configure_download_mode

_ALLOWED_PICKLE_MODULES = frozenset({
    "networkx.classes.digraph",
    "networkx.classes.coreviews",
    "networkx.classes.reportviews",
})


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module in _ALLOWED_PICKLE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to unpickle disallowed class: {module}.{name}"
        )


class GraphBuilder:
    def __init__(
        self,
        tokenizer_name: str = "openai/gpt-oss-20b",
        max_order: int = 5,
        chunk_size: int = 10000,
        hf_token: str | None = None,
        download_mode: Literal["auto", "hf_transfer", "default"] = "auto",
    ):
        self.tokenizer_name = tokenizer_name
        self.max_order = max_order

        configure_download_mode(download_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=hf_token,
            model_max_length=None,  # Suppress max length warnings
        )
        self.chunk_size = chunk_size
        self.graph = nx.DiGraph()
        self.token_counts: dict[int, int] = {}
        self.ngram_transition_counts: dict[tuple, dict[int, int]] = {}
        self.context_index: dict[tuple, int] = {}

    def build_from_files(self, corpus_files: list[str]) -> nx.DiGraph:
        logger.info(f"Building graph from {len(corpus_files)} file(s)...")

        cross_file_context = []

        if self.tokenizer.bos_token_id is not None:
            cross_file_context = [self.tokenizer.bos_token_id]
            logger.debug(f"Added BOS token: {self.tokenizer.bos_token_id}")

        for i, file_path in enumerate(corpus_files):
            logger.info(f"Processing {file_path}...")
            is_last_file = i == len(corpus_files) - 1
            cross_file_context = self._process_file(
                file_path, cross_file_context, is_last_file
            )

        logger.info("Calculating transition probabilities...")
        self._calculate_probabilities()

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        return self.graph

    _READ_CHUNK_CHARS = 1_000_000  # 1MB text chunks for streaming file reads

    def _process_file(
        self, file_path: str, prev_context: list[int], is_last_file: bool
    ) -> list[int]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = path.stat().st_size
        total_token_count = 0
        carry_over = prev_context.copy()

        with open(path, "r", encoding="utf-8") as f, tqdm(
            total=file_size,
            desc=f"  Reading {path.name}",
            unit="B",
            unit_scale=True,
            leave=False,
        ) as pbar:
            leftover = ""
            while True:
                raw = f.read(self._READ_CHUNK_CHARS)
                if not raw:
                    # Process any remaining leftover text
                    if leftover:
                        token_ids = self.tokenizer.encode(
                            leftover, add_special_tokens=False
                        )
                        if is_last_file and self.tokenizer.eos_token_id is not None:
                            token_ids.append(self.tokenizer.eos_token_id)
                        carry_over = self._process_token_chunk(
                            token_ids, carry_over
                        )
                        total_token_count += len(token_ids)
                    elif is_last_file and self.tokenizer.eos_token_id is not None:
                        carry_over = self._process_token_chunk(
                            [self.tokenizer.eos_token_id], carry_over
                        )
                        total_token_count += 1
                    break

                pbar.update(len(raw.encode("utf-8")))
                text = leftover + raw

                # Split at last whitespace to avoid cutting words
                last_ws = text.rfind(" ")
                if last_ws == -1 or not f.readable():
                    chunk_text = text
                    leftover = ""
                else:
                    chunk_text = text[:last_ws]
                    leftover = text[last_ws:]

                token_ids = self.tokenizer.encode(
                    chunk_text, add_special_tokens=False
                )
                carry_over = self._process_token_chunk(token_ids, carry_over)
                total_token_count += len(token_ids)

        if is_last_file and self.tokenizer.eos_token_id is not None:
            logger.debug(f"Added EOS token: {self.tokenizer.eos_token_id}")

        logger.info(
            f"Tokenized {total_token_count:,} tokens from {path.name}"
        )

        return (
            carry_over[-(self.max_order - 1) :]
            if len(carry_over) >= self.max_order - 1
            else carry_over
        )

    def _process_token_chunk(
        self, token_ids: list[int], prev_context: list[int]
    ) -> list[int]:
        combined_tokens = prev_context + token_ids

        for token_id in token_ids:
            self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1

        chunk_overlap = self.max_order - 1
        for chunk_start in range(0, len(combined_tokens), self.chunk_size):
            chunk_end = min(
                chunk_start + self.chunk_size + chunk_overlap, len(combined_tokens)
            )
            chunk_tokens = combined_tokens[chunk_start:chunk_end]

            for order in range(1, self.max_order + 1):
                for i in range(len(chunk_tokens) - order):
                    context = tuple(chunk_tokens[i : i + order])
                    next_token = chunk_tokens[i + order]

                    if context not in self.ngram_transition_counts:
                        self.ngram_transition_counts[context] = {}

                    self.ngram_transition_counts[context][next_token] = (
                        self.ngram_transition_counts[context].get(next_token, 0) + 1
                    )

                    self.context_index[context] = order

        # Return the tail needed for cross-chunk context continuity
        return (
            combined_tokens[-(self.max_order - 1) :]
            if len(combined_tokens) >= self.max_order - 1
            else combined_tokens
        )

    def _calculate_probabilities(self) -> None:
        logger.info(f"Adding {len(self.token_counts)} token nodes...")
        for token_id, count in self.token_counts.items():
            token_text = self.tokenizer.decode([token_id])
            self.graph.add_node(
                token_id, token_id=token_id, text=token_text, count=count
            )

        logger.info(
            f"Processing {len(self.ngram_transition_counts)} n-gram contexts..."
        )

        with tqdm(
            total=len(self.ngram_transition_counts),
            desc="Building graph edges",
            unit="contexts",
            unit_scale=True,
        ) as pbar:
            for context, next_token_counts in self.ngram_transition_counts.items():
                total_count = sum(next_token_counts.values())

                for next_token, count in next_token_counts.items():
                    probability = count / total_count
                    order = len(context)

                    self.graph.add_edge(
                        context,
                        next_token,
                        weight=probability,
                        count=count,
                        order=order,
                    )

                pbar.update(1)

        logger.info(
            f"Built context index with {len(self.context_index)} unique n-grams"
        )

    def save(self, filepath: str) -> None:
        num_nodes_per_order = {}
        num_edges_per_order = {}

        for order in range(1, self.max_order + 1):
            nodes_at_order = [
                n
                for n in self.graph.nodes()
                if isinstance(n, tuple) and len(n) == order
            ]
            edges_at_order = [
                e for e in self.graph.edges(data=True) if e[2].get("order") == order
            ]
            num_nodes_per_order[order] = len(nodes_at_order)
            num_edges_per_order[order] = len(edges_at_order)

        metadata = {
            "graph": self.graph,
            "context_index": self.context_index,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_version": self.tokenizer.name_or_path,
            "build_timestamp": datetime.now().isoformat(),
            "max_order": self.max_order,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_nodes_per_order": num_nodes_per_order,
            "num_edges_per_order": num_edges_per_order,
        }

        with open(filepath, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Graph saved to {filepath}")
        logger.info(f"Max order: {self.max_order}")
        for order in range(1, self.max_order + 1):
            logger.info(
                f"  Order {order}: {num_nodes_per_order[order]} contexts, {num_edges_per_order[order]} edges"
            )

    @staticmethod
    def load(
        filepath: str,
        validate_tokenizer: bool = True,
        expected_tokenizer: str | None = None,
    ) -> tuple:
        with open(filepath, "rb") as f:
            metadata = _RestrictedUnpickler(f).load()

        graph = metadata["graph"]
        saved_tokenizer = metadata.get("tokenizer_name", "unknown")

        if validate_tokenizer and expected_tokenizer:
            if saved_tokenizer != expected_tokenizer:
                raise ValueError(
                    f"Tokenizer mismatch: graph built with '{saved_tokenizer}' "
                    f"but verifier uses '{expected_tokenizer}'. "
                    f"Token IDs will not align. Rebuild the graph with the correct tokenizer."
                )
            logger.info(f"✓ Tokenizer validated: {saved_tokenizer}")
        else:
            logger.info(f"Loaded graph built with tokenizer: {saved_tokenizer}")

        return graph, metadata
