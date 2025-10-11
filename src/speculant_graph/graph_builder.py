import pickle
from pathlib import Path
from datetime import datetime

import networkx as nx
from transformers import AutoTokenizer
from loguru import logger


class GraphBuilder:

    def __init__(
        self,
        tokenizer_name: str = "openai/gpt-oss-20b",
        chunk_size: int = 10000,
        hf_token: str | None = None,
    ):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        self.chunk_size = chunk_size
        self.graph = nx.DiGraph()
        self.token_counts: dict[int, int] = {}
        self.transition_counts: dict[tuple, int] = {}

    def build_from_files(self, corpus_files: list[str]) -> nx.DiGraph:
        logger.info(f"Building graph from {len(corpus_files)} file(s)...")

        for file_path in corpus_files:
            logger.info(f"Processing {file_path}...")
            self._process_file(file_path)

        logger.info("Calculating transition probabilities...")
        self._calculate_probabilities()

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def _process_file(self, file_path: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        token_ids = self.tokenizer.encode(text)
        logger.debug(f"Tokenized {len(token_ids)} tokens from {file_path}")

        for i, token_id in enumerate(token_ids):
            self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1

            if i < len(token_ids) - 1:
                next_token_id = token_ids[i + 1]
                transition = (token_id, next_token_id)
                self.transition_counts[transition] = self.transition_counts.get(transition, 0) + 1

    def _calculate_probabilities(self) -> None:
        for token_id, count in self.token_counts.items():
            token_text = self.tokenizer.decode([token_id])
            self.graph.add_node(
                token_id,
                token_id=token_id,
                text=token_text,
                count=count
            )

        for (from_token, to_token), transition_count in self.transition_counts.items():
            from_count = self.token_counts[from_token]
            probability = transition_count / from_count

            self.graph.add_edge(
                from_token,
                to_token,
                weight=probability,
                count=transition_count
            )

    def save(self, filepath: str) -> None:
        metadata = {
            "graph": self.graph,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_version": self.tokenizer.name_or_path,
            "build_timestamp": datetime.now().isoformat(),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }

        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Graph saved to {filepath}")

    @staticmethod
    def load(filepath: str, validate_tokenizer: bool = True) -> tuple:
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)

        graph = metadata["graph"]

        if validate_tokenizer:
            logger.info(f"Loaded graph built with tokenizer: {metadata['tokenizer_name']}")

        return graph, metadata
