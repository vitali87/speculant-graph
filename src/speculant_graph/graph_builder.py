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
        max_order: int = 5,
        chunk_size: int = 10000,
        hf_token: str | None = None,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_order = max_order
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        self.chunk_size = chunk_size
        self.graph = nx.DiGraph()
        self.token_counts: dict[int, int] = {}
        self.ngram_transition_counts: dict[tuple, dict[int, int]] = {}
        self.context_index: dict[tuple, int] = {}

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

        for order in range(1, self.max_order + 1):
            for i in range(len(token_ids) - order):
                context = tuple(token_ids[i:i + order])
                next_token = token_ids[i + order]

                if context not in self.ngram_transition_counts:
                    self.ngram_transition_counts[context] = {}

                self.ngram_transition_counts[context][next_token] = \
                    self.ngram_transition_counts[context].get(next_token, 0) + 1

                self.context_index[context] = order

    def _calculate_probabilities(self) -> None:
        for token_id, count in self.token_counts.items():
            token_text = self.tokenizer.decode([token_id])
            self.graph.add_node(
                token_id,
                token_id=token_id,
                text=token_text,
                count=count
            )

        logger.info(f"Processing {len(self.ngram_transition_counts)} n-gram contexts...")

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
                    order=order
                )

        logger.debug(f"Built context index with {len(self.context_index)} unique n-grams")

    def save(self, filepath: str) -> None:
        num_nodes_per_order = {}
        num_edges_per_order = {}

        for order in range(1, self.max_order + 1):
            nodes_at_order = [n for n in self.graph.nodes() if isinstance(n, tuple) and len(n) == order]
            edges_at_order = [e for e in self.graph.edges(data=True) if e[2].get('order') == order]
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

        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Graph saved to {filepath}")
        logger.info(f"Max order: {self.max_order}")
        for order in range(1, self.max_order + 1):
            logger.info(f"  Order {order}: {num_nodes_per_order[order]} contexts, {num_edges_per_order[order]} edges")

    @staticmethod
    def load(filepath: str, validate_tokenizer: bool = True) -> tuple:
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)

        graph = metadata["graph"]

        if validate_tokenizer:
            logger.info(f"Loaded graph built with tokenizer: {metadata['tokenizer_name']}")

        return graph, metadata
