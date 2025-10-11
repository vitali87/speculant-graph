import random
from typing import Self

import networkx as nx
from pydantic import BaseModel
from transformers import AutoTokenizer
from loguru import logger


class DraftResult(BaseModel):
    token_ids: list[int]
    strategy: str
    requested_k: int
    actual_length: int
    terminated_early: bool
    termination_reason: str | None = None


class DraftGenerator:

    def __init__(
        self,
        graph: nx.DiGraph,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None
    ):
        self.graph = graph
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        self.most_frequent_token = self._find_most_frequent_token()

    def _find_most_frequent_token(self) -> int:
        max_count = 0
        most_frequent = None

        for node, data in self.graph.nodes(data=True):
            count = data.get('count', 0)
            if count > max_count:
                max_count = count
                most_frequent = node

        logger.debug(f"Most frequent token: {most_frequent} (count: {max_count})")
        return most_frequent

    def generate(
        self,
        prompt: str,
        k: int,
        strategy: str = "greedy"
    ) -> DraftResult:
        token_ids = self.tokenizer.encode(prompt)

        if len(token_ids) == 0:
            logger.warning("Empty prompt, using most frequent token as start")
            start_token = self.most_frequent_token
        else:
            start_token = token_ids[-1]
            start_text = self.tokenizer.decode([start_token])
            logger.debug(f"Last token of prompt: {start_token} ('{start_text}')")

        if start_token not in self.graph:
            start_text = self.tokenizer.decode([start_token])
            fallback_text = self.tokenizer.decode([self.most_frequent_token])
            logger.warning(f"Start token {start_token} ('{start_text}') not in graph, using most frequent token {self.most_frequent_token} ('{fallback_text}')")
            start_token = self.most_frequent_token

        logger.info(f"Drafting {k} tokens using {strategy} strategy...")
        if strategy == "greedy":
            draft_tokens = self._generate_greedy(start_token, k)
        elif strategy == "sampling":
            draft_tokens = self._generate_sampling(start_token, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'greedy' or 'sampling'")

        terminated_early = len(draft_tokens) < k
        termination_reason = None

        if terminated_early:
            if len(draft_tokens) == 0:
                termination_reason = "No successors from start token"
            else:
                termination_reason = "Hit dead end or EOS"

        draft_text = self.tokenizer.decode(draft_tokens) if draft_tokens else "(empty)"
        token_ids_str = ", ".join(str(t) for t in draft_tokens)
        logger.info(f"Draft complete: {len(draft_tokens)}/{k} tokens = '{draft_text}' (IDs: {token_ids_str})")

        return DraftResult(
            token_ids=draft_tokens,
            strategy=strategy,
            requested_k=k,
            actual_length=len(draft_tokens),
            terminated_early=terminated_early,
            termination_reason=termination_reason
        )

    def _generate_greedy(self, start_token: int, k: int) -> list[int]:
        draft = []
        current_token = start_token

        for _ in range(k):
            successors = list(self.graph.successors(current_token))

            if not successors:
                logger.debug(f"No successors for token {current_token}, stopping early")
                break

            best_successor = max(
                successors,
                key=lambda t: self.graph[current_token][t]['weight']
            )

            draft.append(best_successor)
            current_token = best_successor

        return draft

    def _generate_sampling(self, start_token: int, k: int) -> list[int]:
        draft = []
        current_token = start_token

        for _ in range(k):
            successors = list(self.graph.successors(current_token))

            if not successors:
                logger.debug(f"No successors for token {current_token}, stopping early")
                break

            weights = [self.graph[current_token][s]['weight'] for s in successors]

            sampled_successor = random.choices(successors, weights=weights, k=1)[0]

            draft.append(sampled_successor)
            current_token = sampled_successor

        return draft

    @classmethod
    def from_file(
        cls,
        filepath: str,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None
    ) -> Self:
        from speculant_graph.graph_builder import GraphBuilder

        graph, metadata = GraphBuilder.load(filepath)
        return cls(graph, tokenizer_name, hf_token)
