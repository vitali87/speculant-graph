import random
from typing import Self, Literal

import networkx as nx
from pydantic import BaseModel
from transformers import AutoTokenizer
from loguru import logger

from speculant_graph.download_utils import configure_download_mode


class DraftResult(BaseModel):
    token_ids: list[int]
    token_probs: list[float]  # P_draft(token) for each token
    strategy: str
    requested_k: int
    actual_length: int
    terminated_early: bool
    termination_reason: str | None = None


class DraftGenerator:

    def __init__(
        self,
        graph: nx.DiGraph,
        context_index: dict[tuple, int],
        max_order: int,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None,
        download_mode: Literal["auto", "hf_transfer", "default"] = "auto"
    ):
        self.graph = graph
        self.context_index = context_index
        self.max_order = max_order

        configure_download_mode(download_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        self.most_frequent_token = self._find_most_frequent_token()

    def _find_most_frequent_token(self) -> int:
        max_count = 0
        most_frequent = None

        for node, data in self.graph.nodes(data=True):
            if isinstance(node, int):
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
            logger.warning("Empty prompt, cannot draft")
            return DraftResult(
                token_ids=[],
                token_probs=[],
                strategy=strategy,
                requested_k=k,
                actual_length=0,
                terminated_early=True,
                termination_reason="Empty prompt"
            )

        logger.info(f"Drafting up to {k} tokens using {strategy} strategy from multi-order graph...")

        draft_tokens = []
        draft_probs = []
        current_context = token_ids.copy()

        while len(draft_tokens) < k:
            matched_order, context_tuple = self._find_highest_order_match(current_context)

            if matched_order == 0:
                logger.debug("No matching context found in any order, stopping draft")
                break

            logger.debug(f"Matched order-{matched_order} context: {context_tuple}")

            if strategy == "greedy":
                next_tokens, next_probs = self._generate_greedy_from_context(context_tuple, k - len(draft_tokens))
            elif strategy == "sampling":
                next_tokens, next_probs = self._generate_sampling_from_context(context_tuple, k - len(draft_tokens))
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'greedy' or 'sampling'")

            if not next_tokens:
                logger.debug(f"No more tokens from order-{matched_order} context, stopping draft")
                break

            draft_tokens.extend(next_tokens)
            draft_probs.extend(next_probs)
            current_context.extend(next_tokens)

            context_text = self.tokenizer.decode(next_tokens)
            logger.debug(f"Generated {len(next_tokens)} tokens from order-{matched_order}: '{context_text}'")

        terminated_early = len(draft_tokens) < k
        termination_reason = None

        if terminated_early:
            if len(draft_tokens) == 0:
                termination_reason = "No matching context in any order"
            else:
                termination_reason = "Hit dead end in graph traversal"

        draft_text = self.tokenizer.decode(draft_tokens) if draft_tokens else "(empty)"
        token_ids_str = ", ".join(str(t) for t in draft_tokens)
        logger.info(f"Draft complete: {len(draft_tokens)}/{k} tokens = '{draft_text}' (IDs: {token_ids_str})")

        return DraftResult(
            token_ids=draft_tokens,
            token_probs=draft_probs,
            strategy=strategy,
            requested_k=k,
            actual_length=len(draft_tokens),
            terminated_early=terminated_early,
            termination_reason=termination_reason
        )

    def _find_highest_order_match(self, context: list[int]) -> tuple[int, tuple | None]:
        for order in range(self.max_order, 0, -1):
            if len(context) < order:
                continue

            context_tuple = tuple(context[-order:])

            if context_tuple in self.context_index:
                return order, context_tuple

        return 0, None

    def _generate_greedy_from_context(self, context: tuple, max_tokens: int) -> tuple[list[int], list[float]]:
        draft = []
        probs = []
        current_context = context

        for _ in range(max_tokens):
            successors = list(self.graph.successors(current_context))

            if not successors:
                break

            best_successor = max(
                successors,
                key=lambda t: self.graph[current_context][t]['weight']
            )
            best_prob = self.graph[current_context][best_successor]['weight']

            draft.append(best_successor)
            probs.append(best_prob)

            order = len(current_context)
            if order < self.max_order:
                current_context = current_context + (best_successor,)
            else:
                current_context = current_context[1:] + (best_successor,)

            if current_context not in self.context_index:
                break

        return draft, probs

    def _generate_sampling_from_context(self, context: tuple, max_tokens: int) -> tuple[list[int], list[float]]:
        draft = []
        probs = []
        current_context = context

        for _ in range(max_tokens):
            successors = list(self.graph.successors(current_context))

            if not successors:
                break

            weights = [self.graph[current_context][s]['weight'] for s in successors]

            sampled_successor = random.choices(successors, weights=weights, k=1)[0]
            sampled_prob = self.graph[current_context][sampled_successor]['weight']

            draft.append(sampled_successor)
            probs.append(sampled_prob)

            order = len(current_context)
            if order < self.max_order:
                current_context = current_context + (sampled_successor,)
            else:
                current_context = current_context[1:] + (sampled_successor,)

            if current_context not in self.context_index:
                break

        return draft, probs

    @classmethod
    def from_file(
        cls,
        filepath: str,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None,
        download_mode: Literal["auto", "hf_transfer", "default"] = "auto"
    ) -> Self:
        from speculant_graph.graph_builder import GraphBuilder

        graph, metadata = GraphBuilder.load(filepath)
        context_index = metadata.get("context_index", {})
        max_order = metadata.get("max_order", 1)

        return cls(graph, context_index, max_order, tokenizer_name, hf_token, download_mode)
