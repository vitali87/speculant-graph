import math
import random
from collections import defaultdict
from typing import Self, Literal

import networkx as nx
from pydantic import BaseModel
from transformers import AutoTokenizer
from loguru import logger

from speculant_graph.config import DraftConfig
from speculant_graph.download_utils import configure_download_mode


class DraftResult(BaseModel):
    token_ids: list[int]
    token_probs: list[float]
    matched_contexts: list[tuple]
    successors: list[list[int]]
    successor_weights: list[list[float]]
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
        config: DraftConfig | None = None,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None,
        download_mode: Literal["auto", "hf_transfer", "default"] = "auto",
    ):
        self.graph = graph
        self.context_index = context_index
        self.max_order = max_order
        self.config = config or DraftConfig()

        configure_download_mode(download_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

    def generate(self, prompt: str, k: int, strategy: str = "greedy") -> DraftResult:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if len(token_ids) == 0:
            logger.warning("Empty prompt, cannot draft")
            return DraftResult(
                token_ids=[],
                token_probs=[],
                matched_contexts=[],
                successors=[],
                successor_weights=[],
                strategy=strategy,
                requested_k=k,
                actual_length=0,
                terminated_early=True,
                termination_reason="Empty prompt",
            )

        logger.info(
            f"Drafting up to {k} tokens using {strategy} strategy from multi-order graph..."
        )

        draft_tokens = []
        draft_probs = []
        matched_contexts = []
        successors_list = []
        successor_weights_list = []
        current_context = token_ids.copy()

        while len(draft_tokens) < k:
            if self.config.attentive_mix:
                # Use attentive mixing of multiple orders
                q_mix = self._mix_contexts(current_context)

                if not q_mix:
                    logger.debug("No matching context, stopping draft")
                    break

                # Sample or greedy from mixture
                if strategy == "greedy":
                    next_tok = max(q_mix, key=q_mix.get)
                    next_prob = 1.0
                else:
                    toks, probs = list(q_mix.keys()), list(q_mix.values())
                    next_tok = random.choices(toks, weights=probs, k=1)[0]
                    next_prob = q_mix[next_tok]

                draft_tokens.append(next_tok)
                draft_probs.append(next_prob)
                matched_contexts.append(tuple(current_context[-1:]))
                successors_list.append(list(q_mix.keys()))
                successor_weights_list.append(list(q_mix.values()))
                current_context.append(next_tok)

                tok_text = self.tokenizer.decode([next_tok])
                logger.debug(f"Drafted (attentive): '{tok_text}'")

            else:
                # Original: single highest order
                matched_order, context_tuple = self._find_highest_order_match(
                    current_context
                )

                if matched_order == 0:
                    logger.debug("No matching context, stopping draft")
                    break

                logger.debug(f"Matched order-{matched_order} context: {context_tuple}")

                if strategy == "greedy":
                    next_tokens, next_probs, contexts, succs, weights = (
                        self._generate_greedy_from_context(
                            context_tuple, k - len(draft_tokens)
                        )
                    )
                elif strategy == "sampling":
                    next_tokens, next_probs, contexts, succs, weights = (
                        self._generate_sampling_from_context(
                            context_tuple, k - len(draft_tokens)
                        )
                    )
                else:
                    raise ValueError(f"Unknown strategy: '{strategy}'")

                if not next_tokens:
                    logger.debug("No more tokens, stopping draft")
                    break

                draft_tokens.extend(next_tokens)
                draft_probs.extend(next_probs)
                matched_contexts.extend(contexts)
                successors_list.extend(succs)
                successor_weights_list.extend(weights)
                current_context.extend(next_tokens)

                context_text = self.tokenizer.decode(next_tokens)
                logger.debug(f"Generated: '{context_text}'")

        terminated_early = len(draft_tokens) < k
        termination_reason = None

        if terminated_early:
            if len(draft_tokens) == 0:
                termination_reason = "No matching context in any order"
            else:
                termination_reason = "Hit dead end in graph traversal"

        draft_text = self.tokenizer.decode(draft_tokens) if draft_tokens else "(empty)"
        token_ids_str = ", ".join(str(t) for t in draft_tokens)
        logger.info(
            f"Draft complete: {len(draft_tokens)}/{k} tokens = '{draft_text}' (IDs: {token_ids_str})"
        )

        return DraftResult(
            token_ids=draft_tokens,
            token_probs=draft_probs,
            matched_contexts=matched_contexts,
            successors=successors_list,
            successor_weights=successor_weights_list,
            strategy=strategy,
            requested_k=k,
            actual_length=len(draft_tokens),
            terminated_early=terminated_early,
            termination_reason=termination_reason,
        )

    def _find_highest_order_match(self, context: list[int]) -> tuple[int, tuple | None]:
        for order in range(self.max_order, 0, -1):
            if len(context) < order:
                continue

            context_tuple = tuple(context[-order:])

            if context_tuple in self.context_index:
                return order, context_tuple

        return 0, None

    def _generate_greedy_from_context(self, context: tuple, max_tokens: int):
        draft = []
        probs = []
        contexts = []
        successors_list = []
        weights_list = []
        current_context = context

        for _ in range(max_tokens):
            successors = list(self.graph.successors(current_context))

            if not successors:
                break

            weights = [self.graph[current_context][s]["weight"] for s in successors]
            best_successor = max(
                successors, key=lambda t: self.graph[current_context][t]["weight"]
            )

            draft.append(best_successor)
            probs.append(1.0)
            contexts.append(current_context)
            successors_list.append(successors)
            weights_list.append(weights)

            order = len(current_context)
            if order < self.max_order:
                current_context = current_context + (best_successor,)
            else:
                current_context = current_context[1:] + (best_successor,)

            if current_context not in self.context_index:
                break

        return draft, probs, contexts, successors_list, weights_list

    def _generate_sampling_from_context(self, context: tuple, max_tokens: int):
        draft = []
        probs = []
        contexts = []
        successors_list = []
        weights_list = []
        current_context = context

        for _ in range(max_tokens):
            successors = list(self.graph.successors(current_context))

            if not successors:
                break

            weights = [self.graph[current_context][s]["weight"] for s in successors]

            sampled_successor = random.choices(successors, weights=weights, k=1)[0]
            sampled_prob = self.graph[current_context][sampled_successor]["weight"]

            draft.append(sampled_successor)
            probs.append(sampled_prob)
            contexts.append(current_context)
            successors_list.append(successors)
            weights_list.append(weights)

            order = len(current_context)
            if order < self.max_order:
                current_context = current_context + (sampled_successor,)
            else:
                current_context = current_context[1:] + (sampled_successor,)

            if current_context not in self.context_index:
                break

        return draft, probs, contexts, successors_list, weights_list

    def get_most_frequent_token(self) -> int:
        """
        Returns the token with the highest count from the graph.
        Used as a fallback for empty prompts.
        """
        max_count = 0
        most_frequent = 0

        for node in self.graph.nodes():
            # Only check token nodes (int)
            if isinstance(node, int):
                node_data = self.graph.nodes[node]
                count = node_data.get("count", 0)
                if count > max_count:
                    max_count = count
                    most_frequent = node

        logger.debug(f"Most frequent token: {most_frequent} (count: {max_count})")
        return most_frequent

    def _get_successor_dist(self, ctx: tuple[int, ...]) -> dict[int, float]:
        """Get {token_id: prob} distribution for a context."""
        dist = {}
        for _, tok, attr in self.graph.out_edges(ctx, data=True):
            dist[tok] = float(attr.get("weight", 0.0))
        return dist

    def _mix_contexts(self, context: list[int]) -> dict[int, float]:
        """
        Mix multiple order contexts with attention weights.
        Returns q_mix = Σ attention_j * P_j(·)
        """
        orders, contexts = [], []

        # Collect all matching contexts from high to low order
        for o in range(self.max_order, 0, -1):
            if len(context) < o:
                continue
            ctx = tuple(context[-o:])
            if ctx in self.context_index:
                orders.append(o)
                contexts.append(ctx)

        if not contexts:
            return {}

        # Compute attention weights: prefer higher orders
        beta = self.config.order_bias
        tau = self.config.mix_temperature
        scores = [beta * (o - 1) / tau for o in orders]

        # Numerically stable softmax
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        Z = sum(exps)
        attn_weights = [e / Z for e in exps]

        logger.debug(
            f"Mixing {len(contexts)} orders: "
            f"{list(zip(orders, [f'{a:.3f}' for a in attn_weights]))}"
        )

        # Mix successor distributions
        q_mix = defaultdict(float)
        for a, ctx in zip(attn_weights, contexts):
            P_j = self._get_successor_dist(ctx)
            for tok, prob in P_j.items():
                q_mix[tok] += a * prob

        # Normalize (defensive)
        Z_q = sum(q_mix.values())
        if Z_q > 0:
            for tok in list(q_mix.keys()):
                q_mix[tok] /= Z_q

        return dict(q_mix)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        tokenizer_name: str = "openai/gpt-oss-20b",
        hf_token: str | None = None,
        download_mode: Literal["auto", "hf_transfer", "default"] = "auto",
        config: DraftConfig | None = None,
    ) -> Self:
        from speculant_graph.graph_builder import GraphBuilder

        graph, metadata = GraphBuilder.load(
            filepath, validate_tokenizer=True, expected_tokenizer=tokenizer_name
        )
        context_index = metadata.get("context_index", {})
        max_order = metadata.get("max_order", 1)

        return cls(
            graph=graph,
            context_index=context_index,
            max_order=max_order,
            config=config,
            tokenizer_name=tokenizer_name,
            hf_token=hf_token,
            download_mode=download_mode,
        )
