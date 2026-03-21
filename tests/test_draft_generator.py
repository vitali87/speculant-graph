import random

import pytest

from speculant_graph.config import DraftConfig
from speculant_graph.draft_generator import DraftGenerator, DraftResult

from conftest import TOKENIZER_NAME, build_simple_graph


class TestDraftGeneratorInit:
    def test_init_with_graph(self, small_graph):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            tokenizer_name=TOKENIZER_NAME,
        )
        assert gen.graph is graph
        assert gen.context_index is context_index
        assert gen.max_order == 3

    def test_default_config(self, small_graph):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            tokenizer_name=TOKENIZER_NAME,
        )
        assert isinstance(gen.config, DraftConfig)
        assert gen.config.k == 5

    def test_custom_config(self, small_graph):
        graph, context_index = small_graph
        config = DraftConfig(k=10, strategy="sampling")
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=config,
            tokenizer_name=TOKENIZER_NAME,
        )
        assert gen.config.k == 10
        assert gen.config.strategy == "sampling"

    def test_accepts_preloaded_tokenizer(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            tokenizer=tokenizer,
        )
        assert gen.tokenizer is tokenizer


class TestDraftGeneratorFromFile:
    def test_loads_from_file(self, saved_graph_path):
        gen = DraftGenerator.from_file(
            saved_graph_path,
            tokenizer_name=TOKENIZER_NAME,
        )
        assert gen.graph is not None
        assert gen.context_index is not None
        assert gen.max_order == 3

    def test_tokenizer_mismatch_raises(self, saved_graph_path):
        with pytest.raises(ValueError, match="Tokenizer mismatch"):
            DraftGenerator.from_file(
                saved_graph_path,
                tokenizer_name="some-other-tokenizer",
            )


class TestDraftResult:
    def test_draft_result_fields(self):
        result = DraftResult(
            token_ids=[1, 2, 3],
            token_probs=[0.5, 0.3, 0.2],
            matched_contexts=[(1,), (2,), (3,)],
            successors=[[2, 3], [3, 4], [4]],
            successor_weights=[[0.6, 0.4], [0.7, 0.3], [1.0]],
            strategy="greedy",
            requested_k=5,
            actual_length=3,
            terminated_early=True,
            termination_reason="Hit dead end in graph traversal",
        )
        assert result.token_ids == [1, 2, 3]
        assert result.actual_length == 3
        assert result.terminated_early is True
        assert result.termination_reason == "Hit dead end in graph traversal"

    def test_empty_draft_result(self):
        result = DraftResult(
            token_ids=[],
            token_probs=[],
            matched_contexts=[],
            successors=[],
            successor_weights=[],
            strategy="greedy",
            requested_k=5,
            actual_length=0,
            terminated_early=True,
            termination_reason="Empty prompt",
        )
        assert result.actual_length == 0
        assert result.terminated_early is True


class TestGenerate:
    def test_empty_prompt(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            tokenizer=tokenizer,
        )
        result = gen.generate("", k=5, strategy="greedy")

        assert isinstance(result, DraftResult)
        assert result.actual_length == 0
        assert result.terminated_early is True
        assert result.termination_reason == "Empty prompt"

    def test_greedy_generates_tokens(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=5, strategy="greedy")

        assert isinstance(result, DraftResult)
        assert result.strategy == "greedy"
        assert len(result.token_ids) == result.actual_length
        assert len(result.token_probs) == result.actual_length
        assert len(result.matched_contexts) == result.actual_length

    def test_sampling_generates_tokens(self, small_graph, tokenizer):
        graph, context_index = small_graph
        random.seed(42)
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="sampling", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=5, strategy="sampling")

        assert isinstance(result, DraftResult)
        assert result.strategy == "sampling"

    def test_attentive_mix_generates_tokens(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=True),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=5, strategy="greedy")

        assert isinstance(result, DraftResult)

    def test_respects_k_limit(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=100, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat sat on the mat", k=3, strategy="greedy")

        assert result.actual_length <= 3

    def test_no_matching_context(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("xyzzy foobar blargh", k=5, strategy="greedy")

        assert result.terminated_early is True

    def test_successors_and_weights_populated(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=5, strategy="greedy")

        if result.actual_length > 0:
            assert len(result.successors) == result.actual_length
            assert len(result.successor_weights) == result.actual_length
            for succs, weights in zip(result.successors, result.successor_weights):
                assert len(succs) == len(weights)


class TestFindHighestOrderMatch:
    def test_finds_highest_order(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )

        order, ctx = gen._find_highest_order_match([10, 20])
        assert order == 2
        assert ctx == (10, 20)

    def test_falls_back_to_lower_order(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )

        order, ctx = gen._find_highest_order_match([99, 10])
        assert order == 1
        assert ctx == (10,)

    def test_returns_zero_when_no_match(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )

        order, ctx = gen._find_highest_order_match([99, 88])
        assert order == 0
        assert ctx is None

    def test_handles_short_context(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=5,
            tokenizer_name=TOKENIZER_NAME,
        )

        order, ctx = gen._find_highest_order_match([10])
        assert order == 1
        assert ctx == (10,)


class TestMostFrequentToken:
    def test_finds_most_frequent(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )

        token = gen.get_most_frequent_token()
        assert token == 10  # "the" has count=10


class TestMixContexts:
    def test_returns_distribution(self):
        graph, context_index = build_simple_graph()
        config = DraftConfig(
            k=5,
            strategy="greedy",
            attentive_mix=True,
            order_bias=1.0,
            mix_temperature=1.0,
        )
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            config=config,
            tokenizer_name=TOKENIZER_NAME,
        )

        dist = gen._mix_contexts([10])
        assert isinstance(dist, dict)
        assert len(dist) > 0
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_when_no_context_match(self):
        graph, context_index = build_simple_graph()
        config = DraftConfig(k=5, strategy="greedy", attentive_mix=True)
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            config=config,
            tokenizer_name=TOKENIZER_NAME,
        )

        dist = gen._mix_contexts([99, 88])
        assert dist == {}

    def test_higher_order_gets_more_weight(self):
        graph, context_index = build_simple_graph()
        config = DraftConfig(
            k=5,
            strategy="greedy",
            attentive_mix=True,
            order_bias=2.0,
            mix_temperature=1.0,
        )
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            config=config,
            tokenizer_name=TOKENIZER_NAME,
        )

        # Context [10, 20] matches both order-2 (10,20) and order-1 (20,)
        # Order-2 predicts token 30 with prob 1.0
        # Order-1 (20,) predicts token 30 with prob 1.0
        # Both agree, so token 30 should dominate
        dist = gen._mix_contexts([10, 20])
        assert 30 in dist
        assert dist[30] > 0.9


class TestContextEntropy:
    def test_zero_entropy_for_deterministic(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )
        entropy = gen._compute_context_entropy({30: 1.0})
        assert entropy == 0.0

    def test_positive_entropy_for_uniform(self):
        graph, context_index = build_simple_graph()
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=2,
            tokenizer_name=TOKENIZER_NAME,
        )
        entropy = gen._compute_context_entropy({10: 0.5, 20: 0.5})
        assert entropy > 0
