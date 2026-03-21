
import pytest

from speculant_graph.config import DraftConfig
from speculant_graph.draft_generator import DraftGenerator, DraftResult
from speculant_graph.graph_builder import GraphBuilder

from conftest import TOKENIZER_NAME


class TestGraphBuildAndDraft:
    """Integration tests: build graph from corpus then generate drafts."""

    def test_build_and_draft_greedy(self, corpus_file, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        gen = DraftGenerator(
            graph=graph,
            context_index=builder.context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )

        result = gen.generate("The cat", k=5, strategy="greedy")

        assert isinstance(result, DraftResult)
        assert result.actual_length > 0
        assert all(isinstance(t, int) for t in result.token_ids)
        assert all(isinstance(p, float) for p in result.token_probs)

    def test_build_and_draft_sampling(self, corpus_file, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        gen = DraftGenerator(
            graph=graph,
            context_index=builder.context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="sampling", attentive_mix=False),
            tokenizer=tokenizer,
        )

        result = gen.generate("The cat", k=5, strategy="sampling")

        assert isinstance(result, DraftResult)
        assert result.actual_length > 0
        for p in result.token_probs:
            assert 0.0 < p <= 1.0

    def test_build_and_draft_attentive(self, corpus_file, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        gen = DraftGenerator(
            graph=graph,
            context_index=builder.context_index,
            max_order=3,
            config=DraftConfig(
                k=5,
                strategy="greedy",
                attentive_mix=True,
                order_bias=1.0,
                mix_temperature=1.0,
                entropy_penalty=0.5,
            ),
            tokenizer=tokenizer,
        )

        result = gen.generate("The cat", k=5, strategy="greedy")

        assert isinstance(result, DraftResult)

    def test_save_load_and_draft(self, corpus_file, tmp_path, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        gen = DraftGenerator.from_file(
            filepath,
            tokenizer_name=TOKENIZER_NAME,
            tokenizer=tokenizer,
        )

        result = gen.generate("The cat", k=5, strategy="greedy")
        assert isinstance(result, DraftResult)
        assert result.actual_length > 0


class TestMultiFileGraph:
    """Integration tests with multiple corpus files."""

    def test_multi_file_graph_has_more_coverage(
        self, corpus_file, corpus_file_2, tokenizer
    ):
        builder_single = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder_single.build_from_files([corpus_file])

        builder_multi = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder_multi.build_from_files([corpus_file, corpus_file_2])

        assert (
            builder_multi.graph.number_of_nodes()
            >= builder_single.graph.number_of_nodes()
        )

    def test_cross_corpus_drafting(self, corpus_file, corpus_file_2, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file, corpus_file_2])

        gen = DraftGenerator(
            graph=graph,
            context_index=builder.context_index,
            max_order=2,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )

        # Try a prompt from first corpus
        result1 = gen.generate("The cat", k=3, strategy="greedy")
        assert isinstance(result1, DraftResult)

        # Try a prompt from second corpus
        result2 = gen.generate("Machine learning", k=3, strategy="greedy")
        assert isinstance(result2, DraftResult)


class TestGraphOrderVariations:
    """Test different max_order settings."""

    @pytest.mark.parametrize("max_order", [1, 2, 3, 5])
    def test_different_orders(self, corpus_file, tokenizer, max_order):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME,
            max_order=max_order,
            chunk_size=500,
        )
        graph = builder.build_from_files([corpus_file])

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        # Verify no context exceeds max_order
        for ctx in builder.context_index:
            assert len(ctx) <= max_order

        gen = DraftGenerator(
            graph=graph,
            context_index=builder.context_index,
            max_order=max_order,
            config=DraftConfig(k=3, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)


class TestDraftStrategies:
    """Compare greedy vs sampling strategies on same graph."""

    def test_greedy_is_deterministic(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )

        result1 = gen.generate("The cat sat", k=5, strategy="greedy")
        result2 = gen.generate("The cat sat", k=5, strategy="greedy")

        assert result1.token_ids == result2.token_ids

    def test_greedy_probs_are_one(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat sat", k=5, strategy="greedy")

        for p in result.token_probs:
            assert p == 1.0

    def test_sampling_probs_are_valid(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=5, strategy="sampling", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat sat", k=5, strategy="sampling")

        for p in result.token_probs:
            assert 0.0 < p <= 1.0


class TestAttentiveMixIntegration:
    """Test attentive mixing with various config params."""

    def test_low_temperature_sharpens(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(
                k=5,
                strategy="greedy",
                attentive_mix=True,
                mix_temperature=0.1,
            ),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)

    def test_high_order_bias(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(
                k=5,
                strategy="greedy",
                attentive_mix=True,
                order_bias=5.0,
            ),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)

    def test_high_entropy_penalty(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(
                k=5,
                strategy="greedy",
                attentive_mix=True,
                entropy_penalty=5.0,
            ),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)


class TestEdgeCases:
    """Test edge cases in the full pipeline."""

    def test_very_short_prompt(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=3, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("a", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)

    def test_prompt_not_in_corpus(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=3, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("quantum entanglement hypothesis", k=3, strategy="greedy")
        assert isinstance(result, DraftResult)
        assert result.terminated_early is True

    def test_k_equals_one(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=1, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat", k=1, strategy="greedy")
        assert result.actual_length <= 1

    def test_large_k(self, small_graph, tokenizer):
        graph, context_index = small_graph
        gen = DraftGenerator(
            graph=graph,
            context_index=context_index,
            max_order=3,
            config=DraftConfig(k=1000, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )
        result = gen.generate("The cat sat on the mat", k=1000, strategy="greedy")
        assert isinstance(result, DraftResult)
        # Graph may have cycles allowing full generation, or terminate early
        # Either way, should produce a valid result
        assert result.actual_length <= 1000

    def test_chunk_size_one(self, corpus_file, tokenizer):
        """Test graph building with very small chunk size."""
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=1
        )
        graph = builder.build_from_files([corpus_file])
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
