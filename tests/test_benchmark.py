import time

import pytest
from transformers import AutoTokenizer

from speculant_graph.config import DraftConfig
from speculant_graph.draft_generator import DraftGenerator
from speculant_graph.graph_builder import GraphBuilder

from conftest import TOKENIZER_NAME, SAMPLE_CORPUS

PERF_CORPUS = SAMPLE_CORPUS * 20  # ~3k words for meaningful timing


@pytest.fixture(scope="module")
def perf_graph(tmp_path_factory):
    path = tmp_path_factory.mktemp("perf") / "corpus.txt"
    path.write_text(PERF_CORPUS)

    builder = GraphBuilder(tokenizer_name=TOKENIZER_NAME, max_order=5, chunk_size=5000)
    builder.build_from_files([str(path)])
    return builder


class TestGraphBuildPerformance:
    def test_build_completes_under_threshold(self, tmp_path):
        path = tmp_path / "perf_corpus.txt"
        path.write_text(PERF_CORPUS)

        # Pre-load tokenizer so download time isn't included
        AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        start = time.perf_counter()
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=5, chunk_size=5000
        )
        builder.build_from_files([str(path)])
        duration = time.perf_counter() - start

        print(f"\nGraph build: {duration:.3f}s")
        print(f"  Nodes: {builder.graph.number_of_nodes()}")
        print(f"  Edges: {builder.graph.number_of_edges()}")
        print(f"  Contexts: {len(builder.context_index)}")

        # Should complete in reasonable time (generous for CI)
        assert duration < 60, f"Graph build took {duration:.1f}s, expected <60s"

    def test_save_load_roundtrip_performance(self, perf_graph, tmp_path):
        filepath = str(tmp_path / "perf_graph.pkl")

        start = time.perf_counter()
        perf_graph.save(filepath)
        save_duration = time.perf_counter() - start

        start = time.perf_counter()
        GraphBuilder.load(filepath, validate_tokenizer=False)
        load_duration = time.perf_counter() - start

        print(f"\nSave: {save_duration:.3f}s, Load: {load_duration:.3f}s")

        assert save_duration < 30, f"Save took {save_duration:.1f}s"
        assert load_duration < 30, f"Load took {load_duration:.1f}s"


class TestDraftGenerationPerformance:
    def test_greedy_draft_throughput(self, perf_graph):
        tokenizer = perf_graph.tokenizer
        gen = DraftGenerator(
            graph=perf_graph.graph,
            context_index=perf_graph.context_index,
            max_order=5,
            config=DraftConfig(k=20, strategy="greedy", attentive_mix=False),
            tokenizer=tokenizer,
        )

        prompts = ["The cat sat on", "A bird flew over", "The dog ran in"]
        total_tokens = 0
        start = time.perf_counter()

        for prompt in prompts:
            for _ in range(100):
                result = gen.generate(prompt, k=20, strategy="greedy")
                total_tokens += result.actual_length

        duration = time.perf_counter() - start
        throughput = total_tokens / duration

        print(f"\nGreedy draft: {total_tokens} tokens in {duration:.3f}s")
        print(f"  Throughput: {throughput:.0f} tokens/sec")

        assert throughput > 100, (
            f"Greedy throughput {throughput:.0f} tok/s, expected >100"
        )

    def test_attentive_mix_draft_throughput(self, perf_graph):
        tokenizer = perf_graph.tokenizer
        gen = DraftGenerator(
            graph=perf_graph.graph,
            context_index=perf_graph.context_index,
            max_order=5,
            config=DraftConfig(k=20, strategy="greedy", attentive_mix=True),
            tokenizer=tokenizer,
        )

        prompts = ["The cat sat on", "A bird flew over", "The dog ran in"]
        total_tokens = 0
        start = time.perf_counter()

        for prompt in prompts:
            for _ in range(100):
                result = gen.generate(prompt, k=20, strategy="greedy")
                total_tokens += result.actual_length

        duration = time.perf_counter() - start
        throughput = total_tokens / duration

        print(f"\nAttentive mix draft: {total_tokens} tokens in {duration:.3f}s")
        print(f"  Throughput: {throughput:.0f} tokens/sec")

        # Attentive mix is slower but should still be reasonable
        assert throughput > 50, (
            f"Attentive throughput {throughput:.0f} tok/s, expected >50"
        )

    def test_sampling_draft_throughput(self, perf_graph):
        tokenizer = perf_graph.tokenizer
        gen = DraftGenerator(
            graph=perf_graph.graph,
            context_index=perf_graph.context_index,
            max_order=5,
            config=DraftConfig(k=20, strategy="sampling", attentive_mix=False),
            tokenizer=tokenizer,
        )

        prompts = ["The cat sat on", "A bird flew over", "The dog ran in"]
        total_tokens = 0
        start = time.perf_counter()

        for prompt in prompts:
            for _ in range(100):
                result = gen.generate(prompt, k=20, strategy="sampling")
                total_tokens += result.actual_length

        duration = time.perf_counter() - start
        throughput = total_tokens / duration

        print(f"\nSampling draft: {total_tokens} tokens in {duration:.3f}s")
        print(f"  Throughput: {throughput:.0f} tokens/sec")

        assert throughput > 100, (
            f"Sampling throughput {throughput:.0f} tok/s, expected >100"
        )


class TestContextLookupPerformance:
    def test_context_index_lookup_speed(self, perf_graph):
        contexts = list(perf_graph.context_index.keys())[:1000]

        start = time.perf_counter()
        for _ in range(1000):
            for ctx in contexts:
                _ = perf_graph.context_index.get(ctx)
        duration = time.perf_counter() - start

        lookups = 1000 * len(contexts)
        rate = lookups / duration

        print(f"\nContext lookups: {lookups} in {duration:.3f}s")
        print(f"  Rate: {rate:.0f} lookups/sec")

        assert rate > 1_000_000, f"Lookup rate {rate:.0f}/s, expected >1M/s"
