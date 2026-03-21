import os

import networkx as nx
import pytest

from speculant_graph.graph_builder import GraphBuilder

from conftest import TOKENIZER_NAME


class TestGraphBuilderInit:
    def test_creates_empty_graph(self):
        builder = GraphBuilder(tokenizer_name=TOKENIZER_NAME, max_order=3)
        assert isinstance(builder.graph, nx.DiGraph)
        assert builder.graph.number_of_nodes() == 0
        assert builder.graph.number_of_edges() == 0

    def test_stores_config(self):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=4, chunk_size=200
        )
        assert builder.tokenizer_name == TOKENIZER_NAME
        assert builder.max_order == 4
        assert builder.chunk_size == 200

    def test_loads_tokenizer(self):
        builder = GraphBuilder(tokenizer_name=TOKENIZER_NAME)
        assert builder.tokenizer is not None
        tokens = builder.tokenizer.encode("hello")
        assert len(tokens) > 0


class TestBuildFromFiles:
    def test_single_file(self, corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_multiple_files(self, corpus_file, corpus_file_2):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file, corpus_file_2])

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_nonexistent_file_raises(self, tmp_path):
        builder = GraphBuilder(tokenizer_name=TOKENIZER_NAME, max_order=2)
        with pytest.raises(FileNotFoundError):
            builder.build_from_files([str(tmp_path / "nonexistent.txt")])

    def test_graph_has_token_nodes(self, corpus_file, tokenizer):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        token_nodes = [n for n in graph.nodes() if isinstance(n, int)]
        assert len(token_nodes) > 0

        for node in token_nodes:
            data = graph.nodes[node]
            assert "token_id" in data
            assert "text" in data
            assert "count" in data
            assert data["count"] > 0

    def test_graph_has_context_nodes(self, corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        context_nodes = [n for n in graph.nodes() if isinstance(n, tuple)]
        assert len(context_nodes) > 0

    def test_edges_have_weights(self, corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        for u, v, data in graph.edges(data=True):
            assert "weight" in data
            assert "count" in data
            assert "order" in data
            assert 0.0 < data["weight"] <= 1.0
            assert data["count"] > 0
            assert data["order"] >= 1

    def test_edge_weights_sum_to_one_per_context(self, corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([corpus_file])

        context_nodes = [n for n in graph.nodes() if isinstance(n, tuple)]
        for ctx in context_nodes:
            successors = list(graph.successors(ctx))
            if successors:
                weight_sum = sum(graph[ctx][s]["weight"] for s in successors)
                assert abs(weight_sum - 1.0) < 1e-6, (
                    f"Weights for context {ctx} sum to {weight_sum}"
                )

    def test_context_index_populated(self, corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        assert len(builder.context_index) > 0
        for ctx, order in builder.context_index.items():
            assert isinstance(ctx, tuple)
            assert order == len(ctx)

    def test_max_order_respected(self, corpus_file):
        max_order = 2
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=max_order, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        for ctx in builder.context_index:
            assert len(ctx) <= max_order

    def test_empty_file(self, empty_corpus_file):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        graph = builder.build_from_files([empty_corpus_file])
        # Empty file may still have BOS/EOS token edges depending on tokenizer
        # but should have very few nodes compared to a real corpus
        token_nodes = [n for n in graph.nodes() if isinstance(n, int)]
        assert len(token_nodes) <= 2  # at most BOS and EOS


class TestSaveAndLoad:
    def test_save_creates_file(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_load_returns_graph_and_metadata(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        graph, metadata = GraphBuilder.load(
            filepath, validate_tokenizer=False
        )

        assert isinstance(graph, nx.DiGraph)
        assert isinstance(metadata, dict)
        assert "context_index" in metadata
        assert "tokenizer_name" in metadata
        assert "max_order" in metadata
        assert "num_nodes" in metadata
        assert "num_edges" in metadata
        assert "build_timestamp" in metadata

    def test_roundtrip_preserves_graph(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        original_graph = builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        loaded_graph, metadata = GraphBuilder.load(
            filepath, validate_tokenizer=False
        )

        assert loaded_graph.number_of_nodes() == original_graph.number_of_nodes()
        assert loaded_graph.number_of_edges() == original_graph.number_of_edges()

        for node in original_graph.nodes():
            assert node in loaded_graph.nodes()

    def test_roundtrip_preserves_context_index(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=3, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        _, metadata = GraphBuilder.load(filepath, validate_tokenizer=False)

        assert metadata["context_index"] == builder.context_index

    def test_tokenizer_validation_passes(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        graph, metadata = GraphBuilder.load(
            filepath, validate_tokenizer=True, expected_tokenizer=TOKENIZER_NAME
        )
        assert graph is not None

    def test_tokenizer_validation_fails_on_mismatch(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        with pytest.raises(ValueError, match="Tokenizer mismatch"):
            GraphBuilder.load(
                filepath,
                validate_tokenizer=True,
                expected_tokenizer="some-other-tokenizer",
            )

    def test_metadata_has_per_order_stats(self, corpus_file, tmp_path):
        max_order = 3
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=max_order, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        _, metadata = GraphBuilder.load(filepath, validate_tokenizer=False)

        assert "num_nodes_per_order" in metadata
        assert "num_edges_per_order" in metadata
        for order in range(1, max_order + 1):
            assert order in metadata["num_nodes_per_order"]
            assert order in metadata["num_edges_per_order"]
