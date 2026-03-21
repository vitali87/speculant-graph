import os
import tempfile

import networkx as nx
import pytest
from transformers import AutoTokenizer

from speculant_graph.config import DraftConfig


TOKENIZER_NAME = "openai-community/gpt2"

SAMPLE_CORPUS = (
    "The cat sat on the mat. The cat sat on the rug. "
    "The dog sat on the mat. The dog ran in the park. "
    "A bird flew over the fence. The bird sang a song."
)

SAMPLE_CORPUS_2 = (
    "Machine learning is a subset of artificial intelligence. "
    "Deep learning uses neural networks with many layers. "
    "Natural language processing deals with text and speech."
)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture
def corpus_file(tmp_path):
    f = tmp_path / "corpus.txt"
    f.write_text(SAMPLE_CORPUS)
    return str(f)


@pytest.fixture
def corpus_file_2(tmp_path):
    f = tmp_path / "corpus2.txt"
    f.write_text(SAMPLE_CORPUS_2)
    return str(f)


@pytest.fixture
def empty_corpus_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    return str(f)


@pytest.fixture(scope="session")
def small_graph(tokenizer):
    """Build a small graph from sample corpus for reuse across tests."""
    from speculant_graph.graph_builder import GraphBuilder

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_CORPUS)
        corpus_path = f.name

    try:
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME,
            max_order=3,
            chunk_size=500,
        )
        graph = builder.build_from_files([corpus_path])
        context_index = builder.context_index
    finally:
        os.unlink(corpus_path)

    return graph, context_index


@pytest.fixture
def saved_graph_path(small_graph, tmp_path):
    """Save the small graph to a pickle file and return the path."""
    from speculant_graph.graph_builder import GraphBuilder

    graph, context_index = small_graph

    builder = GraphBuilder.__new__(GraphBuilder)
    builder.graph = graph
    builder.context_index = context_index
    builder.tokenizer_name = TOKENIZER_NAME
    builder.max_order = 3
    builder.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    builder.token_counts = {}
    builder.ngram_transition_counts = {}
    for node in graph.nodes():
        if isinstance(node, int):
            builder.token_counts[node] = graph.nodes[node].get("count", 0)

    filepath = str(tmp_path / "test_graph.pkl")
    builder.save(filepath)
    return filepath


@pytest.fixture
def draft_config():
    return DraftConfig(k=5, strategy="greedy", attentive_mix=False)


@pytest.fixture
def draft_config_attentive():
    return DraftConfig(k=5, strategy="greedy", attentive_mix=True)


@pytest.fixture
def draft_config_sampling():
    return DraftConfig(k=5, strategy="sampling", attentive_mix=False)


def build_simple_graph():
    """Build a minimal hand-crafted graph for deterministic tests."""
    g = nx.DiGraph()

    g.add_node(10, token_id=10, text="the", count=10)
    g.add_node(20, token_id=20, text="cat", count=5)
    g.add_node(30, token_id=30, text="sat", count=5)
    g.add_node(40, token_id=40, text="dog", count=3)

    g.add_edge((10,), 20, weight=0.6, count=6, order=1)
    g.add_edge((10,), 40, weight=0.4, count=4, order=1)

    g.add_edge((20,), 30, weight=1.0, count=5, order=1)

    g.add_edge((10, 20), 30, weight=1.0, count=5, order=2)

    context_index = {
        (10,): 1,
        (20,): 1,
        (10, 20): 2,
    }

    return g, context_index
