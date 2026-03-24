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
def small_graph_builder(tokenizer):
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
        builder.build_from_files([corpus_path])
    finally:
        os.unlink(corpus_path)

    return builder


@pytest.fixture(scope="session")
def small_graph(small_graph_builder):
    return small_graph_builder.graph, small_graph_builder.context_index


@pytest.fixture
def saved_graph_path(small_graph_builder, tmp_path):
    filepath = str(tmp_path / "test_graph.pkl")
    small_graph_builder.save(filepath)
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
