import io
import os
import pickle

import pytest

from speculant_graph.graph_builder import GraphBuilder

from conftest import TOKENIZER_NAME


class TestRestrictedUnpickler:
    def test_allows_networkx_digraph(self, saved_graph_path):
        graph, metadata = GraphBuilder.load(saved_graph_path, validate_tokenizer=False)
        assert graph is not None
        assert metadata["tokenizer_name"] == TOKENIZER_NAME

    def test_blocks_os_system(self, tmp_path):
        payload = _build_exploit_pickle(os.system, "echo pwned")
        filepath = str(tmp_path / "malicious.pkl")
        with open(filepath, "wb") as f:
            f.write(payload)

        with pytest.raises(pickle.UnpicklingError, match="disallowed class"):
            GraphBuilder.load(filepath, validate_tokenizer=False)

    def test_blocks_eval(self, tmp_path):
        payload = _build_exploit_pickle(eval, "1+1")
        filepath = str(tmp_path / "malicious.pkl")
        with open(filepath, "wb") as f:
            f.write(payload)

        with pytest.raises(pickle.UnpicklingError, match="disallowed class"):
            GraphBuilder.load(filepath, validate_tokenizer=False)

    def test_blocks_subprocess(self, tmp_path):
        import subprocess

        payload = _build_exploit_pickle(subprocess.check_output, "id")
        filepath = str(tmp_path / "malicious.pkl")
        with open(filepath, "wb") as f:
            f.write(payload)

        with pytest.raises(pickle.UnpicklingError, match="disallowed class"):
            GraphBuilder.load(filepath, validate_tokenizer=False)

    def test_roundtrip_still_works(self, corpus_file, tmp_path):
        builder = GraphBuilder(
            tokenizer_name=TOKENIZER_NAME, max_order=2, chunk_size=500
        )
        builder.build_from_files([corpus_file])

        filepath = str(tmp_path / "graph.pkl")
        builder.save(filepath)

        graph, metadata = GraphBuilder.load(filepath, validate_tokenizer=False)
        assert graph.number_of_nodes() == builder.graph.number_of_nodes()
        assert graph.number_of_edges() == builder.graph.number_of_edges()
        assert metadata["context_index"] == builder.context_index


def _build_exploit_pickle(func, arg):
    """Build a malicious pickle payload that calls func(arg)."""
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dump(_ExploitHelper(func, arg))
    return buf.getvalue()


class _ExploitHelper:
    """Helper class that pickles as a function call for testing."""

    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def __reduce__(self):
        return (self.func, (self.arg,))
