import os

import pytest

from speculant_graph.download_utils import configure_download_mode


class TestConfigureDownloadMode:
    def test_auto_mode(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
        configure_download_mode("auto")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_default_mode(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
        configure_download_mode("default")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_hf_transfer_package_handling(self, monkeypatch):
        import importlib.util

        monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
        configure_download_mode("hf_transfer")

        if importlib.util.find_spec("hf_transfer"):
            assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
        else:
            assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid download_mode"):
            configure_download_mode("turbo")

    def test_auto_clears_env(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
        configure_download_mode("auto")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
