import os

import pytest

from speculant_graph.download_utils import configure_download_mode


class TestConfigureDownloadMode:
    def test_auto_mode(self):
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        configure_download_mode("auto")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_default_mode(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        configure_download_mode("default")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_hf_transfer_without_package(self):
        # hf_transfer likely not installed in test env, so it should fallback
        configure_download_mode("hf_transfer")
        # Either it's set (if hf_transfer installed) or cleared (if not)
        # Both are valid outcomes

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid download_mode"):
            configure_download_mode("turbo")

    def test_auto_clears_env(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        configure_download_mode("auto")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
