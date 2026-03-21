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

    def test_hf_transfer_package_handling(self):
        import importlib.util

        configure_download_mode("hf_transfer")

        if importlib.util.find_spec("hf_transfer"):
            assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
        else:
            assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid download_mode"):
            configure_download_mode("turbo")

    def test_auto_clears_env(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        configure_download_mode("auto")
        assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
