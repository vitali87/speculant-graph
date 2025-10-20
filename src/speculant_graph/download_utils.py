import os
from typing import Literal
from loguru import logger


def configure_download_mode(mode: Literal["auto", "hf_transfer", "default"]) -> None:
    if mode == "hf_transfer":
        try:
            import hf_transfer  # noqa: F401

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            logger.info(
                "Download acceleration enabled: hf_transfer (optimized for high-bandwidth)"
            )
        except ImportError:
            logger.warning(
                "hf_transfer requested but not installed. "
                "Install with: pip install huggingface_hub[hf_transfer]. "
                "Falling back to default download mode."
            )
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    elif mode == "auto":
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        logger.info("Download acceleration: auto (will use hf_xet if available)")
    elif mode == "default":
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        logger.info("Download acceleration: disabled (using standard downloads)")
    else:
        raise ValueError(
            f"Invalid download_mode: {mode}. Must be 'auto', 'hf_transfer', or 'default'"
        )
