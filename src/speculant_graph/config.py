from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Literal


class GraphConfig(BaseModel):
    max_order: int = Field(default=5, ge=1, le=10)
    tokenizer_name: str = Field(default="openai/gpt-oss-20b")
    chunk_size: int = Field(default=10000, gt=0)
    hf_token: str | None = Field(default=None)
    download_mode: Literal["auto", "hf_transfer", "default"] = Field(
        default="auto",
        description="Download acceleration mode: 'auto' uses hf_xet if available, 'hf_transfer' for high-bandwidth, 'default' for standard downloads"
    )


class DraftConfig(BaseModel):
    k: int = Field(default=5, gt=0)
    strategy: str = Field(default="greedy", pattern="^(greedy|sampling)$")


class VerifierConfig(BaseModel):
    model_name: str = Field(default="openai/gpt-oss-20b")
    device: str | None = Field(default=None)
    hf_token: str | None = Field(default=None)
    download_mode: Literal["auto", "hf_transfer", "default"] = Field(
        default="auto",
        description="Download acceleration mode: 'auto' uses hf_xet if available, 'hf_transfer' for high-bandwidth, 'default' for standard downloads"
    )


class GenerationConfig(BaseModel):
    max_tokens: int = Field(default=100, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)


class SpeculativeDecodingConfig(BaseSettings):
    graph: GraphConfig = Field(default_factory=GraphConfig)
    draft: DraftConfig = Field(default_factory=DraftConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    model_config = {
        "env_prefix": "SPECULANT_",
        "env_nested_delimiter": "__",
    }
