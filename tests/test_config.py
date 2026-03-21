import pytest
from pydantic import ValidationError

from speculant_graph.config import (
    DraftConfig,
    GenerationConfig,
    GraphConfig,
    SpeculativeDecodingConfig,
    VerifierConfig,
)


class TestGraphConfig:
    def test_defaults(self):
        config = GraphConfig()
        assert config.max_order == 5
        assert config.tokenizer_name == "openai/gpt-oss-20b"
        assert config.chunk_size == 10000
        assert config.hf_token is None
        assert config.download_mode == "auto"

    def test_custom_values(self):
        config = GraphConfig(max_order=3, tokenizer_name="gpt2", chunk_size=500)
        assert config.max_order == 3
        assert config.tokenizer_name == "gpt2"
        assert config.chunk_size == 500

    def test_max_order_range(self):
        GraphConfig(max_order=1)
        GraphConfig(max_order=10)

        with pytest.raises(ValidationError):
            GraphConfig(max_order=0)
        with pytest.raises(ValidationError):
            GraphConfig(max_order=11)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            GraphConfig(chunk_size=0)
        with pytest.raises(ValidationError):
            GraphConfig(chunk_size=-1)

    def test_invalid_download_mode(self):
        with pytest.raises(ValidationError):
            GraphConfig(download_mode="turbo")


class TestDraftConfig:
    def test_defaults(self):
        config = DraftConfig()
        assert config.k == 5
        assert config.strategy == "greedy"
        assert config.attentive_mix is True
        assert config.order_bias == 1.0
        assert config.mix_temperature == 1.0
        assert config.reliability_weight == 1.0
        assert config.entropy_penalty == 0.5

    def test_k_must_be_positive(self):
        with pytest.raises(ValidationError):
            DraftConfig(k=0)
        with pytest.raises(ValidationError):
            DraftConfig(k=-1)

    def test_strategy_validation(self):
        DraftConfig(strategy="greedy")
        DraftConfig(strategy="sampling")

        with pytest.raises(ValidationError):
            DraftConfig(strategy="beam_search")
        with pytest.raises(ValidationError):
            DraftConfig(strategy="")

    def test_order_bias_must_be_positive(self):
        with pytest.raises(ValidationError):
            DraftConfig(order_bias=0.0)
        with pytest.raises(ValidationError):
            DraftConfig(order_bias=-1.0)

    def test_mix_temperature_must_be_positive(self):
        with pytest.raises(ValidationError):
            DraftConfig(mix_temperature=0.0)

    def test_entropy_penalty_can_be_zero(self):
        config = DraftConfig(entropy_penalty=0.0)
        assert config.entropy_penalty == 0.0

    def test_entropy_penalty_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            DraftConfig(entropy_penalty=-0.1)


class TestVerifierConfig:
    def test_defaults(self):
        config = VerifierConfig()
        assert config.model_name == "openai/gpt-oss-20b"
        assert config.device is None
        assert config.hf_token is None
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "auto"
        assert config.low_cpu_mem_usage is True

    def test_custom_values(self):
        config = VerifierConfig(
            model_name="gpt2",
            device="cpu",
            torch_dtype="float32",
            device_map=None,
            low_cpu_mem_usage=False,
        )
        assert config.model_name == "gpt2"
        assert config.device == "cpu"
        assert config.torch_dtype == "float32"
        assert config.device_map is None
        assert config.low_cpu_mem_usage is False


class TestGenerationConfig:
    def test_defaults(self):
        config = GenerationConfig()
        assert config.max_tokens == 100
        assert config.temperature == 1.0
        assert config.seed is None

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            GenerationConfig(max_tokens=0)

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=0.0)

    def test_seed_optional(self):
        config = GenerationConfig(seed=42)
        assert config.seed == 42


class TestSpeculativeDecodingConfig:
    def test_defaults(self):
        config = SpeculativeDecodingConfig()
        assert isinstance(config.graph, GraphConfig)
        assert isinstance(config.draft, DraftConfig)
        assert isinstance(config.verifier, VerifierConfig)
        assert isinstance(config.generation, GenerationConfig)

    def test_nested_config(self):
        config = SpeculativeDecodingConfig(
            draft=DraftConfig(k=10, strategy="sampling"),
            generation=GenerationConfig(max_tokens=200),
        )
        assert config.draft.k == 10
        assert config.draft.strategy == "sampling"
        assert config.generation.max_tokens == 200

    def test_env_prefix(self):
        assert SpeculativeDecodingConfig.model_config["env_prefix"] == "SPECULANT_"
        assert SpeculativeDecodingConfig.model_config["env_nested_delimiter"] == "__"
