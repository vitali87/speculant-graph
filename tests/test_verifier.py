import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from speculant_graph.config import DraftConfig, GenerationConfig, VerifierConfig
from speculant_graph.verifier import GenerationResult, SpeculativeDecoder


VOCAB_SIZE = 50257  # GPT-2 vocab size


def _make_mock_model_output(batch_size=1, seq_len=1, vocab_size=VOCAB_SIZE):
    logits = torch.randn(batch_size, seq_len, vocab_size)
    past_kv = ((torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),)
    return SimpleNamespace(logits=logits, past_key_values=past_kv)


def _make_decoder(small_graph, tokenizer):
    graph, context_index = small_graph
    from speculant_graph.draft_generator import DraftGenerator

    decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
    decoder.verifier_config = VerifierConfig(model_name="mock")
    decoder.draft_config = DraftConfig(k=3, strategy="greedy", attentive_mix=False)
    decoder.device = "cpu"
    decoder.tokenizer = tokenizer

    mock_model = MagicMock()
    mock_model.return_value = _make_mock_model_output()
    decoder.model = mock_model

    decoder.draft_generator = DraftGenerator(
        graph=graph,
        context_index=context_index,
        max_order=3,
        config=decoder.draft_config,
        tokenizer=tokenizer,
    )

    decoder._reset_verifier_cache()
    return decoder


class TestGenerationResult:
    def test_fields(self):
        result = GenerationResult(
            text="hello world",
            token_ids=[1, 2],
            acceptance_rate=0.5,
            num_accepted=1,
            num_rejected=1,
            total_tokens=2,
            position_acceptance_counts={0: 1},
            position_proposal_counts={0: 1, 1: 1},
        )
        assert result.text == "hello world"
        assert result.acceptance_rate == 0.5
        assert result.total_tokens == 2

    def test_zero_acceptance(self):
        result = GenerationResult(
            text="test",
            token_ids=[1],
            acceptance_rate=0.0,
            num_accepted=0,
            num_rejected=5,
            total_tokens=1,
            position_acceptance_counts={},
            position_proposal_counts={0: 5},
        )
        assert result.acceptance_rate == 0.0
        assert result.num_rejected == 5


class TestResetVerifierCache:
    def test_resets_all_fields(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder.past_key_values = "something"
        decoder.attention_mask = "something"
        decoder.last_logits = "something"

        decoder._reset_verifier_cache()

        assert decoder.past_key_values is None
        assert decoder.attention_mask is None
        assert decoder.last_logits is None


class TestPrimeVerifierState:
    def test_sets_cache(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        input_ids = torch.tensor([[1, 2, 3]])

        decoder._prime_verifier_state(input_ids)

        assert decoder.past_key_values is not None
        assert decoder.attention_mask is not None
        assert decoder.last_logits is not None
        assert decoder.attention_mask.shape == (1, 3)

    def test_calls_model_with_correct_args(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        input_ids = torch.tensor([[10, 20]])

        decoder._prime_verifier_state(input_ids)

        decoder.model.assert_called_once()
        call_kwargs = decoder.model.call_args[1]
        assert call_kwargs["use_cache"] is True
        assert torch.equal(call_kwargs["input_ids"], input_ids)


class TestAppendToken:
    def test_raises_without_cache(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)

        with pytest.raises(RuntimeError, match="cache not initialized"):
            decoder._append_token(42)

    def test_extends_attention_mask(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1, 2]]))

        old_mask_len = decoder.attention_mask.shape[1]
        decoder._append_token(3)

        assert decoder.attention_mask.shape[1] == old_mask_len + 1

    def test_updates_logits(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        decoder._append_token(2)

        assert decoder.last_logits is not None


class TestNextTokenDistribution:
    def test_raises_without_cache(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)

        with pytest.raises(RuntimeError, match="cache not initialized"):
            decoder._next_token_distribution(1.0)

    def test_returns_valid_distribution(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder.last_logits = torch.randn(1, VOCAB_SIZE)

        probs = decoder._next_token_distribution(1.0)

        assert probs.shape == (VOCAB_SIZE,)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all()

    def test_temperature_affects_distribution(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        torch.manual_seed(42)
        fixed_logits = torch.randn(1, VOCAB_SIZE)
        decoder.last_logits = fixed_logits.clone()

        probs_low_t = decoder._next_token_distribution(0.1)
        decoder.last_logits = fixed_logits.clone()
        probs_high_t = decoder._next_token_distribution(10.0)

        assert probs_low_t.max() > probs_high_t.max()


class TestVerifyDraft:
    def test_raises_without_cache(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)

        with pytest.raises(RuntimeError, match="cache not initialized"):
            decoder._verify_draft(
                [1], [2], [1.0], [(1,)], [[2]], [[1.0]], "greedy", 1.0
            )

    def test_greedy_accept(self, small_graph, tokenizer):
        random.seed(0)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -10.0)
        logits[0, 2] = 10.0
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1], [2], [1.0], [(1,)], [[2]], [[1.0]], "greedy", 1.0
        )

        assert accepted == 1
        assert rejected == 0
        assert 2 in tokens
        assert 0 in positions

    def test_greedy_reject_samples_correction(self, small_graph, tokenizer):
        random.seed(42)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -10.0)
        logits[0, 3] = 10.0  # token 3 is likely
        logits[0, 50] = -100.0  # draft token 50 is very unlikely
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1], [50], [1.0], [(1,)], [[50]], [[1.0]], "greedy", 1.0
        )

        assert accepted == 0
        assert rejected == 1
        assert has_corr is True
        assert len(tokens) == 1  # correction token sampled
        assert tokens[0] == 3  # should sample the high-probability token

    def test_sampling_strategy(self, small_graph, tokenizer):
        random.seed(0)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -5.0)
        logits[0, 5] = 5.0
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1],
            [5],
            [0.5],
            [(1,)],
            [[5, 6]],
            [[0.5, 0.5]],
            "sampling",
            1.0,
        )

        assert accepted + rejected == 1
        assert len(tokens) >= 1

    def test_multiple_draft_tokens(self, small_graph, tokenizer):
        random.seed(0)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -100.0)
        logits[0, 2] = 100.0
        decoder.last_logits = logits

        draft_sequence = [3, 4]
        call_count = [0]

        def mock_model_call(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            output_logits = torch.full((1, 1, VOCAB_SIZE), -100.0)
            if idx < len(draft_sequence):
                output_logits[0, 0, draft_sequence[idx]] = 100.0
            past_kv = ((torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),)
            return SimpleNamespace(logits=output_logits, past_key_values=past_kv)

        decoder.model.side_effect = mock_model_call

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1],
            [2, 3, 4],
            [1.0, 1.0, 1.0],
            [(1,), (2,), (3,)],
            [[2], [3], [4]],
            [[1.0], [1.0], [1.0]],
            "greedy",
            1.0,
        )

        assert accepted == 3
        assert rejected == 0
        assert tokens == [2, 3, 4]
        assert positions == [0, 1, 2]
        assert decoder.model.call_count == 4


class TestGenerateFromVerifier:
    def test_raises_without_cache(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)

        with pytest.raises(RuntimeError, match="cache not initialized"):
            decoder._generate_from_verifier(1, 1.0)

    def test_generates_tokens(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        count, tokens = decoder._generate_from_verifier(1, 1.0)

        assert count == 1
        assert len(tokens) == 1
        assert isinstance(tokens[0], int)

    def test_generates_multiple(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        count, tokens = decoder._generate_from_verifier(3, 1.0)

        assert count == 3
        assert len(tokens) == 3


class TestPrepareInputIds:
    def test_normal_prompt(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        input_ids = decoder._prepare_input_ids("The cat")

        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] > 0
        assert input_ids.device.type == "cpu"

    def test_empty_prompt_uses_special_token(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        input_ids = decoder._prepare_input_ids("")

        assert input_ids.shape == (1, 1)
        token_id = input_ids[0, 0].item()
        special_ids = [
            sid
            for sid in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ]
            if sid is not None
        ]
        if special_ids:
            assert token_id in special_ids
        else:
            assert token_id == decoder.draft_generator.get_most_frequent_token()


class TestGenerate:
    def test_returns_generation_result(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=5, temperature=1.0, seed=42)

        result = decoder.generate("The cat", config)

        assert isinstance(result, GenerationResult)
        assert len(result.token_ids) > 0
        assert result.total_tokens > 0
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert isinstance(result.text, str)

    def test_respects_max_tokens(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=3, temperature=1.0, seed=42)

        result = decoder.generate("The cat", config)

        assert result.total_tokens <= 3

    def test_seed_reproducibility(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=5, temperature=1.0, seed=123)

        def mock_model_call(**kwargs):
            torch.manual_seed(0)
            output = _make_mock_model_output()
            return output

        decoder.model.side_effect = mock_model_call
        result1 = decoder.generate("The cat", config)

        decoder.model.side_effect = mock_model_call
        result2 = decoder.generate("The cat", config)

        assert result1.token_ids == result2.token_ids

    def test_position_tracking(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=5, temperature=1.0, seed=42)

        result = decoder.generate("The cat", config)

        assert isinstance(result.position_acceptance_counts, dict)
        assert isinstance(result.position_proposal_counts, dict)


class TestGenerateStream:
    def test_yields_chunks_then_result(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=3, temperature=1.0, seed=42)

        chunks = list(decoder.generate_stream("The cat", config))

        assert len(chunks) > 0
        assert isinstance(chunks[-1], GenerationResult)

        for chunk in chunks[:-1]:
            assert isinstance(chunk, tuple)
            assert len(chunk) == 2
            text, token_ids = chunk
            assert isinstance(text, str)
            assert isinstance(token_ids, list)

    def test_generate_stream_returns_valid_result(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=3, temperature=1.0, seed=42)

        stream_result = None
        for chunk in decoder.generate_stream("The cat", config):
            if isinstance(chunk, GenerationResult):
                stream_result = chunk

        assert stream_result is not None
        assert stream_result.total_tokens > 0
        assert isinstance(stream_result.text, str)

    def test_stream_empty_draft_falls_back_to_verifier(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=2, temperature=1.0, seed=42)

        chunks = list(decoder.generate_stream("xyzzy foobar blargh", config))

        assert isinstance(chunks[-1], GenerationResult)
        assert chunks[-1].total_tokens > 0


class TestGenerateEdgeCases:
    def test_empty_draft_falls_back_to_verifier(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=2, temperature=1.0, seed=42)

        result = decoder.generate("xyzzy foobar blargh", config)

        assert isinstance(result, GenerationResult)
        assert result.total_tokens > 0
        assert result.acceptance_rate == 0.0

    def test_all_drafts_rejected_generates_fallback(self, small_graph, tokenizer):
        random.seed(42)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -100.0)
        logits[0, 99] = 100.0
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1],
            [50],
            [1.0],
            [(1,)],
            [[50]],
            [[1.0]],
            "greedy",
            1.0,
        )

        assert accepted == 0
        assert rejected == 1
        assert has_corr is True
        assert tokens[0] == 99

    def test_generate_without_seed(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=3, temperature=1.0, seed=None)

        result = decoder.generate("The cat", config)
        assert isinstance(result, GenerationResult)

    def test_generate_with_accepted_and_correction(self, small_graph, tokenizer):
        random.seed(0)
        decoder = _make_decoder(small_graph, tokenizer)
        config = GenerationConfig(max_tokens=5, temperature=1.0, seed=42)

        result = decoder.generate("The cat", config)

        assert isinstance(result, GenerationResult)
        assert result.total_tokens > 0


class TestSamplingVerification:
    def test_sampling_zero_draft_prob_accepts(self, small_graph, tokenizer):
        random.seed(0)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -10.0)
        logits[0, 5] = 10.0
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1],
            [5],
            [0.0],
            [(1,)],
            [[5]],
            [[0.0]],
            "sampling",
            1.0,
        )

        assert accepted == 1
        assert tokens[0] == 5

    def test_sampling_residual_fallback(self, small_graph, tokenizer):
        random.seed(42)
        decoder = _make_decoder(small_graph, tokenizer)
        decoder._prime_verifier_state(torch.tensor([[1]]))

        logits = torch.full((1, VOCAB_SIZE), -100.0)
        logits[0, 7] = 100.0
        decoder.last_logits = logits

        accepted, rejected, tokens, has_corr, positions = decoder._verify_draft(
            [1],
            [7],
            [0.99],
            [(1,)],
            [[7, 8]],
            [[0.5, 0.5]],
            "sampling",
            1.0,
        )

        assert len(tokens) >= 1


class TestPrepareInputIdsFallbackChain:
    def test_eos_fallback_when_no_bos(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        original_bos = decoder.tokenizer.bos_token_id
        decoder.tokenizer.bos_token_id = None

        input_ids = decoder._prepare_input_ids("")

        decoder.tokenizer.bos_token_id = original_bos
        assert input_ids.shape == (1, 1)
        token_id = input_ids[0, 0].item()
        assert token_id == tokenizer.eos_token_id

    def test_pad_fallback_when_no_bos_or_eos(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        original_bos = decoder.tokenizer.bos_token_id
        original_eos = decoder.tokenizer.eos_token_id
        decoder.tokenizer.bos_token_id = None
        decoder.tokenizer.eos_token_id = None

        input_ids = decoder._prepare_input_ids("")

        decoder.tokenizer.bos_token_id = original_bos
        decoder.tokenizer.eos_token_id = original_eos
        assert input_ids.shape == (1, 1)

    def test_most_frequent_fallback(self, small_graph, tokenizer):
        decoder = _make_decoder(small_graph, tokenizer)
        original_bos = decoder.tokenizer.bos_token_id
        original_eos = decoder.tokenizer.eos_token_id
        original_pad = decoder.tokenizer.pad_token_id
        decoder.tokenizer.bos_token_id = None
        decoder.tokenizer.eos_token_id = None
        decoder.tokenizer.pad_token_id = None

        input_ids = decoder._prepare_input_ids("")

        decoder.tokenizer.bos_token_id = original_bos
        decoder.tokenizer.eos_token_id = original_eos
        decoder.tokenizer.pad_token_id = original_pad
        assert input_ids.shape == (1, 1)
        token_id = input_ids[0, 0].item()
        assert token_id == decoder.draft_generator.get_most_frequent_token()
