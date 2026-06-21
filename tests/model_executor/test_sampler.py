from unittest.mock import patch

import pytest

from vkwr.engine.request import SamplingParams


@pytest.fixture(autouse=True)
def cuda_device_guard():
    """Override root conftest - engine core tests use mocking."""
    yield


# ── RWKV7Sampler ─────────────────────────────────────────────────────


class TestRWKV7Sampler:
    @patch("vkwr.model_executor.layers.sampler.setup_rand")
    @patch("vkwr.model_executor.layers.sampler.sample_temperature_topk_topp")
    def test_forward_single_request(self, mock_sample, mock_setup):
        import torch

        from vkwr.model_executor.layers.sampler import RWKV7Sampler

        sampler = RWKV7Sampler(seed=42)

        logits = torch.randn(1, 100, dtype=torch.float16)
        mock_states = torch.zeros(64, dtype=torch.int64)
        mock_setup.return_value = mock_states
        mock_sample.return_value = torch.tensor([5])

        sampling_params_list = [SamplingParams(temperature=0.8, top_k=50, top_p=0.9)]
        sampled, logprobs = sampler(logits, sampling_params_list)

        mock_setup.assert_called_once_with(42, 1)
        mock_sample.assert_called_once_with(
            logits,
            mock_states,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        assert sampled.tolist() == [5]
        assert logprobs is None

    @patch("vkwr.model_executor.layers.sampler.setup_rand")
    @patch("vkwr.model_executor.layers.sampler.sample_temperature_topk_topp")
    def test_forward_states_reused(self, mock_sample, mock_setup):
        import torch

        from vkwr.model_executor.layers.sampler import RWKV7Sampler

        sampler = RWKV7Sampler(seed=42)

        logits1 = torch.randn(1, 100, dtype=torch.float16)
        mock_setup.return_value = torch.zeros(64, dtype=torch.int64)
        mock_sample.return_value = torch.tensor([5])

        params = [SamplingParams()]
        sampler(logits1, params)
        sampler(logits1, params)

        assert mock_setup.call_count == 1
        assert mock_sample.call_count == 2

    @patch("vkwr.model_executor.layers.sampler.setup_rand")
    @patch("vkwr.model_executor.layers.sampler.sample_temperature_topk_topp")
    def test_forward_reinit_on_batch_size_change(self, mock_sample, mock_setup):
        import torch

        from vkwr.model_executor.layers.sampler import RWKV7Sampler

        sampler = RWKV7Sampler(seed=42)

        logits1 = torch.randn(1, 100, dtype=torch.float16)
        logits2 = torch.randn(2, 100, dtype=torch.float16)

        mock_setup.side_effect = [
            torch.zeros(64, dtype=torch.int64),
            torch.zeros(128, dtype=torch.int64),
        ]
        mock_sample.return_value = torch.tensor([5])

        params = [SamplingParams()]
        sampler(logits1, params)
        sampler(logits2, params)

        assert mock_setup.call_count == 2

    @patch("vkwr.model_executor.layers.sampler.setup_rand")
    @patch("vkwr.model_executor.layers.sampler.sample_temperature_topk_topp")
    def test_forward_default_params(self, mock_sample, mock_setup):
        import torch

        from vkwr.model_executor.layers.sampler import RWKV7Sampler

        sampler = RWKV7Sampler(seed=123)

        logits = torch.randn(1, 50, dtype=torch.float16)
        mock_setup.return_value = torch.zeros(64, dtype=torch.int64)
        mock_sample.return_value = torch.tensor([10])

        sampling_params_list = [SamplingParams()]
        sampled, _ = sampler(logits, sampling_params_list)

        mock_sample.assert_called_once_with(
            logits,
            mock_setup.return_value,
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
        )
