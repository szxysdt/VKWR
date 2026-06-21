"""Tests for EngineCoreClient ABC, InprocClient, and factory method.

Covers:
- EngineCoreClient ABC abstract methods
- InprocClient add_request / get_output equivalence with direct EngineCore calls
- make_client factory returns correct client types
- make_client raises for invalid asyncio_only combination
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.engine.core_request import EngineCoreRequest
from vkwr.engine.outputs import EngineCoreOutputs
from vkwr.engine.request import SamplingParams


def _make_config() -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
    )


def _make_request(request_id: str = "req-1", prompt_token_ids: list[int] | None = None) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        arrival_time=time.monotonic(),
        priority=0,
        current_wave=0,
    )


class TestEngineCoreClientABC:
    """Verify EngineCoreClient is a proper ABC with required methods."""

    def test_abc_has_abstract_methods(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        abstract = EngineCoreClient.__abstractmethods__
        assert "shutdown" in abstract
        assert len(abstract) >= 1

    def test_cannot_instantiate_abc(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        with pytest.raises(TypeError, match="Can't instantiate abstract"):
            EngineCoreClient()  # type: ignore[misc]


class TestMakeClientFactory:
    """Test EngineCoreClient.make_client factory method."""

    def test_make_client_returns_inproc(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        with patch("vkwr.engine.core_client.InprocClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = EngineCoreClient.make_client(multiprocess_mode=False, vkwr_config=_make_config())
            MockClient.assert_called_once()
            assert isinstance(client, MagicMock)

    def test_make_client_returns_sync_mp(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        with patch("vkwr.engine.core_client.SyncMPClient") as MockClient:
            MockClient.return_value = MagicMock()
            EngineCoreClient.make_client(multiprocess_mode=True, asyncio_mode=False, vkwr_config=_make_config())
            MockClient.assert_called_once()

    def test_make_client_returns_async_mp(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        with patch("vkwr.engine.core_client.AsyncMPClient") as MockClient:
            MockClient.return_value = MagicMock()
            EngineCoreClient.make_client(multiprocess_mode=True, asyncio_mode=True, vkwr_config=_make_config())
            MockClient.assert_called_once()

    def test_make_client_async_only_raises(self) -> None:
        from vkwr.engine.core_client import EngineCoreClient

        with pytest.raises(NotImplementedError, match="asyncio without multiprocessing"):
            EngineCoreClient.make_client(multiprocess_mode=False, asyncio_mode=True, vkwr_config=_make_config())


class TestInprocClientInit:
    """Test InprocClient initialization and lifecycle."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_inproc_creates_engine_core(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        assert client.engine_core is not None
        assert client.engine_core._initialized is True
        client.shutdown()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_inproc_shutdown(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)
        client.engine_core.shutdown = MagicMock()
        client.shutdown()
        client.engine_core.shutdown.assert_called_once()


class TestInprocClientAddRequest:
    """Test InprocClient add_request dispatches to EngineCore."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_add_request_dispatches(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        req = _make_request("add-test")
        client.add_request(req)

        assert "add-test" in client.engine_core.scheduler.running or len(client.engine_core.scheduler.waiting) > 0
        client.shutdown()


class TestInprocClientGetOutput:
    """Test InprocClient get_output returns EngineCoreOutputs."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_get_output_no_requests_returns_empty(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        # No requests added; should return empty outputs
        result = client.get_output(timeout=0.1)
        assert isinstance(result, EngineCoreOutputs)
        assert not result.outputs
        client.shutdown()


class TestInprocClientAbortRequests:
    """Test InprocClient abort_requests dispatches correctly."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_requests_noop_when_empty(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        client.abort_requests([])
        client.shutdown()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_requests_calls_engine_core(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)
        client.engine_core.abort_requests = MagicMock(return_value={0: EngineCoreOutputs()})

        client.abort_requests(["req-abort"])
        client.engine_core.abort_requests.assert_called_once_with(["req-abort"])
        client.shutdown()


class TestInprocClientUtility:
    """Test InprocClient call_utility dispatches to EngineCore."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_call_utility_direct_call(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.core_client import InprocClient

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        client.engine_core.fake_method = MagicMock(return_value="utility-result")
        result = client.call_utility("fake_method", "arg1")
        assert result == "utility-result"
        client.shutdown()
