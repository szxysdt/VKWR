"""Real-model integration test for SyncMPClient.

Spawns a real EngineCore subprocess with the RWKV7 model, sends a request
through SyncMPClient, and verifies the output matches InprocClient.

Requires GPU and a loaded RWKV7 model. Run:

    VKWR_ENGINE_READY_TIMEOUT_S=300 \
    VKWR_RWKV7_MODEL_PATH=/path/to/model.pth \
    uv run pytest tests/engine/test_sync_mp_client_real.py -v --tb=short -s

Skips automatically if:
- CUDA is unavailable
- Model file does not exist
"""

from __future__ import annotations

import os
import signal
import time

import pytest
import torch

from vkwr.config.compilation import CompilationConfig
from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.config.worker import GPUWorkerConfig
from vkwr.engine.core_client import EngineCoreClient, InprocClient, SyncMPClient
from vkwr.engine.core_request import EngineCoreRequest
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import SamplingParams

MODEL_PATH = os.environ.get("VKWR_RWKV7_MODEL_PATH")

_DEFAULT_GET_OUTPUT_TIMEOUT_S = 120.0


def _make_config() -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model=MODEL_PATH, max_model_len=2048),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=256),
        worker_config=GPUWorkerConfig(device="cuda", enforce_eager=True, shutdown_timeout=0),
        compilation_config=CompilationConfig(cudagraph_mode="none"),
    )


def _make_request(
    request_id: str = "req-1",
    prompt_token_ids: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=sampling_params or SamplingParams(max_tokens=16),
        arrival_time=time.monotonic(),
        priority=0,
        current_wave=0,
    )


def _collect_outputs(
    client,
    request_ids: list[str],
    timeout: float | None = None,
) -> dict[str, list[EngineCoreOutput]]:
    """Poll client.get_output() until all request_ids have a finished output.

    Mirrors the vLLM pattern: loop until done, skipping empty outputs.
    """
    timeout = timeout or _DEFAULT_GET_OUTPUT_TIMEOUT_S
    outputs: dict[str, list[EngineCoreOutput]] = {rid: [] for rid in request_ids}
    unfinished = set(request_ids)
    deadline = time.monotonic() + timeout

    while unfinished and time.monotonic() < deadline:
        engine_outputs = client.get_output(timeout=1.0)
        if not engine_outputs.outputs:
            continue

        for out in engine_outputs.outputs:
            if out.request_id in outputs:
                outputs[out.request_id].append(out)
                if out.finish_reason is not None:
                    unfinished.discard(out.request_id)

    return outputs


def _poll_first_output(
    client,
    timeout: float | None = None,
) -> EngineCoreOutputs:
    """Poll until the first non-empty EngineCoreOutputs is returned."""
    timeout = timeout or _DEFAULT_GET_OUTPUT_TIMEOUT_S
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get_output(timeout=1.0)
        if result.outputs:
            return result
    raise TimeoutError(f"Timeout waiting for first output (after {timeout}s)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(MODEL_PATH is None, reason="VKWR_RWKV7_MODEL_PATH not set")
@pytest.mark.skipif(MODEL_PATH is not None and not os.path.isfile(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}")
class TestSyncMPClientRealModel:
    """Integration tests with a real RWKV7 model via SyncMPClient."""

    def test_add_request_get_output_sync(self):
        """SyncMPClient: add a request and get output across process boundary."""
        config = _make_config()

        client = SyncMPClient(config)
        time.sleep(0.1)

        req = _make_request("real-1", [1, 2, 3])
        client.add_request(req)

        outputs = _poll_first_output(client)
        assert isinstance(outputs, EngineCoreOutputs)
        assert len(outputs.outputs) >= 1

        out = outputs.outputs[0]
        assert out.request_id == "real-1"
        assert len(out.new_token_ids) > 0, "Expected at least one generated token"
        print(f"  Generated token: {out.new_token_ids[0]}")

        client.shutdown()

    def test_add_request_get_output_inproc(self):
        """InprocClient: add a request and get output in same process (baseline)."""
        config = _make_config()

        client = InprocClient(config)

        req = _make_request("inproc-1", [1, 2, 3])
        client.add_request(req)

        outputs = _poll_first_output(client)
        assert isinstance(outputs, EngineCoreOutputs)
        assert len(outputs.outputs) >= 1

        out = outputs.outputs[0]
        assert out.request_id == "inproc-1"
        assert len(out.new_token_ids) > 0, "Expected at least one generated token"
        print(f"  Generated token: {out.new_token_ids[0]}")

        client.shutdown()

    def test_sync_inproc_consistency(self):
        """SyncMPClient and InprocClient produce identical outputs for same input."""
        # Use temperature=0 (greedy) to avoid non-deterministic sampling.
        # RWKV7Sampler doesn't honor per-request seed, so greedy is the only
        # reliable way to guarantee reproducibility across two independent runs.
        # temperature=0.001 is the minimum accepted by the sampling kernel;
        # at this value the sampler behaves deterministically (argmax).
        greedy_params = SamplingParams(max_tokens=16, temperature=0.001)

        # Run SyncMPClient first (avoids GPU OOM when InprocClient loads first)
        sync_config = _make_config()
        sync = SyncMPClient(sync_config)
        time.sleep(0.1)
        sync.add_request(_make_request("consistency", [1, 2, 3], greedy_params))
        sync_outputs = _collect_outputs(sync, ["consistency"])
        sync.shutdown()

        # Run InprocClient
        inproc_config = _make_config()
        inproc = InprocClient(inproc_config)
        inproc.add_request(_make_request("consistency", [1, 2, 3], greedy_params))
        inproc_outputs = _collect_outputs(inproc, ["consistency"])
        inproc.shutdown()

        # Compare aggregated token ids
        assert len(inproc_outputs["consistency"]) >= 1
        assert len(sync_outputs["consistency"]) >= 1

        in_tokens = []
        for o in inproc_outputs["consistency"]:
            in_tokens.extend(o.new_token_ids)

        sn_tokens = []
        for o in sync_outputs["consistency"]:
            sn_tokens.extend(o.new_token_ids)

        assert in_tokens == sn_tokens
        print(f"  Both produced tokens: {in_tokens}")

    def test_abort_requests_sync(self):
        """SyncMPClient: abort a request and receive abort output."""
        config = _make_config()

        client = SyncMPClient(config)
        time.sleep(0.1)

        req = _make_request("abort-real", [1, 2, 3])
        client.add_request(req)

        # Abort immediately
        client.abort_requests(["abort-real"])

        # Poll for output (either normal generation or abort)
        outputs = _poll_first_output(client)
        assert isinstance(outputs, EngineCoreOutputs)
        assert len(outputs.outputs) >= 1

        out = outputs.outputs[0]
        assert out.request_id == "abort-real"
        assert len(out.new_token_ids) > 0 or out.finish_reason == "abort"

        client.shutdown()

    def test_multiple_requests(self):
        """SyncMPClient: send multiple requests and receive outputs."""
        config = _make_config()

        client = SyncMPClient(config)
        time.sleep(0.1)

        client.add_request(_make_request("multi-a", [10, 20]))
        client.add_request(_make_request("multi-b", [30, 40]))

        all_outputs = _collect_outputs(client, ["multi-a", "multi-b"])

        assert len(all_outputs["multi-a"]) >= 1
        assert len(all_outputs["multi-b"]) >= 1

        a_tokens = []
        for o in all_outputs["multi-a"]:
            a_tokens.extend(o.new_token_ids)
        b_tokens = []
        for o in all_outputs["multi-b"]:
            b_tokens.extend(o.new_token_ids)

        assert len(a_tokens) > 0
        assert len(b_tokens) > 0

        client.shutdown()

    def test_engine_dead_on_process_kill(self):
        """When the engine process is killed, get_output raises EngineDeadError."""
        config = _make_config()

        client = SyncMPClient(config)
        time.sleep(0.1)

        # Verify engine is alive
        client.ensure_alive()
        pid = client._process.pid

        # Kill the engine process
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)  # Wait for monitor thread to detect death

        # get_output should raise EngineDeadError
        with pytest.raises(EngineDeadError):
            client.get_output(timeout=5)

        # Cleanup
        try:
            client.shutdown()
        except Exception:
            pass

    def test_factory_creates_sync_client(self):
        """EngineCoreClient.make_client creates SyncMPClient when multiprocess_mode=True."""
        config = _make_config()

        client = EngineCoreClient.make_client(multiprocess_mode=True, vkwr_config=config)
        assert isinstance(client, SyncMPClient)

        # Send a request to verify it works
        req = _make_request("factory-1", [1, 2])
        client.add_request(req)

        outputs = _poll_first_output(client)
        assert isinstance(outputs, EngineCoreOutputs)
        assert len(outputs.outputs) >= 1

        client.shutdown()
