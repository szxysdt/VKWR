from __future__ import annotations

import gc
import itertools
import logging
from typing import TYPE_CHECKING

import torch

from vkwr.engine.outputs import ModelRunnerOutput
from vkwr.model_executor.layers.sampler import RWKV7Sampler
from vkwr.model_executor.models.rwkv7_varlen_v2x import RWKV7

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.scheduler.output import SchedulerOutput
    from vkwr.state.state_slot_manager import StateSlotManager

logger = logging.getLogger(__name__)


class GPUModelRunner:
    """GPU Model Runner - key component connecting the scheduler and model layers"""

    def __init__(
        self,
        config: VkwrConfig,
        device: torch.device,
        slot_manager: StateSlotManager | None = None,
    ):
        self.config = config
        self.device = device
        self.model_config = config.model_config
        self.scheduler_config = config.scheduler_config
        self.dtype = getattr(torch, config.model_config.dtype)

        self.max_num_seqs = config.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens

        self.slot_manager: StateSlotManager | None = slot_manager

        self.model: RWKV7 | None = None
        self.sampler: RWKV7Sampler | None = None

        # State buffers (allocated in load_model)
        self._state_shift: torch.Tensor | None = None
        self._state_wkv: torch.Tensor | None = None
        self._state_elapsed: torch.Tensor | None = None

        # Static input buffers (Task 1)
        self._input_ids: torch.Tensor | None = None
        self._query_start_loc: torch.Tensor | None = None
        self._input_ids_host: torch.Tensor | None = None
        self._query_start_loc_host: torch.Tensor | None = None
        self._scatter_idx: torch.Tensor | None = None

        # Uniform decode static buffers (Task 1 + Task 4)
        self._uniform_query_start_loc: torch.Tensor | None = None
        self._uniform_req_id: torch.Tensor | None = None
        self._uniform_tokens: torch.Tensor | None = None
        self._uniform_slot_indices: torch.Tensor | None = None
        self._uniform_x: torch.Tensor | None = None
        self._uniform_x_host: torch.Tensor | None = None
        self._emb_buf_idx: int = 0
        self._emb_dma_event: torch.cuda.Event | None = None

        # CUDA Graph manager (Task 4)
        self.cudagraph_manager: any | None = None
        self._cudagraph_enabled: bool = False

        # State cache manager (Task 6)
        self.state_cache: any | None = None

        # Last sampled token per slot (GPU cache for decode input)
        self._last_sampled_token: torch.Tensor | None = None
        self._decode_tokens_min_prev: torch.Tensor | None = None

        # Pre-allocated decode index buffers (avoids per-step allocation)
        self._decode_slot_indices_i32: torch.Tensor | None = None
        self._decode_pad_indices_i64: torch.Tensor | None = None

    def load_model(self) -> None:
        """Load model weights, initialize RWKV7Model + allocate state buffers"""
        from vkwr.config.model import WeightConfig

        weight_config = WeightConfig()

        self.model = RWKV7(
            self.model_config.model,
            weight_config=weight_config,
        )

        cfg = self.model.config
        self.L, self.C, self.H, self.N = cfg.L, cfg.C, cfg.H, cfg.N

        # Allocate reusable state buffers
        max_bsz = self.max_num_seqs
        self._state_shift = torch.zeros(
            (self.L, 2, max_bsz, self.C),
            dtype=self.dtype,
            device=self.device,
        )
        self._state_wkv = torch.zeros(
            (self.L, max_bsz, self.H, self.N, self.N),
            dtype=torch.float16,
            device=self.device,
        )
        self._state_elapsed = torch.zeros(
            (max_bsz,),
            dtype=torch.int32,
            device=self.device,
        )
        self._last_sampled_token = torch.full(
            (max_bsz,),
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
        )

        # Log state buffer sizes
        per_slot_shift = self.L * 2 * self.C * 2
        per_slot_wkv = self.L * self.H * self.N * self.N * 2
        per_slot_meta = 4 + 8
        per_slot_total = per_slot_shift + per_slot_wkv + per_slot_meta
        mb = 1024 * 1024
        logger.info(
            "State buffers: L=%d C=%d H=%d N=%d max_slots=%d | per_slot=%.2f MB (shift=%.1fKB wkv=%.1fKB meta=%dB) | total=%.2f MB",
            self.L,
            self.C,
            self.H,
            self.N,
            max_bsz,
            per_slot_total / mb,
            per_slot_shift / 1024,
            per_slot_wkv / 1024,
            per_slot_meta,
            per_slot_total * max_bsz / mb,
        )

        # Task 1: Static input buffers (zero-allocation)
        self._input_ids = torch.empty(
            (self.max_num_batched_tokens,),
            dtype=torch.long,
            device=self.device,
        )
        self._query_start_loc = torch.empty(
            (self.max_num_seqs + 1,),
            dtype=torch.int32,
            device=self.device,
        )
        self._input_ids_host = torch.empty(
            (self.max_num_batched_tokens,),
            dtype=torch.long,
            pin_memory=True,
        )
        self._query_start_loc_host = torch.empty(
            (self.max_num_seqs + 1,),
            dtype=torch.int32,
            pin_memory=True,
        )
        self._scatter_idx = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.long,
            device=self.device,
        )

        # Task 4: Uniform decode static buffers
        self._uniform_query_start_loc = torch.arange(self.max_num_seqs + 1, dtype=torch.int32, device=self.device)
        self._uniform_req_id = torch.arange(self.max_num_seqs, dtype=torch.int32, device=self.device)
        # Must use torch.full with a valid token id, NOT torch.empty.
        # Uninitialized garbage values from torch.empty caused embedding lookup OOB
        # during CUDA graph warmup, corrupting downstream cuBLAS.
        self._uniform_tokens = torch.full((self.max_num_seqs,), 1, dtype=torch.long, device=self.device)
        self._uniform_slot_indices = torch.empty((self.max_num_seqs,), dtype=torch.int32, device=self.device)
        self._uniform_x = torch.empty(
            (self.max_num_seqs, self.model.config.C),
            dtype=self.model.inference_config.dtype,
            device=self.device,
        )
        self._uniform_x_host = torch.empty(
            (2, self.max_num_seqs, self.model.config.C),
            dtype=self.model.inference_config.dtype,
            pin_memory=True,
        )
        self._emb_buf_idx = 0
        self._emb_dma_event = torch.cuda.Event()

        # Pre-allocated decode index buffers (reuse across steps, avoids per-step alloc)
        self._decode_slot_indices_i32 = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.int32,
            device=self.device,
        )
        self._decode_pad_indices_i64 = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.long,
            device=self.device,
        )

        # Task 4: CUDA Graph manager
        cudagraph_mode = self.config.compilation_config.cudagraph_mode
        if cudagraph_mode != "none":
            from vkwr.worker.cuda_graph import CUDAGraphManager

            raw_sizes = self.config.compilation_config.cudagraph_capture_size
            cudagraph_shapes = [(1,) * b for b in raw_sizes] if raw_sizes else None
            self.cudagraph_manager = CUDAGraphManager(
                cudagraph_shapes,
                self.device,
                emb_cpu=self.model.emb_cpu,
                max_batch_size=self.max_num_seqs,
            )
            self._cudagraph_enabled = True
        else:
            self.cudagraph_manager = None
            self._cudagraph_enabled = False

        # Task 6: State cache manager
        from vkwr.state.state_cache import StateCacheManager

        self.state_cache = StateCacheManager(
            max_cpu_entries=self.config.state_config.state_cache_max_cpu_entries,
            max_checkpoints_per_req=self.config.state_config.max_checkpoints_per_req,
            device=self.device,
        )

        self.sampler = RWKV7Sampler(seed=self.model_config.seed)
        logger.info(
            "Model loaded: L=%d C=%d H=%d N=%d V=%d",
            self.L,
            self.C,
            self.H,
            self.N,
            cfg.V,
        )

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model forward pass with varlen support."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not scheduler_output.scheduled_req_ids:
            return ModelRunnerOutput(sampled_token_ids={})

        is_uniform = self._is_uniform_decode(scheduler_output)

        if is_uniform:
            return self._execute_uniform_decode(scheduler_output)
        else:
            return self._execute_eager(scheduler_output)

    def _execute_eager(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Eager execution path for non-uniform-decode batches."""
        # Zero state for new requests
        for req_id in scheduler_output.scheduled_req_ids:
            run_data = scheduler_output.request_data[req_id]
            if not run_data.is_decode and run_data.num_computed_tokens == 0:
                slot = run_data.slot_index
                self._state_shift[:, :, slot].zero_()
                self._state_wkv[:, slot].zero_()
                self._state_elapsed[slot].zero_()

        tokens, query_start_loc, max_t = self._prepare_input_ids(scheduler_output)
        indices = [scheduler_output.request_data[rid].slot_index for rid in scheduler_output.scheduled_req_ids]
        B = len(indices)
        self._decode_slot_indices_i32[:B].copy_(torch.tensor(indices, dtype=torch.int32))
        slot_indices = self._decode_slot_indices_i32[:B]
        state = [
            self._state_shift,
            self._state_wkv,
            self._state_elapsed,
        ]

        is_prefill = any(not scheduler_output.request_data[rid].is_decode for rid in scheduler_output.scheduled_req_ids)
        if is_prefill:
            logger.info("Prefill batch: total_tokens=%d, max_t=%d, B=%d", tokens.numel(), max_t, len(scheduler_output.scheduled_req_ids))

        with torch.inference_mode():
            logits = self.model.forward(tokens, state, query_start_loc, max_t, slot_indices)

        torch.cuda.synchronize()

        return ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=logits,
        )

    def _execute_uniform_decode(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Uniform decode fast path with CUDA Graph support.

        v2x: uses slot_indices to address global state buffer directly.
        Supports free slot padding for CUDA graph.
        """
        B = len(scheduler_output.scheduled_req_ids)
        slot_indices = [scheduler_output.request_data[rid].slot_index for rid in scheduler_output.scheduled_req_ids]
        self._decode_slot_indices_i32[:B].copy_(torch.tensor(slot_indices, dtype=torch.int32))
        slot_indices_tensor = self._decode_slot_indices_i32[:B]
        self._decode_pad_indices_i64[:B].copy_(torch.tensor(slot_indices, dtype=torch.long))
        decode_tokens_gpu = self._last_sampled_token.index_select(0, self._decode_pad_indices_i64[:B])

        if hasattr(self, "_decode_tokens_min_prev") and self._decode_tokens_min_prev is not None:
            if self._decode_tokens_min_prev.item() < 0:
                raise RuntimeError("Some slots have no sampled token. This should not happen if is_last_prefill sampling is correct.")
        self._decode_tokens_min_prev = decode_tokens_gpu.min()

        if self._cudagraph_enabled:
            graph, _output_logits, padded_B = self.cudagraph_manager.get_graph_with_padding(B)
        else:
            graph, _output_logits, padded_B = None, None, B  # type: ignore[assignment]

        n_pad = (padded_B - B) if graph is not None else 0
        can_pad = (n_pad <= self.slot_manager.available_pad_count) if (n_pad > 0 and self.slot_manager) else True

        if graph is None or not can_pad:
            logits = self._execute_uniform_decode_eager(B, decode_tokens_gpu, slot_indices_tensor)
        else:
            if n_pad > 0:
                pad_slots = self.slot_manager.reserve_pad_slots(n_pad)
                self._decode_slot_indices_i32[B:padded_B].copy_(torch.tensor(pad_slots, dtype=torch.int32))
                self._prepare_padding_slots(pad_slots)
                self._uniform_slot_indices[:padded_B].copy_(self._decode_slot_indices_i32[:padded_B])
            else:
                self._uniform_slot_indices[:B].copy_(slot_indices_tensor)
            self._uniform_tokens[B:padded_B].fill_(0)

            if self.model.emb_cpu:
                self._prepare_uniform_decode_cpu_emb(B, decode_tokens_gpu, padded_B)
            else:
                self._uniform_tokens[:B].copy_(decode_tokens_gpu, non_blocking=True)
                if padded_B > B:
                    self._uniform_tokens[B:padded_B].fill_(0)

            logits = self.cudagraph_manager.replay(tuple([1] * padded_B))

            if n_pad > 0:
                self._cleanup_dummy_slots(pad_slots)
                self.slot_manager.release_pad_slots()
                logits = logits[:B]

        torch.cuda.synchronize()

        self._maybe_checkpoint(scheduler_output, slot_indices)

        return ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=logits,
        )

    def _execute_uniform_decode_eager(self, B: int, decode_tokens: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        """Eager fallback for uniform decode using global state buffer.

        v2x: uses slot_indices to address global state, no gather needed.
        """
        if self.model.emb_cpu:
            self._prepare_uniform_decode_cpu_emb(B, decode_tokens)
            x = self._uniform_x[:B]
        else:
            x = self.model.embed(decode_tokens)

        state = [
            self._state_shift,
            self._state_wkv,
            self._state_elapsed,
        ]

        query_start_loc = self._uniform_query_start_loc[: B + 1]
        req_id = self._uniform_req_id[:B]
        path = self.model.path_selector.select(B, 1, B)

        with torch.inference_mode():
            logits = self.model.forward_from_x(x, state, path, query_start_loc, req_id, 1, B, slot_indices)
        return logits

    def _prepare_uniform_decode_cpu_emb(self, B: int, decode_tokens: torch.Tensor, padded_B: int = 0) -> None:
        """CPU emb mode: embed outside graph, copy to x buffer before replay.

        If padded_B > B, also embed dummy tokens (token=0) for padding positions.
        """
        idx = self._emb_buf_idx & 1
        buf = self._uniform_x_host[idx]
        self._emb_dma_event.wait()
        flat = decode_tokens[:B].cpu()
        torch.index_select(self.model.z["emb.weight"], 0, flat, out=buf[:B])
        self._uniform_x[:B].copy_(
            buf[:B],
            non_blocking=True,
        )

        if padded_B > B:
            dummy_tokens = torch.zeros(padded_B - B, dtype=torch.long, device="cpu")
            torch.index_select(
                self.model.z["emb.weight"],
                0,
                dummy_tokens,
                out=buf[B:padded_B],
            )
            self._uniform_x[B:padded_B].copy_(buf[B:padded_B], non_blocking=True)

        self._emb_dma_event.record()
        self._emb_buf_idx = 1 - idx

    def _prepare_padding_slots(self, pad_slot_indices: list[int]) -> None:
        """Before replay: zero state for actual pad slot indices."""
        n = len(pad_slot_indices)
        self._decode_pad_indices_i64[:n].copy_(torch.tensor(pad_slot_indices, dtype=torch.long))
        idx = self._decode_pad_indices_i64[:n]
        self._state_shift[:, :, idx].zero_()
        self._state_wkv[:, idx].zero_()
        self._state_elapsed[idx].zero_()
        self._last_sampled_token[idx].fill_(0)

    def _cleanup_dummy_slots(self, pad_slot_indices: list[int]) -> None:
        """After replay: zero state for actual pad slot indices that graph wrote to.

        RNN models (RWKV7) have no attention mask to isolate pad positions,
        so dummy forward writes non-zero state that must be cleaned up.
        """
        n = len(pad_slot_indices)
        self._decode_pad_indices_i64[:n].copy_(torch.tensor(pad_slot_indices, dtype=torch.long))
        idx = self._decode_pad_indices_i64[:n]
        self._state_shift[:, :, idx].zero_()
        self._state_wkv[:, idx].zero_()
        self._state_elapsed[idx].zero_()
        self._last_sampled_token[idx].fill_(0)

    def sample_tokens(
        self,
        model_runner_output: ModelRunnerOutput,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Sample from logits, return sampling results mapped by req_id.

        Requests that are in chunked-prefill (not the last chunk) are skipped:
        they do not produce sampled tokens and their logits are not passed to
        the sampler.
        """
        if self.sampler is None:
            raise RuntimeError("Sampler not initialized. Call load_model() first.")

        if not scheduler_output.scheduled_req_ids:
            return ModelRunnerOutput(sampled_token_ids={})

        logits = model_runner_output.logits
        if logits is None:
            raise RuntimeError("No logits in model_runner_output")

        # Separate requests that need sampling from those that don't
        # (chunked prefill non-last-chunk requests).
        need_sample_indices: list[int] = []
        need_sample_req_ids: list[str] = []
        for idx, req_id in enumerate(scheduler_output.scheduled_req_ids):
            run_data = scheduler_output.request_data[req_id]
            if run_data and (run_data.is_decode or run_data.is_last_prefill):
                need_sample_indices.append(idx)
                need_sample_req_ids.append(req_id)

        if not need_sample_req_ids:
            return ModelRunnerOutput(sampled_token_ids={}, sampled_logprobs=None, logits=None)

        # Slice logits only for requests that need sampling
        sample_indices = torch.tensor(need_sample_indices, dtype=torch.long, device=self.device)
        sample_logits = logits.index_select(0, sample_indices)

        sampling_params_list = [scheduler_output.request_data[req_id].sampling_params for req_id in need_sample_req_ids]
        sampled, logprobs = self.sampler(sample_logits.float(), sampling_params_list)

        # Write sampled tokens to per-slot cache.
        slot_positions = [scheduler_output.request_data[req_id].slot_index for req_id in need_sample_req_ids]
        if len(need_sample_req_ids) > 0 and slot_positions == list(range(len(need_sample_req_ids))):
            self._last_sampled_token[: len(need_sample_req_ids)].copy_(sampled.to(torch.long))
        else:
            slot_indices = torch.tensor(slot_positions, dtype=torch.long, device=self.device)
            self._last_sampled_token.index_copy_(0, slot_indices, sampled.to(torch.long))

        # Build output dict using GPU tensor directly (no .tolist())
        sampled_token_ids: dict[str, list[int]] = {}
        sampled_cpu = sampled.cpu()
        for i, req_id in enumerate(need_sample_req_ids):
            sampled_token_ids[req_id] = [int(sampled_cpu[i])]

        return ModelRunnerOutput(
            sampled_token_ids=sampled_token_ids,
            sampled_logprobs=logprobs,
            logits=None,
        )

    def _prepare_input_ids(self, scheduler_output: SchedulerOutput) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Build flat input_ids, query_start_loc, and max_t using pre-allocated buffers."""
        all_tokens = []
        for req_id in scheduler_output.scheduled_req_ids:
            run_data = scheduler_output.request_data[req_id]
            if run_data.input_token_ids is not None:
                all_tokens.extend(run_data.input_token_ids)
            else:
                slot = run_data.slot_index
                all_tokens.append(int(self._last_sampled_token[slot]))

        total_tokens = len(all_tokens)

        self._input_ids_host[:total_tokens] = torch.as_tensor(all_tokens, dtype=torch.long)
        self._input_ids[:total_tokens].copy_(
            self._input_ids_host[:total_tokens],
            non_blocking=True,
        )

        seq_lens = [scheduler_output.request_data[rid].num_tokens for rid in scheduler_output.scheduled_req_ids]
        offsets = [0] + list(itertools.accumulate(seq_lens))
        self._query_start_loc_host[: len(offsets)] = torch.as_tensor(offsets, dtype=torch.int32)
        self._query_start_loc[: len(offsets)].copy_(
            self._query_start_loc_host[: len(offsets)],
            non_blocking=True,
        )

        max_t = max(seq_lens) if seq_lens else 1

        return (
            self._input_ids[:total_tokens],
            self._query_start_loc[: len(offsets)],
            max_t,
        )

    def _slice_state_from_slot(self, slot: int) -> list[torch.Tensor]:
        """Slice a single request's state from the global GPU buffer."""
        return [
            self._state_shift[:, :, slot],
            self._state_wkv[:, slot],
            self._state_elapsed[slot],
        ]

    def _write_state_to_slot(self, slot: int, state: list[torch.Tensor]) -> None:
        """Write state to a global GPU buffer slot."""
        self._state_shift[:, :, slot] = state[0]
        self._state_wkv[:, slot] = state[1]
        self._state_elapsed[slot] = state[2]

    def _maybe_checkpoint(self, scheduler_output: SchedulerOutput, slot_indices: list[int]) -> None:
        """Optional checkpoint: save state snapshot every N steps."""
        interval = self.config.state_config.checkpoint_interval
        if interval is None or interval <= 0:
            return

        for i, req_id in enumerate(scheduler_output.scheduled_req_ids):
            step = scheduler_output.request_data[req_id].num_computed_tokens
            if step % interval == 0:
                slot = slot_indices[i]
                req_state = self._slice_state_from_slot(slot)
                self.state_cache.checkpoint(req_id, step, req_state)

    def warmup(self) -> None:
        """Basic warmup: run a small-batch forward pass"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Warming up model...")
        with torch.inference_mode():
            tokens = torch.tensor([1, 2, 3], dtype=torch.long, device=self.device)
            state = self.model.zero_state(1)
            query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
            slot_indices = torch.tensor([0], dtype=torch.int32, device=self.device)
            _ = self.model.forward(tokens, state, query_start_loc, 3, slot_indices)
        logger.info("Warmup complete.")

    def _is_uniform_decode(self, scheduler_output: SchedulerOutput) -> bool:
        """Check if this is a uniform decode batch."""
        if scheduler_output.total_num_scheduled_tokens == 0:
            return False
        num_reqs = len(scheduler_output.scheduled_req_ids)
        if scheduler_output.total_num_scheduled_tokens != num_reqs:
            return False
        for req_id in scheduler_output.scheduled_req_ids:
            if not scheduler_output.request_data[req_id].is_decode:
                return False
        return True

    def _capture_for_shape(self, seq_lens: tuple[int]) -> None:
        """Pre-capture CUDA Graph for a varlen shape."""
        model = self.model
        B = len(seq_lens)
        total_tokens = sum(seq_lens)
        max_t = max(seq_lens)

        self._uniform_slot_indices[:B].copy_(torch.arange(B, dtype=torch.int32, device=self.device))

        state = [
            self._state_shift,
            self._state_wkv,
            self._state_elapsed,
        ]

        self._uniform_query_start_loc[: B + 1].copy_(
            torch.tensor(
                [0] + list(itertools.accumulate(seq_lens)),
                dtype=torch.int32,
            ),
        )
        self._uniform_req_id[:total_tokens].copy_(
            torch.tensor(
                [i for i, sl in enumerate(seq_lens) for _ in range(sl)],
                dtype=torch.int32,
            ),
        )
        query_start_loc = self._uniform_query_start_loc[: B + 1]
        req_id = self._uniform_req_id[:total_tokens]
        path = model.path_selector.select(B, max_t, total_tokens)
        slot_indices = self._uniform_slot_indices[:B]

        def forward_fn(x, state, path, query_start_loc, req_id, max_t, padded_B):
            return model.forward_from_x(x, state, path, query_start_loc, req_id, max_t, padded_B, slot_indices)

        if model.emb_cpu:
            x = self._uniform_x[:total_tokens]
            self.cudagraph_manager._capture_for_shape(seq_lens, None, forward_fn, x, state, path, query_start_loc, req_id, max_t, tokens=None)
        else:

            def embed_fn(tokens):
                return model.embed(tokens)

            self.cudagraph_manager._capture_for_shape(
                seq_lens, embed_fn, forward_fn, None, state, path, query_start_loc, req_id, max_t, tokens=self._uniform_tokens
            )

    def determine_available_memory(self) -> int:
        """Profile available GPU memory"""
        gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory
        available_mem = int(gpu_mem_total * self.config.worker_config.gpu_memory_utilization)
        return available_mem

    def shutdown(self) -> None:
        """Release all GPU resources held by this runner."""
        self.model = None
        self.sampler = None
        self.cudagraph_manager = None
        self.state_cache = None
        self._state_shift = None
        self._state_wkv = None
        self._state_elapsed = None
        self._input_ids = None
        self._query_start_loc = None
        self._input_ids_host = None
        self._query_start_loc_host = None
        self._scatter_idx = None
        self._uniform_query_start_loc = None
        self._uniform_req_id = None
        self._uniform_tokens = None
        self._uniform_slot_indices = None
        self._uniform_x = None
        self._uniform_x_host = None
        self._emb_dma_event = None
        self._last_sampled_token = None
        self._decode_tokens_min_prev = None
        self._decode_slot_indices_i32 = None
        self._decode_pad_indices_i64 = None

        gc.collect()
        torch.cuda.empty_cache()
