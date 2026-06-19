from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import torch

from vkwr.model_executor.layers.sampler import RWKV7Sampler
from vkwr.model_executor.models.rwkv7_varlen import RWKV7

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
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
        self._uniform_x: torch.Tensor | None = None
        self._uniform_x_host: torch.Tensor | None = None
        self._emb_buf_idx: int = 0
        self._emb_dma_event: torch.cuda.Event | None = None

        # Decode temp state buffers (Task 2 + Task 4)
        self._decode_state_shift: torch.Tensor | None = None
        self._decode_state_wkv: torch.Tensor | None = None
        self._decode_state_elapsed: torch.Tensor | None = None

        # CUDA Graph manager (Task 4)
        self.cudagraph_manager: any | None = None
        self._cudagraph_enabled: bool = False

        # State cache manager (Task 6)
        self.state_cache: any | None = None

        # Last sampled token per slot (GPU cache for decode input)
        self._last_sampled_token: torch.Tensor | None = None

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
        self._uniform_tokens = torch.empty((self.max_num_seqs,), dtype=torch.long, device=self.device)
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

        # Task 2 + Task 4: Decode temp state buffers
        self._decode_state_shift = torch.empty(
            (self.L, 2, self.max_num_seqs, self.C),
            dtype=self.dtype,
            device=self.device,
        ).zero_()
        self._decode_state_wkv = torch.empty(
            (self.L, self.max_num_seqs, self.H, self.N, self.N),
            dtype=self.dtype,
            device=self.device,
        ).zero_()
        self._decode_state_elapsed = torch.empty(
            (self.max_num_seqs,),
            dtype=torch.int32,
            device=self.device,
        ).zero_()

        # Task 4: CUDA Graph manager
        cudagraph_mode = self.config.compilation_config.cudagraph_mode
        if cudagraph_mode != "none":
            from vkwr.worker.cuda_graph import CUDAGraphManager

            raw_sizes = self.config.compilation_config.cudagraph_capture_size
            cudagraph_shapes = [(1,) * b for b in raw_sizes] if raw_sizes else None
            self.cudagraph_manager = CUDAGraphManager(cudagraph_shapes, self.device, emb_cpu=self.model.emb_cpu)
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

        # Task 5: Apply batch reorder GPU swap
        if scheduler_output.sorted_indices is not None:
            self._apply_batch_reorder(
                scheduler_output,
                scheduler_output.sorted_indices,
                scheduler_output.slot_indices,
            )

        # Task 3: Route to uniform decode or eager path
        is_uniform = self._is_uniform_decode(scheduler_output)
        if is_uniform:
            return self._execute_uniform_decode(scheduler_output)
        else:
            return self._execute_eager(scheduler_output)

    def _execute_eager(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Eager execution path for non-uniform-decode batches."""
        from vkwr.engine.outputs import ModelRunnerOutput

        # Zero state for new requests
        for req_id in scheduler_output.scheduled_req_ids:
            run_data = scheduler_output.request_data[req_id]
            if not run_data.is_decode and run_data.num_computed_tokens == 0:
                slot = run_data.slot_index
                self._state_shift[:, :, slot].zero_()
                self._state_wkv[:, slot].zero_()
                self._state_elapsed[slot].zero_()

        tokens, query_start_loc, max_t = self._prepare_input_ids(scheduler_output)
        state, indices = self._prepare_state(scheduler_output)

        is_prefill = any(not scheduler_output.request_data[rid].is_decode for rid in scheduler_output.scheduled_req_ids)
        if is_prefill:
            logger.info("Prefill batch: total_tokens=%d, max_t=%d, B=%d", tokens.numel(), max_t, len(scheduler_output.scheduled_req_ids))
            try:
                from vkwr.engine.tokenizer import get_tokenizer

                tok = get_tokenizer(self.model_config.tokenizer)
                if tok:
                    for req_id in scheduler_output.scheduled_req_ids:
                        run_data = scheduler_output.request_data[req_id]
                        if not run_data.is_decode:
                            text = tok.decode(run_data.input_token_ids)
                            if len(text) > 200:
                                text = text[:200] + "..."
                            logger.info("Prefill [%s] text: %s", req_id, text)
            except Exception:
                pass

        with torch.inference_mode():
            logits = self.model.forward(tokens, state, query_start_loc, max_t)

        self._scatter_state(state, indices)

        model_output = ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=logits,
        )
        return self.sample_tokens(model_output, scheduler_output)

    def _execute_uniform_decode(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Uniform decode fast path with CUDA Graph support."""
        from vkwr.engine.outputs import ModelRunnerOutput

        B = len(scheduler_output.scheduled_req_ids)
        slot_indices = scheduler_output.slot_indices

        # 1. Gather state into temp buffer (one slot at a time)
        for i in range(B):
            slot = slot_indices[i]
            self._decode_state_shift[:, :, i].copy_(self._state_shift[:, :, slot])
            self._decode_state_wkv[:, i].copy_(self._state_wkv[:, slot])
            self._decode_state_elapsed[i].copy_(self._state_elapsed[slot])

        # 2. Get input tokens from GPU cache (1 per slot)
        decode_tokens = []
        for i in range(B):
            slot = slot_indices[i]
            token = int(self._last_sampled_token[slot])
            if token < 0:
                raise RuntimeError(f"slot {slot} has no sampled token. This should not happen if is_last_prefill sampling is correct.")
            decode_tokens.append(token)

        # 3. Forward: CUDA Graph replay OR eager fallback
        seq_lens = tuple([1] * B)
        entry = self.cudagraph_manager.get_graph(seq_lens) if self._cudagraph_enabled else None
        if entry is None:
            logits = self._execute_uniform_decode_eager(B, decode_tokens)
        else:
            if self.model.emb_cpu:
                self._prepare_uniform_decode_cpu_emb(B, decode_tokens)
            else:
                self._prepare_uniform_decode_gpu_emb(B, decode_tokens)
            logits = self.cudagraph_manager.replay(seq_lens)

        # 4. Scatter temp buffer back to global state (one slot at a time)
        for i in range(B):
            slot = slot_indices[i]
            self._state_shift[:, :, slot].copy_(self._decode_state_shift[:, :, i])
            self._state_wkv[:, slot].copy_(self._decode_state_wkv[:, i])
            self._state_elapsed[slot].copy_(self._decode_state_elapsed[i])

        # 4b. Optional checkpoint
        self._maybe_checkpoint(scheduler_output, slot_indices)

        # 5. Sample
        model_output = ModelRunnerOutput(sampled_token_ids={}, logits=logits)
        return self.sample_tokens(model_output, scheduler_output)

    def _execute_uniform_decode_eager(self, B: int, decode_tokens: list[int]) -> torch.Tensor:
        """Eager fallback for uniform decode when CUDA Graph unavailable."""
        if self.model.emb_cpu:
            self._prepare_uniform_decode_cpu_emb(B, decode_tokens)
            x = self._uniform_x[:B]
        else:
            tokens = torch.as_tensor(decode_tokens, dtype=torch.long, device=self.device)
            x = self.model.embed(tokens)

        state = [
            self._decode_state_shift[:, :, :B],
            self._decode_state_wkv[:, :B],
            self._decode_state_elapsed[:B],
        ]

        query_start_loc = self._uniform_query_start_loc[: B + 1]
        req_id = self._uniform_req_id[:B]
        path = self.model.path_selector.select(B, 1, B)

        with torch.inference_mode():
            logits = self.model.forward_from_x(x, state, path, query_start_loc, req_id, 1, B)
        return logits

    def _prepare_uniform_decode_gpu_emb(self, B: int, decode_tokens: list[int]) -> None:
        """GPU emb mode: update token buffer before replay."""
        self._uniform_tokens[:B].copy_(
            torch.as_tensor(decode_tokens, dtype=torch.long, device="cpu"),
            non_blocking=True,
        )

    def _prepare_uniform_decode_cpu_emb(self, B: int, decode_tokens: list[int]) -> None:
        """CPU emb mode: embed outside graph, copy to x buffer before replay."""
        idx = self._emb_buf_idx & 1
        buf = self._uniform_x_host[idx]
        self._emb_dma_event.wait()
        flat = torch.as_tensor(decode_tokens, dtype=torch.long, device="cpu").reshape(-1)
        torch.index_select(self.model.z["emb.weight"], 0, flat, out=buf[:B])
        self._uniform_x[:B].copy_(
            buf[:B],
            non_blocking=True,
        )
        self._emb_dma_event.record()
        self._emb_buf_idx = 1 - idx

    def sample_tokens(
        self,
        model_runner_output: ModelRunnerOutput,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Sample from logits, return sampling results mapped by req_id."""
        if self.sampler is None:
            raise RuntimeError("Sampler not initialized. Call load_model() first.")

        logits = model_runner_output.logits
        if logits is None:
            raise RuntimeError("No logits in model_runner_output")

        sampling_params_list = []
        req_ids = []
        for req_id in scheduler_output.scheduled_req_ids:
            run_data = scheduler_output.request_data[req_id]
            sampling_params_list.append(run_data.sampling_params)
            req_ids.append(req_id)

        sampled, logprobs = self.sampler(logits.float(), sampling_params_list)

        sampled_token_ids = {}
        for i, req_id in enumerate(req_ids):
            run_data = scheduler_output.request_data.get(req_id)
            if run_data and (run_data.is_decode or run_data.is_last_prefill):
                sampled_token_ids[req_id] = [sampled[i].item()]

        # Write last sampled token to per-slot cache.
        # slot_index here is post-reorder. Since _last_sampled_token
        # is swapped alongside state in _swap_state_slots(), the mapping is consistent.
        for req_id, tokens in sampled_token_ids.items():
            if tokens:
                slot = scheduler_output.request_data[req_id].slot_index
                self._last_sampled_token[slot] = tokens[0]

        from vkwr.engine.outputs import ModelRunnerOutput

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

    def _prepare_state(self, scheduler_output: SchedulerOutput) -> tuple[list[torch.Tensor], list[int]]:
        """Gather state for scheduled requests by slot index, returning contiguous copies."""
        if self._state_shift is None or self._state_wkv is None or self._state_elapsed is None:
            raise RuntimeError("State buffers not allocated. Call load_model() first.")

        req_ids = scheduler_output.scheduled_req_ids
        indices = [scheduler_output.request_data[rid].slot_index for rid in req_ids]

        shift = torch.stack([self._state_shift[:, :, idx] for idx in indices], dim=2).contiguous()
        wkv = torch.stack([self._state_wkv[:, idx] for idx in indices], dim=1).contiguous()
        elapsed = torch.stack([self._state_elapsed[idx] for idx in indices]).contiguous()
        return [shift, wkv, elapsed], indices

    def _scatter_state(self, state: list[torch.Tensor], indices: list[int]) -> None:
        """Write updated state back to global buffer, one slot at a time."""
        B = len(indices)
        for i in range(B):
            slot = indices[i]
            self._state_shift[:, :, slot].copy_(state[0][:, :, i])
            self._state_wkv[:, slot].copy_(state[1][:, i])
            self._state_elapsed[slot].copy_(state[2][i])

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

    def _swap_state_slots(self, i: int, j: int) -> None:
        """Swap two state buffer rows (GPU operation, O(1))."""
        self._state_shift[:, :, [i, j]] = self._state_shift[:, :, [j, i]]
        self._state_wkv[:, [i, j]] = self._state_wkv[:, [j, i]]
        self._state_elapsed[[i, j]] = self._state_elapsed[[j, i]]
        self._last_sampled_token[[i, j]] = self._last_sampled_token[[j, i]]

    def _apply_batch_reorder(
        self,
        scheduler_output: SchedulerOutput,
        sorted_indices,
        slot_indices: list[int],
    ) -> None:
        """Apply GPU-side state swap for batch reorder using cycle-based swap."""
        num_reqs = len(sorted_indices)
        visited = [False] * num_reqs

        for src in range(num_reqs):
            if visited[src]:
                continue
            if sorted_indices[src] == src:
                visited[src] = True
                continue

            cycle = [src]
            visited[src] = True
            dst = sorted_indices[src]
            while dst != src:
                cycle.append(dst)
                visited[dst] = True
                dst = sorted_indices[dst]

            for k in range(len(cycle) - 1):
                a_pos, b_pos = cycle[k], cycle[k + 1]
                a_slot, b_slot = slot_indices[a_pos], slot_indices[b_pos]
                self._swap_state_slots(a_slot, b_slot)
                slot_indices[a_pos], slot_indices[b_pos] = b_slot, a_slot
                req_a = scheduler_output.scheduled_req_ids[a_pos]
                req_b = scheduler_output.scheduled_req_ids[b_pos]
                scheduler_output.request_data[req_a].slot_index = b_slot
                scheduler_output.request_data[req_b].slot_index = a_slot

    def warmup(self) -> None:
        """Basic warmup: run a small-batch forward pass"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Warming up model...")
        with torch.inference_mode():
            tokens = torch.tensor([1, 2, 3], dtype=torch.long, device=self.device)
            state = self.model.zero_state(1)
            query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device=self.device)
            _ = self.model.forward(tokens, state, query_start_loc, 3)
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

        state = [
            self._decode_state_shift[:, :, :B],
            self._decode_state_wkv[:, :B],
            self._decode_state_elapsed[:B],
        ]

        query_start_loc = torch.tensor(
            [0] + list(itertools.accumulate(seq_lens)),
            dtype=torch.int32,
            device=self.device,
        )
        req_id = torch.tensor(
            [i for i, sl in enumerate(seq_lens) for _ in range(sl)],
            dtype=torch.int32,
            device=self.device,
        )
        path = model.path_selector.select(B, max_t, total_tokens)

        def forward_fn(x, state, path, query_start_loc, req_id, max_t, B):
            return model.forward_from_x(x, state, path, query_start_loc, req_id, max_t, B)

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
