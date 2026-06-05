from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from vkwr.engine.outputs import ModelRunnerOutput
from vkwr.model_executor.layers.sampler import RWKV7Sampler
from vkwr.model_executor.models.rwkv7 import RWKV7

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.scheduler.output import SchedulerOutput

logger = logging.getLogger(__name__)


class GPUModelRunner:
    """GPU Model Runner - key component connecting the scheduler and model layers"""

    def __init__(self, config: VkwrConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model_config = config.model_config
        self.scheduler_config = config.scheduler_config
        self.dtype = getattr(torch, config.model_config.dtype)

        self.max_num_seqs = config.scheduler_config.max_num_seqs
        self.max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens

        self.model: RWKV7 | None = None
        self.sampler: RWKV7Sampler | None = None

        # State buffers (allocated in load_model)
        self._state_shift: torch.Tensor | None = None
        self._state_wkv: torch.Tensor | None = None
        self._state_elapsed: torch.Tensor | None = None

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

        # Allocate reusable state buffers (aligned with Albatross zero_state format)
        # state = [shift: (L,2,B,C), wkv: (L,B,H,N,N), elapsed: (B)]
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
        """
        Execute model forward pass.
        Phase 1 simplified flow:
        1. Prepare inputs (token IDs)
        2. Prepare state
        3. Execute forward
        4. Sample
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        input_ids = self._prepare_input_ids(scheduler_output)
        state = self._prepare_state(scheduler_output)

        with torch.inference_mode():
            logits = self.model.forward(input_ids, state)

        model_output = ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=logits,
        )
        model_output = self.sample_tokens(model_output, scheduler_output)
        return model_output

    def sample_tokens(
        self,
        model_runner_output: ModelRunnerOutput,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """
        Sample from logits, return sampling results mapped by req_id.
        """
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

        return ModelRunnerOutput(
            sampled_token_ids=sampled_token_ids,
            sampled_logprobs=logprobs,
            logits=None,
        )

    def _prepare_input_ids(self, scheduler_output: SchedulerOutput) -> torch.Tensor:
        """Build a [B, T] input_ids tensor"""
        input_ids_list = []
        for req_id in scheduler_output.scheduled_req_ids:
            run_data = scheduler_output.request_data[req_id]
            input_ids_list.append(run_data.input_token_ids)

        if len(input_ids_list) > 1:
            seq_len = len(input_ids_list[0])
            for i, ids in enumerate(input_ids_list[1:], 1):
                if len(ids) != seq_len:
                    raise ValueError(f"Request {i} has seq_len={len(ids)} != batch seq_len={seq_len}. Prefill and decode requests cannot be batched together.")

        return torch.tensor(input_ids_list, dtype=torch.long, device=self.device)

    def _prepare_state(self, scheduler_output: SchedulerOutput) -> list[torch.Tensor]:
        """Slice state buffers for the currently scheduled requests (views, not cloned)"""
        if self._state_shift is None or self._state_wkv is None or self._state_elapsed is None:
            raise RuntimeError("State buffers not allocated. Call load_model() first.")

        bsz = len(scheduler_output.scheduled_req_ids)
        shift = self._state_shift[:, :, :bsz]
        wkv = self._state_wkv[:, :bsz]
        elapsed = self._state_elapsed[:bsz]
        return [shift, wkv, elapsed]

    def warmup(self) -> None:
        """Basic warmup: run a small-batch forward pass"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Warming up model...")
        with torch.inference_mode():
            tokens = torch.tensor([[1, 2, 3]], dtype=torch.long, device=self.device)
            state = self.model.zero_state(1)
            _ = self.model.forward(tokens, state)
        logger.info("Warmup complete.")

    def determine_available_memory(self) -> int:
        """Profile available GPU memory"""
        gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory
        available_mem = int(gpu_mem_total * self.config.worker_config.gpu_memory_utilization)
        return available_mem
