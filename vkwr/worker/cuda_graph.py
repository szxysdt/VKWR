"""Inspired by vLLM and Albatross.

CUDA Graph capture and replay for RWKV7 inference.

Manages varlen-aware CUDA Graph entries keyed by tuple[int] (seq_lens).
FULL mode only, no padding, no sampling-in-graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CUDAGraphManager:
    """Manage CUDA Graph capture and replay.

    Each graph is identified by a varlen-aware key (tuple[int] of seq_lens).
    First encounter captures, subsequent calls replay.

    - FULL mode only
    - No padding (exact match, key = tuple(seq_lens))
    - No sampling in graph
    - Embedding inside/outside graph depends on emb_cpu
    """

    def __init__(
        self,
        capture_shapes: list[tuple[int]] | None,
        device: torch.device,
        emb_cpu: bool = False,
    ):
        self.capture_shapes = capture_shapes or [(1,) * b for b in range(1, 65)]
        self.device = device
        self.emb_cpu = emb_cpu
        # shape_key (tuple[int]) -> (graph, output_logits)
        self._entries: dict[tuple[int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._stream: torch.cuda.Stream | None = None
        self._capturing = False

    def get_graph(self, seq_lens: tuple[int]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor] | None:
        """Get a captured CUDA Graph.

        Returns (graph, output_logits) or None (fallback to eager).
        """
        return self._entries.get(seq_lens)

    def _capture_for_shape(
        self,
        seq_lens: tuple[int],
        embed_fn,
        forward_fn,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path,
        query_start_loc: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
        tokens: torch.Tensor | None = None,
    ):
        """Pre-capture CUDA Graph for a varlen shape.

        Capture scope depends on emb_cpu:
        - emb_cpu=False (GPU emb): graph = embed(tokens) + forward_from_x(x, ...)
        - emb_cpu=True  (CPU emb): graph = forward_from_x(x, ...) only
        """
        B = len(seq_lens)
        total_tokens = sum(seq_lens)
        if self._stream is None:
            self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())

        if self.emb_cpu:
            with torch.cuda.stream(self._stream):
                forward_fn(x, state, path, query_start_loc, req_id, max_t, B)

            torch.cuda.current_stream().wait_stream(self._stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._stream):
                output_logits = forward_fn(x, state, path, query_start_loc, req_id, max_t, B)

        else:
            with torch.cuda.stream(self._stream):
                warmup_x = embed_fn(tokens[:total_tokens])
                forward_fn(warmup_x, state, path, query_start_loc, req_id, max_t, B)

            torch.cuda.current_stream().wait_stream(self._stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self._stream):
                captured_x = embed_fn(tokens[:total_tokens])
                output_logits = forward_fn(captured_x, state, path, query_start_loc, req_id, max_t, B)

        self._entries[seq_lens] = (graph, output_logits)
        logger.debug("Captured CUDA Graph for shape B=%d", B)

    def replay(self, seq_lens: tuple[int]) -> torch.Tensor:
        """Replay CUDA Graph. Returns output logits tensor."""
        graph, output_logits = self._entries[seq_lens]
        graph.replay()
        return output_logits
