"""Inspired by vLLM and Albatross.

CUDA Graph capture and replay for RWKV7 inference.

Manages varlen-aware CUDA Graph entries keyed by tuple[int] (seq_lens).
Supports sparse capture with padding: actual batch sizes are mapped to the
next captured size, and dummy positions are padded during replay.
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
    - Sparse capture with padding (vLLM-style)
    - No sampling in graph
    - Embedding inside/outside graph depends on emb_cpu
    """

    def __init__(
        self,
        capture_shapes: list[tuple[int]] | None,
        device: torch.device,
        emb_cpu: bool = False,
        max_batch_size: int = 64,
    ):
        self.capture_shapes = capture_shapes or [(1,) * b for b in range(1, 65)]
        self.device = device
        self.emb_cpu = emb_cpu
        self._max_batch_size = max_batch_size
        # shape_key (tuple[int]) -> (graph, output_logits)
        self._entries: dict[tuple[int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._stream: torch.cuda.Stream | None = None
        self._capturing = False

        self._capture_sizes = sorted(set(len(s) for s in self.capture_shapes))
        self._bs_to_padded: dict[int, int] = self._compute_bs_to_padded()

    def _compute_bs_to_padded(self) -> dict[int, int]:
        """For each batch size 1..max_batch_size, find the smallest capture size >= bs."""
        mapping: dict[int, int] = {}
        for bs in range(1, self._max_batch_size + 1):
            padded = None
            for cs in self._capture_sizes:
                if cs >= bs:
                    padded = cs
                    break
            if padded is not None:
                mapping[bs] = padded
        return mapping

    def get_graph(self, seq_lens: tuple[int]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor] | None:
        """Get a captured CUDA Graph by exact seq_lens match.

        Returns (graph, output_logits) or None (fallback to eager).
        """
        return self._entries.get(seq_lens)

    def get_graph_with_padding(self, B: int) -> tuple[torch.cuda.CUDAGraph | None, torch.Tensor | None, int]:
        """Get CUDA Graph for actual batch size B, with optional padding.

        Returns (graph, output_logits, padded_B) or (None, None, B) for eager fallback.
        padded_B == B means no padding.
        """
        padded_B = self._bs_to_padded.get(B)
        if padded_B is None or padded_B > self._max_batch_size:
            return None, None, B

        seq_lens = tuple([1] * padded_B)
        entry = self._entries.get(seq_lens)
        if entry is None:
            return None, None, B
        return entry[0], entry[1], padded_B

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
