from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StateBufferManager:
    """Manage RWKV7 state buffer allocation and slicing.

    The RWKV7 model maintains three types of state:
      - shift: [L, 2, B, C] — shift register for time/channel mix
      - wkv: [L, B, H, N, N] — WKV state matrix
      - elapsed: [B] — global time step counter (advance_i32)

    Buffers are pre-allocated to max_num_seqs size and sliced
    per inference step by the actual batch size. Slices are views
    (not cloned), so in-place updates from the forward pass persist.
    """

    def __init__(
        self,
        L: int,
        C: int,
        H: int,
        N: int,
        max_bsz: int,
        dtype: torch.dtype,
        wkv_dtype: torch.dtype,
        device: torch.device,
    ):
        self.L = L
        self.C = C
        self.H = H
        self.N = N
        self.max_bsz = max_bsz
        self.dtype = dtype
        self.wkv_dtype = wkv_dtype
        self.device = device

        self._shift: torch.Tensor | None = None
        self._wkv: torch.Tensor | None = None
        self._elapsed: torch.Tensor | None = None

    def allocate(self) -> None:
        """Allocate state buffers"""
        self._shift = torch.zeros(
            (self.L, 2, self.max_bsz, self.C),
            dtype=self.dtype,
            device=self.device,
        )
        self._wkv = torch.zeros(
            (self.L, self.max_bsz, self.H, self.N, self.N),
            dtype=self.wkv_dtype,
            device=self.device,
        )
        self._elapsed = torch.zeros(
            (self.max_bsz,),
            dtype=torch.int32,
            device=self.device,
        )
        logger.info(
            "State buffers allocated: L=%d C=%d H=%d N=%d max_bsz=%d",
            self.L,
            self.C,
            self.H,
            self.N,
            self.max_bsz,
        )

    def slice(self, bsz: int) -> list[torch.Tensor]:
        """Slice by batch size, returning [shift, wkv, elapsed] views.

        Note: Returns slices (views) of the original buffers, NOT clones!
        If cloned, the forward pass in-place state modifications would be lost,
        causing each decode step to produce identical output.
        """
        if self._shift is None or self._wkv is None or self._elapsed is None:
            raise RuntimeError("State buffers not allocated. Call allocate() first.")

        if bsz > self.max_bsz:
            raise ValueError(f"batch_size={bsz} exceeds max_bsz={self.max_bsz}. Increase max_num_seqs in SchedulerConfig.")

        shift = self._shift[:, :, :bsz]
        wkv = self._wkv[:, :bsz]
        elapsed = self._elapsed[:bsz]
        return [shift, wkv, elapsed]

    def reset(self, bsz: int | None = None) -> None:
        """Reset state buffers to zero.

        Args:
            bsz: Batch size to reset. None resets the entire buffer.
        """
        if self._shift is None:
            return

        if bsz is not None:
            self._shift[:, :, :bsz].zero_()
            self._wkv[:, :bsz].zero_()
            self._elapsed[:bsz].zero_()
        else:
            self._shift.zero_()
            self._wkv.zero_()
            self._elapsed.zero_()

    @property
    def is_allocated(self) -> bool:
        return self._shift is not None
