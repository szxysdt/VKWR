from dataclasses import dataclass

from vkwr.config.model import RWKV7InferenceConfig, WeightConfig


@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


@dataclass(frozen=True)
class CmixConfig:
    """Channel mixing path mode labels — read-only constants from reference implementation."""

    CMIX_DENSE: str = "dense"
    # sparse
    CMIX_B1T1_SPARSE: str = "b1t1_sparse"
    CMIX_ROWS2_SPARSE: str = "rows2_sparse"
    # no-fc
    CMIX_B1T1_NOFC: str = "b1t1_nofc"
    CMIX_ROWS2_NOFC: str = "rows2_nofc"


@dataclass(frozen=True)
class CmixThresholds:
    """CMIX no-fc / sparse path thresholds.

    CMIX down-projection ``act @ value_fc`` (F→C) has two implementation paths:

    - **NOFC (SPMV)** – custom CUDA kernel that skips zero activation elements
      via ``__ballot_sync`` + ``atomicAdd``. Complexity ``O(rows × nnz × C)``.
      Per-row overhead (ballot, prefix scan, atomics) is fixed regardless of rows,
      so this path dominates when rows is small and cuBLAS GEMM suffers from low
      occupancy and launch overhead.
    - **DENSE (GEMM)** – ``relu_square`` + cuBLAS. Complexity ``O(rows × F × C)``.
      At large rows Tensor Cores are fully occupied and the per-invocation constant
      is amortised, overtaking SPMV.

    The crossover is around rows≈19–20 (empirically measured). NOFC is also
    unsuitable for activation sparsity on Tensor Core: NVIDIA Sparse Tensor Core
    requires **static 2:4 weight sparsity**, whereas CMIX sparsity is **dynamic,
    unstructured activation sparsity** (60–90% zeros from ReLU, position varies
    per-token).

    Thresholds
    ----------
    nofc_max_rows : int
        rows ≤ 19 → SPMV. Above this GEMM occupancy wins.
    nofc_row20_max_t : int
        rows == 20 AND T ≤ 5 → SPMV. At rows=20 the SPMV grid's z-dimension
        is near its efficient limit; larger T increases shift_state management
        overhead, making GEMM competitive.
    nofc_t512_min_rows : int
        rows ≥ 8 AND C/F aligned to 512 → use the 512-tile SPMV variant with
        256 threads/block for better occupancy on large hidden dimensions.
    """

    nofc_max_rows: int = 19
    nofc_row20_max_t: int = 5
    nofc_t512_min_rows: int = 8


@dataclass(frozen=True)
class LowrankConfig:
    """Lowrank fused path thresholds — controls whether to use
    transposed-weight fused GEMM instead of lowrank decomposition."""

    IN_ROWS_T: int = 7
    OUT_ROWS_T: int = 4
    FUSED_MIN_C: int = 1024

    def can_use_lowrank_fused(self, rows: int, C: int) -> bool:
        return C >= self.FUSED_MIN_C and rows <= self.IN_ROWS_T

    def can_use_lowrank_out_fused(self, rows: int, C: int) -> bool:
        return C >= self.FUSED_MIN_C and rows <= self.OUT_ROWS_T


class PathSelector:
    """Select PathConfig based on B, T and runtime configuration."""

    def __init__(
        self,
        cmix_thresholds: CmixThresholds | None = None,
        weight_config: WeightConfig | None = None,
        inference_config: RWKV7InferenceConfig | None = None,
    ):
        self.cmix_thresholds = cmix_thresholds or CmixThresholds()
        self.cmix_config = CmixConfig()
        self.weight_config = weight_config if weight_config is not None else WeightConfig()
        self.inference_config = inference_config if inference_config is not None else RWKV7InferenceConfig()

    def select(self, B: int, T: int) -> PathConfig:
        rows = B * T

        # --- cmix_mode ---
        if self.inference_config.cmix_sparse == "off":
            cmix_mode = self.cmix_config.CMIX_DENSE
        elif self.inference_config.cmix_sparse == "no-fc":
            th = self.cmix_thresholds
            use_nofc = rows <= th.nofc_max_rows or (rows == 20 and T <= th.nofc_row20_max_t)
            cmix_mode = self.cmix_config.CMIX_B1T1_NOFC if rows == 1 else (self.cmix_config.CMIX_ROWS2_NOFC if use_nofc else self.cmix_config.CMIX_DENSE)
        elif rows == 1:
            cmix_mode = self.cmix_config.CMIX_B1T1_SPARSE
        elif rows == 2:
            cmix_mode = self.cmix_config.CMIX_ROWS2_NOFC
        else:
            cmix_mode = self.cmix_config.CMIX_DENSE

        # --- use_batched_rkv ---
        if self.inference_config.rkv_mode == "auto":
            use_batched_rkv = (rows == 1) or (4 <= rows <= 64)
        elif self.inference_config.rkv_mode == "on":
            use_batched_rkv = True
        else:
            use_batched_rkv = False

        if self.weight_config.use_orig_linear("att_c2c"):
            use_batched_rkv = False

        return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)


lorank_cfg = LowrankConfig()
