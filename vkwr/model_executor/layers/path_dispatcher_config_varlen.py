from dataclasses import dataclass

from vkwr.config.model import RWKV7InferenceConfig, WeightConfig


@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


@dataclass(frozen=True)
class CmixConfig:
    CMIX_DENSE: str = "dense"
    CMIX_B1T1_SPARSE: str = "b1t1_sparse"
    CMIX_ROWS2_SPARSE: str = "rows2_sparse"
    CMIX_B1T1_NOFC: str = "b1t1_nofc"
    CMIX_ROWS2_NOFC: str = "rows2_nofc"


@dataclass(frozen=True)
class CmixThresholds:
    nofc_max_rows: int = 19
    nofc_row20_max_t: int = 5
    nofc_t512_min_rows: int = 8


@dataclass(frozen=True)
class LowrankConfig:
    IN_ROWS_T: int = 7
    OUT_ROWS_T: int = 4
    FUSED_MIN_C: int = 1024

    def can_use_lowrank_fused(self, rows: int, C: int) -> bool:
        return C >= self.FUSED_MIN_C and rows <= self.IN_ROWS_T

    def can_use_lowrank_out_fused(self, rows: int, C: int) -> bool:
        return C >= self.FUSED_MIN_C and rows <= self.OUT_ROWS_T


class PathSelector:
    """Select PathConfig for varlen batch.

    Differs from the v1 PathSelector:
    - select() takes (B, max_t) instead of (B, T)
    - cmix_mode is always CMIX_DENSE (varlen kernel handles boundaries)
    - use_batched_rkv uses B and max_t for decision
    """

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

    def select(self, B: int, max_t: int, total_tokens: int = -1) -> PathConfig:
        if total_tokens < 0:
            total_tokens = B * max_t

        # Varlen always uses dense CMIX — sparse/nofc paths rely on [B,T,C] layout
        cmix_mode = self.cmix_config.CMIX_DENSE

        # use_batched_rkv: decide based on B and max_t
        if self.inference_config.rkv_mode == "auto":
            rows_equiv = B * max_t
            use_batched_rkv = (rows_equiv == 1) or (4 <= rows_equiv <= 64)
        elif self.inference_config.rkv_mode == "on":
            use_batched_rkv = True
        else:
            use_batched_rkv = False

        if self.weight_config.use_orig_linear("att_c2c"):
            use_batched_rkv = False

        return PathConfig(rows=total_tokens, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)


lorank_cfg = LowrankConfig()
