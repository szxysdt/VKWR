from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str


@dataclass(frozen=True)
class CmixConfig:
    """Channel mixing path mode labels — read-only constants from reference implementation."""

    CMIX_DENSE: str = "dense"
    CMIX_B1T1_SPARSE: str = "b1t1_sparse"
    CMIX_ROWS2_SPARSE: str = "rows2_sparse"
    CMIX_B1T1_NOFC: str = "b1t1_nofc"
    CMIX_ROWS2_NOFC: str = "rows2_nofc"
