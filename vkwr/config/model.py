from dataclasses import dataclass

import torch

from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class ModelConfig:
    """vLLM-style model configuration for engine layer.

    Contains model path, tokenizer, dtype, and loading parameters.
    Used by EngineArgs -> VkwrConfig pipeline.
    """

    model: str
    tokenizer: str | None = None
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    dtype: str = "float16"
    load_format: str = "auto"
    seed: int = 42
    max_model_len: int | None = None

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)


@config
class RWKV7Config:
    """RWKV7 architecture dimensions (L, C, H, N, V)."""

    L: int  # number of layers
    C: int  # hidden dim (= H * N)
    H: int  # number of heads
    N: int  # head size (HEAD_SIZE, fixed 64)
    V: int  # vocab size

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)


LOWRANK_WEIGHT_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")
ATT_C2C_WEIGHT_SUFFIXES = ("receptance.weight", "key.weight", "value.weight", "output.weight")
ORIG_LINEAR_GROUPS_DEFAULT = frozenset({"att_c2c", "ffn_key", "head"})


def parse_orig_linear_groups(text: str) -> frozenset[str]:
    """Parse a comma-separated CLI string into a frozenset of group names."""
    groups = frozenset(x.strip() for x in text.replace(",", " ").split() if x.strip())
    if not groups or groups == frozenset({"none"}):
        return frozenset()
    unknown = groups - ORIG_LINEAR_GROUPS_DEFAULT
    if unknown:
        raise ValueError(f"unknown orig linear groups: {sorted(unknown)}")
    return groups


@dataclass(frozen=True)
class WeightConfig:
    # lowrank weight
    LOWRANK_WEIGHT_SUFFIXES: tuple[str, ...] = LOWRANK_WEIGHT_SUFFIXES
    ATT_C2C_WEIGHT_SUFFIXES: tuple[str, ...] = ATT_C2C_WEIGHT_SUFFIXES
    ORIG_LINEAR_GROUPS: frozenset[str] = ORIG_LINEAR_GROUPS_DEFAULT

    @classmethod
    def from_cli_string(cls, text: str) -> "WeightConfig":
        """Create a WeightsConfig from a CLI --orig-linear-groups string."""
        groups = parse_orig_linear_groups(text)
        if groups:
            return cls(ORIG_LINEAR_GROUPS=groups)
        return cls()  # uses default

    def is_lowrank_weight(self, key: str) -> bool:
        return key.endswith(self.LOWRANK_WEIGHT_SUFFIXES)

    def is_att_c2c_weight(self, key: str) -> bool:
        return ".att." in key and key.endswith(self.ATT_C2C_WEIGHT_SUFFIXES)

    def use_orig_linear(self, group: str) -> bool:
        return group in self.ORIG_LINEAR_GROUPS

    def is_orig_linear_weight(self, key: str) -> bool:
        return (
            (self.use_orig_linear("att_c2c") and self.is_att_c2c_weight(key))
            or (self.use_orig_linear("ffn_key") and ".ffn.key.weight" in key)
            or (self.use_orig_linear("head") and key == "head.weight")
        )


@dataclass(frozen=True)
class RWKV7InferenceConfig:
    """RWKV7 runtime inference configuration.

    Controls execution paths, kernel selection, and memory layout choices.
    Must be set before model construction; immutable thereafter.
    """

    dtype: torch.dtype = torch.float16
    wkv_mode: str = "fp16"  # "fp16" | "fp32io16"
    emb_device: str = "gpu"  # "cpu" | "gpu"
    rkv_mode: str = "off"  # "auto" | "on" | "off"
    cmix_sparse: str = "no-fc"  # "auto" | "no-fc" | "off"
    lowrank_weight: str = "both"  # "orig" | "transpose" | "both"
    ln1_tmix_fuse: bool = True  # fused LN1+TMIX for b1t1 path

    def __post_init__(self):
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"invalid dtype: {self.dtype}")
        if self.wkv_mode not in ("fp16", "fp32io16"):
            raise ValueError(f"invalid wkv_mode: {self.wkv_mode}")
        if self.emb_device not in ("cpu", "gpu"):
            raise ValueError(f"invalid emb_device: {self.emb_device}")
        if self.rkv_mode not in ("auto", "on", "off"):
            raise ValueError(f"invalid rkv_mode: {self.rkv_mode}")
        if self.cmix_sparse not in ("auto", "no-fc", "off"):
            raise ValueError(f"invalid cmix_sparse: {self.cmix_sparse}")
        if self.lowrank_weight not in ("orig", "transpose", "both"):
            raise ValueError(f"invalid lowrank_weight: {self.lowrank_weight}")
