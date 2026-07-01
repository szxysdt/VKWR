import pytest
import torch

from vkwr.config.compilation import CompilationConfig
from vkwr.config.engine import VkwrConfig
from vkwr.config.model import (
    ModelConfig,
    RWKV7Config,
    RWKV7InferenceConfig,
    WeightConfig,
    parse_orig_linear_groups,
)
from vkwr.config.parallel import ParallelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.config.utils import (
    config,
    get_default_cudagraph_capture_sizes,
    get_hash_factors,
    hash_factors,
    normalize_value,
    replace,
    update_config,
)
from vkwr.config.vkwr import EngineArgs
from vkwr.config.worker import GPUWorkerConfig

# ── Config decorator ──────────────────────────────────────────────────


class TestConfigDecorator:
    def test_config_decorator_basic(self):
        @config
        class MyConfig:
            name: str
            value: int = 10

        c = MyConfig(name="test")
        assert c.name == "test"
        assert c.value == 10

    def test_config_decorator_forbids_extra(self):
        @config
        class MyConfig:
            name: str

        with pytest.raises(Exception):
            MyConfig(name="test", extra_field="bad")


# ── replace ───────────────────────────────────────────────────────────


class TestReplace:
    def test_replace_single_field(self):
        c = ModelConfig(model="m1", seed=42)
        c2 = replace(c, seed=99)
        assert c2.model == "m1"
        assert c2.seed == 99
        assert c.seed == 42

    def test_replace_multiple_fields(self):
        c = ModelConfig(model="m1", seed=42)
        c2 = replace(c, model="m2", seed=77)
        assert c2.model == "m2"
        assert c2.seed == 77


# ── update_config ─────────────────────────────────────────────────────


class TestUpdateConfig:
    def test_update_config_simple(self):
        c = ModelConfig(model="m1", seed=42)
        c2 = update_config(c, {"seed": 100})
        assert c2.seed == 100
        assert c2.model == "m1"

    def test_update_config_nested_dict(self):
        vc = VkwrConfig(model_config=ModelConfig(model="m1"))
        vc2 = update_config(vc, {"model_config": {"seed": 99}})
        assert vc2.model_config.seed == 99

    def test_update_config_invalid_type(self):
        vc = VkwrConfig(model_config=ModelConfig(model="m1"))
        with pytest.raises(TypeError):
            update_config(vc, {"model_config": "bad"})


# ── normalize_value ───────────────────────────────────────────────────


class TestNormalizeValue:
    def test_none(self):
        assert normalize_value(None) is None

    def test_primitives(self):
        assert normalize_value(42) == 42
        assert normalize_value(3.14) == 3.14
        assert normalize_value("hello") == "hello"
        assert normalize_value(True) is True

    def test_list(self):
        assert normalize_value([1, 2, 3]) == (1, 2, 3)

    def test_dict(self):
        result = normalize_value({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))

    def test_set(self):
        result = normalize_value({1, 2, 3})
        assert result == ("1", "2", "3")

    def test_dataclass(self):
        c = GPUWorkerConfig()
        nv = normalize_value(c)
        assert isinstance(nv, tuple)
        assert len(nv) == 2

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            normalize_value(object())


# ── hash_factors ──────────────────────────────────────────────────────


class TestHashFactors:
    def test_get_hash_factors(self):
        c = GPUWorkerConfig()
        factors = get_hash_factors(c)
        assert "device" in factors
        assert factors["device"] == "cuda"

    def test_get_hash_factors_with_ignored(self):
        c = GPUWorkerConfig()
        factors = get_hash_factors(c, ignored_factors={"device"})
        assert "device" not in factors

    def test_hash_deterministic(self):
        c1 = GPUWorkerConfig()
        c2 = GPUWorkerConfig()
        h1 = hash_factors(get_hash_factors(c1))
        h2 = hash_factors(get_hash_factors(c2))
        assert h1 == h2

    def test_hash_differs(self):
        c1 = GPUWorkerConfig()
        c2 = GPUWorkerConfig(gpu_memory_utilization=0.5)
        h1 = hash_factors(get_hash_factors(c1))
        h2 = hash_factors(get_hash_factors(c2))
        assert h1 != h2


# ── get_default_cudagraph_capture_sizes ──────────────────────────────


class TestGetDefaultCudagraphCaptureSizes:
    def test_small_max_seqs(self):
        sizes = get_default_cudagraph_capture_sizes(4, 8)
        # max_size = min(4, 512, 8) = 4
        # sparse: [1, 2, 4], filtered <=4: [1, 2, 4]
        assert sizes == [1, 2, 4]

    def test_medium_max_seqs(self):
        sizes = get_default_cudagraph_capture_sizes(16, 128)
        # max_size = min(16, 512, 128) = 16
        # sparse: [1, 2, 4] + range(8, 17, 8) = [1, 2, 4, 8, 16]
        assert sizes == [1, 2, 4, 8, 16]

    def test_large_max_seqs(self):
        sizes = get_default_cudagraph_capture_sizes(128, 2048)
        # max_size = min(128, 512, 2048) = 128
        # sparse: [1, 2, 4] + range(8, 129, 8) = [1, 2, 4, 8, ..., 128]
        assert sizes == [1, 2, 4] + list(range(8, 129, 8))

    def test_96_max_seqs(self):
        sizes = get_default_cudagraph_capture_sizes(96, 2048)
        # max_size = 96
        # [1, 2, 4] + range(8, 97, 8) = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
        assert sizes == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
        assert len(sizes) == 15

    def test_zero_returns_empty(self):
        sizes = get_default_cudagraph_capture_sizes(0, 0)
        assert sizes == []

    def test_includes_max_num_batched_tokens(self):
        sizes = get_default_cudagraph_capture_sizes(100, 100)
        # max_size = 100, max_size must be in the list
        assert 100 in sizes

    def test_256_max_seqs(self):
        sizes = get_default_cudagraph_capture_sizes(256, 4096)
        # max_size = 256
        # [1, 2, 4] + range(8, 256, 8) + [256]
        assert sizes == [1, 2, 4] + list(range(8, 256, 8)) + [256]


# ── ModelConfig ───────────────────────────────────────────────────────


class TestModelConfig:
    def test_defaults(self):
        c = ModelConfig(model="test")
        assert c.tokenizer is None
        assert c.trust_remote_code is False
        assert c.dtype == "float16"
        assert c.load_format == "auto"
        assert c.seed == 42
        assert c.max_model_len is None

    def test_custom(self):
        c = ModelConfig(model="m", dtype="float16", max_model_len=4096, seed=1)
        assert c.max_model_len == 4096
        assert c.seed == 1

    def test_compute_hash(self):
        c1 = ModelConfig(model="m")
        c2 = ModelConfig(model="m")
        assert c1.compute_hash() == c2.compute_hash()

    def test_compute_hash_differs(self):
        c1 = ModelConfig(model="m1")
        c2 = ModelConfig(model="m2")
        assert c1.compute_hash() != c2.compute_hash()


# ── SchedulerConfig ───────────────────────────────────────────────────


class TestSchedulerConfig:
    def test_defaults(self):
        c = SchedulerConfig()
        assert c.max_num_batched_tokens == 2048
        assert c.max_num_seqs == 64
        assert c.chunked_prefill_threshold == 512
        assert c.enable_chunked_prefill is True
        assert c.eos_token_id == 0
        assert c.ignore_eos is False
        assert c.default_max_tokens is None

    def test_warning_when_batched_smaller_than_model_len(self):
        """Creating SchedulerConfig with max_num_batched_tokens < max_model_len
        should not raise -- just logs a warning."""
        c = SchedulerConfig(max_model_len=16384, max_num_batched_tokens=2048)
        assert c.max_num_batched_tokens == 2048

    def test_compute_hash(self):
        c1 = SchedulerConfig()
        c2 = SchedulerConfig()
        assert c1.compute_hash() == c2.compute_hash()


# ── GPUWorkerConfig ───────────────────────────────────────────────────


class TestGPUWorkerConfig:
    def test_defaults(self):
        c = GPUWorkerConfig()
        assert c.device == "cuda"
        assert c.gpu_memory_utilization == 0.92
        assert c.enforce_eager is False
        assert c.max_context_len_from_generator is None

    def test_compute_hash(self):
        c1 = GPUWorkerConfig()
        c2 = GPUWorkerConfig()
        assert c1.compute_hash() == c2.compute_hash()


# ── ParallelConfig ────────────────────────────────────────────────────


class TestParallelConfig:
    def test_defaults(self):
        c = ParallelConfig()
        assert c.dp_size == 1
        assert c.tp_size == 1
        assert c.pp_size == 1
        assert c.distributed_executor_backend is None

    def test_tp_rejected(self):
        with pytest.raises(ValueError, match="TP"):
            ParallelConfig(tp_size=2)

    def test_pp_rejected(self):
        with pytest.raises(ValueError, match="PP"):
            ParallelConfig(pp_size=2)

    def test_compute_hash(self):
        c1 = ParallelConfig()
        c2 = ParallelConfig()
        assert c1.compute_hash() == c2.compute_hash()


# ── CompilationConfig ─────────────────────────────────────────────────


class TestCompilationConfig:
    def test_defaults(self):
        c = CompilationConfig()
        assert c.cudagraph_capture_size is None
        assert c.cudagraph_mode == "full"
        assert c.torch_compile_mode is None

    def test_custom_sizes(self):
        c = CompilationConfig(cudagraph_capture_size=[1, 2, 4, 8])
        assert c.cudagraph_capture_size == [1, 2, 4, 8]

    def test_compute_hash(self):
        c1 = CompilationConfig()
        c2 = CompilationConfig()
        assert c1.compute_hash() == c2.compute_hash()


# ── VkwrConfig ────────────────────────────────────────────────────────


class TestVkwrConfig:
    def test_requires_float16(self):
        with pytest.raises(ValueError, match="float16"):
            VkwrConfig(
                model_config=ModelConfig(model="m", dtype="float32"),
            )

    def test_batched_tokens_must_be_ge_seqs(self):
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            VkwrConfig(
                model_config=ModelConfig(model="m"),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=256,
                    max_num_batched_tokens=128,
                ),
            )

    def test_basic_construction(self):
        vc = VkwrConfig(model_config=ModelConfig(model="m"))
        assert vc.model_config.model == "m"

    def test_auto_cudagraph_sizes(self):
        vc = VkwrConfig(
            model_config=ModelConfig(model="m"),
            scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=128),
        )
        assert vc.compilation_config.cudagraph_capture_size is not None
        assert len(vc.compilation_config.cudagraph_capture_size) > 0

    def test_no_auto_cudagraph_when_eager(self):
        vc = VkwrConfig(
            model_config=ModelConfig(model="m"),
            worker_config=GPUWorkerConfig(enforce_eager=True),
        )
        assert vc.compilation_config.cudagraph_capture_size is None

    def test_no_auto_cudagraph_when_mode_none(self):
        vc = VkwrConfig(
            model_config=ModelConfig(model="m"),
            compilation_config=CompilationConfig(cudagraph_mode="none"),
        )
        assert vc.compilation_config.cudagraph_capture_size is None

    def test_compute_hash(self):
        vc1 = VkwrConfig(model_config=ModelConfig(model="m"))
        vc2 = VkwrConfig(model_config=ModelConfig(model="m"))
        assert vc1.compute_hash() == vc2.compute_hash()

    def test_compute_hash_length(self):
        vc = VkwrConfig(model_config=ModelConfig(model="m"))
        h = vc.compute_hash()
        assert len(h) == 10


# ── EngineArgs ────────────────────────────────────────────────────────


class TestEngineArgs:
    def test_defaults(self):
        ea = EngineArgs(model="m")
        assert ea.tokenizer is None
        assert ea.dtype == "float16"
        assert ea.max_num_seqs == 64
        assert ea.gpu_memory_utilization == 0.92

    def test_create_engine_config(self):
        ea = EngineArgs(model="test-model", max_num_seqs=16, max_num_batched_tokens=128)
        vc = ea.create_engine_config()
        assert vc.model_config.model == "test-model"
        assert vc.scheduler_config.max_num_seqs == 16
        assert vc.scheduler_config.max_num_batched_tokens == 128

    def test_create_engine_config_with_max_model_len(self):
        ea = EngineArgs(model="m", max_model_len=4096)
        vc = ea.create_engine_config()
        assert vc.model_config.max_model_len == 4096

    def test_create_engine_config_with_default_max_tokens(self):
        ea = EngineArgs(model="m", default_max_tokens=512)
        vc = ea.create_engine_config()
        assert vc.scheduler_config.default_max_tokens == 512

    def test_create_engine_config_enforce_eager(self):
        ea = EngineArgs(model="m", enforce_eager=True)
        vc = ea.create_engine_config()
        assert vc.worker_config.enforce_eager is True
        assert vc.compilation_config.cudagraph_capture_size is None


# ── RWKV7Config ───────────────────────────────────────────────────────


class TestRWKV7Config:
    def test_construction(self):
        c = RWKV7Config(L=12, C=4096, H=64, N=64, V=65536)
        assert c.L == 12
        assert c.C == 4096
        assert c.H == 64
        assert c.N == 64
        assert c.V == 65536

    def test_compute_hash(self):
        c1 = RWKV7Config(L=12, C=4096, H=64, N=64, V=65536)
        c2 = RWKV7Config(L=12, C=4096, H=64, N=64, V=65536)
        assert c1.compute_hash() == c2.compute_hash()


# ── WeightConfig ──────────────────────────────────────────────────────


class TestWeightConfig:
    def test_default(self):
        wc = WeightConfig()
        assert "att_c2c" in wc.ORIG_LINEAR_GROUPS
        assert "ffn_key" in wc.ORIG_LINEAR_GROUPS
        assert "head" in wc.ORIG_LINEAR_GROUPS

    def test_is_lowrank_weight(self):
        wc = WeightConfig()
        assert wc.is_lowrank_weight("blocks.0.att.w1")
        assert not wc.is_lowrank_weight("blocks.0.att.key.weight")

    def test_is_att_c2c_weight(self):
        wc = WeightConfig()
        assert wc.is_att_c2c_weight("blocks.0.att.receptance.weight")
        assert not wc.is_att_c2c_weight("blocks.0.ffn.key.weight")

    def test_use_orig_linear(self):
        wc = WeightConfig()
        assert wc.use_orig_linear("att_c2c")
        assert not wc.use_orig_linear("unknown")

    def test_is_orig_linear_weight(self):
        wc = WeightConfig()
        assert wc.is_orig_linear_weight("blocks.0.att.receptance.weight")
        assert wc.is_orig_linear_weight("head.weight")
        assert wc.is_orig_linear_weight("blocks.0.att.output.weight")
        assert not wc.is_orig_linear_weight("blocks.0.att.w1")

    def test_from_cli_string(self):
        wc = WeightConfig.from_cli_string("att_c2c,head")
        assert wc.use_orig_linear("att_c2c")
        assert wc.use_orig_linear("head")
        assert not wc.use_orig_linear("ffn_key")

    def test_from_cli_string_none(self):
        """'none' results in empty groups, so from_cli_string returns default WeightConfig."""
        wc = WeightConfig.from_cli_string("none")
        assert wc.use_orig_linear("att_c2c")


# ── parse_orig_linear_groups ─────────────────────────────────────────


class TestParseOrigLinearGroups:
    def test_valid_groups(self):
        g = parse_orig_linear_groups("att_c2c,ffn_key")
        assert "att_c2c" in g
        assert "ffn_key" in g

    def test_empty(self):
        g = parse_orig_linear_groups("")
        assert g == frozenset()

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            parse_orig_linear_groups("bad_group")


# ── RWKV7InferenceConfig ─────────────────────────────────────────────


class TestRWKV7InferenceConfig:
    def test_defaults(self):
        c = RWKV7InferenceConfig()
        assert c.dtype == torch.float16
        assert c.wkv_mode == "fp16"
        assert c.emb_device == "gpu"
        assert c.rkv_mode == "off"
        assert c.cmix_sparse == "no-fc"
        assert c.lowrank_weight == "both"
        assert c.ln1_tmix_fuse is True

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            RWKV7InferenceConfig(dtype=torch.float32)

    def test_invalid_wkv_mode(self):
        with pytest.raises(ValueError, match="wkv_mode"):
            RWKV7InferenceConfig(wkv_mode="bad")

    def test_invalid_emb_device(self):
        with pytest.raises(ValueError, match="emb_device"):
            RWKV7InferenceConfig(emb_device="mps")

    def test_invalid_rkv_mode(self):
        with pytest.raises(ValueError, match="rkv_mode"):
            RWKV7InferenceConfig(rkv_mode="bad")

    def test_invalid_cmix_sparse(self):
        with pytest.raises(ValueError, match="cmix_sparse"):
            RWKV7InferenceConfig(cmix_sparse="bad")

    def test_invalid_lowrank_weight(self):
        with pytest.raises(ValueError, match="lowrank_weight"):
            RWKV7InferenceConfig(lowrank_weight="bad")

    def test_bf16_dtype_accepted(self):
        c = RWKV7InferenceConfig(dtype=torch.bfloat16)
        assert c.dtype == torch.bfloat16
