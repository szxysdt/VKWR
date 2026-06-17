import hashlib
import json
from dataclasses import fields, is_dataclass
from typing import TypeVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

ConfigT = TypeVar("ConfigT")


def config(cls=None, *, config_dict=None, **kwargs):
    """Decorator to create a pydantic dataclass with default config.
    Aligned with vLLM vllm/config/utils.py L52-80. Extra fields are forbidden by default."""
    merged_config = ConfigDict(extra="forbid")
    if config_dict is not None:
        merged_config.update(config_dict)

    def decorator(cls):
        return pydantic_dataclass(cls, config=merged_config, **kwargs)

    if cls is None:
        return decorator
    return decorator(cls)


def replace(config_obj, /, **kwargs):
    """Like dataclasses.replace, but compatible with Pydantic dataclasses.
    Aligned with vLLM vllm/config/utils.py L119-127."""
    cls = type(config_obj)
    result_dict = config_obj.__dict__.copy()
    result_dict.update(kwargs)
    return cls(**result_dict)


def update_config(config_obj, overrides):
    """Recursively apply overrides to a config dataclass.
    Aligned with vLLM vllm/config/utils.py L210-227."""
    processed = {}
    for field_name, value in overrides.items():
        current_value = getattr(config_obj, field_name)
        if is_dataclass(current_value) and not is_dataclass(value):
            if not isinstance(value, dict):
                raise TypeError(f"Overrides to {type(config_obj)}.{field_name} must be a dict or {type(current_value)}, but got {type(value)}")
            value = update_config(current_value, value)
        processed[field_name] = value
    return replace(config_obj, **processed)


def normalize_value(x):
    """Return a stable, JSON-serializable canonical form for hashing.
    Reference: vLLM vllm/config/utils.py L230-321, simplified to VKWR's subset."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if is_dataclass(x):
        type_fqn = f"{x.__class__.__module__}.{x.__class__.__qualname__}"
        items = tuple((f.name, normalize_value(getattr(x, f.name))) for f in sorted(fields(x), key=lambda f: f.name))
        return (type_fqn, items)
    if isinstance(x, dict):
        return tuple(sorted((str(k), normalize_value(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(normalize_value(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted(str(normalize_value(v)) for v in x))
    raise TypeError(f"normalize_value: unsupported type '{type(x).__name__}'")


def get_hash_factors(config_obj, ignored_factors=None):
    """Gets factors used for hashing a config dataclass.
    Aligned with vLLM vllm/config/utils.py L324-342."""
    ignored = ignored_factors or set()
    result = {}
    for dc_field in fields(config_obj):
        if dc_field.name in ignored:
            continue
        value = getattr(config_obj, dc_field.name, None)
        result[dc_field.name] = normalize_value(value)
    return result


def get_default_cudagraph_capture_sizes(max_num_seqs: int, max_num_batched_tokens: int) -> list[int]:
    """Generate CUDA Graph capture sizes.
    Follows vLLM _set_cudagraph_sizes() strategy to dynamically compute capture sizes
    based on max_num_seqs and max_num_batched_tokens."""
    max_size = min(max_num_seqs, 512)
    max_size = min(max_size, max_num_batched_tokens)
    if max_size < 1:
        return []

    sizes: list[int] = [i for i in (1, 2, 4) if i <= max_size]
    if max_size >= 8:
        sizes.extend(range(8, min(max_size + 1, 256), 8))
    if max_size >= 256:
        sizes.extend(range(256, max_size + 1, 16))

    if max_num_batched_tokens <= max_size and max_num_batched_tokens not in sizes:
        sizes.append(max_num_batched_tokens)

    return sorted(set(sizes))


def hash_factors(items):
    """Return a SHA-256 hex digest of the canonical items structure.
    Aligned with vLLM vllm/config/utils.py L345-347."""
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()
