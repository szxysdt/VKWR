"""Inspired by vLLM."""

from __future__ import annotations

import enum
from typing import Any

import msgspec

from vkwr.engine.request import SamplingParams


class EngineCoreRequestType(enum.Enum):
    """Request types as hex bytes, sent directly over ZMQ without encoding."""

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    EXECUTOR_FAILED = b"\x04"
    WAKEUP = b"\x05"


class UtilityResult(msgspec.Struct, gc=False):
    """Wrapper for utility method return value."""

    result: Any


class UtilityOutput(msgspec.Struct, array_like=True, gc=False):
    """Response from a utility method call."""

    call_id: int
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str = ""
    prompt_token_ids: list[int] | None = None
    sampling_params: SamplingParams | None = None
    arrival_time: float = 0.0
    priority: int = 0
    current_wave: int = 0
