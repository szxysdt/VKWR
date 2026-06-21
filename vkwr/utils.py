from __future__ import annotations

import uuid

MASK_64_BITS = (1 << 64) - 1


def _random_uuid() -> str:
    """Generate a random 16-char hex string (lower 64 bits of uuid4)."""
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"


def generate_request_id(prefix: str) -> str:
    """Generate a unique request ID: `{prefix}-{random_uuid[:8]}`."""
    return f"{prefix}-{_random_uuid()[:8]}"
