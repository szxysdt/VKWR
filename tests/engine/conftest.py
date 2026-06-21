from __future__ import annotations

import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_device_guard():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    yield
    gc.collect()
    torch.cuda.empty_cache()
