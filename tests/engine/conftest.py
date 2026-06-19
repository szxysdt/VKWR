from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def cuda_device_guard():
    yield
