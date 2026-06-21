"""Override the CUDA autouse fixture for worker tests that use mocking/CPU."""

import pytest


@pytest.fixture(autouse=True)
def cuda_device_guard():
    """Override root conftest - these tests use CPU/mocked devices."""
    yield
