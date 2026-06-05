import pytest


@pytest.fixture(autouse=True)
def cuda_device_guard():
    """Override root conftest - executor tests use mocking, no CUDA needed."""
    yield
