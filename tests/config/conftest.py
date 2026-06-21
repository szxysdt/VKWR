import pytest


@pytest.fixture(autouse=True)
def cuda_device_guard():
    """Override global CUDA guard - config tests are pure Python."""
    yield
