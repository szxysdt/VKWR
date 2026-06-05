import pytest


@pytest.fixture(autouse=True)
def cuda_device_guard():
    """Override global CUDA guard — entrypoints tests are pure Python logic."""
    yield
