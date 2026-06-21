"""Test for WorkerBase abstract class."""

import pytest

from vkwr.worker.worker_base import WorkerBase


class TestWorkerBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            WorkerBase(None)  # type: ignore

    def test_abstract_methods_exist(self):
        abstract = WorkerBase.__abstractmethods__
        expected = {
            "init_device",
            "load_model",
            "execute_model",
            "sample_tokens",
            "determine_available_memory",
            "compile_or_warm_up_model",
        }
        assert abstract == expected
