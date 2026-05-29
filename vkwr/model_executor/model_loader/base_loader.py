from abc import ABC, abstractmethod
from typing import Any

import torch

from vkwr.model_executor.model_loader.default_loader import DefaultModelLoader


class BaseModelLoader(ABC):
    @abstractmethod
    def load_model(self, model_path: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]: ...


def get_model_loader(load_format: str) -> BaseModelLoader:
    loaders = {"auto": DefaultModelLoader, "raw": DefaultModelLoader}
    return loaders[load_format]()
