import os
from typing import Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

from loguru import logger as logging

if TYPE_CHECKING:
    from gymnasium import Env


class BaseModelManager(ABC):
    """Base interface for all model managers"""

    def __init__(
        self, partial_env: Callable[[int], "Env"], model_name: str, save_dir: str
    ) -> None:
        """
        Args:
            partial_env (Env): Partial environment that only needs level to be instantiated.
        """
        self.partial_env = partial_env
        self.model_name = model_name

        # make the save_dir model specific
        self.save_dir = os.path.join(save_dir, model_name)
        for split in ("train", "test"):
            logging.add(
                os.path.join(self.save_dir, f"{model_name}_{split}.log"),
                level="INFO",
                enqueue=True,
                rotation="100 MB",
                filter=lambda r, m=model_name, s=split: (
                    r["extra"].get("model") == m and r["extra"].get("split") == s
                ),
            )

    def train(self, levels: list[int], render: bool, delay: float) -> None:
        with logging.contextualize(model=self.model_name, split="train"):
            self._train(levels, render, delay)

    @abstractmethod
    def _train(self, levels: list[int], render: bool, delay: float) -> None:
        """Function to train model and save plans/policies/weights etc.

        Args:
            levels (list[int]): List of levels to train on.
            render (bool): Whether to render during training or not.
            delay (float): How much delay to use during rendering.
        """
        pass

    def test(self, levels: list[int], render: bool, delay: float):
        with logging.contextualize(model=self.model_name, split="test"):
            self._test(levels, render, delay)

    @abstractmethod
    def _test(self, levels: list[int], render: bool, delay: float) -> None:
        """Function to use saved plans/policies/weights to get performance metrics
        and visualize solutions.

        Args:
            levels (list[int]): List of levels to train on.
            render (bool): Whether to render during training or not.
            delay (float): How much delay to use during rendering.
        """
        pass
