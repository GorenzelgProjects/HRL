from typing import Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from gymnasium import Env
    
    
class BaseModelManager(ABC):
    """Base interface for all model managers"""
    def __init__(self, partial_env: Callable[[int], "Env"]) -> None:
        """
        Args:
            partial_env (Env): Partial environment that only needs level to be instantiated.
        """
        self.partial_env = partial_env

    @abstractmethod
    def train(self, levels: list[int], render: bool, delay: float) -> None:
        """Function to train model and save plans/policies/weights etc.

        Args:
            levels (list[int]): List of levels to train on.
            render (bool): Whether to render during training or not.
            delay (float): How much delay to use during rendering.
        """
        pass

    @abstractmethod
    def test(self, levels: list[int], render: bool, delay: float) -> None:
        pass