from abc import ABC, abstractmethod

class BaseModelManager(ABC):
    """Base interface for all model managers"""

    @abstractmethod
    def train(self, levels: list[int], env, render, delay):
        pass

    @abstractmethod
    def test(self, levels: list[int], env, render, delay):
        pass