import numpy as np
from abc import ABC, abstractmethod

class BaseIntrinsic(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass
