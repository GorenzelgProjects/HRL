import numpy as np
from abc import ABC, abstractmethod

class BaseIntrinsic(ABC):
    def __init__(self, weight_start = 5, weight_decay_param = 200_000):
        self.weight_start = weight_start
        self.weight = weight_start
        self.weight_decay_param = weight_decay_param
        self.time = 0
    
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass
    
    def update_weight(self):
        """
        lambda_t = lambda_0 * exp(-t / tau)
        """
        self.time += 1
        self.weight = self.weight_start * np.exp(-self.time / self.weight_decay_param)
        
        