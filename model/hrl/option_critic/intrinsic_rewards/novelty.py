import numpy as np
from collections import defaultdict

from model.hrl.option_critic.intrinsic_rewards.base_intrinsic import BaseIntrinsic

class NoveltyIntrinsic(BaseIntrinsic):
    def __init__(self, intrinsic_weight_start: float, intrinsic_decay: float) -> None:
        self.n_visits = defaultdict(lambda: 0)
        super().__init__(weight_start=intrinsic_weight_start, weight_decay_param=intrinsic_decay)
        
    def compute(self, state_idx: int) -> float:
        self.n_visits[state_idx] += 1
        return 1/np.sqrt(self.n_visits[state_idx])
    