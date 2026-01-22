import numpy as np
from collections import defaultdict

from model.hrl.option_critic.intrinsic_rewards.base_intrinsic import BaseIntrinsic

class NoveltyIntrinsic(BaseIntrinsic):
    # Randomly helps a lot or doesn't help at all
    #  Depending on the seed when running lvl 22 thin ice...
    #  Makes the agent stuck and vanishes gradients in lvl 15...
    def __init__(self) -> None:
        self.n_visits = defaultdict(lambda: 0)
        
    def compute(self, state_idx: int, action) -> float:
        if state_idx in self.n_visits:
            reward = -np.sqrt(self.n_visits[state_idx])
        else:
            reward = 1
        self.n_visits[state_idx] += 1
        return reward
        # reward = 1/np.sqrt(self.n_visits[state_idx])
        # return reward
    
    def reset(self):
        self.n_visits = defaultdict(lambda: 0)
