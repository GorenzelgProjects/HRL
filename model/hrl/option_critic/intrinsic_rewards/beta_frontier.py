from collections import defaultdict
from model.hrl.option_critic.intrinsic_rewards.base_intrinsic import BaseIntrinsic

class BetaFrontierIntrinsic(BaseIntrinsic):
    def __init__(self, reward: float, n_actions: float) -> None:
        self.reward = reward
        self.n_actions = n_actions
        self.visited_actions = defaultdict(lambda: set())
        
    def compute(self, state_idx: int, action) -> float:
        self.visited_actions[state_idx].add(action)
        return self.reward if len(self.visited_actions[state_idx]) < self.n_actions else 0.0
    
    def reset(self):
        self.visited_actions = defaultdict(lambda: set())
        self.prev_state_idx = None
