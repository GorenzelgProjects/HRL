import numpy as np
from model.hrl.option_critic.intrinsic_rewards.base_intrinsic import BaseIntrinsic

class IntrinsicComposer:
    def __init__(self, intrinsic_list: list[BaseIntrinsic], weight_start = 5, weight_decay_param = 200_000):
        self.weight_start = weight_start
        self.weight_decay_param = weight_decay_param
        
        self.intrinsic_list = intrinsic_list
        
    def compute(self, new_state_idx, action):
        """
        Computes the sum of the intrinsic rewards
        """
        intrinsic_reward = 0
        for intrinsic_rewarder in self.intrinsic_list:
            intrinsic_reward += intrinsic_rewarder.compute(new_state_idx, action)
        
        return intrinsic_reward

    def weight(self, time):
        """
        lambda_t = lambda_0 * exp(-t / tau)
        """
        return self.weight_start * np.exp(-time / self.weight_decay_param)
        
    def reset(self):
        for intrinsic_rewarder in self.intrinsic_list:
            intrinsic_rewarder.reset()
        