from model.hrl.option_critic.intrinsic_rewards.base_intrinsic import BaseIntrinsic


class RepeatedStateIntrinsic(BaseIntrinsic):
    def __init__(self, penalty: float = -0.1) -> None:
        self.prev_state_idx = None
        self.prev_prev_state_idx = None
        self.penalty = penalty

    def compute(self, new_state_idx: int, action) -> float:
        reward = 0.0

        # Apply penalty if the new state matches the state from two steps ago
        if self.prev_prev_state_idx == new_state_idx:
            reward = self.penalty

        # Update state history
        self.prev_prev_state_idx = self.prev_state_idx
        self.prev_state_idx = new_state_idx

        return reward

    def reset(self):
        self.prev_state_idx = None
        self.prev_prev_state_idx = None