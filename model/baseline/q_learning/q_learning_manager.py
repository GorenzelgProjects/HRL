import os
from typing import Callable

import gymnasium as gym

from model.base_manager import BaseModelManager
from model.baseline.q_learning.q_learning_agent import train_q_learning
from model.utils.save_results import save_results, save_q_table, save_training_results


class QLearningManager(BaseModelManager):
    def __init__(
        self,
        learning_rate: float,
        discount: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        qlearning_episodes: int,
        partial_env: Callable[[int], gym.Env],
        save_dir: str,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.qlearning_episodes = qlearning_episodes

        super().__init__(partial_env, model_name="q_learning", save_dir=save_dir)

    def _train(
        self, levels: list[int], render: bool = False, delay: float = 0.05
    ) -> None:
        level_results = []
        result_dir = os.path.join(self.save_dir, "level_results")
        results_dir = os.path.join(self.save_dir, "results")  # For training results with action sequences
        
        for level_num in levels:
            print(f"\n{'='*60}")
            print(f"Level {level_num}")
            print(f"{'='*60}")
            level_env = self.partial_env(level_num)
            
            result, q_table = train_q_learning(level_env,
                                              self.qlearning_episodes,
                                              self.learning_rate,
                                              self.discount,
                                              self.epsilon,
                                              self.epsilon_decay,
                                              self.epsilon_min,
                                              render,
                                              delay)
            
            # Save results (episode summaries)
            save_results(result, filename=f"lvl-{level_num}_result.json", results_dir=self.save_dir)
            
            # Save Q-table and policy
            save_q_table(q_table, 
                        filename=f"lvl-{level_num}_policy.json", 
                        results_dir=self.save_dir,
                        num_actions=level_env.action_space.n)
            
            # Save training results with action sequences (option-critic format)
            save_training_results(result,
                                 filename=f"training_results_level_{level_num}.json",
                                 results_dir=results_dir,
                                 level=level_num)
            
    def test(self, levels: list[int], render: bool, delay: float) -> None:
        # TODO: Implement or save the plans from q-tables in train() or something 
        #   as current result saving is not enough.
        pass
