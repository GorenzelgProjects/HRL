import os
from typing import Callable

import gymnasium as gym

from model.base_manager import BaseModelManager
from model.baseline.random.random_agent import train_random_agent
from model.utils.save_results import save_results


class RandomManager(BaseModelManager):
    def __init__(
        self, random_episodes: int, partial_env: Callable[[int], gym.Env], save_dir: str
    ) -> None:
        self.random_episodes = random_episodes

        super().__init__(partial_env, model_name="random", save_dir=save_dir)

    def _train(
        self, levels: list[int], render: bool = False, delay: float = 0.05
    ) -> None:
        level_results = []
        result_dir = os.path.join(self.save_dir, "level_results")
        for level_num in levels:
            print(f"\n{'='*60}")
            print(f"Level {level_num}")
            print(f"{'='*60}")
            level_env = self.partial_env(level_num)

            result = train_random_agent(level_env, self.random_episodes, render, delay)

            # level_results.append(result)
            # TODO: Save results in a better way?
            save_results(
                result, filename=f"lvl-{level_num}_result.json", results_dir=result_dir
            )

    def _test(self, levels: list[int], render: bool, delay: float) -> None:
        print(f"Test statistics already saved from training, check {self.save_dir}")
        pass
