from typing import Callable

import gymnasium as gym

from model.base_manager import BaseModelManager
from model.baseline.astar.a_star_agent import train_a_star


class AStarManager(BaseModelManager):
    def __init__(self, heuristic: str, partial_env: Callable[[int],gym.Env]):
        self.heuristic = heuristic
        self.partial_env = partial_env

    def train(self, levels: list[int], render: bool = False, delay: float = 0.05):
        for level in levels:
            lvl_env = self.partial_env(level)
            success, plan = train_a_star(lvl_env, self.heuristic, render, delay)

    def test(self, levels: list[int], env, render, delay):
        pass