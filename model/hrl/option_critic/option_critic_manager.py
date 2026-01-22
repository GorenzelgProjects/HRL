from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, Union

import hydra
import gymnasium as gym
from loguru import logger as logging

from model.base_manager import BaseModelManager
from model.hrl.option_critic.train_agent import train_agent
from model.hrl.option_critic.plot_termination_probs import plot_termination_probabilities

if TYPE_CHECKING:
    from omegaconf import DictConfig


class OptionCriticManager(BaseModelManager):
    def __init__(
        self,
        n_states: int,
        n_options: int,
        n_actions: int,
        n_steps: int,
        n_episodes: int,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        gamma: float,
        alpha_critic: float,
        alpha_theta: float,
        alpha_upsilon: float,
        temperature: float,
        save_frequency: int,
        verbose: bool,
        quiet: bool,
        save_dir: str,
        state_mapping_dir: str,
        partial_env: Callable[[int], gym.Env],
        intrinsic_composer_cfg: Optional[Union["DictConfig",dict]] = None,
    ) -> None:

        self.n_states = n_states
        self.n_options = n_options
        self.n_actions = n_actions

        self.n_steps = n_steps
        self.n_episodes = n_episodes

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha_critic = alpha_critic
        self.alpha_theta = alpha_theta
        self.alpha_upsilon = alpha_upsilon
        self.temperature = temperature

        self.save_frequency = save_frequency
        self.state_mapping_dir = state_mapping_dir
        self.verbose = verbose and not quiet
        
        self.intrinsic_composer = hydra.utils.instantiate(intrinsic_composer_cfg)

        super().__init__(partial_env, model_name="option_critic", save_dir=save_dir)

    def _train(
        self, levels: list[int], render: bool = False, delay: float = 0.05
    ) -> None:

        for level_num in levels:
            logging.info(f"\n{'='*60}")
            logging.info(f"Level {level_num}")
            logging.info(f"{'='*60}")
            level_env = self.partial_env(level_num)

            train_agent(
                env=level_env,
                level=level_num,
                num_episodes=self.n_episodes,
                n_options=self.n_options,
                n_states=self.n_states,
                n_actions=self.n_actions,
                gamma=self.gamma,
                alpha_critic=self.alpha_critic,
                alpha_theta=self.alpha_theta,
                alpha_upsilon=self.alpha_upsilon,
                epsilon=self.epsilon,
                epsilon_decay = self.epsilon_decay,
                epsilon_min = self.epsilon_min,
                n_steps=self.n_steps,
                temperature=self.temperature,
                save_frequency=self.save_frequency,
                output_dir=self.save_dir,
                state_mapping_dir=self.state_mapping_dir,
                verbose=self.verbose,
                render=render,
                delay=delay,
                intrinsic_composer = self.intrinsic_composer
            )

        return

    def _test(self, levels: list[int], render: bool, delay: float):
        episode = self.n_episodes
        for level in levels:
            level_env = self.partial_env(level)
            
            # Plot termination probability of last episode of every level
            saved_agent_file = Path(self.save_dir) / "agents" / f"agent_episode_{episode}_level_{level}.json"
            plot_termination_probabilities(
                level_env,
                saved_agent_file,
                self.state_mapping_dir,
                level,
                episode,
                Path(self.save_dir) / "agent_term_probs"
            )
        
                
        
                
            
        