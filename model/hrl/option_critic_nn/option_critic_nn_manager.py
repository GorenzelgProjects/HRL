from pathlib import Path
from typing import Callable

import gymnasium as gym
from loguru import logger as logging

from model.base_manager import BaseModelManager
from model.hrl.option_critic_nn.train_agent_nn import train_option_critic_nn
from model.hrl.option_critic.plot_termination_probs import plot_termination_probabilities

class OptionCriticNNManager(BaseModelManager):
    def __init__(
        self,
        n_states: int,
        n_options: int,
        n_actions: int,
        n_steps: int,
        n_episodes: int,
        n_filters: list[int],
        conv_sizes: list[int],
        strides: list[int],
        n_neurons: int,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        beta_reg: float,
        entropy_reg: float,
        optimizer_name: str,
        gamma: float,
        lr: float,
        temperature: float,
        img_size: list[int, int],
        save_frequency: int,
        verbose: bool,
        quiet: bool,
        max_history: int,
        cuda: bool,
        save_dir: str,
        partial_env: Callable[[int], gym.Env],
    ) -> None:

        self.n_states = n_states
        self.n_options = n_options
        self.n_actions = n_actions

        self.n_steps = n_steps
        self.n_episodes = n_episodes
        
        self.n_filters = n_filters
        self.conv_sizes = conv_sizes
        self.strides = strides
        self.n_neurons = n_neurons
        
        self.optimizer_name = optimizer_name

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.beta_reg = beta_reg
        self.entropy_reg = entropy_reg
        self.temperature = temperature
        self.img_size = img_size

        self.save_frequency = save_frequency
        self.verbose = verbose and not quiet
        
        self.max_history = max_history
        self.cuda = cuda

        super().__init__(partial_env, model_name="option_critic", save_dir=save_dir)

    def _train(
        self, levels: list[int], render: bool = False, delay: float = 0.05
    ) -> None:

        for level_num in levels:
            logging.info(f"\n{'='*60}")
            logging.info(f"Level {level_num}")
            logging.info(f"{'='*60}")
            level_env = self.partial_env(level_num)

            train_option_critic_nn(
                env=level_env,
                img_size=self.img_size,
                level=level_num,
                num_episodes=self.n_episodes,
                n_options=self.n_options,
                n_states=self.n_states,
                n_actions=self.n_actions,
                n_filters=self.n_filters,
                conv_sizes=self.conv_sizes,
                strides=self.strides,
                n_neurons=self.n_neurons,
                optimizer_name=self.optimizer_name,
                gamma=self.gamma,
                lr=self.lr,
                epsilon=self.epsilon,
                epsilon_decay = self.epsilon_decay,
                epsilon_min = self.epsilon_min,
                beta_reg=self.beta_reg,
                entropy_reg=self.entropy_reg,
                n_steps=self.n_steps,
                temperature=self.temperature,
                save_frequency=self.save_frequency,
                output_dir=self.save_dir,
                verbose=self.verbose,
                render=render,
                delay=delay,
                max_history=self.max_history,
                cuda=self.cuda
            )

        return

    def _test(self, levels: list[int], render: bool, delay: float):
        logging.warning("Gorenzelg will implement this test")
        # episode = self.n_episodes
        # for level in levels:
        #     level_env = self.partial_env(level)
            
        #     # Plot termination probability of last episode of every level
        #     saved_agent_file = Path(self.save_dir) / "agents" / f"agent_episode_{episode}_level_{level}.json"
        #     plot_termination_probabilities(
        #         level_env,
        #         saved_agent_file,
        #         self.state_mapping_dir,
        #         level,
        #         episode,
        #         Path(self.save_dir) / "agent_term_probs"
        #     )