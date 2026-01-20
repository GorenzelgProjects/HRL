from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from loguru import logger as logging

from model.base_manager import BaseModelManager
from model.hrl.option_critic.train_agent import train_agent
from model.hrl.option_critic.plot_termination_probs import plot_termination_probabilities

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
        
        # Simple-SF parameters (will be set from config)
        self.use_simple_sf = False
        self.sf_d = 256
        self.lambda_sf = 0.1
        self.alpha_w = 0.1
        self.sf_lr_main = 1e-3
        self.sf_lr_w = 1e-2
        
        # Transfer learning parameters
        self.sf_transfer_from_level = None  # Level to load shared components from
        self.sf_freeze_encoder = True  # Freeze encoder when using transfer
        
        # Rollout collection parameters
        self.collect_rollouts = False
        self.rollout_save_dir = None
        
        # Transfer learning: shared SF components across levels (internal state)
        self.shared_sf_module = None
        self.level_w_tasks = {}  # Dict: level -> w_task parameter

        super().__init__(partial_env, model_name="option_critic", save_dir=save_dir)

    def _train(
        self, levels: list[int], render: bool = False, delay: float = 0.05
    ) -> None:
        
        for level_idx, level_num in enumerate(levels):
            logging.info(f"\n{'='*60}")
            logging.info(f"Level {level_num}")
            logging.info(f"{'='*60}")
            level_env = self.partial_env(level_num)
            
            # Setup transfer learning components
            shared_sf_module = None
            shared_w_task = None
            freeze_encoder = False
            transfer_from_level = None
            
            # Check if transfer learning is explicitly enabled for this level
            if self.use_simple_sf and self.sf_transfer_from_level is not None:
                transfer_from_level = self.sf_transfer_from_level
                freeze_encoder = self.sf_freeze_encoder
                logging.info(f"Transfer Learning: Loading SF components from level {transfer_from_level}")
                logging.info(f"Transfer Learning: {'Freezing' if freeze_encoder else 'Fine-tuning'} encoder, training w_task[{level_num}] only")
            elif self.use_simple_sf and self.shared_sf_module is not None:
                # Automatic transfer: reuse from previous level in same run
                shared_sf_module = self.shared_sf_module
                if level_num in self.level_w_tasks:
                    shared_w_task = self.level_w_tasks[level_num]
                else:
                    device = next(self.shared_sf_module.encoder.parameters()).device
                    shared_w_task = nn.Parameter(torch.zeros(self.sf_d, device=device))
                    self.level_w_tasks[level_num] = shared_w_task
                freeze_encoder = True
                logging.info(f"Transfer Learning: Reusing encoder + SF heads from previous level, training w_task[{level_num}] only")

            agent, results, sf_components = train_agent(
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
                # Simple-SF parameters (from config if available)
                use_simple_sf=getattr(self, 'use_simple_sf', False),
                sf_d=getattr(self, 'sf_d', 256),
                lambda_sf=getattr(self, 'lambda_sf', 0.1),
                alpha_w=getattr(self, 'alpha_w', 0.1),
                sf_lr_main=getattr(self, 'sf_lr_main', 1e-3),
                sf_lr_w=getattr(self, 'sf_lr_w', 1e-2),
                # Transfer learning parameters
                shared_sf_module=shared_sf_module,
                shared_w_task=shared_w_task,
                freeze_encoder=freeze_encoder,
                transfer_from_level=transfer_from_level,
                # Rollout collection parameters
                collect_rollouts=getattr(self, 'collect_rollouts', False),
                rollout_save_dir=getattr(self, 'rollout_save_dir', None),
            )
            
            # Store shared components after first level for automatic transfer learning
            # (when training multiple levels in one run without explicit sf_transfer_from_level)
            if self.use_simple_sf and level_idx == 0 and sf_components is not None:
                self.shared_sf_module = sf_components['sf_module']
                # Store w_task for first level
                self.level_w_tasks[level_num] = sf_components['w_task']
                logging.info(f"Stored shared SF module for transfer learning to next levels")

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
        
                
        
                
            
        