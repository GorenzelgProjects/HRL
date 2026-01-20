# Inspired by https://github.com/alversafa/option-critic-arch/blob/master/utils.py
# and https://github.com/theophilegervet/options-hierarchical-rl/blob/master/implementations/option_critic.ipynb

import copy

import time
import torch
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from loguru import logger as logging

from environment.thin_ice.thin_ice_env import ThinIceEnv
from model.hrl.option_critic_nn.oc_network import Encoder

from typing import Optional
from collections import defaultdict


class OptionCritic:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_options: int,
        encoder: Encoder,
        gamma: float = 0.99,
        alpha_critic: float = 0.5,
        alpha_theta: float = 0.25,
        alpha_upsilon: float = 0.25,
        epsilon: float = 0.1,
        n_steps: int = 1000,
    ):
        """The class for the OptionCritic architecture

        Args:
            n_states (int): The number of unique states in the environment
            n_actions (int): The number of unique actions in the environment
            n_options (int): The number of options to create (hyperparam)
            gamma (float, optional): The discount factor. Defaults to 0.99.
            alpha_critic (float, optional): The learning rate for the option-critic. Defaults to 0.5.
            alpha_theta (float, optional): The learning rate for the intra-policy. Defaults to 0.25.
            alpha_upsilon (float, optional): The learning rate for the termination function. Defaults to 0.25.
            epsilon (float, optional): Defines the probability of taking a random action. Defaults to 0.9.
            n_steps (int, optional): The maximum number of steps allowed in an episode. Defaults to 1000.
        """

        if n_states <= 0 or n_actions <= 0:
            raise ValueError(
                f"Number of states and actions must be positive. \
                Current choice of n_states: {n_states}, and n_action: {n_actions}"
            )
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.encoder = encoder

        if n_options <= 0:
            raise ValueError("A positive number of options must be set")
        self.n_options = n_options
        
        self.oc_prime = copy.deepcopy(self) # For more stable Q-values
        
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon  # Exploration/Exploitation param

        self.n_steps = n_steps

    def choose_new_option(self, state: np.ndarray | torch.Tensor) -> int:
        """Chooses a new option according to an epsilon-greedy strategy

        Args:
            state (np.ndarray | torch.Tensor): Either the state image or the encoded image

        Returns:
            int: The index of the option to be followed
        """
        Q_Omega = self.encoder.pi_options(state).detach()
        
        if torch.rand(1).item() < self.epsilon:
            # Explore
            option_id = torch.randint(0, self.n_options, (1,)).item()
        else:
            # Exploit
            option_id = torch.argmax(Q_Omega)

        return option_id
    
    def choose_action(self, state: np.ndarray | torch.Tensor, option_idx: int) -> int:
        q_u = self.encoder.intra_options(state)[option_idx]
        action_distribution = td.Categorical(probs = q_u)
        
        action = action_distribution.sample().item()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        
        return action, log_prob, entropy

    def train(self, 
              env: ThinIceEnv, 
              optimizer: torch.optim,
              render: bool = False, delay: float = 0.02
    ) -> tuple[list, dict[list]]:
        """Train the Optic-Critic for a maximum of n_steps

        Args:
            env (ThinIceEnv): The ThinIceEnv
            temperature (float): The temperature for logit calculation in π_theta
            save_mapping (bool, optional): Whether to save the state mapping after training. Defaults to True.

        Returns:
            dict: A dictionary containing the summary of the training of an episode. Including:
                - "level": the level it was trained on,
                - "option_sequence": a list of all options,
                - "action_sequence": A nested dictionary where the first key is the option index, and the second key is the
                number in the option sequence e.g {"0": {"3": [1, 2]}} mean option one had an action sequence at the third option switch that
                executed acion 1 and 2,
                - "total_reward": The total reward,
                - "num_options_used": The number of options used,
                - "total_steps": Total amount of steps used prior to termination,
                - "num_options_switches": The number of times the agent switched option strategy,
                - "terminated": Whether the episode terminated with a win,
                - "truncated": Whether an error happened, terminating the game preemptively,
        """
        # Get initial state
        obs, info = env.reset()
        level = info["level"]

        state = self.encoder(obs)

        # Store option and action sequence
        option_sequence = []
        action_sequence = defaultdict(lambda: defaultdict(list))
        flat_action_sequence = []  # Flat list for easier replay
        
        # Pick an initial option
        option = self.choose_new_option(state)
        option_sequence.append(option)

        total_reward = 0.0

        # Continuous updates
        terminated = False
        truncated = False
        step = 0
        option_switches = 0
        while not (terminated or truncated):
            if step >= self.n_steps:
                logging.info(
                    f"Number of steps exceeds maximum number of steps: {self.n_steps}.\nTerminating..."
                )
                break

            action, log_prob, entropy = self.choose_action(state, option)
            action_sequence[str(option.idx)][option_switches].append(action)
            flat_action_sequence.append(action)  # Also save in flat format for replay
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Options evaluation
            critic_loss = self.critic_loss(
                state, reward, new_state, option, action, terminated
                )
            
            # Options improvement
            actor_loss = self.actor_loss(state,
                                         reward,
                                         new_state,
                                         option,
                                         log_prob,
                                         entropy,
                                         terminated)

            loss = critic_loss + actor_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Pick new option if the previous terminates
            termination_prob = self.encoder.beta(new_state).detach()
            if torch.rand(1).item() < termination_prob:
                option = self.choose_new_option(new_state)
                option_sequence.append(option.idx)
                option_switches += 1

            state = new_state
            step += 1
            
            if render:  
                env.render()
                time.sleep(delay)

        episode_stats = {
            'level': level,
            'option_sequence': option_sequence,
            'action_sequence': action_sequence,  # Keep for backward compatibility
            'flat_action_sequence': flat_action_sequence,  # Add flat sequence for easier replay
            'total_reward': total_reward,
            'num_options_used': len(set(option_sequence)),
            'total_steps': step,
            'num_options_switches': len(option_sequence) - 1,
            'terminated': terminated,
            'truncated': truncated,
        }

        return episode_stats

    def actor_loss(
        self,
        state: int,
        reward: float,
        new_state: int,
        option: int,
        log_prob: float,
        entropy: float,
        terminated: bool
    ) -> None:
        """Implements the option evaluation from Algorithm 1: Option-critic with tabular intra-option Q-learning

        Args:
            state_idx (int): The index of the state
            reward (float): The reward for going from state to new_state
            new_state_idx (int): The index of the new state
            option (Option): The current selected option
            action (int): The action that was just taken
            terminated (bool): Whether the episode has terminated
            temperature (float, optional): The logit sensitivity
        """
        beta_t = self.encoder.beta(state)[option]
        beta_t_plus_1 = self.encoder.beta(state)[option].detach()
        
        Q_Omega_t = self.encoder.pi_options(state).detach()
        Q_Omega_t_plus_1 = self.oc_prime.encoder.pi_options(new_state)[option].detach()
        
        # Target update (no backprop)
        g_t = reward + (1 - terminated) * self.gamma * \
            ((1 - beta_t_plus_1) * Q_Omega_t_plus_1[option] + \
                beta_t_plus_1 * Q_Omega_t_plus_1.max(dim=-1)[0])
        
        # Termination loss
        beta_loss = (1 - terminated) * beta_t * \
            (Q_Omega_t[option].detach() - Q_Omega_t.max(dim=-1)[0].detach() + self.beta_reg)
        
        # Actor-critic policy gradient
        pi_loss = -log_prob * (g_t.detach() - Q_Omega_t[option]) - \
            self.entropy_reg * entropy
            
        actor_loss = beta_loss + pi_loss
        
        return actor_loss

    def critic_loss(
        self,
        state: np.ndarray | torch.Tensor,
        reward: float,
        new_state: np.ndarray | torch.Tensor,
        option: int,
        terminated: bool
    ) -> None:
        """Implements the options improvement from Algorithm 1: Option-critic with tabular intra-option Q-learning

        Args:
            state_idx (int): The index of the state
            new_state_idx (int): The index of the new state
            option (Option): The currently selected option
            action (int): The action just taken
            temperature (float): Hyperparameter for logit calculation sensitivity
        """
        Q_Omega_t = self.encoder.pi_options(state)
        Q_Omega_t_plus_1 = self.oc_prime.encoder.pi_options(new_state) # Using the copied network for stability
        
        beta_t_plus_1 = self.encoder.beta(new_state)[option]
        
        # Estimate Q_U with g_t
        g_t = reward + (1 - terminated) * self.gamma * \
            ((1 - beta_t_plus_1) * Q_Omega_t_plus_1[option] + \
                beta_t_plus_1 * Q_Omega_t_plus_1.max(dim=-1)[0])
        
        # Get the TD error: Q_Omega - Q_U
        td_err = (Q_Omega_t - g_t)**2
        
        return td_err