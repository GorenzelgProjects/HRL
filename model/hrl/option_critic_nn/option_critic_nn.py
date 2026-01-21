# Inspired by https://github.com/alversafa/option-critic-arch/blob/master/utils.py
# and https://github.com/theophilegervet/options-hierarchical-rl/blob/master/implementations/option_critic.ipynb

import copy

import time
import torch
import torch.distributions as td
import numpy as np
from loguru import logger as logging

from environment.thin_ice.thin_ice_env import ThinIceEnv
from model.hrl.option_critic_nn.oc_network import Encoder
from model.hrl.option_critic_nn.replay_buffer import ReplayBuffer

from collections import defaultdict


class OptionCritic:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_options: int,
        encoder: Encoder,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        beta_reg: float = 0.01,
        entropy_reg: float = 0.01,
        n_steps: int = 1000,
        downsample: list[int, int] = [84, 84],
        device: torch.device = torch.device("cpu")
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
        
        self.encoder_prime = copy.deepcopy(self.encoder) # For more stable Q-values
        self.encoder_prime.to(device)
        
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon  # Exploration/Exploitation param
        self.beta_reg = beta_reg # Termination regularization parameter
        self.entropy_reg = entropy_reg # Entropy regularization parameter

        self.n_steps = n_steps
        self.downsample_size = downsample
        
        self.device = device

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
            option_id = torch.argmax(Q_Omega).item()

        return option_id
    
    def choose_action(self, state: np.ndarray | torch.Tensor, option_idx: int) -> int:
        q_u = self.encoder.intra_options(state)[option_idx]
        action_distribution = td.Categorical(probs = q_u)
        
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        
        return action.item(), log_prob, entropy

    def train(self, 
              env: ThinIceEnv, 
              optimizer: torch.optim,
              render: bool = False, 
              delay: float = 0.02,
              max_history: int = 10000,
              batch_size: int = 32,
              update_freq: int = 4,
              freeze_interval: int = 200
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
        buffer = ReplayBuffer(max_history)
        
        # Get initial state
        state, info = env.reset()
        level = info["level"]
        
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
            action_sequence[str(option)][option_switches].append(action)
            flat_action_sequence.append(action)  # Also save in flat format for replay
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, option, reward, new_state, terminated)
            total_reward += reward
            
            if len(buffer) > batch_size:
                # Options improvement
                actor_loss = self.actor_loss(state,
                                            reward,
                                            new_state,
                                            option,
                                            log_prob,
                                            entropy,
                                            terminated)
                loss = actor_loss
                
                if step % update_freq == 0:
                    data_batch = buffer.sample(batch_size)
                    critic_loss = self.critic_loss(data_batch)
                    loss += critic_loss                    
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % freeze_interval == 0:
                    self.encoder_prime.load_state_dict(self.encoder.state_dict())
            
            # Pick new option if the previous terminates
            termination_prob = self.encoder.beta(new_state)[:, option].detach()
            if torch.rand(1).item() < termination_prob:
                option = self.choose_new_option(new_state)
                option_sequence.append(option)
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
        state: np.ndarray | torch.Tensor,
        reward: float,
        new_state: np.ndarray | torch.Tensor,
        option: int,
        log_prob: float,
        entropy: float,
        terminated: bool
    ) -> None:
        """Implements the actor loss (akin to option improvement)"""
        s = self.encoder.encode_state(state)
        new_s = self.encoder.encode_state(new_state)
        new_s_prime = self.encoder_prime.encode_state(new_state)
        
        beta_t = self.encoder.beta(s)[:, option]
        beta_t_plus_1 = self.encoder.beta(new_s)[:, option].detach()
        
        Q_Omega_t = self.encoder.pi_options(s).detach().squeeze()
        Q_Omega_t_plus_1 = self.encoder_prime.pi_options(new_s_prime).detach().squeeze()
        
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
        data_batch
    ) -> None:
        """Implements the critic loss from algorithm 1
        """
        states, options, rewards, new_states, terminates = data_batch
        
        batch_idx = torch.arange(len(options)).long()
        options   = torch.LongTensor(options).to(self.device)
        rewards   = torch.FloatTensor(rewards).to(self.device)
        masks     = 1 - torch.FloatTensor(terminates).to(self.device)
        
        s = self.encoder.encode_state(states).squeeze(0)
        Q_Omega_t = self.encoder.pi_options(s)
        
        new_s_prime = self.encoder_prime.encode_state(new_states).squeeze(0)
        Q_Omega_t_plus_1 = self.encoder_prime.pi_options(new_s_prime) # Using the copied network for stability
        
        new_s = self.encoder.encode_state(new_states).squeeze(0)
        beta_t_plus_1 = self.encoder.beta(new_s).detach()[batch_idx, options]
        
        # Estimate Q_U with g_t
        g_t = rewards + masks * self.gamma * \
            ((1 - beta_t_plus_1) * Q_Omega_t_plus_1[batch_idx, options] + \
                beta_t_plus_1 * Q_Omega_t_plus_1.max(dim=-1)[0])
        
        # Get the TD error: Q_Omega - Q_U
        td_err = torch.mean((Q_Omega_t[batch_idx, options] - g_t.detach())**2)
        
        return td_err