# Inspired by https://github.com/alversafa/option-critic-arch/blob/master/utils.py
# and https://github.com/theophilegervet/options-hierarchical-rl/blob/master/implementations/option_critic.ipynb

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path

from environment.thin_ice.thin_ice_env import ThinIceEnv

from typing import Optional
from collections import defaultdict
import logging

class Option():
    def __init__(self, idx: int, n_states: int, n_actions: int) -> None:
        """The class containing attributes and methods for an Option

        Args:
            idx (int): What index value to assign an Option to better distinguish them from one another
            n_states (int): The number of states in the environment
            n_actions (int): The number of actions in the environment
        """
        self.idx = idx
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.theta = torch.nn.Parameter(torch.rand(n_states, n_actions))    # Intra policy parameters
        self.upsilon = torch.nn.Parameter(torch.rand(n_states,))            # Termination parameters
        
    def pi(self, state_idx: int, temperature: float = 1.0) -> torch.Tensor:
        """The option policy for a particular state

        Args:
            state_idx (int): The index of the state in the environment
            temperature (float, optional): Hyperparam for the temperature. Defaults to 1.0.

        Returns:
            torch.Tensor: The probabilities of each action given a state (tensor of shape (n_actions,))
        """
        logits = self.theta[state_idx] / temperature
        probs = F.softmax(logits, dim = -1)
        
        return probs
    
    def log_pi(self, state_idx: int, action: int, temperature: float = 1.0) -> torch.Tensor:
        """Computes the log probability of an action given a state

        Args:
            state_idx (int): The index of the state
            action (int): The taken action
            temperature (float, optional): Hyperparam for logit calculation. Defaults to 1.0.

        Returns:
            torch.Tensor: The log probability of an action given a state (scalar tensor)
        """
        logits = self.theta[state_idx] / temperature
        log_probs = F.log_softmax(logits, dim = -1)
        
        return log_probs[action]
    
    def choose_action(self, state_idx: int, temperature: float = 1.0) -> int:
        """Choose an action acoording to a multinomial distribution

        Args:
            state_idx (int): The index of the state
            temperature (float, optional): Hyperparam for logit calculation. Defaults to 1.0.

        Returns:
            int: An action given as an integer (should match the action representation)
        """
        return torch.multinomial(self.pi(state_idx, temperature), 1).item()
    
    def beta(self, state_idx: int) -> float:
        return F.sigmoid(self.upsilon[state_idx]).item()


class OptionCritic():
    def __init__(self, 
                 n_states: int,
                 n_actions: int,
                 n_options: int,
                 gamma: float = 0.99, 
                 alpha_critic: float = 0.5,
                 alpha_theta: float = 0.25,
                 alpha_upsilon: float = 0.25,
                 epsilon: float = 0.9,
                 n_steps: int = 1000):
        """The class for the OptionCritic architecture

        Args:
            n_states (int): The number of unique states in the environment
            n_actions (int): The number of unique actions in the environment
            n_options (int): The number of options to create (hyperparam)
            gamma (float, optional): The discount factor. Defaults to 0.99.
            alpha_critic (float, optional): The learning rate for the option-critic. Defaults to 0.5.
            alpha_theta (float, optional): The learning rate for the intra-policy. Defaults to 0.25.
            alpha_upsilon (float, optional): The learning rate for the termination function. Defaults to 0.25.
            epsilon (float, optional): The exploration/exploitation trade-off paramater. Defaults to 0.9.
            n_steps (int, optional): The maximum number of steps allowed in an episode. Defaults to 1000.
        """
        
        if n_states <= 0 or n_actions <= 0:
            raise ValueError(f"Number of states and actions must be positive. \
                Current choice of n_states: {n_states}, and n_action: {n_actions}")
        self.n_states = n_states
        self.n_actions = n_actions
        
        if n_options <= 0:
            raise ValueError("A positive number of options must be set")
        self.n_options = n_options
        self.options = [Option(idx, n_states, n_actions) for idx in range(n_options)]
        
        self.Q_Omega_table = torch.zeros((n_states, n_options))           # Option values
        self.Q_U_table = torch.zeros((n_states, n_options, n_actions))    # Action values for state-option pair  
        
        self.gamma = gamma                  # Discount factor
        self.alpha_critic = alpha_critic    # Critic lr
        self.alpha_theta = alpha_theta      # Intra-policy lr
        self.alpha_upsilon = alpha_upsilon  # Termination function lr
        
        self.epsilon = epsilon              # Exploration/Exploitation param
        
        self.n_steps = n_steps
        
        # State-to-index mapping for tabular methods
        # Maps state tuples (hashable representation) to unique integer indices
        self.state_to_idx_dict: dict[tuple, int] = {}
        self.n_unique_states = 0  # Counter for unique states encountered
        
        # Directory for saving/loading state mappings (default: hrl_models/option_critic/state_mappings/)
        self.state_mapping_dir = Path(__file__).parent / "state_mappings"
        self.state_mapping_dir.mkdir(exist_ok=True)
        self.current_level: Optional[int] = None
    
    def get_Q_Omega(self, state_idx: int, option_idx: Optional[int] = None) -> torch.Tensor | float:
        """Getter function for retrieving the state-option values

        Args:
            state_idx (int): The index of the state
            option_idx (Optional[int], optional): Which option among the n_options. 
            If not specified, it will return π(*|s). Defaults to None.

        Returns:
            torch.Tensor | float: Either the Q-values for each option given a state or the Q-value for a state-option pair
        """
        if option_idx is None:
            return self.Q_Omega_table[state_idx]
        
        return self.Q_Omega_table[state_idx, option_idx]
    
    def get_Q_U(self, state_idx: int, option_idx: int, action_idx: Optional[int] = None) -> torch.Tensor | float:
        """Getter function for retrieving state-option pairs for a specific action

        Args:
            state_idx (int): The index of the state
            option_idx (int): The index of the option
            action_idx (Optional[int], optional): The action (assumed discrete numbers as representation).
            If not specified, it returns all Q-values for the state-option pair. Defaults to None.

        Returns:
            torch.Tensor | float: Either all Q-values for an action in the context of a state-option pair, 
            or one value for that specific action
        """
        if action_idx is None:
            return self.Q_U_table[state_idx, option_idx]
        
        return self.Q_U_table[state_idx, option_idx, action_idx]
    
    def set_Q_Omega(self, state_idx: int, option: Option) -> None:
        """Updates the Q_Omega table according to its definition
        
        Q_Omega(s, omega) = ∑_a π_(omega, theta)(a|s) * Q_U(s, omega, a)

        Args:
            state_idx (int): The index of the state
            option (Option): The current option
        """
        self.Q_Omega_table[state_idx, option.idx] = torch.sum(option.pi(state_idx) * self.get_Q_U(state_idx, option.idx))
        
        return

    def set_Q_U(self, state_idx: int, option_idx: int, action_idx: int, new_value: float) -> None:
        """Updates the Q_U table according to a new specific value

        Args:
            state_idx (int): The state index
            option_idx (int): The option index
            action_idx (int): The action index (representation should be discrete numbers)
            new_value (float): The new value to set Q_U[s_idx, o_idx, a_idx] to
        """
        self.Q_U_table[state_idx, option_idx, action_idx] = new_value
        
        return
    
    def choose_new_option(self, state_idx: int) -> Option:
        """Chooses a new option according to an epsilon-greedy strategy

        Args:
            state_idx (int): The index of the state representation

        Returns:
            Option: The option to be followed
        """
        
        if torch.rand(1).item() < self.epsilon:
            # Explore
            random_option_id = torch.randint(0, self.n_options, (1,)).item()
            return self.options[random_option_id]
        else:
            # Exploit
            best_option_id = torch.argmax(self.get_Q_Omega(state_idx))
            return self.options[best_option_id]


    def train(self, env: ThinIceEnv, temperature: float, save_mapping: bool = True) -> tuple[list, dict[list]]:
        """Train the Optic-Critic for a maximum of n_steps

        Args:
            env (ThinIceEnv): The ThinIceEnv
            temperature (float): The temperature for logit calculation in π_theta
            save_mapping (bool, optional): Whether to save the state mapping after training. Defaults to True.
        
        Returns:
            tuple[list, dict[list]]: The list of pursued options and a dictionary with each option's action sequence
        """
        # Get level from environment
        level = getattr(env, 'level', None) or getattr(env, 'current_level', None)
        
        # Get initial state
        state, info = env.reset()
        # Try to get level from info if not available from env
        if level is None and 'level' in info:
            level = info['level']
        
        state_idx = self._state_to_idx(state, level=level)
        
        # Store option and action sequence
        option_sequence = []
        action_sequence = defaultdict(list)
        
        # Pick an initial option
        option = self.choose_new_option(state_idx)
        option_sequence.append(option.idx)
        
        # Continuous updates
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            if step >= self.n_steps:
                logging.info(f"Number of steps exceeds maximum number of steps: {self.n_steps}.\nTerminating...")
                break
            
            action = option.choose_action(state_idx, temperature)
            action_sequence[option.idx].append(action)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_idx = self._state_to_idx(new_state, level=level)
            
            # Options evaluation
            self.options_evaluation(state_idx, reward, new_state_idx, option, action, terminated)
            
            # Options improvement
            self.options_improvement(state_idx, new_state_idx, option, action, temperature)
            
            # Pick new option if the previous terminates
            termination_prob = option.beta(new_state_idx)
            if torch.rand(1).item() < termination_prob:
                option = self.choose_new_option(new_state_idx)
                option_sequence.append(option.idx)
            
            state_idx = new_state_idx
            step += 1
        
        # Save state mapping after training if requested
        if save_mapping and level is not None:
            self._save_state_mapping(level)
        
        return option_sequence, action_sequence
    
    def options_evaluation(self, 
                          state_idx: int, 
                          reward: float, 
                          new_state_idx: int, 
                          option: Option, 
                          action: int,
                          terminated: bool) -> None:
        """Implements the option evaluation from Algorithm 1: Option-critic with tabular intra-option Q-learning

        Args:
            state_idx (int): The index of the state
            reward (float): The reward for going from state to new_state
            new_state_idx (int): The index of the new state
            option (Option): The current selected option
            action (int): The action that was just taken
            terminated (bool): Whether the episode has terminated
        """
        option_idx = option.idx
        
        delta = reward - self.get_Q_U(state_idx, option_idx, action)
        
        if not terminated:
            with torch.no_grad():
                delta = delta + self.gamma * (1 - option.beta(new_state_idx)) * self.get_Q_Omega(new_state_idx, option_idx) \
                    + self.gamma * option.beta(new_state_idx) * torch.max(self.get_Q_Omega(new_state_idx))
            
        # Update Q_U
        updated_Q_U_value = self.get_Q_U(state_idx, option_idx, action) + self.alpha_critic * delta
        self.set_Q_U(state_idx, option_idx, action, updated_Q_U_value)
        
        # Update Q_Omega
        self.set_Q_Omega(state_idx, option)
        
        return
    
    def options_improvement(self, 
                            state_idx: int, 
                            new_state_idx: int, 
                            option: Option, 
                            action: int,
                            temperature: float) -> None:
        """Implements the options improvement from Algorithm 1: Option-critic with tabular intra-option Q-learning

        Args:
            state_idx (int): The index of the state
            new_state_idx (int): The index of the new state
            option (Option): The currently selected option
            action (int): The action just taken
            temperature (float): Hyperparameter for logit calculation sensitivity
        """
        option_idx = option.idx
        
        # Update for theta
        option.theta.grad = None # Clear old gradients
        
        # Negative sign due to pytorch minimization
        # Detach Q_U value since it's from the critic (not part of policy gradient)
        q_u_value = self.get_Q_U(state_idx, option_idx, action)
        if isinstance(q_u_value, torch.Tensor):
            q_u_value = q_u_value.detach()
        else:
            q_u_value = torch.tensor(q_u_value, dtype=torch.float32)
        
        log_pi = option.log_pi(state_idx, action, temperature)
        loss = -log_pi * q_u_value
        loss.backward()
        
        with torch.no_grad():
            option.theta += self.alpha_theta * option.theta.grad
        
        # Update for upsilon
        option.upsilon.grad = None
        advantage = (self.get_Q_Omega(new_state_idx, option_idx) - torch.max(self.get_Q_Omega(new_state_idx))).detach()
        
        # Compute beta directly from upsilon parameter to maintain gradient flow
        # (beta() method calls .item() which detaches, so we compute it here)
        upsilon_value = option.upsilon[new_state_idx]
        beta_value = F.sigmoid(upsilon_value)
        
        loss = beta_value * advantage
        loss.backward()
        
        with torch.no_grad():
            option.upsilon -= self.alpha_upsilon * option.upsilon.grad
        
        return
    
    def _get_state_mapping_file(self, level: int) -> Path:
        """Get the file path for a level's state mapping
        
        Args:
            level (int): The level number
            
        Returns:
            Path: Path to the JSON file for this level's state mapping
        """
        return self.state_mapping_dir / f"level_{level}_state_mapping.json"
    
    def _load_state_mapping(self, level: int) -> bool:
        """Load state-to-index mapping from file if it exists
        
        Args:
            level (int): The level number to load mapping for
            
        Returns:
            bool: True if mapping was loaded successfully, False otherwise
        """
        mapping_file = self._get_state_mapping_file(level)
        
        if not mapping_file.exists():
            return False
        
        try:
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            
            # Convert list of [state_list, index] pairs back to dictionary with tuple keys
            # JSON doesn't support tuple keys, so we stored as list of pairs
            state_mapping_list = data['state_to_idx_dict']
            self.state_to_idx_dict = {tuple(k): v for k, v in state_mapping_list}
            self.n_unique_states = data['n_unique_states']
            
            # Update n_states if needed to accommodate loaded states
            if self.n_unique_states > self.n_states:
                old_size = self.n_states
                new_size = max(self.n_unique_states, int(self.n_states * 1.5))
                
                # Expand Q-tables
                expanded_Q_Omega = torch.zeros((new_size, self.n_options))
                expanded_Q_Omega[:old_size] = self.Q_Omega_table
                self.Q_Omega_table = expanded_Q_Omega
                
                expanded_Q_U = torch.zeros((new_size, self.n_options, self.n_actions))
                expanded_Q_U[:old_size] = self.Q_U_table
                self.Q_U_table = expanded_Q_U
                
                self.n_states = new_size
                
                # Expand option parameters
                for option in self.options:
                    old_theta = option.theta.clone()
                    old_upsilon = option.upsilon.clone()
                    
                    expanded_theta = torch.nn.Parameter(torch.rand(new_size, self.n_actions))
                    expanded_theta[:old_size] = old_theta
                    option.theta = expanded_theta
                    
                    expanded_upsilon = torch.nn.Parameter(torch.rand(new_size,))
                    expanded_upsilon[:old_size] = old_upsilon
                    option.upsilon = expanded_upsilon
            
            logging.info(f"Loaded state mapping for level {level}: {self.n_unique_states} unique states")
            return True
            
        except Exception as e:
            logging.warning(f"Failed to load state mapping for level {level}: {e}")
            return False
    
    def _save_state_mapping(self, level: int) -> bool:
        """Save state-to-index mapping to file
        
        Args:
            level (int): The level number to save mapping for
            
        Returns:
            bool: True if mapping was saved successfully, False otherwise
        """
        mapping_file = self._get_state_mapping_file(level)
        
        try:
            # Convert tuple keys to lists for JSON serialization
            # Store as list of [state_list, index] pairs since JSON keys must be strings
            state_mapping_list = [[list(k), v] for k, v in self.state_to_idx_dict.items()]
            
            data = {
                'state_to_idx_dict': state_mapping_list,
                'n_unique_states': self.n_unique_states,
                'level': level
            }
            
            with open(mapping_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved state mapping for level {level}: {self.n_unique_states} unique states to {mapping_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save state mapping for level {level}: {e}")
            return False
    
    def _state_to_idx(self, state: torch.Tensor, level: Optional[int] = None) -> int:
        """Converts an observation from the environment to an index

        Args:
            state (torch.Tensor): The observation from env.step() or env.reset()
                                  Can be a numpy array or torch tensor representing
                                  a flattened grid of shape (grid_height * grid_width,)
            level (Optional[int]): The level number. If provided and mapping is empty,
                                  will attempt to load existing mapping for this level.

        Returns:
            int: The index of the state. If the state has been seen before, returns
                 its existing index. Otherwise, assigns a new index and returns it.
        """
        # Try to load state mapping if we have a level and mapping is empty
        if level is not None and level != self.current_level:
            self.current_level = level
            # Only load if we don't have any states yet (fresh start)
            if len(self.state_to_idx_dict) == 0:
                self._load_state_mapping(level)
        
        # Convert state to a hashable tuple representation
        # Handle both numpy arrays and torch tensors
        if isinstance(state, torch.Tensor):
            state_tuple = tuple(state.cpu().numpy().flatten().tolist())
        elif isinstance(state, np.ndarray):
            state_tuple = tuple(state.flatten().tolist())
        else:
            # Try to convert to list/tuple directly
            state_tuple = tuple(state.flatten() if hasattr(state, 'flatten') else state)
        
        # Check if we've seen this state before
        if state_tuple not in self.state_to_idx_dict:
            # Assign a new index for this unique state
            self.state_to_idx_dict[state_tuple] = self.n_unique_states
            self.n_unique_states += 1
            
            # Dynamically expand Q-tables if needed
            # Note: This assumes states are encountered incrementally
            # For very large state spaces, consider pre-allocating or using sparse representations
            if self.n_unique_states > self.n_states:
                # Expand Q_Omega table
                old_size = self.Q_Omega_table.shape[0]
                new_size = max(self.n_unique_states, int(self.n_states * 1.5))
                expanded_Q_Omega = torch.zeros((new_size, self.n_options))
                expanded_Q_Omega[:old_size] = self.Q_Omega_table
                self.Q_Omega_table = expanded_Q_Omega
                
                # Expand Q_U table
                expanded_Q_U = torch.zeros((new_size, self.n_options, self.n_actions))
                expanded_Q_U[:old_size] = self.Q_U_table
                self.Q_U_table = expanded_Q_U
                
                # Update n_states to reflect new capacity
                self.n_states = new_size
                
                # Expand theta and upsilon for each option
                for option in self.options:
                    old_theta = option.theta.clone()
                    old_upsilon = option.upsilon.clone()
                    
                    # Expand theta
                    expanded_theta = torch.nn.Parameter(torch.rand(new_size, self.n_actions))
                    expanded_theta[:old_size] = old_theta
                    option.theta = expanded_theta
                    
                    # Expand upsilon
                    expanded_upsilon = torch.nn.Parameter(torch.rand(new_size,))
                    expanded_upsilon[:old_size] = old_upsilon
                    option.upsilon = expanded_upsilon
        
        return self.state_to_idx_dict[state_tuple]

# TODO: Comment and annotate this file

# TODO: In a new file
# train_agent.py: Train for X episodes and save results (the agent, the options, the options and action sequences)
# plotting.py: Make plotting of the results (options, q-values for analysis)
# experiment_suite.py: Gather all baseline methods / HRL methods
#       NOTE: Consider Hydra for easier config / experiment management