# Inspired by https://github.com/alversafa/option-critic-arch/blob/master/utils.py
# and https://github.com/theophilegervet/options-hierarchical-rl/blob/master/implementations/option_critic.ipynb

import torch
import torch.nn.functional as F

from thin_ice.thin_ice_env import ThinIceEnv

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
        
    def pi(self, state_idx: int, temperature: float = 1.0) -> float:
        """The option policy for a particular state

        Args:
            state_idx (int): The index of the state in the environment
            temperature (float, optional): Hyperparam for the temperature. Defaults to 1.0.

        Returns:
            float: The probabilities of each action given a state
        """
        logits = self.theta[state_idx] / temperature
        probs = F.softmax(logits, dim = -1).item()
        
        return probs
    
    def log_pi(self, state_idx: int, action: int, temperature: float = 1.0) -> float:
        """Computes the log probability of an action given a state

        Args:
            state_idx (int): The index of the state
            action (int): The taken action
            temperature (float, optional): Hyperparam for logit calculation. Defaults to 1.0.

        Returns:
            float: The log probability of an action given a state
        """
        logits = self.theta[state_idx] / temperature
        probs = F.log_softmax(logits, dim = -1).item()
        
        return probs[action]
    
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
        
        if torch.rand() < self.epsilon:
            # Explore
            random_option_id = torch.randint(0, self.n_options, (1,)).item()
            return self.options[random_option_id]
        else:
            # Exploit
            best_option_id = torch.argmax(self.get_Q_Omega(state_idx))
            return self.options[best_option_id]


    def train(self, env: ThinIceEnv, temperature: float) -> tuple[list, dict[list]]:
        """Train the Optic-Critic for a maximum of n_steps

        Args:
            env (ThinIceEnv): The ThinIceEnv
            temperature (float): The temperature for logit calculation in π_theta
        
        Returns:
            tuple[list, dict[list]]: The list of pursued options and a dictionary with each option's action sequence
        """
        # Get initial state
        state, _ = env.reset()
        state_idx = self._state_to_idx(state)
        
        # Store option and action sequence
        option_sequence = []
        action_sequence = defaultdict(list)
        
        # Pick an initial option
        option = self.choose_new_option(state_idx)
        option_sequence.append(option.idx)
        
        # Continuous updates
        terminated = False
        step = 0
        while not terminated or truncated:
            if step >= self.n_steps:
                logging.info(f"Number of steps exceeds maximum number of steps: {self.n_steps}.\nTerminating...")
                break
            
            action = option.choose_action(state_idx, temperature)
            action_sequence[option.idx].append(action)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_idx = self._state_to_idx(new_state)
            
            # Options evaluation
            self.options_evaluation(state_idx, reward, new_state_idx, option, action, terminated)
            
            # Options improvement
            self.options_improvement(state_idx, new_state_idx, option, action, temperature)
            
            # Pick new option if the previous terminates
            termination_prob = option.beta(new_state_idx)
            if torch.rand() < termination_prob:
                option = self.choose_new_option(new_state_idx)
                option_sequence.append(option.idx)
        
        return option_sequence, action_sequence
    
    def options_evaluation(self, 
                          state_idx: int, 
                          reward: float, 
                          new_state_idx: int, 
                          option: Option, 
                          action: torch.Tensor[int],
                          terminated: bool) -> None:
        """Implements the option evaluation from Algorithm 1: Option-critic with tabular intra-option Q-learning

        Args:
            state_idx (int): The index of the state
            reward (float): The reward for going from state to new_state
            new_state_idx (int): The index of the new state
            option (Option): The current selected option
            action (torch.Tensor[int]): The action that was just taken
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
        loss = -option.log_pi(state_idx, action, temperature) * self.get_Q_U(state_idx, option_idx, action)
        loss.backward()
        
        with torch.no_grad():
            option.theta += self.alpha_theta * option.theta.grad
        
        # Update for upsilon
        option.upsilon.grad = None
        advantage = (self.get_Q_Omega(new_state_idx, option_idx) - torch.max(self.get_Q_Omega(new_state_idx))).detach()
        
        loss = option.beta(new_state_idx) * advantage
        loss.backward()
        
        with torch.no_grad():
            option.upsilon -= self.alpha_upsilon * option.upsilon.grad
        
        return
    
    def _state_to_idx(self, state: torch.Tensor) -> int:
        """Converts an observation from the environment to an index

        Args:
            state (torch.Tensor): The observation from env.step()

        Returns:
            int: The index of the state
        """
        # TODO: 
        # 1. Initialize a dictionary outside of OptionCritic
        # 2. Calculate number of states
        # 3. Map each state in this function to an integer using the dict
        # 4. Pre-initialize for all levels - past or future
        
        raise NotImplementedError()

# TODO: Comment and annotate this file

# TODO: In a new file
# train_agent.py: Train for X episodes and save results (the agent, the options, the options and action sequences)
# plotting.py: Make plotting of the results (options, q-values for analysis)
# experiment_suite.py: Gather all baseline methods / HRL methods
#       NOTE: Consider Hydra for easier config / experiment management