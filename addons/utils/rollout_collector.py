"""
Rollout collection utilities for GAS and Simple-SF.

Provides a replay-buffer-like structure to collect rich trajectories
for offline analysis and mining.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


class RolloutCollector:
    """
    Collects rich trajectory data during training for offline analysis.
    
    Stores per-step: obs, z (if encoder used), option, action, reward, done, info.
    """
    
    def __init__(self):
        """Initialize empty collector."""
        self.rollouts: List[Dict] = []
        self.current_episode: Dict = {
            'obs': [],
            'z': [],
            'option_sequence': [],
            'action_sequence': [],
            'rewards': [],
            'dones': [],
            'agent_pos': [],
            'info': []
        }
    
    def add_step(
        self,
        obs: np.ndarray,
        option_idx: int,
        action: int,
        reward: float,
        done: bool,
        info: Dict,
        z: Optional[np.ndarray] = None
    ):
        """
        Add a step to the current episode.
        
        Args:
            obs: Observation (can be pixels or grid)
            option_idx: Current active option index
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            info: Info dictionary from environment
            z: Optional encoded features (if encoder is used)
        """
        self.current_episode['obs'].append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        if z is not None:
            self.current_episode['z'].append(z.copy() if isinstance(z, np.ndarray) else z)
        self.current_episode['option_sequence'].append(option_idx)
        self.current_episode['action_sequence'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)
        
        # Extract agent position from info if available
        agent_pos = info.get('agent_pos') or info.get('player_pos') or None
        self.current_episode['agent_pos'].append(agent_pos)
        
        self.current_episode['info'].append(info.copy() if isinstance(info, dict) else info)
    
    def finish_episode(self, terminated: bool, truncated: bool, total_reward: float):
        """
        Finish current episode and add to rollouts.
        
        Args:
            terminated: Whether episode terminated successfully
            truncated: Whether episode was truncated
            total_reward: Total episode reward
        """
        episode = {
            'obs': self.current_episode['obs'],
            'z': self.current_episode['z'] if len(self.current_episode['z']) > 0 else None,
            'option_sequence': self.current_episode['option_sequence'],
            'action_sequence': self.current_episode['action_sequence'],
            'rewards': self.current_episode['rewards'],
            'dones': self.current_episode['dones'],
            'agent_pos': self.current_episode['agent_pos'],
            'info': self.current_episode['info'],
            'terminated': terminated,
            'truncated': truncated,
            'total_reward': total_reward,
            'num_steps': len(self.current_episode['obs'])
        }
        
        self.rollouts.append(episode)
        
        # Reset for next episode
        self.current_episode = {
            'obs': [],
            'z': [],
            'option_sequence': [],
            'action_sequence': [],
            'rewards': [],
            'dones': [],
            'agent_pos': [],
            'info': []
        }
    
    def get_rollouts(self) -> List[Dict]:
        """Get all collected rollouts."""
        return self.rollouts
    
    def clear(self):
        """Clear all collected rollouts."""
        self.rollouts = []
        self.current_episode = {
            'obs': [],
            'z': [],
            'option_sequence': [],
            'action_sequence': [],
            'rewards': [],
            'dones': [],
            'agent_pos': [],
            'info': []
        }
