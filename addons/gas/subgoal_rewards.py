"""
Subgoal rewards and option assignment utilities.

Implements latching (assigning subgoals to options) and option-conditioned intrinsic rewards.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def latch_subgoals_to_options(
    rollouts: List[Dict],
    subgoals: List[Dict],
    position_tolerance: float = 1.0
) -> Dict[int, int]:
    """
    Assign subgoals to options based on which option was active when subgoal was reached.
    
    For each mined subgoal g, count how often it's reached under each option ω_t.
    Assign g to the option with highest reach count.
    
    Args:
        rollouts: List of episode dictionaries with 'option', 'agent_pos', etc.
        subgoals: List of subgoal dictionaries with 'position', 'id', etc.
        position_tolerance: Distance tolerance for considering a subgoal "reached"
    
    Returns:
        Dictionary mapping subgoal_id -> option_idx
    """
    # Count reaches per (subgoal, option) pair
    reach_counts = defaultdict(lambda: defaultdict(int))
    
    for episode in rollouts:
        option_sequence = episode.get('option_sequence', [])
        agent_positions = episode.get('agent_pos', [])
        
        if len(option_sequence) == 0 or len(agent_positions) == 0:
            continue
        
        # For each step, check if any subgoal is reached
        for step_idx, (option_idx, pos) in enumerate(zip(option_sequence, agent_positions)):
            if pos is None:
                continue
            
            # Check distance to each subgoal
            for subgoal in subgoals:
                subgoal_pos = subgoal.get('position')
                if subgoal_pos is None:
                    continue
                
                # Compute distance (assuming pos is (x, y) tuple or list)
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    dist = np.sqrt((pos[0] - subgoal_pos[0])**2 + (pos[1] - subgoal_pos[1])**2)
                else:
                    continue
                
                if dist <= position_tolerance:
                    reach_counts[subgoal['id']][option_idx] += 1
    
    # Assign each subgoal to option with highest count
    assignments = {}
    for subgoal_id, option_counts in reach_counts.items():
        if len(option_counts) > 0:
            best_option = max(option_counts.items(), key=lambda x: x[1])[0]
            assignments[subgoal_id] = best_option
        else:
            # No reaches found, assign to option 0 as default
            assignments[subgoal_id] = 0
    
    return assignments


class SubgoalRewards:
    """
    Option-conditioned intrinsic reward computation.
    
    Provides intrinsic rewards based on subgoal assignments to options.
    """
    
    def __init__(
        self,
        subgoals: List[Dict],
        assignments: Dict[int, int],
        reward_scale: float = 1.0,
        position_tolerance: float = 1.0,
        use_smooth: bool = False
    ):
        """
        Initialize subgoal rewards.
        
        Args:
            subgoals: List of subgoal dictionaries
            assignments: Dictionary mapping subgoal_id -> option_idx
            reward_scale: Scale factor eta for intrinsic rewards
            position_tolerance: Distance tolerance for subgoal reach
            use_smooth: If True, use smooth reward based on distance
        """
        self.subgoals = subgoals
        self.assignments = assignments
        self.reward_scale = reward_scale
        self.position_tolerance = position_tolerance
        self.use_smooth = use_smooth
        
        # Create reverse mapping: option -> list of subgoals
        self.option_to_subgoals = defaultdict(list)
        for subgoal_id, option_idx in assignments.items():
            self.option_to_subgoals[option_idx].append(subgoal_id)
    
    def compute_intrinsic_reward(
        self,
        option_idx: int,
        agent_pos: Tuple[float, float]
    ) -> float:
        """
        Compute intrinsic reward for current option and position.
        
        Returns: r_intr = eta * indicator(pos == g_assigned_to_ω) or smooth variant.
        
        Args:
            option_idx: Current active option
            agent_pos: Current agent position (x, y)
        
        Returns:
            Intrinsic reward value
        """
        if agent_pos is None:
            return 0.0
        
        # Get subgoals assigned to this option
        assigned_subgoals = self.option_to_subgoals.get(option_idx, [])
        
        if len(assigned_subgoals) == 0:
            return 0.0
        
        max_reward = 0.0
        
        for subgoal_id in assigned_subgoals:
            subgoal = self.subgoals[subgoal_id]
            subgoal_pos = subgoal.get('position')
            
            if subgoal_pos is None:
                continue
            
            # Compute distance
            if isinstance(agent_pos, (list, tuple)) and len(agent_pos) >= 2:
                dist = np.sqrt((agent_pos[0] - subgoal_pos[0])**2 + (agent_pos[1] - subgoal_pos[1])**2)
            else:
                continue
            
            if self.use_smooth:
                # Smooth reward: exponential decay with distance
                reward = self.reward_scale * np.exp(-dist / self.position_tolerance)
            else:
                # Binary reward: 1 if within tolerance, 0 otherwise
                reward = self.reward_scale if dist <= self.position_tolerance else 0.0
            
            max_reward = max(max_reward, reward)
        
        return max_reward
    
    def compute_total_reward(
        self,
        env_reward: float,
        option_idx: int,
        agent_pos: Tuple[float, float]
    ) -> float:
        """
        Compute total reward: environment reward + intrinsic reward.
        
        Args:
            env_reward: Environment reward
            option_idx: Current active option
            agent_pos: Current agent position
        
        Returns:
            Total reward
        """
        r_intr = self.compute_intrinsic_reward(option_idx, agent_pos)
        return env_reward + r_intr
