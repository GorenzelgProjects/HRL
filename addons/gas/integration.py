"""
Integration utilities for GAS mining and fine-tuning.

Provides helper functions to run the complete GAS pipeline:
1. Load logged rollouts
2. Run GAS miner to extract subgoals
3. Assign subgoals to options
4. Fine-tune Option-Critic with intrinsic rewards
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from addons.encoder.pixel_encoder import make_pixel_encoder
from addons.gas.gas_miner import GASMiner
from addons.gas.subgoal_rewards import latch_subgoals_to_options, SubgoalRewards


def load_rollouts_from_json(filepath: Path) -> List[Dict]:
    """
    Load logged rollouts from JSON file.
    
    Expected format: training_results_level_X.json with episodes containing:
    - 'obs' or observations
    - 'option_sequence'
    - 'action_sequence'
    - 'agent_pos' (if available)
    - 'reward', 'terminated', etc.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        List of episode dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    
    # Convert to format expected by GAS miner
    rollouts = []
    for ep in episodes:
        rollout = {
            'obs': ep.get('obs', []),
            'option_sequence': ep.get('option_sequence', []),
            'action_sequence': ep.get('action_sequence', []),
            'flat_action_sequence': ep.get('flat_action_sequence', []),
            'reward': ep.get('total_reward', 0.0),
            'terminated': ep.get('terminated', False),
            'agent_pos': ep.get('agent_pos', [])  # May need to be extracted from info
        }
        rollouts.append(rollout)
    
    return rollouts


def run_gas_mining(
    rollouts: List[Dict],
    encoder,
    output_dir: Path,
    num_epochs: int = 50,
    way_steps: int = 8,
    te_threshold: float = -1.0,  # Default: auto-determine
    num_subgoals: int = 10,
    batch_size: int = 1024
) -> tuple[List[Dict], Dict[int, int]]:
    """
    Run complete GAS mining pipeline.
    
    Args:
        rollouts: List of episode dictionaries
        encoder: Pixel encoder instance
        output_dir: Directory to save results
        num_epochs: TDR training epochs
        way_steps: Steps ahead for TE computation and distance cutoff
        te_threshold: Temporal Efficiency threshold
        num_subgoals: Target number of subgoals
        batch_size: Batch size for graph construction
    
    Returns:
        Tuple of (subgoals list, assignments dict)
    """
    # Create GAS miner
    miner = GASMiner(encoder, z_dim=256, h_dim=128)
    
    # Run mining
    subgoals = miner.mine(
        rollouts,
        num_epochs=num_epochs,
        way_steps=way_steps,
        te_threshold=te_threshold,
        num_subgoals=num_subgoals,
        batch_size=batch_size
    )
    
    # Assign subgoals to options
    assignments = latch_subgoals_to_options(rollouts, subgoals)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save subgoals
    subgoals_file = output_dir / "subgoals.json"
    with open(subgoals_file, 'w') as f:
        json.dump(subgoals, f, indent=2, default=str)
    
    # Save assignments
    assignments_file = output_dir / "subgoal_assignments.json"
    with open(assignments_file, 'w') as f:
        json.dump(assignments, f, indent=2)
    
    print(f"Saved subgoals to {subgoals_file}")
    print(f"Saved assignments to {assignments_file}")
    
    return subgoals, assignments


def create_subgoal_reward_module(
    subgoals: List[Dict],
    assignments: Dict[int, int],
    reward_scale: float = 1.0
) -> SubgoalRewards:
    """
    Create SubgoalRewards module for fine-tuning.
    
    Args:
        subgoals: List of subgoal dictionaries
        assignments: Dictionary mapping subgoal_id -> option_idx
        reward_scale: Scale factor for intrinsic rewards
    
    Returns:
        SubgoalRewards instance
    """
    return SubgoalRewards(
        subgoals=subgoals,
        assignments=assignments,
        reward_scale=reward_scale
    )
