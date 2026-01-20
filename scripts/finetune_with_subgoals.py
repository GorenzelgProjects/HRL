"""
Script to fine-tune Option-Critic with subgoal-based intrinsic rewards.

Usage:
    python scripts/finetune_with_subgoals.py --subgoals-file path/to/subgoals.json --assignments-file path/to/assignments.json
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from addons.gas.integration import create_subgoal_reward_module
from addons.gas.subgoal_rewards import SubgoalRewards


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Option-Critic with subgoal rewards')
    parser.add_argument('--subgoals-file', type=str, required=True,
                       help='Path to subgoals JSON file')
    parser.add_argument('--assignments-file', type=str, required=True,
                       help='Path to subgoal assignments JSON file')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                       help='Scale factor for intrinsic rewards')
    
    args = parser.parse_args()
    
    # Load subgoals and assignments
    with open(args.subgoals_file, 'r') as f:
        subgoals = json.load(f)
    with open(args.assignments_file, 'r') as f:
        assignments = json.load(f)
    
    # Create reward module
    reward_module = create_subgoal_reward_module(
        subgoals=subgoals,
        assignments=assignments,
        reward_scale=args.reward_scale
    )
    
    print(f"Created subgoal reward module with {len(subgoals)} subgoals")
    print(f"Reward scale: {args.reward_scale}")
    print("\nTo use this in training, integrate SubgoalRewards into the Option-Critic training loop.")
    print("The reward_module.compute_total_reward() method should be called during training.")


if __name__ == "__main__":
    main()
