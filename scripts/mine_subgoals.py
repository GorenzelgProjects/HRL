"""
Script to run GAS mining on collected rollouts.

Usage:
    python scripts/mine_subgoals.py --rollout-file path/to/rollouts.npz --output-dir path/to/output
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from addons.encoder.pixel_encoder import make_pixel_encoder
from addons.gas.integration import run_gas_mining, load_rollouts_from_json


def main():
    parser = argparse.ArgumentParser(description='Run GAS mining on collected rollouts')
    parser.add_argument('--rollout-file', type=str, required=True,
                       help='Path to rollout file (JSON or NPZ)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save mined subgoals')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='TDR training epochs')
    parser.add_argument('--way-steps', type=int, default=8,
                       help='Steps ahead for TE computation and distance cutoff')
    parser.add_argument('--te-threshold', type=float, default=-1.0,
                       help='Temporal Efficiency threshold (negative = auto-determine from data, default: auto)')
    parser.add_argument('--num-subgoals', type=int, default=10,
                       help='Target number of subgoals to extract')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for graph construction')
    
    args = parser.parse_args()
    
    # Create encoder - use feature vector mode since observations are 104-element vectors
    encoder = make_pixel_encoder(z_dim=256, grid_height=17, grid_width=19, feature_vector_size=104)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    
    # Load rollouts
    rollout_path = Path(args.rollout_file)
    if not rollout_path.exists():
        print(f"Error: Rollout file not found: {rollout_path}")
        print(f"\nTo collect rollouts, enable rollout collection in your config:")
        print(f"  models.option_critic.collect_rollouts: true")
        print(f"  models.option_critic.rollout_save_dir: \"logs/your_experiment/rollouts\"")
        print(f"\nOr use an existing training results file (JSON format):")
        # Try to find example files
        logs_dir = Path("logs")
        if logs_dir.exists():
            example_files = list(logs_dir.glob("option_critic_*/option_critic/results/training_results_level_*.json"))
            if example_files:
                print(f"  Example: --rollout-file {example_files[0]}")
        return
    
    if rollout_path.suffix == '.json':
        rollouts = load_rollouts_from_json(rollout_path)
    else:
        # Load from NPZ
        try:
            data = np.load(rollout_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading NPZ file: {e}")
            return
        # Convert to expected format
        rollouts = []
        
        # Handle both object arrays (variable length) and regular arrays
        obs_data = data['obs']
        if obs_data.dtype == object:
            # Object array - each element is a list/array
            num_episodes = len(obs_data)
        else:
            # Regular array - need to check shape
            num_episodes = obs_data.shape[0] if len(obs_data.shape) > 1 else 1
        
        for i in range(num_episodes):
            rollout = {
                'obs': obs_data[i],
            }
            
            # Handle optional fields - check if they're object arrays
            if 'option_sequence' in data:
                opt_seq = data['option_sequence']
                rollout['option_sequence'] = opt_seq[i] if opt_seq.dtype == object else (opt_seq[i].tolist() if hasattr(opt_seq[i], 'tolist') else opt_seq[i])
            else:
                rollout['option_sequence'] = []
            
            if 'action_sequence' in data:
                act_seq = data['action_sequence']
                rollout['action_sequence'] = act_seq[i] if act_seq.dtype == object else (act_seq[i].tolist() if hasattr(act_seq[i], 'tolist') else act_seq[i])
            else:
                rollout['action_sequence'] = []
            
            if 'rewards' in data:
                rew = data['rewards']
                rollout['rewards'] = rew[i] if rew.dtype == object else (rew[i].tolist() if hasattr(rew[i], 'tolist') else rew[i])
            else:
                rollout['rewards'] = []
            
            if 'agent_pos' in data:
                pos = data['agent_pos']
                rollout['agent_pos'] = pos[i] if pos.dtype == object else (pos[i].tolist() if hasattr(pos[i], 'tolist') else pos[i])
            else:
                rollout['agent_pos'] = []
            
            rollout['terminated'] = data['terminated'][i] if 'terminated' in data else False
            rollouts.append(rollout)
    
    print(f"Loaded {len(rollouts)} rollouts")
    
    # Check if rollouts have required data
    if len(rollouts) > 0:
        first_rollout = rollouts[0]
        if not first_rollout.get('obs') or len(first_rollout.get('obs', [])) == 0:
            print("\nWarning: Rollouts don't contain observations (obs field).")
            print("GAS mining requires observations to compute embeddings.")
            print("\nTo collect rollouts with observations, enable in config:")
            print("  models.option_critic.collect_rollouts: true")
            print("  models.option_critic.rollout_save_dir: \"logs/your_experiment/rollouts\"")
            print("\nThen run training to collect rollouts before mining.")
            return
    
    # Run mining
    output_dir = Path(args.output_dir)
    subgoals, assignments = run_gas_mining(
        rollouts=rollouts,
        encoder=encoder,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        way_steps=args.way_steps,
        te_threshold=args.te_threshold,
        num_subgoals=args.num_subgoals,
        batch_size=args.batch_size
    )
    
    print(f"\nMining complete!")
    print(f"Extracted {len(subgoals)} subgoals")
    print(f"Assignments: {assignments}")


if __name__ == "__main__":
    main()
