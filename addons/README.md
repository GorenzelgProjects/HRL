# Addons for Option-Critic

This directory contains addons to enhance the Option-Critic implementation with:
1. **Simple Successor Features (Simple-SF)** - For continual learning and transfer
2. **GAS (Graph-Assisted Stitching)** - For offline subgoal mining and intrinsic rewards

## Structure

```
addons/
├── encoder/           # Shared pixel encoder (CNN-style mockup)
├── simple_sf/         # Simple Successor Features addon
├── gas/              # GAS subgoal mining addon
└── utils/            # Utility modules (rollout collection)
```

## Installation

The addons require additional dependencies:
- `torch` (PyTorch) - for neural network modules
- `networkx` - for GAS graph construction (already in requirements.txt)

## Usage

### Simple Successor Features

Enable Simple-SF in your config:

```yaml
models:
  option_critic:
    use_simple_sf: true
    sf_d: 256
    lambda_sf: 0.1
    alpha_w: 0.1
    sf_lr_main: 1e-3
    sf_lr_w: 1e-2
```

Then run training as usual:
```bash
python main.py experiment=option_critic_example
```

### GAS Subgoal Mining

**Important:** GAS mining requires rollouts with observations. The standard `training_results_level_X.json` files don't include observations, so you need to enable rollout collection first.

1. **Collect rollouts during training:**
```yaml
models:
  option_critic:
    collect_rollouts: true
    rollout_save_dir: "logs/my_experiment/rollouts"
```

Then run training:
```bash
python main.py experiment=option_critic_example
```

This will save rollouts to the specified directory.

2. **Mine subgoals from collected rollouts:**
```bash
# Use the saved NPZ files from rollout collection
python scripts/mine_subgoals.py --rollout-file logs/your_experiment/rollouts/rollouts_level_21_episode_100.npz --output-dir logs/my_experiment/subgoals --num-subgoals 10

# Or use JSON training results (if they contain obs field)
python scripts/mine_subgoals.py --rollout-file logs/option_critic_T2026-01-20-11-31-50/option_critic/results/training_results_level_1.json --output-dir logs/my_experiment/subgoals --num-subgoals 10
```

3. **Fine-tune with subgoal rewards:**
```bash
python scripts/finetune_with_subgoals.py \
    --subgoals-file logs/my_experiment/subgoals/subgoals.json \
    --assignments-file logs/my_experiment/subgoals/subgoal_assignments.json \
    --reward-scale 1.0
```

## Architecture

The addons use a shared pixel encoder that converts observations to feature vectors `z_t`. This encoder is used by:
- Simple-SF: Computes basis features `phi(z)` and successor features `psi(z,a)`
- GAS: Uses encoder outputs for Temporal Distance Representation (TDR)

The encoder is a mockup CNN that can be easily swapped for a real implementation later.

## Integration

The addons are integrated into Option-Critic training through:
- `addons/simple_sf/integration.py` - Wires Simple-SF losses into training
- `addons/gas/integration.py` - Provides utilities for mining and fine-tuning
- Modified `train_agent.py` - Collects observations and computes SF losses

All addon code is optional - if addons are not available, training proceeds normally.
