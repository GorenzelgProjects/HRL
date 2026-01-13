# OptionCritic for Thin Ice

This directory contains an implementation of the Option-Critic architecture for hierarchical reinforcement learning on the Thin Ice environment.

## Overview

The Option-Critic architecture is a hierarchical reinforcement learning method that learns both:
- **Options**: Temporally extended actions (policies over primitive actions)
- **Option policies**: Intra-option policies that select primitive actions
- **Termination functions**: Determine when an option should terminate

This implementation uses tabular Q-learning for the option-value functions and policy gradient methods for the option policies and termination functions.

## Files

### Core Implementation

- **`option_critic.py`**: Main implementation of the OptionCritic architecture
  - `Option` class: Represents a single option with intra-option policy (theta) and termination function (upsilon)
  - `OptionCritic` class: Main agent class with Q-tables and option management
  - State-to-index mapping with automatic save/load functionality

### Training

- **`train_agent.py`**: Training script for the OptionCritic agent
  - Trains agent for specified number of episodes
  - Saves agent checkpoints, training results, and state mappings
  - Comprehensive logging and progress tracking

### Visualization

- **`plotting.py`**: Visualization utilities for analyzing training results
  - Training metrics plots (rewards, steps, completion rate)
  - Option usage analysis
  - Q-value visualizations
  - Option sequence timelines

## Installation

Make sure you have the required dependencies:

```bash
pip install torch numpy matplotlib gymnasium pygame pyyaml
```

## Quick Start

### 1. Train an Agent

Train an OptionCritic agent on level 1 for 100 episodes:

```bash
python hrl_models/option_critic/train_agent.py \
    --level 1 \
    --num_episodes 100 \
    --n_options 4 \
    --output_dir training_output \
    --verbose
```

### 2. Visualize Results

Generate plots from training results:

```bash
python hrl_models/option_critic/plotting.py \
    --results_file training_output/results/training_results_level_1.json \
    --agent_file training_output/agents/agent_episode_100_level_1.json \
    --output_dir plots
```

## Detailed Usage

### Training Script (`train_agent.py`)

#### Basic Usage

```bash
python hrl_models/option_critic/train_agent.py --level 1 --num_episodes 100
```

#### Command-Line Arguments

**Environment Parameters:**
- `--level`: Level number to train on (1-19, default: 1)
- `--num_episodes`: Number of training episodes (default: 100)

**Agent Parameters:**
- `--n_options`: Number of options to use (default: 4)
- `--n_states`: Initial estimate of number of states (default: 1000)
- `--n_actions`: Number of actions (default: 4, should match environment)
- `--gamma`: Discount factor (default: 0.99)
- `--alpha_critic`: Learning rate for critic/Q-tables (default: 0.5)
- `--alpha_theta`: Learning rate for intra-option policies (default: 0.25)
- `--alpha_upsilon`: Learning rate for termination functions (default: 0.25)
- `--epsilon`: Exploration parameter for option selection (default: 0.9)
- `--n_steps`: Maximum steps per episode (default: 1000)
- `--temperature`: Temperature for option policy (default: 1.0)

**Training Parameters:**
- `--save_frequency`: Save agent every N episodes (default: 10)
- `--output_dir`: Output directory for results (default: "training_output")
- `--verbose`: Print detailed logs
- `--quiet`: Suppress output

#### Example: Advanced Training

```bash
python hrl_models/option_critic/train_agent.py \
    --level 1 \
    --num_episodes 200 \
    --n_options 6 \
    --gamma 0.95 \
    --alpha_critic 0.3 \
    --alpha_theta 0.2 \
    --alpha_upsilon 0.2 \
    --epsilon 0.8 \
    --temperature 1.5 \
    --save_frequency 20 \
    --output_dir my_training_run \
    --verbose
```

#### Output Structure

After training, the following structure is created:

```
training_output/
├── agents/
│   ├── agent_episode_10_level_1.json    # Human-readable agent state
│   ├── agent_episode_10_level_1.pt      # PyTorch format (for loading)
│   ├── agent_episode_20_level_1.json
│   └── ...
├── results/
│   └── training_results_level_1.json    # Training statistics and sequences
└── logs/
    └── training_YYYYMMDD_HHMMSS.log      # Training logs
```

### Plotting Script (`plotting.py`)

#### Basic Usage

```bash
python hrl_models/option_critic/plotting.py \
    --results_file training_output/results/training_results_level_1.json \
    --agent_file training_output/agents/agent_episode_100_level_1.json
```

#### Command-Line Arguments

- `--results_file`: Path to training results JSON file (required)
- `--agent_file`: Path to agent JSON file (optional, needed for Q-value plots)
- `--output_dir`: Directory to save plots (default: "plots")
- `--no-show`: Don't display plots (only save to files)
- `--plots`: Which plots to generate (choices: `metrics`, `options`, `qvalues`, `sequences`, `all`, default: `all`)

#### Example: Generate Specific Plots

```bash
# Only generate training metrics and option usage plots
python hrl_models/option_critic/plotting.py \
    --results_file training_output/results/training_results_level_1.json \
    --plots metrics options \
    --output_dir plots
```

#### Generated Plots

1. **Training Metrics** (`training_metrics_level_{level}.png`)
   - Reward per episode (with moving average)
   - Steps per episode
   - Options used per episode
   - Option switches per episode
   - Completion rate over time
   - Summary statistics

2. **Option Usage Analysis** (`option_usage_level_{level}.png`)
   - Option frequency bar chart
   - Option transition matrix heatmap
   - Option duration distribution
   - Option usage over episodes

3. **Q-Value Analysis** (`q_values_episode_{episode}_level_{level}.png`)
   - Q_Omega heatmap (state × option)
   - Mean Q-value per option
   - Q_U heatmap (option × action)
   - Q-value distribution

4. **Option Sequences** (`option_sequences_level_{level}.png`)
   - Timeline visualization of option sequences
   - Shows option switches across episodes

## Python API

### Using OptionCritic Programmatically

```python
from thin_ice.thin_ice_env import ThinIceEnv
from hrl_models.option_critic.option_critic import OptionCritic

# Create environment
env = ThinIceEnv(level=1, render_mode=None, headless=True)

# Create agent
agent = OptionCritic(
    n_states=1000,      # Initial state estimate
    n_actions=4,        # 4 actions: up, down, left, right
    n_options=4,        # Number of options
    gamma=0.99,         # Discount factor
    alpha_critic=0.5,    # Critic learning rate
    alpha_theta=0.25,   # Policy learning rate
    alpha_upsilon=0.25, # Termination learning rate
    epsilon=0.9,       # Exploration parameter
    n_steps=1000       # Max steps per episode
)

# Train for one episode
option_sequence, action_sequence = agent.train(env, temperature=1.0)

# Get Q-values
state_idx = 0
q_omega = agent.get_Q_Omega(state_idx)  # Q-values for all options
q_u = agent.get_Q_U(state_idx, option_idx=0, action_idx=1)  # Specific Q-value

# Choose an option
option = agent.choose_new_option(state_idx)

# Get option policy
probs = option.pi(state_idx, temperature=1.0)  # Action probabilities
action = option.choose_action(state_idx, temperature=1.0)  # Sample action

env.close()
```

## State Mapping

The implementation automatically saves and loads state-to-index mappings for each level. This ensures:
- Consistent state indices across training runs
- Faster training on subsequent runs (no need to rebuild mapping)
- Efficient state space management

State mappings are saved to:
```
hrl_models/option_critic/state_mappings/level_{level}_state_mapping.json
```

## Algorithm Details

### Option Evaluation (Critic Update)

The Q-values are updated using intra-option Q-learning:

```
δ = r + γ[(1-β(s'))Q_Ω(s',ω) + β(s')max_ω' Q_Ω(s',ω')] - Q_U(s,ω,a)
Q_U(s,ω,a) ← Q_U(s,ω,a) + α_critic * δ
Q_Ω(s,ω) ← Σ_a π_ω(a|s) * Q_U(s,ω,a)
```

### Option Improvement (Policy Update)

**Intra-option policy (theta):**
```
∇_θ J = -log π_ω(a|s) * Q_U(s,ω,a)
θ ← θ + α_θ * ∇_θ J
```

**Termination function (upsilon):**
```
∇_υ J = β(s') * [Q_Ω(s',ω) - max_ω' Q_Ω(s',ω')]
υ ← υ - α_υ * ∇_υ J
```

## Key Features

1. **Automatic State Mapping**: Dynamically maps environment states to indices with save/load functionality
2. **Dynamic Table Expansion**: Q-tables and option parameters expand automatically as new states are encountered
3. **Comprehensive Logging**: Detailed logs of training progress and statistics
4. **Checkpointing**: Regular saves of agent state for resuming training
5. **Rich Visualizations**: Multiple plot types for analyzing training results
6. **Flexible Configuration**: Extensive command-line arguments for hyperparameter tuning

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Memory Issues**: Reduce `n_states` initial estimate or use fewer options
3. **Slow Training**: Enable headless mode and reduce `save_frequency`
4. **Plotting Errors**: Ensure matplotlib is installed: `pip install matplotlib`

### Performance Tips

- Start with fewer options (2-4) for faster initial training
- Use lower learning rates (0.1-0.3) for more stable learning
- Adjust `epsilon` based on exploration needs (higher = more exploration)
- Save state mappings are reused automatically on subsequent runs

## References

- [Option-Critic Architecture](https://arxiv.org/abs/1609.05140) - Bacon et al., 2017
- Implementation inspired by:
  - https://github.com/alversafa/option-critic-arch
  - https://github.com/theophilegervet/options-hierarchical-rl

## License

This implementation is part of the HRL project for Thin Ice environment.
