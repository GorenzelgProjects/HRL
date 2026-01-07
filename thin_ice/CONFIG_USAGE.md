# Configuration File Usage

The `config.yaml` file controls all aspects of the Thin Ice environment, including RL methods, reward functions, and test settings.

## Quick Start

1. Edit `config.yaml` to customize settings
2. Run tests: `python test_env.py`
3. Or override with command-line: `python test_env.py --render --level 5`

## Configuration Sections

### Environment Settings
```yaml
environment:
  level: 1              # Level to test (1-19)
  max_episode_steps: 1000
  headless: true        # Run without display
  render_mode: null     # "human", "ansi", or null
```

### Test Settings
```yaml
test:
  render: false         # Enable visual rendering
  delay: 0.1           # Delay between frames (seconds)
  random_episodes: 3   # Number of random agent episodes
  qlearning_episodes: 50
```

### RL Method Selection
```yaml
rl_method:
  method: "both"        # "random", "qlearning", or "both"
  
  qlearning:
    learning_rate: 0.1
    discount: 0.95
    epsilon: 0.2        # Exploration rate
    epsilon_decay: 0.995
    epsilon_min: 0.01
```

### Reward Function Configuration
```yaml
rewards:
  new_tile_reward: 0.1              # Reward for visiting new tile
  level_completion_reward: 5.0      # Reward for reaching exit
  perfect_completion_bonus: 10.0    # Bonus if all tiles visited
  key_collection_reward: 1.0
  keyhole_unlock_reward: 1.0
  treasure_collection_reward: 2.0
  invalid_move_penalty: -0.01
  death_penalty: -5.0
  use_distance_reward: false        # Enable distance shaping
  distance_reward_scale: -0.01
```

## Command Line Overrides

All config settings can be overridden via command line:

```bash
# Use config defaults
python test_env.py

# Override specific settings
python test_env.py --render --level 5 --method random

# Use custom config file
python test_env.py --config my_config.yaml
```

## Examples

### Test with visual rendering:
```yaml
# In config.yaml
test:
  render: true
  delay: 0.2
```
Then run: `python test_env.py`

### Custom reward function:
```yaml
# In config.yaml
rewards:
  new_tile_reward: 0.5        # Higher reward for exploration
  level_completion_reward: 20.0  # Much higher completion reward
  death_penalty: -10.0         # Stronger penalty
```

### Q-learning with high exploration:
```yaml
# In config.yaml
rl_method:
  qlearning:
    epsilon: 0.5        # More exploration
    learning_rate: 0.2  # Faster learning
```

## Notes

- Command-line arguments override config file settings
- Config file is loaded from the `thin_ice/` directory
- Use `--config` to specify a different config file
- All numeric values in rewards can be adjusted to shape learning behavior
