# Thin Ice Gymnasium Environment

This directory contains a Gymnasium environment wrapper for the Thin Ice game, allowing you to train reinforcement learning agents on the game.

## Installation

Make sure you have the required dependencies:

```bash
pip install gymnasium numpy pygame pyyaml
```

## Usage

### Basic Usage

```python
import gymnasium as gym
from thin_ice_env import ThinIceEnv

# Register the environment
gym.register(
    id="ThinIce-v0",
    entry_point=ThinIceEnv,
    max_episode_steps=1000,
)

# Create environment
env = gym.make("ThinIce-v0", level=1, render_mode=None, headless=True)

# Reset environment
obs, info = env.reset(seed=42)

# Run an episode
done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### Environment Details

**Action Space:** `Discrete(4)`
- 0: Move right
- 1: Move up
- 2: Move left
- 3: Move down

**Observation Space:** `Box(0, 7, (grid_height * grid_width,), int32)`
- Flattened grid representation where each cell is encoded:
  - 0 = empty/free
  - 1 = wall
  - 2 = ice
  - 3 = water
  - 4 = player
  - 5 = exit
  - 6 = key
  - 7 = keyhole

**Rewards:**
- +0.1 for visiting a new tile
- +1.0 for collecting a key
- +1.0 for unlocking a keyhole
- +2.0 for collecting treasure
- +5.0 for reaching the exit (regular completion)
- +10.0 for reaching the exit after visiting all tiles (perfect completion)
- -0.01 for invalid moves
- -5.0 for dying (getting stuck)

**Episode Termination:**
- Episode ends when the player reaches the exit tile
- Episode ends when the player dies (gets stuck surrounded by walls)
- Episode is truncated after 1000 steps (configurable)

### Testing

Run the test script to validate the environment and see it in action:

```bash
cd thin_ice
python test_env.py
```

This will:
1. Validate the environment setup
2. Test with a random agent
3. Train a simple Q-learning agent

### Training Agents

You can use any Gymnasium-compatible RL library to train agents:

#### Example with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from thin_ice_env import ThinIceEnv

# Register environment
gym.register(
    id="ThinIce-v0",
    entry_point=ThinIceEnv,
    max_episode_steps=1000,
)

# Create vectorized environment
env = make_vec_env("ThinIce-v0", n_envs=4, env_kwargs={"level": 1, "headless": True})

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones.any():
        obs = env.reset()
```

#### Example with Custom Training Loop

```python
import gymnasium as gym
from thin_ice_env import ThinIceEnv
import numpy as np

gym.register(
    id="ThinIce-v0",
    entry_point=ThinIceEnv,
    max_episode_steps=1000,
)

env = gym.make("ThinIce-v0", level=1)

# Simple training loop
for episode in range(100):
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Your agent here
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"Episode {episode}: Reward = {total_reward:.2f}")

env.close()
```

### Rendering

You can render the environment in different modes:

```python
# Human rendering (requires display)
env = gym.make("ThinIce-v0", level=1, render_mode="human", headless=False)

# ASCII rendering
env = gym.make("ThinIce-v0", level=1, render_mode="ansi", headless=True)
env.render()  # Prints ASCII representation

# No rendering (fastest, for training)
env = gym.make("ThinIce-v0", level=1, render_mode=None, headless=True)
```

### Level Selection

You can train on different levels:

```python
# Train on level 1
env = gym.make("ThinIce-v0", level=1, headless=True)

# Train on level 5
env = gym.make("ThinIce-v0", level=5, headless=True)

# Levels 1-19 are available
```

### Environment Parameters

- `level` (int): Level number (1-19), default=1
- `render_mode` (str): Rendering mode ("human", "rgb_array", "ansi", or None), default=None
- `headless` (bool): If True, disable pygame display for faster training, default=True

## File Structure

- `thin_ice_env.py`: Main Gymnasium environment implementation
- `test_env.py`: Test script with random and Q-learning agents
- `GYMNASIUM_README.md`: This file

## Notes

- The environment runs in headless mode by default for faster training
- The observation space uses a flattened grid representation for simplicity
- For more complex agents, you might want to use a neural network-based approach (e.g., PPO, DQN) instead of tabular Q-learning
- The environment is compatible with all Gymnasium wrappers (e.g., TimeLimit, FrameStack, etc.)

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Creating Custom Environments](https://gymnasium.farama.org/introduction/create_custom_env/)
