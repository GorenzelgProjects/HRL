"""
Simple example of using the Thin Ice Gymnasium environment
"""

import gymnasium as gym
from thin_ice_env import ThinIceEnv

# Register the environment
gym.register(
    id="ThinIce-v0",
    entry_point=ThinIceEnv,
    max_episode_steps=1000,
)

def main():
    # Create environment
    print("Creating Thin Ice environment...")
    env = gym.make("ThinIce-v0", level=1, render_mode=None, headless=True)
    
    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps with random actions
    print("\nRunning 10 random steps...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, "
              f"Distance={info['distance']}, Terminated={terminated}")
        
        if terminated or truncated:
            print("Episode ended!")
            obs, info = env.reset(seed=42)
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    # Close environment
    env.close()
    print("\nEnvironment closed successfully!")

if __name__ == "__main__":
    main()
