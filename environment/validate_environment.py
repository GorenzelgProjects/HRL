import gymnasium as gym

def validate_environment(id: str, entry_point: type[gym.Env]) -> None:
    """Basic environment validation"""
    print("=" * 60)
    print("Basic Environment Validation")
    print("=" * 60)
    
    # Register environment
    gym.register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=1000,
    )
    
    # Create environment
    try:
        env = gym.make(id, level=1, render_mode=None, headless=True)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return
    
    # Test reset
    try:
        obs, info = env.reset(seed=42)
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Initial info: {info}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        env.close()
        return
    
    # Test step
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        env.close()
        return
    
    # Test all actions
    try:
        obs, info = env.reset(seed=42)
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset(seed=42)
        print(f"✓ All actions work correctly")
    except Exception as e:
        print(f"✗ Action test failed: {e}")
        env.close()
        return
    
    env.close()
    print("=" * 60)
    print("All basic tests passed!")
    print("=" * 60)