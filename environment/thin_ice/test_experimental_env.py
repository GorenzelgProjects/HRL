"""
Test script for the experimental Thin Ice environment
Extended with longer tests, multiple episodes, and Q-learning
"""

import gymnasium as gym
import numpy as np
from thin_ice_env_experimental import ThinIceEnvExperimental
import time
import json
import os
from datetime import datetime
import argparse

def test_random_agent(env, num_episodes=10, render=False, delay=0.1, max_steps=1000):
    """Test the environment with a random agent for multiple episodes"""
    print("=" * 60)
    print("Testing Experimental Environment with Random Agent")
    print(f"Running {num_episodes} episodes (max {max_steps} steps each)")
    print("=" * 60)
    
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Initial distance to exit: {info['distance']}")
        print(f"Total tiles to visit: {info['total_tiles']}")
        
        if render:
            try:
                env.render()
                time.sleep(delay)
            except Exception as e:
                print(f"Error during rendering: {e}")
        
        while not terminated and not truncated:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            if steps % 100 == 0:
                print(f"  Step {steps}: Reward={total_reward:.2f}, Distance={info['distance']}, "
                      f"Visited={info['visited_tiles']}/{info['total_tiles']}")
            
            if steps >= max_steps:
                truncated = True
                break
        
        episode_time = time.time() - episode_start_time
        episode_result = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "terminated": terminated,
            "truncated": truncated,
            "visited_tiles": info['visited_tiles'],
            "total_tiles": info['total_tiles'],
            "complete_tiles": info['complete_tiles'],
            "final_distance": info['distance'],
            "has_key": info['has_key'],
            "level": info['level'],
            "time_seconds": episode_time
        }
        results.append(episode_result)
        
        print(f"Episode finished: Steps={steps}, Total Reward={total_reward:.2f}, "
              f"Terminated={terminated}, Visited Tiles={info['visited_tiles']}/{info['total_tiles']}, "
              f"Time={episode_time:.2f}s")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_visited = np.mean([r['visited_tiles'] for r in results])
    avg_time = np.mean([r['time_seconds'] for r in results])
    completion_rate = sum(1 for r in results if r['terminated']) / len(results)
    
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Visited Tiles: {avg_visited:.1f}")
    print(f"Average Time per Episode: {avg_time:.2f}s")
    print(f"Completion Rate: {completion_rate * 100:.1f}%")
    print("=" * 60)
    
    return results


def test_q_learning(env, num_episodes=200, learning_rate=0.1, discount=0.95, 
                    epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                    render=False, delay=0.05, max_steps=1000):
    """Test with a simple Q-learning agent"""
    print("=" * 60)
    print("Testing Experimental Environment with Q-Learning Agent")
    print(f"Running {num_episodes} episodes")
    print("=" * 60)
    
    # Initialize Q-table
    q_table = {}
    epsilon = epsilon_start
    
    def get_state_key(obs):
        """Convert observation to a hashable state key"""
        # Use a simplified state: hash of the observation
        return hash(obs.tobytes())
    
    def get_q_value(state_key, action):
        """Get Q-value for state-action pair"""
        if state_key not in q_table:
            q_table[state_key] = np.zeros(env.action_space.n)
        return q_table[state_key][action]
    
    def update_q_value(state_key, action, reward, next_state_key, done):
        """Update Q-value using Q-learning"""
        current_q = get_q_value(state_key, action)
        
        if done:
            target_q = reward
        else:
            next_q_values = [get_q_value(next_state_key, a) for a in range(env.action_space.n)]
            target_q = reward + discount * max(next_q_values)
        
        q_table[state_key][action] = current_q + learning_rate * (target_q - current_q)
    
    results = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        state_key = get_state_key(obs)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        if (episode + 1) % 20 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}, Epsilon: {epsilon:.3f}, "
                  f"Q-table size: {len(q_table)}")
        
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [get_q_value(state_key, a) for a in range(env.action_space.n)]
                action = np.argmax(q_values)
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state_key = get_state_key(next_obs)
            
            # Update Q-value
            update_q_value(state_key, action, reward, next_state_key, terminated or truncated)
            
            state_key = next_state_key
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            if steps >= max_steps:
                truncated = True
                break
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_time = time.time() - episode_start_time
        episode_rewards.append(total_reward)
        
        episode_result = {
            "episode": episode + 1,
            "steps": steps,
            "total_reward": float(total_reward),
            "terminated": terminated,
            "truncated": truncated,
            "visited_tiles": next_info['visited_tiles'],
            "total_tiles": next_info['total_tiles'],
            "epsilon": epsilon,
            "q_table_size": len(q_table),
            "time_seconds": episode_time
        }
        results.append(episode_result)
        
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(f"  Recent 50-episode average reward: {recent_avg:.2f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Q-Learning Summary")
    print("=" * 60)
    print(f"Final Q-table size: {len(q_table)}")
    print(f"Final epsilon: {epsilon:.3f}")
    print(f"Average reward (first 50): {np.mean(episode_rewards[:50]):.2f}")
    print(f"Average reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Completion rate: {sum(1 for r in results if r['terminated']) / len(results) * 100:.1f}%")
    print("=" * 60)
    
    return results


def test_multiple_levels(levels=[1, 2, 3, 5, 7, 10], num_episodes_per_level=5, render=False):
    """Test multiple levels"""
    print("=" * 60)
    print("Testing Multiple Levels")
    print("=" * 60)
    
    all_results = {}
    
    for level in levels:
        print(f"\n{'=' * 60}")
        print(f"Testing Level {level}")
        print(f"{'=' * 60}")
        
        try:
            env = ThinIceEnvExperimental(level=level, render_mode="semi" if render else None, headless=True)
            results = test_random_agent(env, num_episodes=num_episodes_per_level, 
                                        render=render, delay=0.1, max_steps=500)
            all_results[level] = results
            env.close()
        except Exception as e:
            print(f"Error testing level {level}: {e}")
            all_results[level] = None
    
    # Print cross-level summary
    print("\n" + "=" * 60)
    print("Cross-Level Summary")
    print("=" * 60)
    for level, results in all_results.items():
        if results:
            avg_reward = np.mean([r['total_reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])
            completion = sum(1 for r in results if r['terminated']) / len(results) * 100
            print(f"Level {level}: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.1f}, "
                  f"Completion={completion:.1f}%")
    print("=" * 60)
    
    return all_results


def test_experimental_environment():
    """Quick test of the experimental environment"""
    print("=" * 60)
    print("Quick Test: Experimental Thin Ice Environment")
    print("=" * 60)
    
    env = ThinIceEnvExperimental(level=1, render_mode="semi", headless=True)
    
    obs, info = env.reset(seed=42)
    print(f"Environment reset successfully")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    print()
    
    env.render()
    
    print("Running 10 random steps...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.2)
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
    print("=" * 60)


def test_with_rendering():
    """Test with full visual rendering (pygame sprites)"""
    print("=" * 60)
    print("Testing Experimental Environment with Full Visual Rendering")
    print("=" * 60)
    
    env = ThinIceEnvExperimental(level=1, render_mode="human", headless=False)
    
    obs, info = env.reset(seed=42)
    print(f"Initial distance: {info['distance']}")
    print("Window should open showing the game with sprites...")
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        time.sleep(0.1)
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    env.close()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the experimental Thin Ice environment")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--random", type=int, default=0, metavar="N", 
                       help="Run random agent for N episodes (default: 0)")
    parser.add_argument("--qlearning", type=int, default=0, metavar="N",
                       help="Run Q-learning for N episodes (default: 0)")
    parser.add_argument("--levels", action="store_true", 
                       help="Test multiple levels")
    parser.add_argument("--level", type=int, default=1, help="Starting level (default: 1)")
    parser.add_argument("--max-steps", type=int, default=1000, 
                       help="Maximum steps per episode (default: 1000)")
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else ("semi" if args.random > 0 or args.qlearning > 0 else None)
    
    if args.random > 0:
        env = ThinIceEnvExperimental(level=args.level, render_mode=render_mode, headless=not args.render)
        test_random_agent(env, num_episodes=args.random, render=args.render, 
                         delay=0.1, max_steps=args.max_steps)
        env.close()
    elif args.qlearning > 0:
        env = ThinIceEnvExperimental(level=args.level, render_mode=render_mode, headless=not args.render)
        test_q_learning(env, num_episodes=args.qlearning, render=args.render, 
                       delay=0.05, max_steps=args.max_steps)
        env.close()
    elif args.levels:
        test_multiple_levels(render=args.render)
    elif args.render:
        test_with_rendering()
    else:
        test_experimental_environment()
        print("\n")
        print("Usage examples:")
        print("  python test_experimental_env.py --random 50          # Run 50 random episodes")
        print("  python test_experimental_env.py --qlearning 200     # Run 200 Q-learning episodes")
        print("  python test_experimental_env.py --levels            # Test multiple levels")
        print("  python test_experimental_env.py --render            # Test with visual rendering")
        print("  python test_experimental_env.py --random 10 --render # Random with rendering")
