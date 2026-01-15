"""
Test script for Thin Ice Gymnasium Environment
Tests the environment with a simple random agent and Q-learning agent
"""

import gymnasium as gym
import copy
import sys
import numpy as np
import time
from thin_ice.thin_ice_env import ThinIceEnv
from baseline_models.astar_utils import FrontierAStar, ThinIceState, ThinIceGoal, Heuristic
from typing import Optional, Dict, Tuple
import json
import os
from datetime import datetime
import argparse
import yaml
from thin_ice.data.classes.Player import Player
from baseline_models.random_agent import test_random_agent
from baseline_models.q_learning_agent import test_q_learning
from baseline_models.a_star_agent import test_a_star


def test_environment_basic():
    """Basic environment validation"""
    print("=" * 60)
    print("Basic Environment Validation")
    print("=" * 60)
    
    # Register environment
    gym.register(
        id="ThinIce-v0",
        entry_point=ThinIceEnv,
        max_episode_steps=1000,
    )
    
    # Create environment
    try:
        env = gym.make("ThinIce-v0", level=1, render_mode=None, headless=True)
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


def save_results(results, filename="test_results.json"):
    """Save test results to a JSON file"""
    # Create results directory if it doesn't exist
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_episodes": len(results),
        "episodes": results,
        "summary": {
            "avg_reward": float(np.mean([r["total_reward"] for r in results])),
            "avg_steps": float(np.mean([r["steps"] for r in results])),
            "avg_time": float(np.mean([r["time_seconds"] for r in results])),
            "success_rate": float(sum(1 for r in results if r["terminated"]) / len(results)),
            "avg_visited_tiles": float(np.mean([r["visited_tiles"] for r in results])),
            "avg_total_tiles": float(np.mean([r["total_tiles"] for r in results])),
        }
    }
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    print(f"Summary:")
    print(f"  Average Reward: {output['summary']['avg_reward']:.2f}")
    print(f"  Average Steps: {output['summary']['avg_steps']:.1f}")
    print(f"  Success Rate: {output['summary']['success_rate']*100:.1f}%")
    print(f"  Average Visited Tiles: {output['summary']['avg_visited_tiles']:.1f}/{output['summary']['avg_total_tiles']:.1f}")
    
    return filepath


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    # Load configuration from YAML first
    config = load_config("config.yaml")
    
    # Get defaults from config or use command-line defaults
    env_config = config.get("environment", {})
    test_config = config.get("test", {})
    rl_config = config.get("rl_method", {})
    qlearning_config = rl_config.get("qlearning", {})
    
    parser = argparse.ArgumentParser(description='Test Thin Ice Gymnasium Environment')
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config YAML file')
    parser.add_argument('--render', action='store_true', 
                       default=test_config.get("render", False),
                       help='Render the environment visually')
    parser.add_argument('--level', type=int, 
                       default=env_config.get("level", 1),
                       help='Level to test (1-19), or 0 to test all levels sequentially')
    parser.add_argument('--random-episodes', type=int, 
                       default=test_config.get("random_episodes", 3),
                       help='Number of random agent episodes')
    parser.add_argument('--qlearning-episodes', type=int, 
                       default=test_config.get("qlearning_episodes", 50),
                       help='Number of Q-learning episodes')
    parser.add_argument('--heuristic', type=str,
                        default=test_config.get("heuristic", "manhattan"),
                        help="Which heuristic function to use in the a-star algorithm")
    parser.add_argument('--delay', type=float, 
                       default=test_config.get("delay", 0.1),
                       help='Delay between rendered frames (seconds)')
    parser.add_argument('--method', type=str,
                       default=rl_config.get("method", "all"),
                       choices=["random", "qlearning", "astar", "all"],
                       help='RL method to use')
    args = parser.parse_args()
    
    # Reload config if custom path provided
    if args.config != "config.yaml":
        config = load_config(args.config)
        env_config = config.get("environment", {})
        test_config = config.get("test", {})
        rl_config = config.get("rl_method", {})
        qlearning_config = rl_config.get("qlearning", {})
    
    # Register environment
    gym.register(
        id="ThinIce-v0",
        entry_point=ThinIceEnv,
        max_episode_steps=env_config.get("max_episode_steps", 1000),
    )
    
    # Run basic validation
    test_environment_basic()
    
    print("\n")
    
    # Create environment for testing
    # If --render is specified, force render_mode to "human" and headless to False
    if args.render:
        render_mode = "human"
        headless = False
    else:
        render_mode = env_config.get("render_mode")
        headless = env_config.get("headless", True)
    
    # Pass reward config to environment if available
    reward_config = config.get("rewards", {})
    
    all_results = {}
    
    # Handle A* method separately if testing all levels
    if args.method in ["astar", "all"] and args.level == 0:
        # Test all levels sequentially with A*
        levels_to_test = list(range(1, 20))  # Levels 1-19
        print("=" * 60)
        print(f"Running A* on all levels (1-19) sequentially")
        print("=" * 60)
        
        all_level_results = []
        
        for level_num in levels_to_test:
            print(f"\n{'='*60}")
            print(f"Level {level_num}")
            print(f"{'='*60}")
            
            # Create environment for this level
            level_env = ThinIceEnv(level=level_num, render_mode=render_mode, 
                                  headless=headless, reward_config=reward_config)
            
            try:
                success, plan = test_a_star(level_env, 
                                           heuristic=args.heuristic,
                                           render=args.render,
                                           delay=args.delay)
                
                level_result = {
                    "level": level_num,
                    "success": success,
                    "plan_length": len(plan) if plan else 0,
                    "plan": plan if plan else []
                }
                all_level_results.append(level_result)
                
                if success:
                    print(f"✓ Level {level_num}: Solution found with {len(plan)} steps")
                else:
                    print(f"✗ Level {level_num}: No solution found")
                    
            except Exception as e:
                print(f"✗ Level {level_num}: Error - {e}")
                import traceback
                traceback.print_exc()
                all_level_results.append({
                    "level": level_num,
                    "success": False,
                    "plan_length": 0,
                    "plan": [],
                    "error": str(e)
                })
            finally:
                level_env.close()
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary of All Levels")
        print(f"{'='*60}")
        successful = sum(1 for r in all_level_results if r["success"])
        print(f"Successfully solved: {successful}/{len(all_level_results)} levels")
        if successful > 0:
            avg_steps = np.mean([r["plan_length"] for r in all_level_results if r["success"]])
            print(f"Average solution length: {avg_steps:.1f} steps")
        
        # Store results
        all_results["astar_all_levels"] = all_level_results
        
        # Skip other methods if only running A* on all levels
        if args.method == "astar":
            env = None  # No need to create env for other methods
        else:
            # Still need to run other methods on a single level
            env = ThinIceEnv(level=args.level, render_mode=render_mode, 
                            headless=headless, reward_config=reward_config)
    else:
        # Create environment for single level or other methods
        env = ThinIceEnv(level=args.level, render_mode=render_mode, 
                        headless=headless, reward_config=reward_config)
    
    # Test based on method selection (only if not already handled above)
    if env is not None:
        if args.method in ["random", "both"]:
            print("\n")
            random_results = test_random_agent(env, num_episodes=args.random_episodes, 
                                               render=args.render, delay=args.delay)
            all_results["random_agent"] = random_results
        
        if args.method in ["qlearning", "both"]:
            print("\n")
            # Get Q-learning parameters from config
            learning_rate = qlearning_config.get("learning_rate", 0.1)
            discount = qlearning_config.get("discount", 0.95)
            epsilon = qlearning_config.get("epsilon", 0.2)
            
            qlearning_results = test_q_learning(env, num_episodes=args.qlearning_episodes, 
                                                learning_rate=learning_rate, 
                                                discount=discount, 
                                                epsilon=epsilon,
                                                render=args.render, delay=args.delay)
            all_results["qlearning_agent"] = qlearning_results
        
        if args.method in ["astar", "all"] and args.level != 0:
            # Single level A* test
            print("\n")
            astar_results = test_a_star(env, 
                                        heuristic=args.heuristic,
                                        render=args.render,
                                        delay=args.delay)
            all_results["astar_single"] = astar_results
        
        env.close()
    
    # Save all results
    results_config = config.get("results", {})
    if results_config.get("save_results", True):
        # Handle A* all levels results separately
        if "astar_all_levels" in all_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = results_config.get("results_dir", "test_results")
            os.makedirs(results_dir, exist_ok=True)
            filename = f"astar_all_levels_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            output = {
                "timestamp": datetime.now().isoformat(),
                "method": "astar",
                "heuristic": args.heuristic if args.method in ["astar", "all"] else None,
                "levels": all_results["astar_all_levels"],
                "summary": {
                    "total_levels": len(all_results["astar_all_levels"]),
                    "successful_levels": sum(1 for r in all_results["astar_all_levels"] if r["success"]),
                    "failed_levels": sum(1 for r in all_results["astar_all_levels"] if not r["success"]),
                    "avg_plan_length": float(np.mean([r["plan_length"] for r in all_results["astar_all_levels"] if r["success"]])) if any(r["success"] for r in all_results["astar_all_levels"]) else 0.0
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nResults saved to: {filepath}")
            print(f"Summary:")
            print(f"  Successful: {output['summary']['successful_levels']}/{output['summary']['total_levels']} levels")
            if output['summary']['avg_plan_length'] > 0:
                print(f"  Average plan length: {output['summary']['avg_plan_length']:.1f} steps")
        
        # Also save other results if present
        other_results = {k: v for k, v in all_results.items() if k != "astar_all_levels"}
        if other_results:
            all_episode_results = []
            for method_results in other_results.values():
                if isinstance(method_results, list):
                    all_episode_results.extend(method_results)
            
            if all_episode_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = results_config.get("results_dir", "test_results")
                filename = f"test_results_{timestamp}.json"
                save_results(all_episode_results, filename)
    
    print("\nAll tests completed!")
