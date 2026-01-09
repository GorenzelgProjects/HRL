"""
Test script for Thin Ice Gymnasium Environment
Tests the environment with a simple random agent and Q-learning agent
"""

import gymnasium as gym
import copy
import sys
import numpy as np
import time
from thin_ice_env import ThinIceEnv
from astar_utils import FrontierAStar, ThinIceState, ThinIceGoal, Heuristic
from typing import Optional, Dict, Tuple
import json
import os
from datetime import datetime
import argparse
import yaml
from data.classes.Player import Player

def test_random_agent(env, num_episodes=5, render=False, delay=0.1):
    """Test the environment with a random agent"""
    print("=" * 60)
    print("Testing with Random Agent")
    print("=" * 60)
    
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        print(f"\nEpisode {episode + 1}")
        print(f"Initial distance to exit: {info['distance']}")
        
        if render:
            try:
                env.render()
                time.sleep(delay)
            except Exception as e:
                print(f"Error during rendering: {e}")
                import traceback
                traceback.print_exc()
        
        while not terminated and not truncated:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            if steps % 50 == 0:
                print(f"  Step {steps}: Reward={total_reward:.2f}, Distance={info['distance']}")
            
            if steps >= 500:  # Safety limit
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
    
    print("\n" + "=" * 60)
    return results


def test_q_learning(env, num_episodes=100, learning_rate=0.1, discount=0.95, epsilon=0.1, render=False, delay=0.05):
    """Test with a simple Q-learning agent"""
    print("=" * 60)
    print("Testing with Q-Learning Agent")
    print("=" * 60)
    
    # Initialize Q-table
    # State space is large, so we'll use a simplified representation
    # We'll use a hash of the observation as the state key
    q_table = {}
    
    def get_state_key(obs):
        """Convert observation to a hashable state key"""
        # Use a simplified state: player position + exit position + has_key
        # This is a simplified representation for demonstration
        # In practice, you might want to use a neural network instead
        return hash(obs.tobytes())
    
    def get_q_value(state_key, action):
        """Get Q-value for state-action pair"""
        if state_key not in q_table:
            q_table[state_key] = np.zeros(env.action_space.n)
        return q_table[state_key][action]
    
    def update_q_value(state_key, action, reward, next_state_key, terminated):
        """Update Q-value using Q-learning"""
        current_q = get_q_value(state_key, action)
        if terminated:
            target_q = reward
        else:
            next_q_values = [get_q_value(next_state_key, a) for a in range(env.action_space.n)]
            target_q = reward + discount * max(next_q_values)
        
        q_table[state_key][action] = current_q + learning_rate * (target_q - current_q)
    
    episode_rewards = []
    episode_lengths = []
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state_key = get_state_key(obs)
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        episode_start_time = time.time()
        
        if render and episode == num_episodes - 1:  # Render only last episode
            env.render()
            time.sleep(delay)
        
        while not terminated and not truncated:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [get_q_value(state_key, a) for a in range(env.action_space.n)]
                action = np.argmax(q_values)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_key = get_state_key(next_obs)
            
            # Update Q-table
            update_q_value(state_key, action, reward, next_state_key, terminated)
            
            state_key = next_state_key
            total_reward += reward
            steps += 1
            
            if render and episode == num_episodes - 1:  # Render only last episode
                env.render()
                time.sleep(delay)
            
            if steps >= 500:  # Safety limit
                truncated = True
                break
        
        episode_time = time.time() - episode_start_time
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
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
            "time_seconds": episode_time,
            "q_table_size": len(q_table)
        }
        results.append(episode_result)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Avg Reward={avg_reward:.2f}, Avg Length={avg_length:.1f}, "
                  f"Q-table size={len(q_table)}")
    
    print(f"\nTraining complete!")
    print(f"Average reward over last 10 episodes: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    print("=" * 60)
    
    return results

def test_a_star(env, heuristic: str = "manhattan", render: bool = False, delay: float = 0.05):

    # Set up
    initial_state, _ = env.reset()
    
    # Create initial state with a new player instance to avoid mutations
    # We need to create a new Player instance for the initial state
    initial_player = Player(env.game, env.player.x, env.player.y, env.settings)
    initial_state = ThinIceState(env.reward_config, initial_player)
    goal_state = ThinIceGoal(env.end_tile.x, env.end_tile.y)
    heuristic_obj = Heuristic(heuristic)
    action_set_converter = {
            0: (1, 0),   # right
            1: (0, -1),  # up
            2: (-1, 0),  # left
            3: (0, 1),   # down
        }
    frontier = FrontierAStar(heuristic_obj)
    
    frontier.prepare(goal_state)

    # Clear the parent pointer and cost in order make sure that the initial state is a root node
    initial_state.parent = None
    initial_state.path_cost = 0

    frontier.add(initial_state)
    expanded = set()
    
    max_iterations = 1000000  # Safety limit
    iteration = 0
    
    print("Searching for solution with A* algorithm...")
    
    while iteration < max_iterations:
        iteration += 1
        
        if frontier.is_empty():
            print("-" * 40, file=sys.stderr)
            print("FAILED TO FIND SOLUTION!", file=sys.stderr)
            print("-" * 40, file=sys.stderr)
            return False, [] # Failure to find a solution
        
        # Dequeue state from frontier and add to expanded set
        new_state = frontier.pop()
        expanded.add(new_state)
        
        # Check goal state
        if goal_state.is_goal(new_state):
            plan = new_state.extract_plan()
            print(f"Solution found in {iteration} iterations with {len(plan)} steps")
            if new_state.moving_block_pos:
                print(f"Final moving block position: {new_state.moving_block_pos}")
            
            # If rendering is enabled, execute the plan step by step
            if render:
                print(f"\nExecuting solution with {len(plan)} steps...")
                print("=" * 60)
                
                # Reset environment to initial state
                env.reset()
                
                # Convert direction strings to action integers
                direction_to_action = {
                    "right": 0,
                    "up": 1,
                    "left": 2,
                    "down": 3
                }
                
                # Render initial state
                env.render()
                time.sleep(delay)
                
                # Execute plan step by step
                total_reward = 0
                for step_idx, direction in enumerate(plan, 1):
                    action = direction_to_action[direction]
                    
                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    # Render the step
                    env.render()
                    print(f"Step {step_idx}/{len(plan)}: {direction.upper()} - Reward: {reward:.2f}, Total: {total_reward:.2f}")
                    time.sleep(delay)
                    
                    if terminated or truncated:
                        print(f"\nEpisode completed! Final reward: {total_reward:.2f}")
                        if terminated:
                            print("✓ Successfully reached the goal!")
                        break
                
                # Keep window open for a moment to see final state
                if not terminated and not truncated:
                    print(f"\nPlan executed. Final reward: {total_reward:.2f}")
                time.sleep(delay * 2)
                print("=" * 60)
            
            return True, plan
        
        # Expand new_state (0=right, 1=up, 2=left, 3=down)
        for (action, (dx, dy)) in new_state.get_applicable_actions(action_set_converter):
            
            # CRITICAL FIX: Create a new Player instance for each child state
            # This prevents mutations from affecting parent states
            # The Player shares the game object (for collision checking) but has its own position
            child_player = Player(
                new_state.player.game, 
                new_state.player.x, 
                new_state.player.y, 
                new_state.player.settings
            )
            
            # Create child state with new player instance
            child = ThinIceState(
                new_state.reward_config,  # Dict is immutable enough for our purposes
                child_player,
                action,
                new_state,
                new_state.has_key,  # These are primitives, no need to copy
                new_state.can_teleport,
                new_state.reset_once,
                new_state.moved,
                new_state.keyhole_unlocked,  # Track keyhole unlock status
                new_state.moving_block_pos  # Track moving block position
            )
            
            # Apply the action
            child = child.result(dx, dy)
            
            if child not in expanded and not frontier.contains(child):
                frontier.add(child)
                
            # Update priority if we found a better path to this state
            elif hasattr(frontier, "priority_queue") and frontier.contains(child):
                current_priority = frontier.priority_queue.get_priority(child)
                new_priority = frontier.f(child, goal_state)
                if current_priority > new_priority:
                    frontier.priority_queue.change_priority(child, new_priority)
    
    print("-" * 40, file=sys.stderr)
    print(f"MAX ITERATIONS ({max_iterations}) REACHED!", file=sys.stderr)
    print("-" * 40, file=sys.stderr)
    return False, []


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
