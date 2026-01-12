"""
Test script for Thin Ice Gymnasium Environment
Tests the environment with a simple random agent and Q-learning agent
"""

import gymnasium as gym
import numpy as np
from thin_ice.thin_ice_env import ThinIceEnv
import time
import json
import os
from datetime import datetime
import argparse
import yaml
import networkx as nx

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


def test_astar(env, render=False, delay=0.05, max_steps=1000, replan_interval=1):
    """
    Test with A* pathfinding algorithm using NetworkX.
    
    Args:
        env: The environment to test
        render: Whether to render the environment
        delay: Delay between rendered frames
        max_steps: Maximum steps per episode
        replan_interval: Replan every N steps (1 = replan after each step)
    """
    print("=" * 60)
    print("Testing with A* Algorithm")
    print("=" * 60)
    
    def manhattan_distance(pos1, pos2):
        """Heuristic function: Manhattan distance"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_walkable(grid, x, y, grid_width, grid_height, has_key=False):
        """Check if a position is walkable"""
        # Check bounds
        if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
            return False
        
        # Check tile type
        tile_type = grid[y, x]
        
        # Walls are not walkable
        if tile_type == 1:  # WALL
            return False
        
        # Keyholes require a key (tile type 7)
        if tile_type == 7 and not has_key:
            return False
        
        # All other tiles are walkable (empty, ice, exit, key, etc.)
        return True
    
    def build_graph_from_observation(obs, grid_width, grid_height, has_key=False):
        """Build a NetworkX graph from the current observation"""
        G = nx.Graph()
        grid = obs.reshape(grid_height, grid_width)
        
        # Add nodes for all walkable positions
        for y in range(grid_height):
            for x in range(grid_width):
                if is_walkable(grid, x, y, grid_width, grid_height, has_key):
                    G.add_node((x, y))
        
        # Add edges between adjacent walkable positions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        for node in G.nodes():
            x, y = node
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in G:
                    # Edge weight is 1 (uniform cost)
                    G.add_edge(node, neighbor, weight=1.0)
        
        return G
    
    def find_positions_in_grid(grid, grid_width, grid_height):
        """Find player and exit positions in the grid"""
        player_pos = None
        exit_pos = None
        
        for y in range(grid_height):
            for x in range(grid_width):
                tile_type = grid[y, x]
                if tile_type == 4:  # PLAYER
                    player_pos = (x, y)
                elif tile_type == 5:  # EXIT
                    exit_pos = (x, y)
        
        return player_pos, exit_pos
    
    def path_to_actions(path):
        """Convert a path (list of positions) to a list of actions"""
        if len(path) < 2:
            return []
        
        actions = []
        action_map = {
            (1, 0): 0,   # right
            (-1, 0): 2,  # left
            (0, 1): 3,   # down
            (0, -1): 1,  # up
        }
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]
            action = action_map.get((dx, dy))
            if action is not None:
                actions.append(action)
        
        return actions
    
    results = []
    
    # Run one episode
    obs, info = env.reset(seed=42)
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False
    episode_start_time = time.time()
    
    print(f"Initial distance to exit: {info['distance']}")
    print(f"Total tiles to visit: {info['total_tiles']}")
    
    if render:
        env.render()
        time.sleep(delay)
    
    plan_step = 0
    
    while not terminated and not truncated:
        # Replan if needed
        if plan_step % replan_interval == 0:
            # Build graph from current observation
            grid = obs.reshape(env.grid_height, env.grid_width)
            G = build_graph_from_observation(obs, env.grid_width, env.grid_height, 
                                            has_key=info.get('has_key', False))
            
            # Find player and exit positions
            player_pos, exit_pos = find_positions_in_grid(grid, env.grid_width, env.grid_height)
            
            if player_pos is None or exit_pos is None:
                print("Warning: Could not find player or exit position")
                break
            
            if player_pos not in G or exit_pos not in G:
                print(f"Warning: Source or target not in graph. Source: {player_pos}, Target: {exit_pos}")
                # Try to find a path to any reachable position
                if player_pos in G:
                    # Find closest reachable position to target
                    if len(G.nodes()) > 0:
                        closest = min(G.nodes(), 
                                     key=lambda n: manhattan_distance(n, exit_pos) if n != player_pos else float('inf'))
                        exit_pos = closest
                    else:
                        break
                else:
                    break
            
            try:
                # Use A* to find path
                path = nx.astar_path(G, player_pos, exit_pos, 
                                    heuristic=lambda u, v: manhattan_distance(u, v),
                                    weight='weight')
                
                # Convert path to actions
                actions = path_to_actions(path)
                
                if len(actions) == 0:
                    print("Warning: No path found or path is empty")
                    # Try a random action as fallback
                    action = env.action_space.sample()
                else:
                    action = actions[0]  # Take first action from path
                    
            except nx.NetworkXNoPath:
                print("Warning: No path found to exit")
                # Try a random action as fallback
                action = env.action_space.sample()
        else:
            # Continue with previous plan (if we had one)
            if 'actions' in locals() and len(actions) > 0:
                action = actions.pop(0)
            else:
                action = env.action_space.sample()
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        plan_step += 1
        
        if render:
            env.render()
            time.sleep(delay)
        
        if steps % 50 == 0:
            print(f"  Step {steps}: Reward={total_reward:.2f}, Distance={info['distance']}, "
                  f"Visited={info['visited_tiles']}/{info['total_tiles']}")
        
        if steps >= max_steps:
            truncated = True
            break
    
    episode_time = time.time() - episode_start_time
    episode_result = {
        "episode": 1,
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
    
    print(f"\nEpisode finished: Steps={steps}, Total Reward={total_reward:.2f}, "
          f"Terminated={terminated}, Visited Tiles={info['visited_tiles']}/{info['total_tiles']}, "
          f"Time={episode_time:.2f}s")
    
    # Print summary
    print("\n" + "=" * 60)
    print("A* Algorithm Summary")
    print("=" * 60)
    print(f"Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Success: {terminated}")
    print(f"Visited Tiles: {info['visited_tiles']}/{info['total_tiles']}")
    print(f"Time: {episode_time:.2f}s")
    print("=" * 60)
    
    return results


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
                       help='Level to test (1-19)')
    parser.add_argument('--random-episodes', type=int, 
                       default=test_config.get("random_episodes", 3),
                       help='Number of random agent episodes')
    parser.add_argument('--qlearning-episodes', type=int, 
                       default=test_config.get("qlearning_episodes", 50),
                       help='Number of Q-learning episodes')
    parser.add_argument('--delay', type=float, 
                       default=test_config.get("delay", 0.1),
                       help='Delay between rendered frames (seconds)')
    parser.add_argument('--method', type=str,
                       default=rl_config.get("method", "both"),
                       choices=["random", "qlearning", "astar", "both"],
                       help='RL method to use')
    parser.add_argument('--astar', action='store_true',
                       help='Run A* pathfinding algorithm')
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
    
    # Create environment with custom reward config
    env = ThinIceEnv(level=args.level, render_mode=render_mode, headless=headless, reward_config=reward_config)
    
    all_results = {}
    
    # Test based on method selection
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
    
    if args.method == "astar" or args.astar:
        print("\n")
        astar_results = test_astar(env, render=args.render, delay=args.delay, max_steps=500)
        all_results["astar_agent"] = astar_results
    
    env.close()
    
    # Save all results
    results_config = config.get("results", {})
    if results_config.get("save_results", True):
        # Combine all results
        all_episode_results = []
        for method_results in all_results.values():
            all_episode_results.extend(method_results)
        
        if all_episode_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = results_config.get("results_dir", "test_results")
            filename = f"test_results_{timestamp}.json"
            save_results(all_episode_results, filename)
    
    print("\nAll tests completed!")
