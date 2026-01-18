"""
Utility script to replay saved playthroughs from training logs.

Supports:
- Q-learning training_results_level_X.json format
- Option-critic training_results_level_X.json format
- A* astar_all_levels_*.json format
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Callable
import gymnasium as gym


def load_training_results(filepath: str) -> Dict:
    """Load training results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def replay_episode(env: gym.Env, action_sequence: List[int], 
                   render: bool = True, delay: float = 0.1, 
                   verbose: bool = True) -> Dict:
    """Replay a single episode from an action sequence
    
    Args:
        env: The environment to replay in
        action_sequence: List of actions to execute
        render: Whether to render the environment
        delay: Delay between steps in seconds
        verbose: Whether to print step information
        
    Returns:
        Dictionary with replay statistics
    """
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False
    
    if render:
        env.render()
        time.sleep(delay)
    
    action_names = {0: "right", 1: "up", 2: "left", 3: "down"}
    
    for step_idx, action in enumerate(action_sequence, 1):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if verbose:
            action_name = action_names.get(action, f"action_{action}")
            print(f"Step {step_idx}/{len(action_sequence)}: {action_name.upper()} - "
                  f"Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if render:
            env.render()
            time.sleep(delay)
        
        if terminated or truncated:
            if verbose:
                print(f"\nEpisode completed!")
                if terminated:
                    print("✓ Successfully reached the goal!")
                else:
                    print("⚠ Episode truncated")
            break
    
    if render:
        time.sleep(delay * 2)  # Keep window open
    
    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "visited_tiles": info.get('visited_tiles', 0),
        "total_tiles": info.get('total_tiles', 0),
        "complete_tiles": info.get('complete_tiles', 0),
    }


def replay_from_qlearning_file(filepath: str, episode_num: Optional[int] = None,
                                render: bool = True, delay: float = 0.1,
                                env_factory: Optional[Callable] = None) -> None:
    """Replay episodes from a Q-learning training results file
    
    Args:
        filepath: Path to training_results_level_X.json file
        episode_num: Specific episode to replay (None = replay all)
        render: Whether to render the environment
        delay: Delay between steps in seconds
        env_factory: Function that creates the environment (takes level as arg)
    """
    # Load training results
    results = load_training_results(filepath)
    level = results['level']
    episodes = results['episodes']
    
    if env_factory is None:
        raise ValueError("env_factory must be provided to create environment")
    
    # Create environment
    env = env_factory(level)
    
    # Determine which episodes to replay
    if episode_num is not None:
        # Find specific episode
        episode_to_replay = None
        for ep in episodes:
            if ep['episode'] == episode_num:
                episode_to_replay = ep
                break
        
        if episode_to_replay is None:
            print(f"Episode {episode_num} not found in file")
            return
        
        episodes_to_replay = [episode_to_replay]
    else:
        episodes_to_replay = episodes
    
    print(f"\n{'='*60}")
    print(f"Replaying {len(episodes_to_replay)} episode(s) from level {level}")
    print(f"{'='*60}")
    
    # Replay each episode
    for ep in episodes_to_replay:
        print(f"\nEpisode {ep['episode']}:")
        print(f"  Original reward: {ep['total_reward']:.2f}")
        print(f"  Original steps: {ep['steps']}")
        print(f"  Original terminated: {ep['terminated']}")
        
        action_sequence = ep.get('action_sequence', [])
        if not action_sequence:
            print("  ⚠ No action sequence found in episode data")
            continue
        
        print(f"  Replaying {len(action_sequence)} actions...")
        replay_stats = replay_episode(env, action_sequence, render, delay, verbose=True)
        
        print(f"\n  Replay Results:")
        print(f"    Reward: {replay_stats['total_reward']:.2f} (original: {ep['total_reward']:.2f})")
        print(f"    Steps: {replay_stats['steps']} (original: {ep['steps']})")
        print(f"    Terminated: {replay_stats['terminated']} (original: {ep['terminated']})")
    
    env.close()


def replay_from_option_critic_file(filepath: str, episode_num: Optional[int] = None,
                                   render: bool = True, delay: float = 0.1,
                                   env_factory: Optional[Callable] = None) -> None:
    """Replay episodes from an option-critic training results file
    
    Args:
        filepath: Path to training_results_level_X.json file
        episode_num: Specific episode to replay (None = replay all)
        render: Whether to render the environment
        delay: Delay between steps in seconds
        env_factory: Function that creates the environment (takes level as arg)
    """
    # Load training results
    results = load_training_results(filepath)
    level = results['level']
    episodes = results['episodes']
    
    if env_factory is None:
        raise ValueError("env_factory must be provided to create environment")
    
    # Create environment
    env = env_factory(level)
    
    # Determine which episodes to replay
    if episode_num is not None:
        # Find specific episode
        episode_to_replay = None
        for ep in episodes:
            if ep['episode'] == episode_num:
                episode_to_replay = ep
                break
        
        if episode_to_replay is None:
            print(f"Episode {episode_num} not found in file")
            return
        
        episodes_to_replay = [episode_to_replay]
    else:
        episodes_to_replay = episodes
    
    print(f"\n{'='*60}")
    print(f"Replaying {len(episodes_to_replay)} episode(s) from level {level}")
    print(f"{'='*60}")
    
    # Replay each episode
    for ep in episodes_to_replay:
        print(f"\nEpisode {ep['episode']}:")
        print(f"  Original reward: {ep['total_reward']:.2f}")
        print(f"  Original steps: {ep['total_steps']}")
        print(f"  Original terminated: {ep['terminated']}")
        
        option_sequence = ep.get('option_sequence', [])
        
        # Option-critic: prefer flat_action_sequence if available (new format)
        # Otherwise, reconstruct from option_sequence and action_sequence dict (old format)
        if 'flat_action_sequence' in ep:
            # New format: use flat action sequence directly
            action_sequence = ep['flat_action_sequence']
            print(f"  Using flat action sequence ({len(action_sequence)} actions)")
        else:
            # Old format: reconstruct from option_sequence and action_sequence dict
            # This is a fallback for older saved results
            action_sequence_dict = ep.get('action_sequence', {})
            
            # Track how many actions we've consumed from each option
            option_action_counts = {str(opt): 0 for opt in action_sequence_dict.keys()}
            
            # Reconstruct action sequence by following option_sequence
            # Each time an option appears, we take the next action from that option's list
            action_sequence = []
            for opt_idx in option_sequence:
                opt_str = str(opt_idx)
                if opt_str in action_sequence_dict:
                    actions_for_option = action_sequence_dict[opt_str]
                    consumed_count = option_action_counts[opt_str]
                    
                    # Take the next action from this option's list
                    if consumed_count < len(actions_for_option):
                        action_sequence.append(actions_for_option[consumed_count])
                        option_action_counts[opt_str] += 1
                    else:
                        print(f"  ⚠ Warning: Option {opt_idx} has no more actions (consumed {consumed_count}/{len(actions_for_option)})")
                else:
                    print(f"  ⚠ Warning: Option {opt_idx} not found in action_sequence")
            
            print(f"  Reconstructed from option sequence ({len(action_sequence)} actions from {len(option_sequence)} options)")
        
        if not action_sequence:
            print("  ⚠ No action sequence found in episode data")
            continue
        
        print(f"  Replaying {len(action_sequence)} actions from {len(option_sequence)} options...")
        replay_stats = replay_episode(env, action_sequence, render, delay, verbose=True)
        
        print(f"\n  Replay Results:")
        print(f"    Reward: {replay_stats['total_reward']:.2f} (original: {ep['total_reward']:.2f})")
        print(f"    Steps: {replay_stats['steps']} (original: {ep['total_steps']})")
        print(f"    Terminated: {replay_stats['terminated']} (original: {ep['terminated']})")
    
    env.close()


def replay_from_astar_file(filepath: str, level: Optional[int] = None,
                           render: bool = True, delay: float = 0.1,
                           env_factory: Optional[Callable] = None) -> None:
    """Replay plan from an A* results file
    
    Args:
        filepath: Path to astar_all_levels_*.json file
        level: Specific level to replay (None = replay all)
        render: Whether to render the environment
        delay: Delay between steps in seconds
        env_factory: Function that creates the environment (takes level as arg)
    """
    # Load A* results
    results = load_training_results(filepath)
    levels_data = results.get('levels', [])
    
    if env_factory is None:
        raise ValueError("env_factory must be provided to create environment")
    
    # Determine which levels to replay
    if level is not None:
        levels_to_replay = [lvl for lvl in levels_data if lvl['level'] == level]
    else:
        levels_to_replay = levels_data
    
    print(f"\n{'='*60}")
    print(f"Replaying {len(levels_to_replay)} level(s) from A* results")
    print(f"{'='*60}")
    
    # Direction to action mapping
    direction_to_action = {
        "right": 0,
        "up": 1,
        "left": 2,
        "down": 3
    }
    
    # Replay each level
    for level_data in levels_to_replay:
        level_num = level_data['level']
        plan = level_data.get('plan', [])
        success = level_data.get('success', False)
        
        if not plan:
            print(f"\nLevel {level_num}: No plan found")
            continue
        
        print(f"\nLevel {level_num}:")
        print(f"  Plan length: {len(plan)}")
        print(f"  Success: {success}")
        
        # Create environment
        env = env_factory(level_num)
        
        # Convert plan to action sequence
        action_sequence = [direction_to_action.get(direction.lower(), 0) 
                           for direction in plan]
        
        print(f"  Replaying {len(action_sequence)} actions...")
        replay_stats = replay_episode(env, action_sequence, render, delay, verbose=True)
        
        print(f"\n  Replay Results:")
        print(f"    Steps: {replay_stats['steps']} (original: {len(plan)})")
        print(f"    Terminated: {replay_stats['terminated']} (original: {success})")
        
        env.close()


def main():
    """Command-line interface for replay utility"""
    parser = argparse.ArgumentParser(description='Replay saved playthroughs from training logs')
    parser.add_argument('filepath', type=str, help='Path to training results JSON file')
    parser.add_argument('--episode', type=int, default=None, 
                       help='Specific episode number to replay (default: all)')
    parser.add_argument('--level', type=int, default=None,
                       help='Specific level number to replay (for A* files)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (faster replay)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps in seconds (default: 0.1)')
    parser.add_argument('--env-type', type=str, default='thin_ice',
                       choices=['thin_ice'],
                       help='Environment type (default: thin_ice)')
    parser.add_argument('--config-path', type=str, default='config',
                       help='Path to config directory (default: config)')
    
    args = parser.parse_args()
    
    # Determine file type and create appropriate environment factory
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return
    
    # Load config to get environment settings
    generate_water = False  # Default value
    try:
        import yaml
        # Try to load thin_ice.yaml (which includes base.yaml defaults)
        config_path = Path(args.config_path) / "environment" / "thin_ice.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
            env_specific_settings = env_config.get('env_specific_settings', {})
            generate_water = env_specific_settings.get('generate_water', False)
            print(f"Loaded config from {config_path}: generate_water = {generate_water}")
        else:
            # Fallback: try base.yaml
            base_config_path = Path(args.config_path) / "environment" / "base.yaml"
            if base_config_path.exists():
                with open(base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f) or {}
                env_specific_settings = base_config.get('env_specific_settings', {})
                generate_water = env_specific_settings.get('generate_water', False)
                print(f"Loaded config from {base_config_path}: generate_water = {generate_water}")
            else:
                print(f"Warning: Config files not found, using default: generate_water = {generate_water}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}, using default: generate_water = {generate_water}")
    
    # Create environment factory
    if args.env_type == 'thin_ice':
        from environment.thin_ice.thin_ice_env import ThinIceEnv
        
        def env_factory(level: int):
            render_mode = "human" if not args.no_render else None
            headless = args.no_render
            return ThinIceEnv(level=level, render_mode=render_mode, headless=headless, 
                            generate_water=generate_water)
    else:
        raise ValueError(f"Unknown environment type: {args.env_type}")
    
    # Determine file type and call appropriate replay function
    filename = filepath.name.lower()
    
    if 'training_results_level' in filename:
        # Check if it's option-critic format (has option_sequence) or Q-learning format
        results = load_training_results(str(filepath))
        if results.get('episodes') and len(results['episodes']) > 0:
            first_ep = results['episodes'][0]
            if 'option_sequence' in first_ep:
                # Option-critic format
                replay_from_option_critic_file(str(filepath), args.episode, 
                                              not args.no_render, args.delay, env_factory)
            else:
                # Q-learning format
                replay_from_qlearning_file(str(filepath), args.episode,
                                           not args.no_render, args.delay, env_factory)
        else:
            print("Error: No episodes found in file")
    elif 'astar' in filename:
        # A* format
        replay_from_astar_file(str(filepath), args.level,
                              not args.no_render, args.delay, env_factory)
    else:
        print(f"Error: Unknown file format. Expected 'training_results_level_*.json' or 'astar_*.json'")
        print(f"Got: {filename}")


if __name__ == "__main__":
    main()
