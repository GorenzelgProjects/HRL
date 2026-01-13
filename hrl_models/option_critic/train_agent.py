"""
Training script for OptionCritic agent on Thin Ice environment.

This script trains an OptionCritic agent for a specified number of episodes
and saves the trained agent, options, and training results.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from thin_ice.thin_ice_env import ThinIceEnv
from hrl_models.option_critic.option_critic import OptionCritic


def setup_logging(log_dir: Path, verbose: bool = True):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ]
    )
    return log_file


def train_episode(agent: OptionCritic, env: ThinIceEnv, temperature: float, episode: int) -> Dict:
    """Train the agent for one episode
    
    Args:
        agent: The OptionCritic agent
        env: The ThinIceEnv environment
        temperature: Temperature parameter for option policy
        episode: Current episode number
        
    Returns:
        Dictionary with episode statistics
    """
    # Get level from environment
    level = getattr(env, 'level', None) or getattr(env, 'current_level', None)
    
    # Reset environment
    state, info = env.reset()
    if level is None and 'level' in info:
        level = info['level']
    
    state_idx = agent._state_to_idx(state, level=level)
    
    # Store episode data
    option_sequence = []
    action_sequence = defaultdict(list)
    rewards = []
    total_reward = 0.0
    step = 0
    
    # Pick an initial option
    option = agent.choose_new_option(state_idx)
    option_sequence.append(option.idx)
    
    # Run episode
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if step >= agent.n_steps:
            break
        
        action = option.choose_action(state_idx, temperature)
        action_sequence[option.idx].append(action)
        
        new_state, reward, terminated, truncated, step_info = env.step(action)
        new_state_idx = agent._state_to_idx(new_state, level=level)
        
        rewards.append(reward)
        total_reward += reward
        
        # Options evaluation
        agent.options_evaluation(state_idx, reward, new_state_idx, option, action, terminated)
        
        # Options improvement
        agent.options_improvement(state_idx, new_state_idx, option, action, temperature)
        
        # Pick new option if the previous terminates
        termination_prob = option.beta(new_state_idx)
        if torch.rand(1).item() < termination_prob:
            option = agent.choose_new_option(new_state_idx)
            option_sequence.append(option.idx)
        
        state_idx = new_state_idx
        step += 1
    
    # Save state mapping after episode
    if level is not None:
        agent._save_state_mapping(level)
    
    # Calculate episode statistics
    episode_stats = {
        'episode': episode,
        'level': level,
        'option_sequence': option_sequence,
        'action_sequence': {str(k): v for k, v in action_sequence.items()},  # Convert to string keys for JSON
        'rewards': rewards,
        'total_reward': total_reward,
        'num_options_used': len(set(option_sequence)),
        'total_steps': step,
        'num_options_switches': len(option_sequence) - 1,
        'terminated': terminated,
        'truncated': truncated,
    }
    
    return episode_stats


def save_agent(agent: OptionCritic, save_dir: Path, episode: int, level: int):
    """Save the trained agent to disk
    
    Args:
        agent: The OptionCritic agent to save
        save_dir: Directory to save the agent
        episode: Current episode number
        level: Current level number
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dictionary with all agent components
    agent_state = {
        'episode': episode,
        'level': level,
        'n_states': agent.n_states,
        'n_actions': agent.n_actions,
        'n_options': agent.n_options,
        'gamma': agent.gamma,
        'alpha_critic': agent.alpha_critic,
        'alpha_theta': agent.alpha_theta,
        'alpha_upsilon': agent.alpha_upsilon,
        'epsilon': agent.epsilon,
        'n_steps': agent.n_steps,
        'n_unique_states': agent.n_unique_states,
        'Q_Omega_table': agent.Q_Omega_table.detach().cpu().numpy().tolist(),
        'Q_U_table': agent.Q_U_table.detach().cpu().numpy().tolist(),
    }
    
    # Save options (theta and upsilon for each option)
    options_data = []
    for option in agent.options:
        options_data.append({
            'idx': option.idx,
            'theta': option.theta.detach().cpu().numpy().tolist(),
            'upsilon': option.upsilon.detach().cpu().numpy().tolist(),
        })
    agent_state['options'] = options_data
    
    # Save to JSON
    agent_file = save_dir / f"agent_episode_{episode}_level_{level}.json"
    with open(agent_file, 'w') as f:
        json.dump(agent_state, f, indent=2)
    
    # Also save as PyTorch state dict for easier loading
    torch_file = save_dir / f"agent_episode_{episode}_level_{level}.pt"
    torch.save({
        'Q_Omega_table': agent.Q_Omega_table,
        'Q_U_table': agent.Q_U_table,
        'options': {opt.idx: {'theta': opt.theta, 'upsilon': opt.upsilon} for opt in agent.options},
        'config': {
            'n_states': agent.n_states,
            'n_actions': agent.n_actions,
            'n_options': agent.n_options,
            'gamma': agent.gamma,
            'alpha_critic': agent.alpha_critic,
            'alpha_theta': agent.alpha_theta,
            'alpha_upsilon': agent.alpha_upsilon,
            'epsilon': agent.epsilon,
            'n_steps': agent.n_steps,
        }
    }, torch_file)
    
    logging.info(f"Saved agent to {agent_file} and {torch_file}")


def save_training_results(results: List[Dict], save_dir: Path, level: int):
    """Save training results (option and action sequences) to file
    
    Args:
        results: List of episode statistics dictionaries
        save_dir: Directory to save results
        level: Level number
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = save_dir / f"training_results_level_{level}.json"
    
    training_data = {
        'level': level,
        'timestamp': datetime.now().isoformat(),
        'num_episodes': len(results),
        'episodes': results,
        'summary': {
            'avg_steps_per_episode': float(np.mean([r['total_steps'] for r in results])) if results else 0,
            'avg_reward_per_episode': float(np.mean([r['total_reward'] for r in results])) if results else 0,
            'max_reward': float(np.max([r['total_reward'] for r in results])) if results else 0,
            'min_reward': float(np.min([r['total_reward'] for r in results])) if results else 0,
            'avg_options_per_episode': float(np.mean([r['num_options_used'] for r in results])) if results else 0,
            'avg_option_switches': float(np.mean([r['num_options_switches'] for r in results])) if results else 0,
            'completion_rate': float(np.mean([1 if r['terminated'] else 0 for r in results])) if results else 0,
            'total_episodes': len(results),
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logging.info(f"Saved training results to {results_file}")


def train_agent(
    level: int = 1,
    num_episodes: int = 100,
    n_options: int = 4,
    n_states: int = 1000,
    n_actions: int = 4,
    gamma: float = 0.99,
    alpha_critic: float = 0.5,
    alpha_theta: float = 0.25,
    alpha_upsilon: float = 0.25,
    epsilon: float = 0.9,
    n_steps: int = 1000,
    temperature: float = 1.0,
    save_frequency: int = 10,
    output_dir: str = "training_output",
    verbose: bool = True,
) -> Tuple[OptionCritic, List[Dict]]:
    """Train OptionCritic agent for specified number of episodes
    
    Args:
        level: Level number to train on (1-19)
        num_episodes: Number of training episodes
        n_options: Number of options to use
        n_states: Initial estimate of number of states
        n_actions: Number of actions (should be 4 for Thin Ice)
        gamma: Discount factor
        alpha_critic: Learning rate for critic
        alpha_theta: Learning rate for intra-option policy
        alpha_upsilon: Learning rate for termination function
        epsilon: Exploration parameter
        n_steps: Maximum steps per episode
        temperature: Temperature for option policy
        save_frequency: Save agent every N episodes
        output_dir: Directory to save outputs
        verbose: Whether to print detailed logs
        
    Returns:
        Tuple of (trained agent, list of episode results)
    """
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_dir = output_path / "agents"
    results_dir = output_path / "results"
    log_dir = output_path / "logs"
    
    # Setup logging
    log_file = setup_logging(log_dir, verbose)
    logging.info(f"Starting training for level {level}")
    logging.info(f"Training parameters: {locals()}")
    
    # Create environment
    logging.info("Creating environment...")
    env = ThinIceEnv(level=level, render_mode=None, headless=True)
    
    # Create agent
    logging.info("Initializing OptionCritic agent...")
    agent = OptionCritic(
        n_states=n_states,
        n_actions=n_actions,
        n_options=n_options,
        gamma=gamma,
        alpha_critic=alpha_critic,
        alpha_theta=alpha_theta,
        alpha_upsilon=alpha_upsilon,
        epsilon=epsilon,
        n_steps=n_steps,
    )
    
    # Training loop
    logging.info(f"Starting training for {num_episodes} episodes...")
    all_results = []
    
    for episode in range(1, num_episodes + 1):
        logging.info(f"Episode {episode}/{num_episodes}")
        
        # Train one episode
        episode_stats = train_episode(agent, env, temperature, episode)
        all_results.append(episode_stats)
        
        # Log episode statistics
        if verbose:
            logging.info(
                f"Episode {episode}: Steps={episode_stats['total_steps']}, "
                f"Reward={episode_stats['total_reward']:.2f}, "
                f"Options used={episode_stats['num_options_used']}, "
                f"Option switches={episode_stats['num_options_switches']}, "
                f"Terminated={episode_stats['terminated']}"
            )
        
        # Save agent periodically
        if episode % save_frequency == 0 or episode == num_episodes:
            save_agent(agent, save_dir, episode, level)
            save_training_results(all_results, results_dir, level)
            logging.info(f"Saved checkpoint at episode {episode}")
    
    # Final save
    logging.info("Training completed. Saving final agent and results...")
    save_agent(agent, save_dir, num_episodes, level)
    save_training_results(all_results, results_dir, level)
    
    # Close environment
    env.close()
    
    logging.info(f"Training complete! Results saved to {output_path}")
    logging.info(f"Log file: {log_file}")
    
    return agent, all_results


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description="Train OptionCritic agent on Thin Ice")
    
    # Environment parameters
    parser.add_argument("--level", type=int, default=1, help="Level number (1-19)")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes")
    
    # Agent parameters
    parser.add_argument("--n_options", type=int, default=4, help="Number of options")
    parser.add_argument("--n_states", type=int, default=1000, help="Initial estimate of number of states")
    parser.add_argument("--n_actions", type=int, default=4, help="Number of actions")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--alpha_critic", type=float, default=0.5, help="Critic learning rate")
    parser.add_argument("--alpha_theta", type=float, default=0.25, help="Intra-option policy learning rate")
    parser.add_argument("--alpha_upsilon", type=float, default=0.25, help="Termination function learning rate")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Exploration parameter")
    parser.add_argument("--n_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for option policy")
    
    # Training parameters
    parser.add_argument("--save_frequency", type=int, default=10, help="Save agent every N episodes")
    parser.add_argument("--output_dir", type=str, default="training_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    # Train agent
    agent, results = train_agent(
        level=args.level,
        num_episodes=args.num_episodes,
        n_options=args.n_options,
        n_states=args.n_states,
        n_actions=args.n_actions,
        gamma=args.gamma,
        alpha_critic=args.alpha_critic,
        alpha_theta=args.alpha_theta,
        alpha_upsilon=args.alpha_upsilon,
        epsilon=args.epsilon,
        n_steps=args.n_steps,
        temperature=args.temperature,
        save_frequency=args.save_frequency,
        output_dir=args.output_dir,
        verbose=verbose,
    )
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total episodes: {len(results)}")
        if results:
            avg_steps = np.mean([r['total_steps'] for r in results])
            avg_reward = np.mean([r['total_reward'] for r in results])
            avg_options = np.mean([r['num_options_used'] for r in results])
            completion_rate = np.mean([1 if r['terminated'] else 0 for r in results])
            print(f"Average steps per episode: {avg_steps:.2f}")
            print(f"Average reward per episode: {avg_reward:.2f}")
            print(f"Average options used per episode: {avg_options:.2f}")
            print(f"Completion rate: {completion_rate*100:.1f}%")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
