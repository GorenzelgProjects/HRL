"""
Training script for OptionCritic agent on Thin Ice environment.

This script trains an OptionCritic agent for a specified number of episodes
and saves the trained agent, options, and training results.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from loguru import logger as logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.thin_ice.thin_ice_env import ThinIceEnv
from model.hrl.option_critic.option_critic import OptionCritic
from model.hrl.option_critic.state_manager import StateManager


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
        "episode": episode,
        "level": level,
        "n_states": agent.n_states,
        "n_actions": agent.n_actions,
        "n_options": agent.n_options,
        "gamma": agent.gamma,
        "alpha_critic": agent.alpha_critic,
        "alpha_theta": agent.alpha_theta,
        "alpha_upsilon": agent.alpha_upsilon,
        "epsilon": agent.epsilon,
        "n_steps": agent.n_steps,
        "n_unique_states": agent.state_manager.n_unique_states,
        "Q_Omega_table": agent.Q_Omega_table.detach().cpu().numpy().tolist(),
        "Q_U_table": agent.Q_U_table.detach().cpu().numpy().tolist(),
    }

    # Save options (theta and upsilon for each option)
    options_data = []
    for option in agent.options:
        options_data.append(
            {
                "idx": option.idx,
                "theta": option.theta.detach().cpu().numpy().tolist(),
                "upsilon": option.upsilon.detach().cpu().numpy().tolist(),
            }
        )
    agent_state["options"] = options_data

    # Save to JSON
    agent_file = save_dir / f"agent_episode_{episode}_level_{level}.json"
    with open(agent_file, "w") as f:
        json.dump(agent_state, f, indent=2)

    # Also save as PyTorch state dict for easier loading
    torch_file = save_dir / f"agent_episode_{episode}_level_{level}.pt"
    torch.save(
        {
            "Q_Omega_table": agent.Q_Omega_table,
            "Q_U_table": agent.Q_U_table,
            "options": {
                opt.idx: {"theta": opt.theta, "upsilon": opt.upsilon}
                for opt in agent.options
            },
            "config": {
                "n_states": agent.n_states,
                "n_actions": agent.n_actions,
                "n_options": agent.n_options,
                "gamma": agent.gamma,
                "alpha_critic": agent.alpha_critic,
                "alpha_theta": agent.alpha_theta,
                "alpha_upsilon": agent.alpha_upsilon,
                "epsilon": agent.epsilon,
                "n_steps": agent.n_steps,
            },
        },
        torch_file,
    )

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
        "level": level,
        "timestamp": datetime.now().isoformat(),
        "num_episodes": len(results),
        "episodes": results,
        "summary": {
            "avg_steps_per_episode": (
                float(np.mean([r["total_steps"] for r in results])) if results else 0
            ),
            "avg_reward_per_episode": (
                float(np.mean([r["total_reward"] for r in results])) if results else 0
            ),
            "max_reward": (
                float(np.max([r["total_reward"] for r in results])) if results else 0
            ),
            "min_reward": (
                float(np.min([r["total_reward"] for r in results])) if results else 0
            ),
            "avg_options_per_episode": (
                float(np.mean([r["num_options_used"] for r in results]))
                if results
                else 0
            ),
            "avg_option_switches": (
                float(np.mean([r["num_options_switches"] for r in results]))
                if results
                else 0
            ),
            "completion_rate": (
                float(np.mean([1 if r["terminated"] else 0 for r in results]))
                if results
                else 0
            ),
            "total_episodes": len(results),
        },
    }

    with open(results_file, "w") as f:
        json.dump(training_data, f, indent=2)

    logging.info(f"Saved training results to {results_file}")


def train_agent(
    env: ThinIceEnv,
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
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    n_steps: int = 1000,
    temperature: float = 1.0,
    save_frequency: int = 10,
    output_dir: str = "training_output",
    state_mapping_dir: str = "environment/state_mapping",
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
        state_mapping_dir: Directory to where the state_to_idx dict is stored for a level
        reward_config: Directory to where the env config is stored
        verbose: Whether to print detailed logs

    Returns:
        Tuple of (trained agent, list of episode results)
    """
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dir = output_path / "agents"
    results_dir = output_path / "results"

    # Create state manager
    state_manager = StateManager(Path(state_mapping_dir))

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
        state_manager=state_manager,
    )

    # Training loop
    all_results = []

    for episode in range(1, num_episodes + 1):
        logging.info(f"Episode {episode}/{num_episodes}")

        # Train one episode
        episode_stats = agent.train(env, temperature, save_mapping=True)
        episode_stats["episode"] = episode
        all_results.append(episode_stats)
        
        # Decay the exploration parameter
        agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)
        logging.debug(f"Decayed epsilon to {agent.epsilon}")

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

    return agent, all_results
