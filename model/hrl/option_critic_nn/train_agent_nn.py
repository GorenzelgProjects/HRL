"""
Training script for OptionCritic agent on Thin Ice environment.

This script trains an OptionCritic agent for a specified number of episodes
and saves the trained agent, options, and training results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import tqdm
from loguru import logger as logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.thin_ice.thin_ice_env import ThinIceEnv
from model.hrl.option_critic_nn.oc_network import Encoder
from model.hrl.option_critic_nn.option_critic_nn import OptionCritic


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
        "epsilon": agent.epsilon,
        "n_steps": agent.n_steps,
        "beta_reg": agent.beta_reg,
        "entropy_reg": agent.entropy_reg,
        "img_size": tuple(agent.downsample_size),
    }

    # Save to JSON
    agent_file = save_dir / f"agent_episode_{episode}_level_{level}.json"
    with open(agent_file, "w") as f:
        json.dump(agent_state, f, indent=2)

    # Save encoder state dict
    torch_file = save_dir / f"agent_episode_{episode}_level_{level}.pt"
    encoder = agent.encoder
    torch.save(encoder, torch_file)

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


def train_option_critic_nn(
    env: ThinIceEnv,
    img_size: list[int, int],
    level: int = 1,
    num_episodes: int = 100,
    n_options: int = 4,
    n_states: int = 1000,
    n_actions: int = 4,
    n_filters: list[int] = [32, 64, 64],
    conv_sizes: list[int] = [8, 4, 3],
    strides: list[int] = [4, 2, 1],
    n_neurons: int = 512,
    optimizer_name: str = "RMSProp",
    gamma: float = 0.99,
    lr: float = 0.00025,
    epsilon: float = 0.9,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    beta_reg: float = 0.01,
    entropy_reg: float = 0.01,
    n_steps: int = 1000,
    temperature: float = 1.0,
    save_frequency: int = 10,
    output_dir: str = "training_output",
    verbose: bool = True,
    render: bool = False,
    delay: bool = 0.05,
    max_history: int = 10000,
    cuda: bool = False,
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
    # Set up device
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda")
    elif torch.mps.is_available() and cuda:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Running on {device}")

    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dir = output_path / "agents"
    results_dir = output_path / "results"

    # Initialize the encoder
    encoder = Encoder(
        image_size=img_size,
        n_filters=n_filters,
        conv_sizes=conv_sizes,
        strides=strides,
        n_neurons=n_neurons,
        n_options=n_options,
        n_actions=n_actions,
        temperature=temperature,
        device=device,
    )
    encoder.to(device)

    if optimizer_name == "RMSProp":
        optimizer = torch.optim.RMSprop(encoder.parameters(), lr=lr)
    else:
        raise ValueError("Cannot identify optimizer")

    agent = OptionCritic(
        n_states=n_states,
        n_actions=n_actions,
        n_options=n_options,
        encoder=encoder,
        gamma=gamma,
        epsilon_start=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        beta_reg=beta_reg,
        entropy_reg=entropy_reg,
        n_steps=n_steps,
        downsample=img_size,
        device=device,
    )

    # Training loop
    all_results = []
    pbar = tqdm.tqdm(range(1, num_episodes + 1))
    for episode in pbar:
        # Train one episode
        episode_stats = agent.train(
            env,
            optimizer=optimizer,
            render=render if episode == num_episodes - 1 else False,
            delay=delay,
            max_history=max_history,
        )

        episode_stats["episode"] = episode
        all_results.append(episode_stats)

        # NOTE: Changed to step decay instead
        # Decay the exploration parameter
        # agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)
        # logging.debug(f"Decayed epsilon to {agent.epsilon}")

        pbar.set_postfix(
            dict(R=episode_stats["total_reward"], steps=episode_stats["total_steps"])
        )
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
