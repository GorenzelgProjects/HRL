"""
Training script for OptionCritic agent on Thin Ice environment.

This script trains an OptionCritic agent for a specified number of episodes
and saves the trained agent, options, and training results.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import tqdm
from loguru import logger as logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.thin_ice.thin_ice_env import ThinIceEnv
from model.hrl.option_critic.option_critic import OptionCritic
from model.hrl.option_critic.state_manager import StateManager

# Addons imports (optional)
try:
    from addons.simple_sf.integration import (
        create_simple_sf_module,
        compute_sf_losses,
        create_sf_optimizers
    )
    from addons.utils.rollout_collector import RolloutCollector
    ADDONS_AVAILABLE = True
except ImportError:
    ADDONS_AVAILABLE = False


def save_sf_components(sf_module, w_task, output_path: Path, level: int):
    """Save Simple-SF components (encoder + SF heads + w_task) for transfer learning.
    
    Args:
        sf_module: SimpleSF module to save
        w_task: Task vector parameter to save
        output_path: Base output directory (will save to output_path/agents/)
        level: Level number
    """
    if sf_module is None or w_task is None:
        return
    
    save_dir = output_path / "agents"
    save_dir.mkdir(parents=True, exist_ok=True)
    sf_file = save_dir / f"sf_components_level_{level}.pt"
    
    torch.save({
        'encoder_state_dict': sf_module.encoder.state_dict(),
        'phi_state_dict': sf_module.phi.state_dict(),
        'psi_state_dict': sf_module.psi.state_dict(),
        'w_task': w_task.data,
        'sf_d': sf_module.d_sf,
        'level': level
    }, sf_file)
    
    logging.info(f"Saved SF components to {sf_file}")


def load_sf_components(load_dir: Path, level: int, device, num_actions: int, sf_d: int):
    """Load Simple-SF components from a previous level for transfer learning.
    
    Args:
        load_dir: Directory containing saved SF components
        level: Level number to load from
        device: Device to load components on
        num_actions: Number of actions
        sf_d: Successor features dimension (will use saved value if available)
    
    Returns:
        Tuple of (sf_module, w_task) or (None, None) if not found
    """
    sf_file = load_dir / f"sf_components_level_{level}.pt"
    
    if not sf_file.exists():
        logging.warning(f"SF components file not found: {sf_file}")
        return None, None
    
    try:
        checkpoint = torch.load(sf_file, map_location=device)
        
        # Use saved sf_d if available, otherwise use provided value
        saved_sf_d = checkpoint.get('sf_d', sf_d)
        
        # Create new SF module with same architecture
        from addons.simple_sf.integration import create_simple_sf_module
        sf_module, _ = create_simple_sf_module(
            num_actions=num_actions,
            z_dim=256,
            d_sf=saved_sf_d,
            grid_height=17,
            grid_width=19,
            device=device
        )
        
        # Load state dicts
        sf_module.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        sf_module.phi.load_state_dict(checkpoint['phi_state_dict'])
        sf_module.psi.load_state_dict(checkpoint['psi_state_dict'])
        
        # Create w_task parameter and load its value
        import torch.nn as nn
        w_task = nn.Parameter(checkpoint['w_task'].to(device))
        
        logging.info(f"Loaded SF components from level {level} for transfer learning (sf_d={saved_sf_d})")
        return sf_module, w_task
        
    except Exception as e:
        logging.error(f"Error loading SF components: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None


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
    render: bool = False,
    delay: bool = 0.05,
    # Simple-SF parameters
    use_simple_sf: bool = False,
    sf_d: int = 256,
    lambda_sf: float = 0.1,
    alpha_w: float = 0.1,
    sf_lr_main: float = 1e-3,
    sf_lr_w: float = 1e-2,
    # Transfer learning: shared components (None = create new, not None = reuse)
    shared_sf_module = None,
    shared_w_task = None,
    freeze_encoder: bool = False,  # If True, freeze encoder on later levels
    transfer_from_level: Optional[int] = None,  # Level to load components from (if specified)
    # Rollout collection parameters
    collect_rollouts: bool = False,
    rollout_save_dir: Optional[str] = None,
) -> Tuple[OptionCritic, List[Dict], Optional[Dict]]:
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

    # Initialize Simple-SF if enabled
    sf_module = None
    w_task = None
    sf_opt_main = None
    sf_opt_w = None
    encoder_target = None
    sf_target = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_simple_sf and ADDONS_AVAILABLE:
        # Check if we should load from a saved level
        if transfer_from_level is not None and shared_sf_module is None:
            logging.info(f"Loading SF components from level {transfer_from_level} for transfer learning...")
            loaded_sf, loaded_w_task = load_sf_components(
                load_dir=output_path / "agents",
                level=transfer_from_level,
                device=device,
                num_actions=n_actions,
                sf_d=sf_d
            )
            if loaded_sf is not None:
                shared_sf_module = loaded_sf
                shared_w_task = loaded_w_task
        
        if shared_sf_module is not None:
            # Transfer learning: reuse shared encoder + SF heads
            logging.info(f"Using shared Simple-SF module for level {level} (transfer learning)")
            sf_module = shared_sf_module
            
            # Use provided w_task or create new one for this level
            if shared_w_task is not None:
                w_task = shared_w_task
                logging.info(f"Using provided w_task for level {level}")
            else:
                # Create new w_task for this level
                import torch.nn as nn
                w_task = nn.Parameter(torch.zeros(sf_d, device=device))
                logging.info(f"Created new w_task for level {level}")
            
            # Freeze encoder and SF heads if requested
            if freeze_encoder:
                for param in sf_module.encoder.parameters():
                    param.requires_grad = False
                for param in sf_module.phi.parameters():
                    param.requires_grad = False
                for param in sf_module.psi.parameters():
                    param.requires_grad = False
                logging.info("Frozen encoder and SF heads (transfer learning mode)")
                # Only create optimizer for w_task
                sf_opt_main = None
                sf_opt_w = torch.optim.Adam([w_task], lr=sf_lr_w)
            else:
                # Fine-tune mode: update everything but with lower learning rate
                sf_opt_main, sf_opt_w = create_sf_optimizers(
                    sf_module, w_task, lr_main=sf_lr_main * 0.1, lr_w=sf_lr_w  # Lower LR for fine-tuning
                )
                logging.info("Fine-tuning encoder and SF heads with reduced learning rate")
            
            # Reuse or create target networks
            encoder_target = None  # Can reuse from previous level if needed
        else:
            # Standard: create new for this level
            logging.info(f"Initializing Simple Successor Features addon for level {level}...")
            sf_module, w_task = create_simple_sf_module(
                num_actions=n_actions,
                z_dim=256,
                d_sf=sf_d,
                grid_height=env.grid_height if hasattr(env, 'grid_height') else 17,
                grid_width=env.grid_width if hasattr(env, 'grid_width') else 19,
                device=device
            )
            sf_opt_main, sf_opt_w = create_sf_optimizers(
                sf_module, w_task, lr_main=sf_lr_main, lr_w=sf_lr_w
            )
            # Create target networks (EMA-style updates)
            encoder_target = type(sf_module.encoder)(
                z_dim=256,
                grid_height=env.grid_height if hasattr(env, 'grid_height') else 17,
                grid_width=env.grid_width if hasattr(env, 'grid_width') else 19
            ).to(device)
            encoder_target.load_state_dict(sf_module.encoder.state_dict())
            encoder_target.eval()
            logging.info("Simple-SF initialized")

    # Initialize rollout collector if enabled
    rollout_collector = None
    if collect_rollouts and ADDONS_AVAILABLE:
        rollout_collector = RolloutCollector()
        logging.info("Rollout collection enabled")

    # Training loop
    all_results = []
    pbar = tqdm.tqdm(range(1, num_episodes + 1))
    
    # Replay buffer for Simple-SF (collect transitions for batch updates)
    sf_replay_buffer = []
    sf_batch_size = 32
    
    for episode in pbar:
        # logging.info(f"Episode {episode}/{num_episodes}")

        # Train one episode (with observation collection if enabled)
        encoder_for_collection = sf_module.encoder if (use_simple_sf and ADDONS_AVAILABLE) else None
        episode_stats = agent.train(
            env,
            temperature,
            save_mapping=True,
            render=render if episode == num_episodes - 1 else False,
            delay=delay,
            observation_collector=rollout_collector if collect_rollouts and ADDONS_AVAILABLE else None,
            encoder=encoder_for_collection
        )
        
        episode_stats["episode"] = episode
        
        # Add SF loss info to episode stats if available (will be updated after SF update)
        all_results.append(episode_stats)
        
        # Collect transitions for Simple-SF from rollout collector
        if use_simple_sf and ADDONS_AVAILABLE and rollout_collector:
            rollouts = rollout_collector.get_rollouts()
            if len(rollouts) > 0:
                # Get the last episode's transitions
                last_episode = rollouts[-1]
                obs_list = last_episode.get('obs', [])
                action_list = last_episode.get('action_sequence', [])
                if len(obs_list) > 1 and len(action_list) > 0:
                    # device is already defined above
                    for i in range(len(obs_list) - 1):
                        if i >= len(action_list):
                            break
                        obs = torch.from_numpy(obs_list[i]).float().unsqueeze(0).to(device)
                        next_obs = torch.from_numpy(obs_list[i+1]).float().unsqueeze(0).to(device)
                        action = action_list[i]
                        reward = last_episode['rewards'][i] if i < len(last_episode['rewards']) else 0.0
                        done = last_episode['dones'][i] if i < len(last_episode['dones']) else False
                        
                        sf_replay_buffer.append({
                            'obs': obs,
                            'action': action,
                            'reward': reward,
                            'next_obs': next_obs,
                            'done': done
                        })
        
        # Update Simple-SF if enabled and we have enough transitions
        sf_losses_computed = False
        if use_simple_sf and ADDONS_AVAILABLE and len(sf_replay_buffer) >= sf_batch_size:
            # Sample batch
            batch_indices = np.random.choice(len(sf_replay_buffer), size=sf_batch_size, replace=False)
            batch = {
                'obs': torch.stack([sf_replay_buffer[i]['obs'] for i in batch_indices]).to(device),
                'action': torch.tensor([sf_replay_buffer[i]['action'] for i in batch_indices], dtype=torch.long).to(device),
                'reward': torch.tensor([sf_replay_buffer[i]['reward'] for i in batch_indices], dtype=torch.float32).to(device),
                'next_obs': torch.stack([sf_replay_buffer[i]['next_obs'] for i in batch_indices]).to(device),
                'done': torch.tensor([sf_replay_buffer[i]['done'] for i in batch_indices], dtype=torch.float32).to(device),
            }
            
            # Compute SF losses
            sf_losses = compute_sf_losses(
                sf_module, w_task, batch, gamma=gamma,
                encoder_target=encoder_target, sf_target=sf_target
            )
            sf_losses_computed = True
            
            # Update main network (only if optimizer exists, i.e., not frozen)
            total_sf_loss = sf_losses['L_psi'] + alpha_w * sf_losses['L_w']
            
            # Log SF losses periodically
            if episode % 10 == 0:
                transfer_mode = " (transfer)" if freeze_encoder or shared_sf_module is not None else ""
                logging.info(
                    f"Episode {episode}: SF Losses{transfer_mode} - "
                    f"L_psi={sf_losses['L_psi'].item():.4f}, "
                    f"L_w={sf_losses['L_w'].item():.4f}, "
                    f"Total_SF={total_sf_loss.item():.4f}, "
                    f"w_task_norm={torch.norm(w_task).item():.4f}"
                )
            
            if sf_opt_main is not None:
                sf_opt_main.zero_grad()
                total_sf_loss.backward()
                sf_opt_main.step()
            elif freeze_encoder:
                # In transfer mode with frozen encoder, only L_w contributes
                pass  # w_task is updated separately below
            
            # Update task vector (only from L_w)
            sf_opt_w.zero_grad()
            sf_losses['L_w'].backward()
            sf_opt_w.step()
            
            # Update target networks (EMA-style, every few steps)
            if episode % 10 == 0 and encoder_target is not None:
                tau = 0.005  # Soft update coefficient
                for target_param, param in zip(encoder_target.parameters(), sf_module.encoder.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Update episode stats with SF loss info if computed
        if sf_losses_computed:
            all_results[-1]['sf_L_psi'] = float(sf_losses['L_psi'].item())
            all_results[-1]['sf_L_w'] = float(sf_losses['L_w'].item())
            all_results[-1]['sf_w_task_norm'] = float(torch.norm(w_task).item())
            all_results[-1]['sf_transfer_mode'] = freeze_encoder or shared_sf_module is not None
        
        # Save rollouts if enabled
        if collect_rollouts and rollout_collector and episode % save_frequency == 0:
            rollouts = rollout_collector.get_rollouts()
            if rollout_save_dir and len(rollouts) > 0:
                rollout_path = Path(rollout_save_dir)
                rollout_path.mkdir(parents=True, exist_ok=True)
                rollout_file = rollout_path / f"rollouts_level_{level}_episode_{episode}.npz"
                
                # Prepare data for saving - handle variable-length sequences
                save_data = {}
                for key in rollouts[0].keys():
                    if key == 'info':
                        continue  # Skip info dicts (can't be easily serialized to NPZ)
                    values = [r[key] for r in rollouts]
                    
                    # For variable-length sequences (like obs lists), save as object array
                    # For fixed-length arrays, try to stack them
                    try:
                        # First, try to see if we can stack (all same length)
                        if key in ['obs', 'z', 'agent_pos'] and len(values) > 0:
                            # Check if all have same length
                            lengths = [len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in values]
                            if len(set(lengths)) == 1:
                                # All same length - try to stack
                                if isinstance(values[0], np.ndarray):
                                    # Try to stack arrays
                                    try:
                                        save_data[key] = np.stack(values)
                                    except ValueError:
                                        # Can't stack - save as object array
                                        save_data[key] = np.array(values, dtype=object)
                                else:
                                    # Convert lists to arrays first
                                    arr_values = [np.array(v) for v in values]
                                    try:
                                        save_data[key] = np.stack(arr_values)
                                    except ValueError:
                                        save_data[key] = np.array(values, dtype=object)
                            else:
                                # Different lengths - save as object array
                                save_data[key] = np.array(values, dtype=object)
                        else:
                            # For scalar or simple arrays, try direct conversion
                            arr = np.array(values)
                            save_data[key] = arr
                    except (ValueError, TypeError) as e:
                        # If anything fails, save as object array
                        save_data[key] = np.array(values, dtype=object)
                
                # Save as compressed numpy arrays (object arrays are supported)
                np.savez_compressed(rollout_file, **save_data)
                logging.info(f"Saved {len(rollouts)} rollouts to {rollout_file}")
        
        # Decay the exploration parameter
        agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)
        logging.debug(f"Decayed epsilon to {agent.epsilon}")

        pbar.set_postfix(dict(R=episode_stats['total_reward'], steps=episode_stats['total_steps']))
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

    # Save SF components for potential transfer learning
    if use_simple_sf and ADDONS_AVAILABLE and sf_module is not None:
        save_sf_components(sf_module, w_task, output_path, level)
    
    # Return agent, results, and SF components for transfer learning
    sf_components = None
    if use_simple_sf and ADDONS_AVAILABLE:
        sf_components = {
            'sf_module': sf_module,
            'w_task': w_task,
            'level': level
        }
    
    return agent, all_results, sf_components
