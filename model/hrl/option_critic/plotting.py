"""
Plotting utilities for OptionCritic training results.

This module provides functions to visualize training results, option usage,
Q-values, and other metrics from OptionCritic training.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_training_results(results_file: Path) -> Dict:
    """Load training results from JSON file
    
    Args:
        results_file: Path to training results JSON file
        
    Returns:
        Dictionary containing training results
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def load_agent(agent_file: Path) -> Dict:
    """Load agent state from JSON file
    
    Args:
        agent_file: Path to agent JSON file
        
    Returns:
        Dictionary containing agent state
    """
    with open(agent_file, 'r') as f:
        return json.load(f)


def plot_training_metrics(results: Dict, save_path: Optional[Path] = None, show: bool = True):
    """Plot training metrics over episodes
    
    Args:
        results: Training results dictionary
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    episodes_data = results['episodes']
    
    episodes = [e['episode'] for e in episodes_data]
    rewards = [e['total_reward'] for e in episodes_data]
    steps = [e['total_steps'] for e in episodes_data]
    options_used = [e['num_options_used'] for e in episodes_data]
    option_switches = [e['num_options_switches'] for e in episodes_data]
    completed = [1 if e['terminated'] else 0 for e in episodes_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics - Level {results["level"]}', fontsize=16, fontweight='bold')
    
    # Reward over episodes
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.6, linewidth=1)
    ax.plot(episodes, np.convolve(rewards, np.ones(10)/10, mode='same'), 
            color='red', linewidth=2, label='Moving Average (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps over episodes
    ax = axes[0, 1]
    ax.plot(episodes, steps, alpha=0.6, linewidth=1, color='green')
    ax.plot(episodes, np.convolve(steps, np.ones(10)/10, mode='same'), 
            color='darkgreen', linewidth=2, label='Moving Average (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Options used per episode
    ax = axes[0, 2]
    ax.plot(episodes, options_used, alpha=0.6, linewidth=1, color='purple')
    ax.plot(episodes, np.convolve(options_used, np.ones(10)/10, mode='same'), 
            color='darkviolet', linewidth=2, label='Moving Average (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Options Used')
    ax.set_title('Options Used per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Option switches per episode
    ax = axes[1, 0]
    ax.plot(episodes, option_switches, alpha=0.6, linewidth=1, color='orange')
    ax.plot(episodes, np.convolve(option_switches, np.ones(10)/10, mode='same'), 
            color='darkorange', linewidth=2, label='Moving Average (10)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Option Switches')
    ax.set_title('Option Switches per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Completion rate
    ax = axes[1, 1]
    completion_rate = np.convolve(completed, np.ones(10)/10, mode='same')
    ax.plot(episodes, completion_rate, linewidth=2, color='teal')
    ax.axhline(y=results['summary']['completion_rate'], color='red', 
               linestyle='--', label=f'Overall: {results["summary"]["completion_rate"]:.1%}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Completion Rate')
    ax.set_title('Completion Rate (Moving Average)')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Summary Statistics:
    
    Total Episodes: {results['summary']['total_episodes']}
    Avg Reward: {results['summary']['avg_reward_per_episode']:.2f}
    Max Reward: {results['summary']['max_reward']:.2f}
    Min Reward: {results['summary']['min_reward']:.2f}
    
    Avg Steps: {results['summary']['avg_steps_per_episode']:.1f}
    Avg Options Used: {results['summary']['avg_options_per_episode']:.2f}
    Avg Option Switches: {results['summary']['avg_option_switches']:.2f}
    
    Completion Rate: {results['summary']['completion_rate']:.1%}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_option_usage(results: Dict, save_path: Optional[Path] = None, show: bool = True):
    """Plot option usage analysis
    
    Args:
        results: Training results dictionary
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    episodes_data = results['episodes']
    
    # Collect option usage statistics
    option_counts = Counter()
    option_transitions = defaultdict(int)
    option_durations = defaultdict(list)
    
    for episode in episodes_data:
        option_sequence = episode['option_sequence']
        action_sequence = episode['action_sequence']
        
        # Count option usage
        for opt in option_sequence:
            option_counts[opt] += 1
        
        # Count transitions
        for i in range(len(option_sequence) - 1):
            transition = (option_sequence[i], option_sequence[i+1])
            option_transitions[transition] += 1
        
        # Calculate option durations (number of actions per option)
        for opt_idx, actions in action_sequence.items():
            opt_idx = int(opt_idx)
            option_durations[opt_idx].append(len(actions))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Option Usage Analysis - Level {results["level"]}', fontsize=16, fontweight='bold')
    
    # Option frequency
    ax = axes[0, 0]
    options = sorted(option_counts.keys())
    counts = [option_counts[opt] for opt in options]
    colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
    bars = ax.bar([f'Option {opt}' for opt in options], counts, color=colors)
    ax.set_xlabel('Option')
    ax.set_ylabel('Usage Count')
    ax.set_title('Option Usage Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Option transitions heatmap
    ax = axes[0, 1]
    if option_transitions:
        max_opt = max(max(t) for t in option_transitions.keys())
        transition_matrix = np.zeros((max_opt + 1, max_opt + 1))
        for (from_opt, to_opt), count in option_transitions.items():
            transition_matrix[from_opt, to_opt] = count
        
        im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('To Option')
        ax.set_ylabel('From Option')
        ax.set_title('Option Transition Matrix')
        ax.set_xticks(range(max_opt + 1))
        ax.set_yticks(range(max_opt + 1))
        ax.set_xticklabels([f'Opt {i}' for i in range(max_opt + 1)])
        ax.set_yticklabels([f'Opt {i}' for i in range(max_opt + 1)])
        
        # Add text annotations
        for i in range(max_opt + 1):
            for j in range(max_opt + 1):
                text = ax.text(j, i, int(transition_matrix[i, j]),
                             ha="center", va="center", color="black" if transition_matrix[i, j] < transition_matrix.max()/2 else "white")
        
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No transitions', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Option Transition Matrix')
    
    # Option duration distribution
    ax = axes[1, 0]
    if option_durations:
        positions = []
        data = []
        labels = []
        for opt in sorted(option_durations.keys()):
            positions.append(opt)
            data.append(option_durations[opt])
            labels.append(f'Option {opt}')
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Option')
        ax.set_ylabel('Duration (Actions)')
        ax.set_title('Option Duration Distribution')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Option Duration Distribution')
    
    # Option usage over episodes
    ax = axes[1, 1]
    episodes = [e['episode'] for e in episodes_data]
    option_usage_over_time = defaultdict(list)
    
    for episode in episodes_data:
        episode_num = episode['episode']
        option_sequence = episode['option_sequence']
        unique_options = set(option_sequence)
        for opt in range(max(option_counts.keys()) + 1):
            option_usage_over_time[opt].append(1 if opt in unique_options else 0)
    
    for opt in sorted(option_usage_over_time.keys()):
        ax.plot(episodes, np.convolve(option_usage_over_time[opt], np.ones(10)/10, mode='same'),
               label=f'Option {opt}', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Usage (Moving Average)')
    ax.set_title('Option Usage Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_q_values(agent: Dict, save_path: Optional[Path] = None, show: bool = True, 
                  max_states: int = 50):
    """Plot Q-value analysis
    
    Args:
        agent: Agent state dictionary
        save_path: Optional path to save the figure
        show: Whether to display the plot
        max_states: Maximum number of states to display (for readability)
    """
    q_omega = np.array(agent['Q_Omega_table'])
    q_u = np.array(agent['Q_U_table'])
    
    n_states, n_options = q_omega.shape
    n_actions = q_u.shape[2]
    
    # Limit states for visualization
    states_to_plot = min(n_states, max_states)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Q-Value Analysis - Episode {agent["episode"]}, Level {agent["level"]}', 
                 fontsize=16, fontweight='bold')
    
    # Q_Omega heatmap (state x option)
    ax = axes[0, 0]
    q_omega_plot = q_omega[:states_to_plot, :]
    im = ax.imshow(q_omega_plot, cmap='viridis', aspect='auto')
    ax.set_xlabel('Option')
    ax.set_ylabel('State Index')
    ax.set_title(f'Q_Omega: State-Option Values (first {states_to_plot} states)')
    ax.set_xticks(range(n_options))
    ax.set_xticklabels([f'Opt {i}' for i in range(n_options)])
    plt.colorbar(im, ax=ax, label='Q-Value')
    
    # Q_Omega statistics per option
    ax = axes[0, 1]
    option_means = np.mean(q_omega, axis=0)
    option_stds = np.std(q_omega, axis=0)
    x_pos = np.arange(n_options)
    bars = ax.bar(x_pos, option_means, yerr=option_stds, capsize=5, 
                  color=plt.cm.Set3(np.linspace(0, 1, n_options)), alpha=0.7)
    ax.set_xlabel('Option')
    ax.set_ylabel('Mean Q-Value')
    ax.set_title('Q_Omega: Mean Q-Value per Option')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Option {i}' for i in range(n_options)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, option_means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + option_stds[i],
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Q_U statistics per option-action pair
    ax = axes[1, 0]
    q_u_means = np.mean(q_u, axis=0)  # Average over states: (n_options, n_actions)
    im = ax.imshow(q_u_means, cmap='plasma', aspect='auto')
    ax.set_xlabel('Action')
    ax.set_ylabel('Option')
    ax.set_title('Q_U: Mean State-Option-Action Values')
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels([f'Action {i}' for i in range(n_actions)])
    ax.set_yticks(range(n_options))
    ax.set_yticklabels([f'Option {i}' for i in range(n_options)])
    
    # Add text annotations
    for i in range(n_options):
        for j in range(n_actions):
            text = ax.text(j, i, f'{q_u_means[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if q_u_means[i, j] < q_u_means.max()/2 else "black",
                         fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Mean Q-Value')
    
    # Q-value distribution
    ax = axes[1, 1]
    q_omega_flat = q_omega.flatten()
    q_u_flat = q_u.flatten()
    
    ax.hist(q_omega_flat, bins=50, alpha=0.6, label='Q_Omega', color='blue', density=True)
    ax.hist(q_u_flat, bins=50, alpha=0.6, label='Q_U', color='red', density=True)
    ax.set_xlabel('Q-Value')
    ax.set_ylabel('Density')
    ax.set_title('Q-Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_option_sequences(results: Dict, num_episodes: int = 10, 
                         save_path: Optional[Path] = None, show: bool = True):
    """Plot option sequences for selected episodes
    
    Args:
        results: Training results dictionary
        num_episodes: Number of episodes to visualize
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    episodes_data = results['episodes']
    episodes_to_plot = min(num_episodes, len(episodes_data))
    
    # Select episodes evenly spaced
    if episodes_to_plot < len(episodes_data):
        indices = np.linspace(0, len(episodes_data) - 1, episodes_to_plot, dtype=int)
        selected_episodes = [episodes_data[i] for i in indices]
    else:
        selected_episodes = episodes_data
    
    fig, axes = plt.subplots(episodes_to_plot, 1, figsize=(14, 2 * episodes_to_plot))
    if episodes_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(f'Option Sequences - Level {results["level"]}', fontsize=16, fontweight='bold')
    
    # Get all unique options for consistent coloring
    all_options = set()
    for episode in episodes_data:
        all_options.update(episode['option_sequence'])
    all_options = sorted(all_options)
    color_map = {opt: plt.cm.Set3(i / len(all_options)) for i, opt in enumerate(all_options)}
    
    for idx, (episode, ax) in enumerate(zip(selected_episodes, axes)):
        option_sequence = episode['option_sequence']
        episode_num = episode['episode']
        
        # Create timeline plot
        y_pos = 0.5
        x_start = 0
        current_option = option_sequence[0]
        
        for i, opt in enumerate(option_sequence):
            if opt != current_option or i == len(option_sequence) - 1:
                # Draw segment for previous option
                width = i - x_start + (1 if i == len(option_sequence) - 1 else 0)
                ax.barh(y_pos, width, left=x_start, height=0.4, 
                       color=color_map[current_option], alpha=0.7, edgecolor='black', linewidth=1)
                ax.text(x_start + width/2, y_pos, f'Opt {current_option}', 
                       ha='center', va='center', fontweight='bold', fontsize=8)
                x_start = i
                current_option = opt
        
        ax.set_xlim([0, len(option_sequence)])
        ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.set_xlabel('Step' if idx == episodes_to_plot - 1 else '')
        ax.set_ylabel(f'Ep {episode_num}', rotation=0, ha='right', va='center')
        ax.set_title(f'Episode {episode_num} - {len(option_sequence)} steps, '
                    f'{len(set(option_sequence))} unique options', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(description="Plot OptionCritic training results")
    
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to training results JSON file")
    parser.add_argument("--agent_file", type=str, default=None,
                       help="Path to agent JSON file (for Q-value plots)")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (only save)")
    parser.add_argument("--plots", type=str, nargs="+", 
                       choices=['metrics', 'options', 'qvalues', 'sequences', 'all'],
                       default=['all'],
                       help="Which plots to generate")
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    # Load results
    print(f"Loading results from {results_file}...")
    results = load_training_results(results_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    level = results['level']
    show = not args.no_show
    
    # Generate plots
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['metrics', 'options', 'qvalues', 'sequences']
    
    if 'metrics' in plots_to_generate:
        print("Generating training metrics plot...")
        plot_training_metrics(
            results, 
            save_path=output_dir / f"training_metrics_level_{level}.png",
            show=show
        )
    
    if 'options' in plots_to_generate:
        print("Generating option usage plot...")
        plot_option_usage(
            results,
            save_path=output_dir / f"option_usage_level_{level}.png",
            show=show
        )
    
    if 'qvalues' in plots_to_generate:
        if args.agent_file:
            agent_file = Path(args.agent_file)
            if agent_file.exists():
                print(f"Loading agent from {agent_file}...")
                agent = load_agent(agent_file)
                print("Generating Q-value analysis plot...")
                plot_q_values(
                    agent,
                    save_path=output_dir / f"q_values_episode_{agent['episode']}_level_{level}.png",
                    show=show
                )
            else:
                print(f"Warning: Agent file not found: {agent_file}. Skipping Q-value plot.")
        else:
            print("Warning: No agent file provided. Skipping Q-value plot.")
    
    if 'sequences' in plots_to_generate:
        print("Generating option sequences plot...")
        plot_option_sequences(
            results,
            num_episodes=10,
            save_path=output_dir / f"option_sequences_level_{level}.png",
            show=show
        )
    
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()