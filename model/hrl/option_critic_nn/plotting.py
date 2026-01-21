"""
Plotting utilities for OptionCritic NN training results.

This module provides functions to visualize training results, option usage,
and other metrics from OptionCritic NN training.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import torch

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


def load_agent(agent_file: Path, encoder_file: Path) -> Dict:
    """Load agent state from JSON and encoder from PyTorch file
    
    Args:
        agent_file: Path to agent JSON file
        encoder_file: Path to encoder PyTorch file
        
    Returns:
        Dictionary containing agent state and encoder
    """
    with open(agent_file, 'r') as f:
        agent_state = json.load(f)
    
    encoder = torch.load(encoder_file, map_location='cpu')
    agent_state['encoder'] = encoder
    
    return agent_state


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
    
    # Empty subplot (removed completion rate)
    ax = axes[1, 1]
    ax.axis('off')
    
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
    
    # Only use last 5 episodes
    episodes_data = episodes_data[-5:] if len(episodes_data) > 5 else episodes_data
    
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
        
        # Calculate option durations (length of each action sequence before option switch)
        # action_sequence structure: {"0": {"0": [1,2,3], "1": [4,5]}, "1": {"0": [6,7]}}
        # We want the length of each innermost list
        for opt_idx, segments in action_sequence.items():
            opt_idx = int(opt_idx)
            # segments is a dict like {"0": [1,2,3], "1": [4,5]}
            for segment_key, action_list in segments.items():
                if isinstance(action_list, list):
                    option_durations[opt_idx].append(len(action_list))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Option Usage Analysis - Level {results["level"]}', fontsize=16, fontweight='bold')
    
    # Option frequency
    ax = axes[0]
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
    ax = axes[1]
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
    ax = axes[2]
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


def plot_encoder_output(agent: Dict, test_image: np.ndarray, 
                       save_path: Optional[Path] = None, show: bool = True):
    """Plot encoder output visualization for a test image
    
    Args:
        agent: Agent state dictionary with encoder
        test_image: Test grayscale image (H, W) or (1, H, W)
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    encoder = agent['encoder']
    encoder.eval()
    device = encoder.device
    
    # Prepare image - ensure it's (H, W) format for grayscale
    if test_image.ndim == 2:
        # Already (H, W) - good
        pass
    elif test_image.ndim == 3:
        if test_image.shape[0] == 1:
            # (1, H, W) -> (H, W)
            test_image = test_image.squeeze(0)
        elif test_image.shape[2] == 1:
            # (H, W, 1) -> (H, W)
            test_image = test_image.squeeze(2)
        else:
            raise ValueError(f"Expected grayscale image, got shape {test_image.shape}")
    
    # Ensure image is 84x84
    if test_image.shape != (84, 84):
        h, w = test_image.shape
        # Use simple resizing with numpy/scipy if available, otherwise use PIL
        try:
            from scipy.ndimage import zoom
            scale_h = 84 / h
            scale_w = 84 / w
            test_image = zoom(test_image, (scale_h, scale_w), order=1)
            test_image = test_image.astype(np.uint8)
        except ImportError:
            try:
                from PIL import Image
                img = Image.fromarray(test_image)
                img = img.resize((84, 84), Image.Resampling.LANCZOS)
                test_image = np.array(img)
            except ImportError:
                # Fallback: simple numpy-based resizing
                from scipy import ndimage
                scale_h = 84 / h
                scale_w = 84 / w
                test_image = ndimage.zoom(test_image, (scale_h, scale_w), order=1)
                test_image = test_image.astype(np.uint8)
        print(f"Resized image from {h}x{w} to {test_image.shape}")
    
    # Prepare as numpy array with channel dimension: (1, H, W) for grayscale
    # The encoder expects (channels, H, W) format
    image_np = test_image[np.newaxis, :, :].astype(np.float32)  # (1, 84, 84)
    
    with torch.no_grad():
        # Get encoder features using encode_state (which handles the conversion)
        features = encoder.encode_state(image_np)  # (1, n_neurons)
        features = features.squeeze(0).cpu().numpy()  # (n_neurons,)
        
        # Get option probabilities - pass numpy array so it uses encode_state
        option_logits = encoder.pi_options(image_np)  # (1, n_options)
        option_probs = torch.softmax(option_logits, dim=-1).squeeze(0).cpu().numpy()  # (n_options,)
        
        # Get termination probabilities
        term_probs = encoder.beta(image_np)  # (1, n_options)
        term_probs = term_probs.squeeze(0).cpu().numpy()  # (n_options,)
        
        # Get action probabilities for each option
        action_probs = encoder.intra_options(image_np)  # (n_options, n_actions)
        action_probs = action_probs.cpu().numpy()  # (n_options, n_actions)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Encoder Output Visualization', fontsize=16, fontweight='bold')
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(test_image, cmap='gray')
    ax.set_title(f'Input Grayscale Image ({test_image.shape[0]}x{test_image.shape[1]})')
    ax.axis('off')
    
    # Encoder features (first 100 features as bar chart)
    ax = axes[0, 1]
    n_features_to_show = min(100, len(features))
    ax.bar(range(n_features_to_show), features[:n_features_to_show], alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    ax.set_title(f'Encoder Features (first {n_features_to_show} of {len(features)})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Option probabilities
    ax = axes[0, 2]
    n_options = len(option_probs)
    bars = ax.bar(range(n_options), option_probs, color=plt.cm.Set3(np.linspace(0, 1, n_options)))
    ax.set_xlabel('Option')
    ax.set_ylabel('Probability')
    ax.set_title('Option Selection Probabilities')
    ax.set_xticks(range(n_options))
    ax.set_xticklabels([f'Opt {i}' for i in range(n_options)])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, option_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Termination probabilities
    ax = axes[1, 0]
    bars = ax.bar(range(n_options), term_probs, color=plt.cm.Set3(np.linspace(0, 1, n_options)))
    ax.set_xlabel('Option')
    ax.set_ylabel('Termination Probability')
    ax.set_title('Option Termination Probabilities')
    ax.set_xticks(range(n_options))
    ax.set_xticklabels([f'Opt {i}' for i in range(n_options)])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, term_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Action probabilities heatmap
    ax = axes[1, 1]
    n_actions = action_probs.shape[1]
    im = ax.imshow(action_probs, cmap='viridis', aspect='auto')
    ax.set_xlabel('Action')
    ax.set_ylabel('Option')
    ax.set_title('Action Probabilities per Option')
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels([f'Action {i}' for i in range(n_actions)])
    ax.set_yticks(range(n_options))
    ax.set_yticklabels([f'Option {i}' for i in range(n_options)])
    # Add text annotations
    for i in range(n_options):
        for j in range(n_actions):
            text = ax.text(j, i, f'{action_probs[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if action_probs[i, j] < action_probs.max()/2 else "black",
                         fontsize=8)
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Feature statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
    Encoder Statistics:
    
    Feature Dimension: {len(features)}
    Feature Mean: {features.mean():.4f}
    Feature Std: {features.std():.4f}
    Feature Min: {features.min():.4f}
    Feature Max: {features.max():.4f}
    
    Image Shape: {test_image.shape}
    Image Size: {test_image.shape[0]}x{test_image.shape[1]}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    parser = argparse.ArgumentParser(description="Plot OptionCritic NN training results")
    
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to training results JSON file")
    parser.add_argument("--agent_file", type=str, default=None,
                       help="Path to agent JSON file (for encoder visualization)")
    parser.add_argument("--encoder_file", type=str, default=None,
                       help="Path to encoder PyTorch file (for encoder visualization)")
    parser.add_argument("--output_dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (only save)")
    parser.add_argument("--plots", type=str, nargs="+", 
                       choices=['metrics', 'options', 'sequences', 'encoder', 'all'],
                       default=['all'],
                       help="Which plots to generate")
    parser.add_argument("--test_image", type=str, default=None,
                       help="Path to test grayscale image for encoder visualization (optional)")
    
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
        plots_to_generate = ['metrics', 'options', 'sequences']
        if args.agent_file and args.encoder_file:
            plots_to_generate.append('encoder')
    
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
    
    if 'sequences' in plots_to_generate:
        print("Generating option sequences plot...")
        plot_option_sequences(
            results,
            num_episodes=10,
            save_path=output_dir / f"option_sequences_level_{level}.png",
            show=show
        )
    
    if 'encoder' in plots_to_generate:
        if args.agent_file and args.encoder_file:
            agent_file = Path(args.agent_file)
            encoder_file = Path(args.encoder_file)
            if agent_file.exists() and encoder_file.exists():
                print(f"Loading agent from {agent_file} and encoder from {encoder_file}...")
                agent = load_agent(agent_file, encoder_file)
                
                # Generate test image if not provided
                if args.test_image:
                    try:
                        import imageio
                        test_image = imageio.imread(args.test_image)
                        if test_image.ndim == 3:
                            # Convert RGB to grayscale
                            test_image = np.mean(test_image, axis=2)
                    except ImportError:
                        print("Warning: imageio not available. Using random test image instead.")
                        img_size = agent.get('img_size', [84, 84])
                        test_image = np.random.rand(img_size[0], img_size[1]) * 255
                        test_image = test_image.astype(np.uint8)
                else:
                    # Create a simple test grayscale image
                    print("No test image provided, generating simple grayscale test image...")
                    img_size = agent.get('img_size', [84, 84])
                    test_image = np.random.rand(img_size[0], img_size[1]) * 255
                    test_image = test_image.astype(np.uint8)
                
                print("Generating encoder output visualization...")
                plot_encoder_output(
                    agent,
                    test_image,
                    save_path=output_dir / f"encoder_output_level_{level}.png",
                    show=show
                )
            else:
                print(f"Warning: Agent or encoder file not found. Skipping encoder visualization.")
        else:
            print("Warning: Agent and encoder files required for encoder visualization.")
    
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
