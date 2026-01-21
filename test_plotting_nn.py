"""
Test script for OptionCritic NN plotting functionality.

This script:
1. Generates sample training results
2. Creates a simple grayscale test image
3. Tests the encoder with the image
4. Generates all plots
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.hrl.option_critic_nn.plotting import (
    plot_training_metrics,
    plot_option_usage,
    plot_option_sequences,
    plot_encoder_output,
    load_training_results
)
from model.hrl.option_critic_nn.oc_network import Encoder


def generate_sample_training_results(num_episodes=50, level=1):
    """Generate sample training results for testing"""
    
    episodes = []
    np.random.seed(42)  # For reproducibility
    
    for episode_num in range(1, num_episodes + 1):
        # Generate option sequence (3-8 options per episode)
        num_options = np.random.randint(3, 9)
        option_sequence = []
        for _ in range(num_options):
            option_sequence.append(np.random.randint(0, 4))  # 4 options
        
        # Generate action sequence (nested dict)
        action_sequence = {}
        option_switches = 0
        for opt_idx in option_sequence:
            opt_key = str(opt_idx)
            if opt_key not in action_sequence:
                action_sequence[opt_key] = {}
            
            # Generate 5-15 actions per option segment
            num_actions = np.random.randint(5, 16)
            action_sequence[opt_key][option_switches] = [
                np.random.randint(0, 4) for _ in range(num_actions)  # 4 actions
            ]
            option_switches += 1
        
        # Generate episode metrics
        total_steps = np.random.randint(20, 100)
        total_reward = np.random.uniform(-10, 50)
        terminated = np.random.random() > 0.3  # 70% success rate
        num_options_used = len(set(option_sequence))
        num_options_switches = len(option_sequence) - 1
        
        episode = {
            "episode": episode_num,
            "level": level,
            "option_sequence": option_sequence,
            "action_sequence": action_sequence,
            "total_reward": float(total_reward),
            "total_steps": total_steps,
            "num_options_used": num_options_used,
            "num_options_switches": num_options_switches,
            "terminated": terminated,
            "truncated": False,
        }
        
        episodes.append(episode)
    
    # Calculate summary statistics
    summary = {
        "avg_steps_per_episode": float(np.mean([e["total_steps"] for e in episodes])),
        "avg_reward_per_episode": float(np.mean([e["total_reward"] for e in episodes])),
        "max_reward": float(np.max([e["total_reward"] for e in episodes])),
        "min_reward": float(np.min([e["total_reward"] for e in episodes])),
        "avg_options_per_episode": float(np.mean([e["num_options_used"] for e in episodes])),
        "avg_option_switches": float(np.mean([e["num_options_switches"] for e in episodes])),
        "completion_rate": float(np.mean([1 if e["terminated"] else 0 for e in episodes])),
        "total_episodes": len(episodes),
    }
    
    training_data = {
        "level": level,
        "timestamp": datetime.now().isoformat(),
        "num_episodes": len(episodes),
        "episodes": episodes,
        "summary": summary,
    }
    
    return training_data


def create_test_encoder(img_size=[84, 84], n_options=4, n_actions=4, device='cpu'):
    """Create a test encoder for visualization"""
    encoder = Encoder(
        image_size=img_size,
        n_filters=[32, 64, 64],
        conv_sizes=[8, 4, 3],
        strides=[4, 2, 1],
        n_neurons=512,
        n_options=n_options,
        n_actions=n_actions,
        temperature=1.0,
        device=device
    )
    encoder.eval()
    return encoder


def create_test_grayscale_image(img_size=[84, 84], pattern='checkerboard'):
    """Create a simple test grayscale image
    
    Args:
        img_size: Image dimensions [height, width]
        pattern: Pattern type ('checkerboard', 'gradient', 'random', 'circle')
    """
    h, w = img_size
    
    if pattern == 'checkerboard':
        # Create checkerboard pattern
        square_size = 10
        image = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
    
    elif pattern == 'gradient':
        # Create horizontal gradient
        image = np.linspace(0, 255, w, dtype=np.uint8)
        image = np.tile(image, (h, 1))
    
    elif pattern == 'random':
        # Random noise
        image = np.random.rand(h, w) * 255
        image = image.astype(np.uint8)
    
    elif pattern == 'circle':
        # Create circle pattern
        image = np.zeros((h, w), dtype=np.uint8)
        center = (h // 2, w // 2)
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        image[mask] = 255
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return image


def test_plotting():
    """Test all plotting functions"""
    
    print("=" * 60)
    print("Testing OptionCritic NN Plotting Functions")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path("test_plots_nn")
    test_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    print("\n1. Generating sample training results...")
    training_data = generate_sample_training_results(num_episodes=50, level=1)
    
    # Save to file
    results_file = test_dir / "test_training_results_level_1.json"
    with open(results_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"   Saved to {results_file}")
    
    # Test training metrics plot
    print("\n2. Testing training metrics plot...")
    plot_training_metrics(
        training_data,
        save_path=test_dir / "test_training_metrics.png",
        show=False
    )
    print("   ✓ Training metrics plot generated")
    
    # Test option usage plot
    print("\n3. Testing option usage plot...")
    plot_option_usage(
        training_data,
        save_path=test_dir / "test_option_usage.png",
        show=False
    )
    print("   ✓ Option usage plot generated")
    
    # Test option sequences plot
    print("\n4. Testing option sequences plot...")
    plot_option_sequences(
        training_data,
        num_episodes=10,
        save_path=test_dir / "test_option_sequences.png",
        show=False
    )
    print("   ✓ Option sequences plot generated")
    
    # Test encoder visualization
    print("\n5. Testing encoder visualization with grayscale image...")
    
    # Create test encoder
    img_size = [84, 84]
    n_options = 4
    n_actions = 4
    device = torch.device('cpu')
    
    encoder = create_test_encoder(
        img_size=img_size,
        n_options=n_options,
        n_actions=n_actions,
        device=device
    )
    
    # Create test images with different patterns
    patterns = ['checkerboard', 'gradient', 'random', 'circle']
    
    for pattern in patterns:
        print(f"   Testing with {pattern} pattern...")
        test_image = create_test_grayscale_image(img_size=img_size, pattern=pattern)
        
        # Create agent dict
        agent = {
            'encoder': encoder,
            'img_size': img_size,
            'n_options': n_options,
            'n_actions': n_actions,
            'episode': 100,
            'level': 1
        }
        
        # Test encoder output visualization
        plot_encoder_output(
            agent,
            test_image,
            save_path=test_dir / f"test_encoder_output_{pattern}.png",
            show=False
        )
        print(f"   ✓ Encoder visualization with {pattern} pattern generated")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print(f"Plots saved to: {test_dir}")
    print("=" * 60)


if __name__ == "__main__":
    test_plotting()
