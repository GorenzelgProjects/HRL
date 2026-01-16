import os
import json
from datetime import datetime

import numpy as np


def save_results(results, filename="test_results.json", results_dir= "test_results"):
    """Save test results to a JSON file"""
    # Create results directory if it doesn't exist
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


def save_q_table(q_table, filename="q_table.json", results_dir="test_results", num_actions=None):
    """Save Q-table and extracted policy to a JSON file
    
    Args:
        q_table: Dictionary mapping state keys (hashes) to numpy arrays of Q-values
        filename: Name of the file to save
        results_dir: Directory to save the file
        num_actions: Number of actions (inferred from Q-table if not provided)
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Handle empty Q-table
        if not q_table:
            print(f"\nWARNING: Q-table is empty, saving empty policy file")
            output = {
                "timestamp": datetime.now().isoformat(),
                "num_states": 0,
                "num_actions": num_actions or 0,
                "q_table": {},
                "policy": {}
            }
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Empty Q-table saved to: {filepath}")
            return filepath
        
        # Convert Q-table to JSON-serializable format
        # State keys are hashes (integers), Q-values are numpy arrays
        q_table_serializable = {}
        policy = {}  # Extract policy: best action for each state
        
        for state_key, q_values in q_table.items():
            # Ensure q_values is a numpy array
            if not isinstance(q_values, np.ndarray):
                q_values = np.array(q_values)
            
            # Convert state_key to string (handle numpy int64 types)
            state_key_str = str(int(state_key)) if isinstance(state_key, (np.integer, np.int64, np.int32)) else str(state_key)
            
            # Convert numpy array to list, handling NaN and inf values
            q_values_list = []
            for val in q_values:
                if np.isnan(val) or np.isinf(val):
                    q_values_list.append(0.0)  # Replace NaN/inf with 0
                else:
                    # Ensure we convert numpy types to native Python types
                    q_values_list.append(float(val))
            
            q_table_serializable[state_key_str] = q_values_list
            
            # Extract policy: action with highest Q-value
            # Replace NaN/inf with -inf so argmax works correctly
            q_values_clean = np.where(np.isfinite(q_values), q_values, -np.inf)
            best_action_idx = np.argmax(q_values_clean)
            # Convert numpy int64 to native Python int
            best_action = int(best_action_idx.item() if hasattr(best_action_idx, 'item') else best_action_idx)
            best_q_value = q_values[best_action]
            
            # Handle NaN/inf in best_q_value
            if np.isnan(best_q_value) or np.isinf(best_q_value):
                best_q_value = 0.0
            else:
                # Ensure native Python float
                best_q_value = float(best_q_value.item() if hasattr(best_q_value, 'item') else best_q_value)
            
            policy[state_key_str] = {
                "best_action": best_action,
                "best_q_value": best_q_value,
                "q_values": q_values_list
            }
        
        # Determine number of actions from first entry if not provided
        if num_actions is None and q_table:
            first_q_values = next(iter(q_table.values()))
            if isinstance(first_q_values, np.ndarray):
                num_actions = int(len(first_q_values))
            elif isinstance(first_q_values, (list, tuple)):
                num_actions = int(len(first_q_values))
            else:
                num_actions = 0
        else:
            # Ensure num_actions is a native Python int
            num_actions = int(num_actions) if num_actions is not None else 0
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "num_states": int(len(q_table)),  # Ensure native Python int
            "num_actions": int(num_actions),  # Ensure native Python int
            "q_table": q_table_serializable,
            "policy": policy
        }
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nQ-table and policy saved to: {filepath}")
        print(f"  Number of states: {len(q_table)}")
        print(f"  Number of actions: {num_actions}")
        
        return filepath
    
    except Exception as e:
        print(f"\nERROR: Failed to save Q-table to {filename}: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_training_results(results, filename="training_results_level_X.json", results_dir="test_results", level=None):
    """Save training results with action sequences (similar to option-critic format)
    
    Args:
        results: List of episode dictionaries with action_sequence
        filename: Name of the file to save
        results_dir: Directory to save the file
        level: Level number (inferred from results if not provided)
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Infer level from results if not provided
    if level is None and results:
        level = results[0].get('level', 1)
    
    # Add metadata and summary
    training_data = {
        'level': level,
        'timestamp': datetime.now().isoformat(),
        'num_episodes': len(results),
        'episodes': results,
        'summary': {
            'avg_steps_per_episode': float(np.mean([r['steps'] for r in results])) if results else 0,
            'avg_reward_per_episode': float(np.mean([r['total_reward'] for r in results])) if results else 0,
            'max_reward': float(np.max([r['total_reward'] for r in results])) if results else 0,
            'min_reward': float(np.min([r['total_reward'] for r in results])) if results else 0,
            'completion_rate': float(np.mean([1 if r['terminated'] else 0 for r in results])) if results else 0,
            'total_episodes': len(results),
        }
    }
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\nTraining results saved to: {filepath}")
    print(f"  Level: {level}")
    print(f"  Episodes: {len(results)}")
    
    return filepath