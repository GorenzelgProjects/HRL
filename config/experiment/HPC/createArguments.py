import json
import os

def createArguments():
    """
    Creates JSON argument files with different hyperparameter combinations
    for option_critic and option_critic_nn experiments.
    """
    
    # Create directories if they don't exist
    os.makedirs('config/experiment/HPC/arguments', exist_ok=True)
    
    arg_idx = 0
    
    # ========== Option Critic Hyperparameter Combinations ==========
    # Base hyperparameters for option_critic
    base_oc = {
        'model': 'option_critic',
        'environment': 'four_rooms',  # or 'thin_ice'
        'levels': [1],
        'render': False,
        'run_training': True,
        'run_testing': True,
    }
    
    # Hyperparameter grids for option_critic
    oc_hyperparams = {
        'n_options': [2, 4, 8],
        'epsilon': [0.9, 0.95],
        'epsilon_decay': [0.995, 0.998],
        'epsilon_min': [0.05, 0.1],
        'gamma': [0.95, 0.99],
        'alpha_critic': [0.25, 0.5],
        'alpha_theta': [0.1, 0.25],
        'alpha_upsilon': [0.1, 0.25],
        'temperature': [1e-3, 1e-2],
        'n_episodes': [1000, 2000],
        'n_steps': [1000],
    }
    
    # Create option_critic argument files
    # We'll create a reasonable subset of combinations
    oc_combinations = [
        # Combination 1: Conservative learning
        {**base_oc, **{
            'n_options': 4,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'alpha_critic': 0.5,
            'alpha_theta': 0.25,
            'alpha_upsilon': 0.25,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
        }},
        # Combination 2: More exploration
        {**base_oc, **{
            'n_options': 4,
            'epsilon': 0.95,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.1,
            'gamma': 0.99,
            'alpha_critic': 0.5,
            'alpha_theta': 0.25,
            'alpha_upsilon': 0.25,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
        }},
        # Combination 3: Faster learning
        {**base_oc, **{
            'n_options': 4,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'alpha_critic': 0.5,
            'alpha_theta': 0.25,
            'alpha_upsilon': 0.25,
            'temperature': 1e-3,
            'n_episodes': 2000,
            'n_steps': 1000,
        }},
        # Combination 4: More options
        {**base_oc, **{
            'n_options': 8,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'alpha_critic': 0.5,
            'alpha_theta': 0.25,
            'alpha_upsilon': 0.25,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
        }},
        # Combination 5: Fewer options
        {**base_oc, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.95,
            'alpha_critic': 0.25,
            'alpha_theta': 0.1,
            'alpha_upsilon': 0.1,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
        }},
    ]
    
    for idx, combo in enumerate(oc_combinations):
        with open(f'config/experiment/HPC/arguments/arguments_oc_{idx}.json', 'w') as fp:
            json.dump(combo, fp, indent=2)
        arg_idx += 1
    
    # ========== Option Critic NN Hyperparameter Combinations ==========
    # Base hyperparameters for option_critic_nn
    base_ocnn = {
        'model': 'option_critic_nn',
        'environment': 'thin_ice',
        'levels': list(range(1, 22)),  # All 21 levels for thin_ice
        'render': False,
        'run_training': True,
        'run_testing': True,
        'cuda': True,
        'use_image_state_representation': True,
        'use_coord_state_representation': False,
    }
    
    # Create option_critic_nn argument files
    ocnn_combinations = [
        # Combination 1: Default settings
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.00025,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
        # Combination 2: Higher learning rate
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.0005,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
        # Combination 3: Lower learning rate
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.0001,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
        # Combination 4: More regularization
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.00025,
            'beta_reg': 0.05,
            'entropy_reg': 0.05,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
        # Combination 5: Different optimizer
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.00025,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'Adam',
        }},
        # Combination 6: More episodes
        {**base_ocnn, **{
            'n_options': 2,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.00025,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 2000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
        # Combination 7: More options
        {**base_ocnn, **{
            'n_options': 4,
            'epsilon': 0.9,
            'epsilon_decay': 0.998,
            'epsilon_min': 0.05,
            'gamma': 0.99,
            'lr': 0.00025,
            'beta_reg': 0.01,
            'entropy_reg': 0.01,
            'temperature': 1e-2,
            'n_episodes': 1000,
            'n_steps': 1000,
            'optimizer_name': 'RMSProp',
        }},
    ]
    
    for idx, combo in enumerate(ocnn_combinations):
        with open(f'config/experiment/HPC/arguments/arguments_ocnn_{idx}.json', 'w') as fp:
            json.dump(combo, fp, indent=2)
        arg_idx += 1
    
    print(f"Created {len(oc_combinations)} option_critic argument files")
    print(f"Created {len(ocnn_combinations)} option_critic_nn argument files")
    print(f"Total argument files: {arg_idx}")

if __name__ == "__main__":
    createArguments()
