import os
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from model.hrl.option_critic.state_manager import StateManager
from model.hrl.option_critic.option_critic import Option

def plot_termination_probabilities(level_env, saved_agent_file, state_mapping_dir, level, episode, output_path):
    output_path = str(output_path)
    saved_agent_file = str(saved_agent_file)
    
    state_manager = StateManager(Path(state_mapping_dir))

    with open(saved_agent_file, "r") as jsonfile: 
        agent_data = json.load(jsonfile)
    options_data = agent_data["options"]
    
    state_manager.load_state_mapping(level)
    
    wall_mask = level_env.get_wall_mask()

    for option_data in options_data:
        termination_probs = -1 * wall_mask
        termination_probs = termination_probs.astype(float)
        
        theta = torch.tensor(option_data["theta"])
        upsilon = torch.tensor(option_data["upsilon"])
        option = Option(option_data["idx"], n_states=theta.shape[0], n_actions=theta.shape[1])
        option.theta = theta
        option.upsilon = upsilon
        
        for state, state_idx in state_manager.state_to_idx_dict.items():
            # TODO: Update to be environment specific wrt.
            #    states having more info than just player location
            termination_prob = option.beta(state_idx)
            player_location = level_env.get_player_loc_from_state(state)
            termination_probs[tuple(player_location)] = termination_prob.detach().item()

        plt.imshow(termination_probs, cmap="viridis", origin="lower")
        plt.colorbar(label="Probability") # TODO: Fix that it makes a bunch of these lol

        plt.title(f"Termination Probabilities Option {option_data['idx']}")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"term-prob_lvl-{level}_ep-{episode}_op-{option_data['idx']}.png"))
        plt.show()