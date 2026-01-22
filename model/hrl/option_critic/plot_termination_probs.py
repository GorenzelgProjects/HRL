import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt

from model.hrl.option_critic.state_manager import StateManager
from model.hrl.option_critic.option_critic import Option
from environment.base_env import BaseDiscreteEnv

def plot_termination_probabilities(level_env: BaseDiscreteEnv, saved_agent_file, state_mapping_dir, level, episode, fig_root_path):
    level_env.reset()
    fig_root_path = str(fig_root_path)
    output_path = os.path.join(fig_root_path, str(level))
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
        termination_probs_dict = defaultdict(lambda: termination_probs.copy())
        
        theta = torch.tensor(option_data["theta"])
        upsilon = torch.tensor(option_data["upsilon"])
        option = Option(option_data["idx"], n_states=theta.shape[0], n_actions=theta.shape[1])
        option.theta = theta
        option.upsilon = upsilon
        
        for state, state_idx in state_manager.state_to_idx_dict.items():
            # TODO: Update to be environment specific wrt.
            #    states having more info than just player location
            termination_prob = option.beta(state_idx)
            player_location, info = level_env.get_player_loc_from_state(state)
            termination_probs_dict[info][tuple(player_location)] = termination_prob.detach().item()

        for info, termination_prob_arr in termination_probs_dict.items():
            info_str = str(info)
            if len(termination_probs_dict) > 1:
                fig_dir = os.path.join(output_path, info_str.replace(" ", "-"))
            else:
                fig_dir = output_path
            plt.imshow(termination_prob_arr, cmap="viridis", origin="upper")
            plt.colorbar(label="Probability")

            plt.title(f"Termination Probs Option: {option_data['idx']}, Info: {info_str}")
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"term-prob_ep-{episode}_op-{option_data['idx']}.png"))
            # plt.show()
            plt.close()