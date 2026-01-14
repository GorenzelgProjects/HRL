"""
Main script
"""
from typing import TYPE_CHECKING
from functools import partial


import hydra
import gymnasium as gym
from omegaconf import DictConfig

from model.baseline.astar.astar_manager import AStarManager
from model.baseline.q_learning.q_learning_manager import QLearningManager
from model.baseline.random.random_manager import RandomManager 

from environment.validate_environment import validate_environment

if TYPE_CHECKING:
    from model.base_manager import BaseModelManager


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    match cfg.environment.env_name:
        case "thin_ice":
            from environment.thin_ice.thin_ice_env import ThinIceEnv
            env_id = "ThinIce-v0"
            env_entrypoint = ThinIceEnv
        case _:
            raise ValueError(f"Unknown environment name: {cfg.environment.name}")

    gym.register(
        id=env_id,
        entry_point=env_entrypoint,
        max_episode_steps=cfg.environment.max_episode_steps,
    )
    
    # Run basic validation
    validate_environment(id=env_id, entry_point=env_entrypoint)
    print("\n")

    if cfg.experiment.render:
        render_mode = cfg.environment.render_true_settings.render_mode
        headless = cfg.environment.render_true_settings.headless
        delay = cfg.environment.render_true_settings.delay
    else:
        render_mode = cfg.environment.render_false_settings.render_mode
        headless = cfg.environment.render_false_settings.headless
        delay = cfg.environment.render_false_settings.delay

    # Partially instantiate environment
    partial_env = partial(env_entrypoint, 
                                    render_mode=render_mode, 
                                    headless=headless, 
                                    reward_config=cfg.environment.rewards,
                                    **cfg.environment.env_specific_settings)
    
    # Instantiate model managers
    models: list["BaseModelManager"] = []
    for model_name, model_cfg in cfg.models.items():
        match model_name:
            case "astar":
                models.append(AStarManager(model_cfg.heuristic, partial_env, cfg.save_dir))
            case "q_learning":
                models.append(QLearningManager(
                    model_cfg.learning_rate,
                    model_cfg.discount,
                    model_cfg.epsilon,
                    model_cfg.epsilon_decay,
                    model_cfg.epsilon_min,
                    model_cfg.qlearning_episodes,
                    partial_env, 
                    cfg.save_dir
                    ))
            case "random":
                models.append(RandomManager(model_cfg.random_episodes, partial_env, cfg.save_dir))
            case _:
                raise ValueError(f"{model_name=} not recognized. Try astar/q_learning/random")

    # Find out which levels to run
    if cfg.experiment.levels == "all":
        levels = list(range(1,20)) # TODO: Hardcoded
    else:
        levels = list(cfg.experiment.levels)
    
    if len(levels) <= 0 or not isinstance(levels[0], int):
        raise ValueError(levels)
    
    print("Starting training")
    if cfg.experiment.run_training:
        for model in models:
            model.train(levels, cfg.experiment.render, delay)
    print("Training finished")
    
    print("Starting testing")
    if cfg.experiment.run_testing:
        for model in models:
            model.test(levels, cfg.experiment.render, delay)
    print("Testing finished")
            
    print("\nAll runs completed!")

if __name__ == "__main__":
    run()

