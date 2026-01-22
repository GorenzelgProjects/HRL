"""
Main script
"""

import sys
from typing import TYPE_CHECKING
from functools import partial

import hydra
import gymnasium as gym
from loguru import logger as logging
from omegaconf import DictConfig

from model.baseline.astar.astar_manager import AStarManager
from model.hrl.option_critic.option_critic_manager import OptionCriticManager
from model.baseline.q_learning.q_learning_manager import QLearningManager
from model.baseline.random.random_manager import RandomManager

from environment.validate_environment import validate_environment

if TYPE_CHECKING:
    from model.base_manager import BaseModelManager


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    # Setup stdout logger
    logging.remove()
    logging.add(
        sys.stdout,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )

    match cfg.environment.env_name:
        case "thin_ice":
            from environment.thin_ice.thin_ice_env import ThinIceEnv

            env_id = "ThinIce-v0"
            env_entrypoint = ThinIceEnv
        case "four_rooms":
            from environment.four_rooms.four_rooms_env import Fourrooms
            env_id = "FourRooms"
            env_entrypoint = Fourrooms
        case _:
            raise ValueError(f"Unknown environment name: {cfg.environment.name}")

    gym.register(
        id=env_id,
        entry_point=env_entrypoint,
        max_episode_steps=cfg.environment.max_episode_steps,
    )

    # Run basic validation
    validate_environment(id=env_id)
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
    partial_env = partial(
        env_entrypoint,
        render_mode=render_mode,
        headless=headless,
        reward_config=cfg.environment.rewards,
        **cfg.environment.env_specific_settings,
    )

    # Instantiate model managers
    models: list["BaseModelManager"] = []
    for model_name, model_cfg in cfg.models.items():
        match model_name:
            case "astar":
                models.append(
                    AStarManager(model_cfg.heuristic, partial_env, cfg.save_dir)
                )
            case "q_learning":
                models.append(
                    QLearningManager(
                        model_cfg.learning_rate,
                        model_cfg.discount,
                        model_cfg.epsilon,
                        model_cfg.epsilon_decay,
                        model_cfg.epsilon_min,
                        model_cfg.qlearning_episodes,
                        partial_env,
                        cfg.save_dir,
                    )
                )
            case "random":
                models.append(
                    RandomManager(model_cfg.random_episodes, partial_env, cfg.save_dir)
                )
            case "option_critic":
                models.append(
                    OptionCriticManager(
                        model_cfg.n_states,
                        model_cfg.n_options,
                        model_cfg.n_actions,
                        model_cfg.n_steps,
                        model_cfg.n_episodes,
                        model_cfg.epsilon,
                        model_cfg.epsilon_decay,
                        model_cfg.epsilon_min,
                        model_cfg.gamma,
                        model_cfg.alpha_critic,
                        model_cfg.alpha_theta,
                        model_cfg.alpha_upsilon,
                        model_cfg.temperature,
                        model_cfg.save_frequency,
                        model_cfg.verbose,
                        model_cfg.quiet,
                        cfg.save_dir,
                        cfg.environment.state_mapping_dir,
                        partial_env,
                        intrinsic_composer_cfg=model_cfg.intrinsic_reward
                    )
                )
            case _:
                raise ValueError(
                    f"{model_name=} not recognized. Try astar/q_learning/random"
                )

    # Find out which levels to run
    if cfg.experiment.levels == "all":
        levels = cfg.environment.level_list
    else:
        levels = list(cfg.experiment.levels)

    if len(levels) <= 0 or not isinstance(levels[0], int):
        raise ValueError(levels)

    logging.info("Starting training")
    if cfg.experiment.run_training:
        for model in models:
            model.train(levels, cfg.experiment.render, delay)
    logging.info("Training finished")

    logging.info("Starting testing")
    if cfg.experiment.run_testing:
        for model in models:
            model.test(levels, cfg.experiment.render, delay)
    logging.info("Testing finished")

    logging.info("\nAll runs completed!")


if __name__ == "__main__":
    run()
