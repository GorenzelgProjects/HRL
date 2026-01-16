# How to run training and testing

Everything is controlled by hydra. All the base configs can be seen [here](../config/config.yaml).
In order to override settings you can either run with an experiment file like [astar_example](../config/experiment/astar_example.yaml) by:
```bash
python main.py experiment=astar_example
```
and/or you can override using through arguments in CLI:
```bash
python main.py experiment=astar_example experiment.levels=[1,2,3]
```

_Note that any nested configs can be overridden using dots: `.`_

## Replaying Saved Playthroughs

After training, you can replay saved action sequences using the replay utility:

### Q-Learning and Option-Critic Replay

```bash
# Replay all episodes from a Q-learning training results file
python -m model.utils.replay logs/q_learning_T2026-01-16-12-55-34/q_learning/results/training_results_level_1.json

# Replay all episodes from an option-critic training results file
python -m model.utils.replay logs/option_critic_T2026-01-16-12-57-11/option_critic/results/training_results_level_1.json

# Replay a specific episode
python -m model.utils.replay logs/q_learning_T2026-01-16-12-55-34/q_learning/results/training_results_level_1.json --episode 10
python -m model.utils.replay logs/option_critic_T2026-01-16-12-57-11/option_critic/results/training_results_level_1.json --episode 25

# Replay without rendering (faster)
python -m model.utils.replay logs/q_learning_T2026-01-16-12-55-34/q_learning/results/training_results_level_1.json --no-render

# Adjust replay speed
python -m model.utils.replay logs/q_learning_T2026-01-16-12-55-34/q_learning/results/training_results_level_1.json --delay 0.2
```

### A* Replay

```bash
# Replay all levels from A* results
python -m model.utils.replay logs/astar_T2026-01-16-11-49-37/astar/astar_all_levels_20260116_115020.json

# Replay a specific level
python -m model.utils.replay logs/astar_T2026-01-16-11-49-37/astar/astar_all_levels_20260116_115020.json --level 5
```

### Replay Options

- `--episode N`: Replay only episode N (for Q-learning/Option-Critic files)
- `--level N`: Replay only level N (for A* files)
- `--no-render`: Disable visual rendering (faster)
- `--delay SECONDS`: Delay between steps in seconds (default: 0.1)
- `--config-path PATH`: Path to config directory (default: config)

**Note:** The replay utility automatically loads environment settings (like `generate_water`) from `config/environment/thin_ice.yaml` to ensure the replay matches the original training conditions.