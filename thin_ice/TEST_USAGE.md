# Testing the Thin Ice Environment

## Basic Usage

Run tests without rendering (fast, for training):
```bash
cd thin_ice
python test_env.py
```

## Visual Play-through

To see the agent play visually:
```bash
python test_env.py --render
```

## Command Line Options

```bash
python test_env.py [OPTIONS]

Options:
  --render              Render the environment visually (opens pygame window)
  --level LEVEL         Level to test (1-19, default: 1)
  --random-episodes N   Number of random agent episodes (default: 3)
  --qlearning-episodes N Number of Q-learning episodes (default: 50)
  --delay SECONDS       Delay between rendered frames in seconds (default: 0.1)
```

## Examples

### Quick test with rendering:
```bash
python test_env.py --render --random-episodes 1 --delay 0.2
```

### Test specific level:
```bash
python test_env.py --level 5 --render
```

### Fast training test (no rendering):
```bash
python test_env.py --random-episodes 10 --qlearning-episodes 100
```

## Results

Test results are automatically saved to `test_results/` directory as JSON files with timestamps.

Each result file contains:
- Timestamp of the test
- All episode data (steps, rewards, visited tiles, etc.)
- Summary statistics (averages, success rate, etc.)

Example output file: `test_results/test_results_20250115_143022.json`

## Output Format

The JSON file contains:
```json
{
  "timestamp": "2025-01-15T14:30:22.123456",
  "total_episodes": 53,
  "episodes": [
    {
      "episode": 1,
      "steps": 45,
      "total_reward": 5.2,
      "terminated": true,
      "truncated": false,
      "visited_tiles": 12,
      "total_tiles": 12,
      "complete_tiles": 12,
      "final_distance": 0,
      "has_key": false,
      "level": 1,
      "time_seconds": 2.34
    },
    ...
  ],
  "summary": {
    "avg_reward": 4.5,
    "avg_steps": 42.3,
    "avg_time": 2.1,
    "success_rate": 0.85,
    "avg_visited_tiles": 11.2,
    "avg_total_tiles": 12.0
  }
}
```

## Notes

- When using `--render`, the pygame window will open and show the game visually
- Rendering slows down execution significantly - use only for visualization
- Results are saved automatically after all tests complete
- The test_results directory is created automatically if it doesn't exist
