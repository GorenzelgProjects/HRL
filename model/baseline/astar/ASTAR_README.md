# A* Pathfinding Algorithm for Thin Ice

This document explains the custom A* algorithm implementation for solving Thin Ice levels optimally. The implementation uses a custom priority queue and frontier-based search, specifically designed to handle the game's unique mechanics including keys, keyholes, moving blocks, and teleporters.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Algorithm Details](#algorithm-details)
- [Limitations and Notes](#limitations-and-notes)

## Installation

### Prerequisites

The A* implementation requires the following dependencies:

```bash
pip install numpy gymnasium pygame pyyaml
```

All dependencies should already be listed in the project's `requirements.txt`.

### Setup

No additional installation steps are required. The A* algorithm is implemented in:
- `thin_ice/astar_utils.py` - Core A* implementation classes
- `thin_ice/test_env.py` - Integration with ThinIceEnv and test function

Simply ensure you're in the `thin_ice` directory or have the module in your Python path.

## Overview

The A* algorithm finds optimal paths from the player's starting position to the exit tile in Thin Ice levels. It handles:

- **Static obstacles**: Walls and impassable tiles
- **Dynamic elements**: Ice tiles, keys, keyholes
- **Moving blocks**: Blocks that can be pushed (level 12+)
- **Teleporters**: Teleportation between two points (level 16+)
- **State tracking**: Visited tiles, key possession, unlocked keyholes

### Key Features

- ✅ **Optimal pathfinding**: Guaranteed to find shortest path when heuristic is admissible
- ✅ **State-aware**: Tracks game state including keys, keyholes, moving blocks
- ✅ **Memory efficient**: Uses priority queue with lazy deletion
- ✅ **Heuristic-based**: Manhattan distance heuristic for efficient search
- ✅ **Extensible**: Easy to add custom heuristics

## Architecture

The implementation follows a standard A* search pattern:

```
FrontierAStar (Priority Queue)
    ↓
ThinIceState (Search Nodes)
    ↓
ThinIceGoal (Goal Condition)
    ↓
Heuristic (Manhattan Distance)
```

### Search Process

1. **Initialization**: Create initial state from player position
2. **Frontier Setup**: Initialize priority queue with initial state
3. **Expansion Loop**:
   - Pop state with lowest f-value (g + h)
   - Check if goal state reached
   - Generate child states for all applicable actions
   - Add new states to frontier
   - Update priority if better path found
4. **Solution Extraction**: Backtrack through parent pointers to extract action plan

## Core Components

### 1. PriorityQueue

A priority queue implementation supporting:
- Adding elements with priorities
- Popping lowest priority element
- Changing priority of existing elements (lazy deletion)
- O(log n) operations

```python
class PriorityQueue:
    def add(self, element, priority: int)
    def pop(self) -> element
    def change_priority(self, element, new_priority: int)
    def size(self) -> int
    def is_empty(self) -> bool
```

### 2. FrontierAStar

Extends `FrontierBestFirst` to implement A* search frontier:
- Uses priority queue ordered by f(n) = g(n) + h(n)
- g(n): Actual cost from start to state n (path_cost)
- h(n): Heuristic estimate from state n to goal

```python
class FrontierAStar(FrontierBestFirst):
    def __init__(self, heuristic: Heuristic)
    def f(self, state: ThinIceState, goal: ThinIceGoal) -> int
```

### 3. ThinIceState

Represents a search state in the game, including:
- Player position (x, y)
- Game state flags (has_key, keyhole_unlocked, can_teleport)
- Moving block position (if applicable)
- Visited tiles set
- Path cost from start
- Parent state pointer (for backtracking)

**Key Methods**:
- `get_applicable_actions()`: Returns valid actions from current state
- `result(dx, dy)`: Applies action and returns new state
- `extract_plan()`: Backtracks to extract action sequence
- `__eq__()` and `__hash__()`: State comparison for duplicate detection

### 4. ThinIceGoal

Defines the goal condition:

```python
class ThinIceGoal:
    def __init__(self, x: int, y: int)
    def is_goal(self, state: ThinIceState) -> bool
```

### 5. Heuristic

Provides heuristic function for A*:

```python
class Heuristic:
    def __init__(self, name: str)  # Currently supports "manhattan"
    def manhattan_distance(self, state: ThinIceState, goal: ThinIceGoal) -> int
```

## Usage

### Command Line Interface

Run A* search using the test script:

```bash
# Basic usage - test on level 1
python test_env.py --method astar

# Test on specific level
python test_env.py --method astar --level 5

# Test with rendering (visual display)
python test_env.py --method astar --level 3 --render

# Test with custom heuristic
python test_env.py --method astar --heuristic manhattan

# Test all levels sequentially
python test_env.py --method astar --level 0

# Custom render delay
python test_env.py --method astar --render --delay 0.1
```

### Python API

#### Basic Usage

```python
from thin_ice_env import ThinIceEnv
from astar_utils import FrontierAStar, ThinIceState, ThinIceGoal, Heuristic
from test_env import test_a_star

# Create environment
env = ThinIceEnv(level=1, render_mode=None, headless=True)

# Run A* search
success, plan = test_a_star(
    env, 
    heuristic="manhattan",
    render=False,
    delay=0.05
)

if success:
    print(f"Solution found with {len(plan)} steps")
    print(f"Plan: {plan}")
else:
    print("No solution found")

env.close()
```

#### Custom Search

For more control, you can implement your own search loop:

```python
from thin_ice_env import ThinIceEnv
from astar_utils import FrontierAStar, ThinIceState, ThinIceGoal, Heuristic
from environment.thin_ice.data.classes.Player import Player

env = ThinIceEnv(level=1, render_mode=None, headless=True)
obs, info = env.reset()

# Create initial state
initial_player = Player(env.game, env.player.x, env.player.y, env.settings)
initial_state = ThinIceState(env.reward_config, initial_player)

# Create goal
goal_state = ThinIceGoal(env.end_tile.x, env.end_tile.y)

# Setup heuristic and frontier
heuristic_obj = Heuristic("manhattan")
frontier = FrontierAStar(heuristic_obj)
frontier.prepare(goal_state)

# Action set mapping
action_set = {
    0: (1, 0),   # right
    1: (0, -1),  # up
    2: (-1, 0),  # left
    3: (0, 1),   # down
}

# Add initial state
initial_state.parent = None
initial_state.path_cost = 0
frontier.add(initial_state)
expanded = set()

# Search loop
while not frontier.is_empty():
    current = frontier.pop()
    expanded.add(current)
    
    # Check goal
    if goal_state.is_goal(current):
        plan = current.extract_plan()
        print(f"Solution found: {plan}")
        break
    
    # Expand children
    for action, (dx, dy) in current.get_applicable_actions(action_set):
        child_player = Player(
            current.player.game,
            current.player.x,
            current.player.y,
            current.player.settings
        )
        
        child = ThinIceState(
            current.reward_config,
            child_player,
            action,
            current,
            current.has_key,
            current.can_teleport,
            current.reset_once,
            current.moved,
            current.keyhole_unlocked,
            current.moving_block_pos
        )
        
        child = child.result(dx, dy)
        
        if child not in expanded and not frontier.contains(child):
            frontier.add(child)
        elif frontier.contains(child):
            # Update if better path found
            current_priority = frontier.priority_queue.get_priority(child)
            new_priority = frontier.f(child, goal_state)
            if current_priority > new_priority:
                frontier.priority_queue.change_priority(child, new_priority)

env.close()
```

## API Reference

### `test_a_star(env, heuristic="manhattan", render=False, delay=0.05)`

Runs A* search on the given environment.

**Parameters**:
- `env` (ThinIceEnv): The game environment instance
- `heuristic` (str): Heuristic name (currently only "manhattan")
- `render` (bool): Whether to visually render the solution
- `delay` (float): Delay between rendered frames in seconds

**Returns**:
- `(bool, list[str])`: Tuple of (success, plan)
  - `success`: True if solution found, False otherwise
  - `plan`: List of action strings ("right", "up", "left", "down")

**Example**:
```python
success, plan = test_a_star(env, heuristic="manhattan", render=True)
```

### ThinIceState Methods

#### `get_applicable_actions(action_set) -> list[tuple]`

Returns list of valid (action, (dx, dy)) tuples from current state.

**Parameters**:
- `action_set` (dict): Mapping of action indices to (dx, dy) tuples

**Returns**: List of (action_index, (dx, dy)) tuples

#### `result(dx, dy) -> ThinIceState`

Applies action and returns new state. Handles:
- Player movement
- Wall collisions
- Key collection
- Keyhole unlocking
- Teleportation
- Moving block pushing
- Visited tile tracking
- Reward calculation

**Parameters**:
- `dx` (int): X direction (-1, 0, or 1)
- `dy` (int): Y direction (-1, 0, or 1)

**Returns**: Modified state (self is modified in-place)

#### `extract_plan() -> list[str]`

Extracts action plan by backtracking through parent states.

**Returns**: List of action strings ("right", "up", "left", "down")

### State Comparison

States are compared based on:
- Player position (x, y)
- Key possession (`has_key`)
- Keyhole unlock status (`keyhole_unlocked`)
- Moving block position (`moving_block_pos`)
- Teleport availability (`can_teleport`)

This ensures states with different game conditions are treated as distinct.

## Examples

### Example 1: Find Solution for Level 1

```python
from thin_ice_env import ThinIceEnv
from test_env import test_a_star

env = ThinIceEnv(level=1, render_mode=None, headless=True)
success, plan = test_a_star(env)

if success:
    print(f"Found solution with {len(plan)} steps")
    print(f"First 10 actions: {plan[:10]}")
else:
    print("No solution found")

env.close()
```

### Example 2: Visualize Solution

```python
from thin_ice_env import ThinIceEnv
from test_env import test_a_star

env = ThinIceEnv(level=3, render_mode="human", headless=False)
success, plan = test_a_star(env, render=True, delay=0.2)

env.close()
```

### Example 3: Test Multiple Levels

```python
from thin_ice_env import ThinIceEnv
from test_env import test_a_star

results = {}
for level in range(1, 6):
    env = ThinIceEnv(level=level, render_mode=None, headless=True)
    success, plan = test_a_star(env)
    results[level] = {
        "success": success,
        "steps": len(plan) if success else None
    }
    env.close()

print("Results:")
for level, result in results.items():
    status = f"{result['steps']} steps" if result['success'] else "Failed"
    print(f"Level {level}: {status}")
```

### Example 4: Custom Heuristic

To add a custom heuristic, extend the `Heuristic` class:

```python
class CustomHeuristic(Heuristic):
    def __init__(self):
        super().__init__("custom")
    
    def custom_distance(self, state: ThinIceState, goal: ThinIceGoal) -> int:
        # Your custom heuristic calculation
        dx = abs(state.player.x - goal.goal_coords[0])
        dy = abs(state.player.y - goal.goal_coords[1])
        # Example: weighted distance
        return dx * 2 + dy
    
    def h(self, state: ThinIceState, goal: ThinIceGoal) -> int:
        return self.custom_distance(state, goal)
```

## Algorithm Details

### A* Search Properties

1. **Admissibility**: Manhattan distance heuristic is admissible (never overestimates), ensuring optimal solutions
2. **Completeness**: Guaranteed to find solution if one exists (given infinite time/memory)
3. **Optimality**: Guaranteed optimal path when heuristic is admissible

### State Space

The state space includes:
- **Position**: 19 × 17 = 323 possible positions
- **Key status**: 2 states (has/doesn't have)
- **Keyhole status**: 2 states (unlocked/locked)
- **Teleport status**: 2 states (can/can't teleport)
- **Moving block**: Variable positions depending on level

Total theoretical state space can be very large, but A* efficiently explores only relevant states.

### Path Cost Calculation

Path cost (`path_cost`) represents:
- **Base cost**: Number of steps taken (incremented by 1 per action)
- **Reward adjustment**: Rewards reduce path cost (optimal paths should maximize rewards)
- **Final cost**: `path_cost = steps - total_rewards`

This makes the algorithm prefer paths with higher rewards.

### Action Applicability

Actions are filtered based on:
1. **Wall collisions**: Cannot move into walls
2. **Moving block collisions**: Cannot move into block unless pushing it
3. **Block pushing**: Can push block if adjacent and moving towards it
4. **Keyhole access**: Must unlock keyhole with key before passing

## Limitations and Notes

### Known Limitations

1. **Memory usage**: Can be high for complex levels with many states
2. **Computation time**: May take significant time for very complex levels
3. **Heuristic dependency**: Solution quality depends on heuristic choice
4. **Static planning**: Plans based on current state, doesn't predict future ice breaks

### Performance Considerations

- **Level complexity**: Simple levels (1-5) solve quickly (<1 second)
- **Complex levels**: Levels 15-19 may take longer due to larger state space
- **Moving blocks**: Levels with moving blocks (12+) have larger state space
- **Memory**: Keep an eye on memory usage for very large searches

### Best Practices

1. **Use appropriate heuristics**: Manhattan distance works well for grid-based games
2. **Limit search space**: Consider time/memory limits for complex levels
3. **State uniqueness**: Ensure `__hash__` and `__eq__` properly distinguish states
4. **Debugging**: Use `render=True` to visualize the solution path

### Troubleshooting

**Problem**: Search takes too long or runs out of memory
- **Solution**: Add iteration limit or memory limit checks

**Problem**: Solution seems suboptimal
- **Solution**: Verify heuristic is admissible, check state comparison logic

**Problem**: No solution found when solution exists
- **Solution**: Check action applicability logic, verify goal condition

## References

- [A* Search Algorithm - Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Introduction to A* - Red Blob Games](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [Thin Ice Game Rules](README.md)

## License

This implementation is part of the Thin Ice game project. See the main project README for license information.
