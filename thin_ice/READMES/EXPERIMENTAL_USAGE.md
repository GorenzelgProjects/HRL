# Experimental Environment Usage Guide

## Configuration from config.yaml

The experimental environment automatically loads settings from `config.yaml`:

```yaml
environment:
  level: 1
  render_mode: "semi"  # Options: "human", "ansi", "semi", or null
  headless: true
```

## Render Modes

### 1. `"semi"` - Semi-render mode (ASCII with screen clearing)
Shows an experimental ASCII representation that updates in place:
```
...................
...................
.###############...
.#E.......P....#...
.###############...
...................
===================
```

**Features:**
- Clears screen before each render
- Shows grid with symbols (., #, E, P, etc.)
- Experimental, minimal output
- Perfect for watching agent play in terminal

### 2. `"ansi"` - ANSI render mode (ASCII with borders)
Shows ASCII with borders and info:
```
===================
...................
.###############...
.#E.......P....#...
.###############...
===================
```

### 3. `"human"` - Full pygame rendering
Shows colored rectangles in a pygame window (requires display)

### 4. `null` - No rendering
Fastest mode for training

## Usage Examples

### Using config.yaml

```python
from thin_ice_env_experimental import ThinIceEnvExperimental

# Automatically loads from config.yaml
env = ThinIceEnvExperimental()  # Uses settings from config.yaml

# Or override specific settings
env = ThinIceEnvExperimental(level=5, render_mode="semi")
```

### Semi-render mode example

```python
from thin_ice_env_experimental import ThinIceEnvExperimental
import time

env = ThinIceEnvExperimental(level=1, render_mode="semi", headless=True)
obs, info = env.reset()

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()  # Shows experimental ASCII grid
    time.sleep(0.2)
    
    if terminated or truncated:
        break

env.close()
```

### With config.yaml settings

Edit `config.yaml`:
```yaml
environment:
  level: 1
  render_mode: "semi"  # Use semi-render mode
  headless: true
```

Then:
```python
env = ThinIceEnvExperimental()  # Automatically uses config.yaml
```

## Symbol Legend

- `.` = Empty/Free tile
- `#` = Wall
- `I` = Ice
- `~` = Water
- `P` = Player
- `E` = Exit
- `K` = Key
- `H` = Keyhole
- `T` = Treasure
- `B` = Moving Block
- `b` = Moving Block Tile
- `1` = Teleporter 1
- `2` = Teleporter 2

## Comparison

| Mode | Speed | Visual | Use Case |
|------|-------|--------|----------|
| `null` | Fastest | None | Training |
| `semi` | Fast | ASCII (experimental) | Watching in terminal |
| `ansi` | Fast | ASCII (with borders) | Debugging |
| `human` | Slowest | Full graphics | Visualization |

## Tips

- Use `"semi"` mode for an experimental terminal view of agent play
- Use `"human"` mode for full visual rendering
- Use `null` for fastest training
- All settings can be overridden via constructor parameters
