# Experimental Environment Rendering Guide

## Render Modes

The experimental environment supports multiple render modes:

### 1. `"semi"` - Semi-render (ASCII with screen clearing)
Experimental ASCII output that updates in place:
```
...................
.###############...
.#E.......P....#...
.###############...
===================
```

**Usage:**
```python
env = ThinIceEnvExperimental(level=1, render_mode="semi", headless=True)
env.render()  # Shows experimental ASCII grid
```

### 2. `"ansi"` - ANSI render (ASCII with borders)
ASCII output with borders and info:
```
===================
...................
.###############...
.#E.......P....#...
===================
```

### 3. `"human"` - Full pygame sprite rendering ⭐ NEW!
Full visual rendering with actual game sprites, just like the original implementation:
- Real game sprites (player, walls, ice, etc.)
- Animated sprites (player animation, water, keys, teleporters)
- Exact visual match to original game
- Pygame window with full graphics

**Usage:**
```python
env = ThinIceEnvExperimental(level=1, render_mode="human", headless=False)
env.render()  # Shows full pygame window with sprites
```

### 4. `null` - No rendering
Fastest mode for training (no rendering overhead)

## Configuration

Set render mode in `config.yaml`:

```yaml
environment:
  render_mode: "human"  # Options: "human", "semi", "ansi", or null
  headless: false       # Must be false for "human" mode
```

## Examples

### Full Visual Rendering (Human Mode)

```python
from thin_ice_env_experimental import ThinIceEnvExperimental
import time

# Create environment with full rendering
env = ThinIceEnvExperimental(level=1, render_mode="human", headless=False)

obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()  # Shows full pygame window with sprites
    time.sleep(0.1)
    
    if terminated or truncated:
        break

env.close()
```

### Semi-render Mode

```python
env = ThinIceEnvExperimental(level=1, render_mode="semi", headless=True)

obs, info = env.reset()
env.render()  # Shows experimental ASCII grid
```

### Using config.yaml

```yaml
# config.yaml
environment:
  level: 1
  render_mode: "human"  # Full sprite rendering
  headless: false
```

```python
# Automatically uses config.yaml
env = ThinIceEnvExperimental()
env.render()  # Full sprite rendering
```

## Features of Human Mode

✅ **Full sprite rendering** - All game sprites loaded and displayed
✅ **Animated sprites** - Player, water, keys, teleporters animate
✅ **Exact visual match** - Looks identical to original game
✅ **Lazy loading** - Sprites only loaded when rendering is needed
✅ **Configurable** - Controlled via config.yaml

## Performance

- **Human mode**: Slower (sprite rendering), but visually accurate
- **Semi mode**: Fast (ASCII), good for terminal viewing
- **Null mode**: Fastest (no rendering), best for training

## Testing

Test full rendering:
```bash
cd thin_ice
python test_experimental_env.py --render
```

Test semi-render:
```bash
python test_experimental_env.py
```
