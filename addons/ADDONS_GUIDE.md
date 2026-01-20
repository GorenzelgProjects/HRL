# Addons Guide: Simple Successor Features & GAS

This guide explains how to use both addons: **Simple Successor Features (Simple-SF)** and **Graph-Assisted Stitching (GAS)**.

---

## Table of Contents

1. [Simple Successor Features](#simple-successor-features)
   - [Overview](#overview)
   - [Configuration](#configuration)
   - [Training](#training)
   - [How It Works](#how-it-works)

2. [Graph-Assisted Stitching (GAS)](#graph-assisted-stitching-gas)
   - [Overview](#overview-1)
   - [Complete Pipeline](#complete-pipeline)
   - [Step 1: Collect Rollouts](#step-1-collect-rollouts)
   - [Step 2: Mine Subgoals](#step-2-mine-subgoals)
   - [Step 3: Fine-tune with Subgoals](#step-3-fine-tune-with-subgoals)
   - [Configuration Options](#configuration-options)

---

## Simple Successor Features

### Overview

Simple Successor Features (Simple-SF) is an auxiliary learning module that helps with continual learning and transfer. It learns:
- **Basis features** `φ(z)`: L2-normalized features that capture task-agnostic state properties
- **Successor features** `ψ(z,a)`: Features that predict future basis features
- **Task vector** `w_task`: Maps basis features to rewards

The key idea is that rewards can be decomposed as `r = φ(z) · w_task`, allowing the agent to quickly adapt to new tasks by only updating `w_task`.

### Configuration

Enable Simple-SF in your config file (`config/models/option_critic.yaml`):

```yaml
models:
  option_critic:
    # Simple Successor Features
    use_simple_sf: true
    sf_d: 256              # Dimension of successor features
    lambda_sf: 0.1         # Weight for SF losses in total loss
    alpha_w: 0.1           # Weight for reward prediction loss (L_w)
    sf_lr_main: 1e-3       # Learning rate for main network (encoder + SF heads)
    sf_lr_w: 1e-2          # Learning rate for task vector w_task
```

**Parameters:**
- `use_simple_sf`: Enable/disable Simple-SF (default: `false`)
- `sf_d`: Dimension of successor features (default: `256`)
- `lambda_sf`: Weight for combining SF losses with Option-Critic loss (default: `1.0`)
- `alpha_w`: Weight for reward prediction loss relative to Q-SF TD loss (default: `1.0`)
- `sf_lr_main`: Learning rate for encoder and SF heads (default: `1e-3`)
- `sf_lr_w`: Learning rate for task vector (default: `1e-2`)

### Training

Simply run training as usual:

```bash
python main.py experiment=option_critic_example
```

Simple-SF will automatically:
1. Compute `φ(z)` and `ψ(z,a)` for each state-action pair
2. Compute Q-SF values: `Q_SF(s,a) = ψ(s,a) · w_task`
3. Train using TD loss on Q-SF: `L_ψ = MSE(Q_SF(s,a), r + γ·max_a' Q_SF(s',a'))`
4. Train task vector: `L_w = MSE(φ(s') · w_task, r)`
5. Combine with Option-Critic loss: `L_total = L_OC + λ_sf · (L_ψ + α_w · L_w)`

**Observing SF Losses:**

During training, you'll see SF loss logs every 10 episodes:
```
Episode 10: SF Losses - L_psi=0.1234, L_w=0.0567, Total_SF=0.1801, w_task_norm=2.3456
```

These losses are also saved in the episode stats in `training_results_level_X.json`:
- `sf_L_psi`: Q-SF TD loss
- `sf_L_w`: Reward prediction loss
- `sf_w_task_norm`: Norm of task vector (useful for monitoring)
- `sf_transfer_mode`: Whether transfer learning is active

### How It Works

**Architecture:**
```
Observation → Pixel Encoder → z_t
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              φ_head(z)          ψ_head(z)
                    ↓                   ↓
              φ(z) [normalized]  ψ(z,a) [for all actions]
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                        w_task (learnable)
                              ↓
                        Reward prediction
```

**Loss Functions:**

1. **Q-SF TD Loss** (`L_ψ`):
   ```
   Q_SF(s,a) = ψ(s,a) · w_task
   target = r + γ · max_a' Q_SF(s',a')
   L_ψ = 0.5 · MSE(Q_SF(s,a), target)
   ```

2. **Reward Prediction Loss** (`L_w`):
   ```
   r_pred = φ(s') · w_task  (with stop gradient on φ)
   L_w = 0.5 · MSE(r_pred, r)
   ```

3. **Total Loss**:
   ```
   L_total = L_OC + λ_sf · (L_ψ + α_w · L_w)
   ```

**Benefits:**
- Faster adaptation to new tasks (only update `w_task`)
- Better transfer learning across similar environments
- More stable training through auxiliary objectives

### Transfer Learning Across Levels

Simple-SF supports explicit transfer learning by loading shared components from a previously trained level:

**How It Works:**

1. **Step 1: Train on Level 1** (train encoder + SF heads + w_task):
   ```yaml
   # config/models/option_critic.yaml
   use_simple_sf: true
   sf_transfer_from_level: null  # No transfer, train from scratch
   ```
   ```bash
   python main.py experiment=option_critic_example
   ```
   - Trains encoder + SF heads (`φ` and `ψ`) from scratch
   - Trains `w_task[1]` for level 1
   - **Automatically saves** SF components to `logs/.../agents/sf_components_level_1.pt`

2. **Step 2: Train on Level 2** (reuse encoder + SF heads, train only w_task):
   ```yaml
   # config/models/option_critic.yaml
   use_simple_sf: true
   sf_transfer_from_level: 1  # Load from level 1
   sf_freeze_encoder: true     # Freeze encoder (default: true)
   ```
   ```bash
   python main.py experiment=option_critic_example
   ```
   - **Loads** encoder + SF heads from level 1
   - **Freezes** encoder + SF heads (no gradients)
   - **Trains only** `w_task[2]` for level 2 (~256 parameters)

**Configuration:**

```yaml
models:
  option_critic:
    use_simple_sf: true
    # ... other SF params ...
    
    # Transfer learning: load from a previously trained level
    sf_transfer_from_level: 1    # Level to load from (null = no transfer)
    sf_freeze_encoder: true      # Freeze encoder when transferring (default: true)
```

**What You'll See:**

**Level 1 (Training from scratch):**
```
Level 1
Episode 10: SF Losses - L_psi=0.1234, L_w=0.0567, Total_SF=0.1801, w_task_norm=2.3456
Saved SF components to logs/.../agents/sf_components_level_1.pt
```

**Level 2 (Transfer learning):**
```
Level 2
Loading SF components from level 1 for transfer learning...
Transfer Learning: Freezing encoder, training w_task[2] only
Episode 10: SF Losses (transfer) - L_psi=0.0890, L_w=0.0345, Total_SF=0.1235, w_task_norm=1.9876
```

**Transfer Learning Modes:**

- **Frozen Encoder** (`sf_freeze_encoder: true`, default):
  - Encoder and SF heads are frozen (no gradients)
  - Only `w_task` is trained
  - Fastest adaptation (~256 parameters)
  - Best when levels are similar

- **Fine-tuning** (`sf_freeze_encoder: false`):
  - Encoder and SF heads use reduced learning rate (10% of normal)
  - `w_task` uses normal learning rate
  - Slower but may help if levels differ significantly

**Benefits of Transfer Learning:**

- **Faster training**: Later levels train much faster (only `w_task`)
- **Better performance**: Encoder learns general representations across levels
- **Parameter efficiency**: Share most parameters, only adapt small `w_task` per level
- **Continual learning**: Can add new levels without retraining everything
- **Explicit control**: You decide which level to transfer from

**Example Workflow:**

1. Train level 1: `sf_transfer_from_level: null`, `levels: [1]`
2. Train level 2: `sf_transfer_from_level: 1`, `levels: [2]`
3. Train level 3: `sf_transfer_from_level: 1`, `levels: [3]` (or use level 2 if you prefer)

---

## Graph-Assisted Stitching (GAS)

### Overview

GAS is an offline subgoal mining technique that:
1. Learns Temporal Distance Representation (TDR) from rollouts
2. Filters states by Temporal Efficiency (TE)
3. Clusters states using TD-aware clustering
4. Builds a graph of key states
5. Extracts subgoals from the graph
6. Assigns subgoals to options for intrinsic rewards

The goal is to discover important waypoints (subgoals) that help the agent navigate more efficiently.

### Complete Pipeline

The GAS pipeline consists of three main steps:

```
1. Collect Rollouts (during training)
   ↓
2. Mine Subgoals (offline)
   ↓
3. Fine-tune with Subgoals (optional)
```

---

### Step 1: Collect Rollouts

**Purpose:** Collect rich trajectory data with observations for offline analysis.

**Configuration:**

Enable rollout collection in your config (`config/models/option_critic.yaml`):

```yaml
models:
  option_critic:
    collect_rollouts: true
    rollout_save_dir: "logs/my_experiment/rollouts"
    rollout_save_frequency: 10  # Save every N episodes
```

**Parameters:**
- `collect_rollouts`: Enable/disable rollout collection (default: `false`)
- `rollout_save_dir`: Directory to save rollouts (default: `logs/rollouts`)
- `rollout_save_frequency`: Save rollouts every N episodes (default: `10`)

**Run Training:**

```bash
python main.py experiment=option_critic_example
```

This will save rollouts to `logs/my_experiment/rollouts/rollouts_level_X_episode_Y.npz`.

**Rollout Format:**

Each rollout file contains:
- `obs`: List of observations (one per step)
- `option_sequence`: List of option indices
- `action_sequence`: List of actions
- `rewards`: List of rewards
- `dones`: List of done flags
- `agent_pos`: List of agent positions (if available)
- `terminated`: Whether episode terminated successfully

---

### Step 2: Mine Subgoals

**Purpose:** Extract important subgoals from collected rollouts using GAS mining.

**Command:**

```bash
python scripts/mine_subgoals.py \
    --rollout-file logs/my_experiment/rollouts/rollouts_level_21_episode_100.npz \
    --output-dir logs/my_experiment/subgoals \
    --num-subgoals 10
```

**Parameters:**

- `--rollout-file`: Path to rollout file (NPZ or JSON format)
- `--output-dir`: Directory to save mined subgoals
- `--num-subgoals`: Target number of subgoals to extract (default: `10`)
- `--num-epochs`: TDR training epochs (default: `50`)
- `--way-steps`: Steps ahead for TE computation and distance cutoff (default: `8`)
- `--te-threshold`: Temporal Efficiency threshold (default: auto-determine)
  - Negative value = auto-determine from data
  - Positive value = use that threshold
  - Lower = more permissive (accepts more states)
- `--batch-size`: Batch size for graph construction (default: `1024`)

**What Happens:**

1. **Train TDR Encoder** (50 epochs by default):
   - Learns embeddings where distances correlate with temporal distance
   - Pre-computes all encoder embeddings in batches (fast)
   - Samples transition pairs to avoid O(N²) explosion

2. **Compute Temporal Efficiency**:
   - For each state, computes cosine similarity between:
     - Step vector: `obs[t+way_steps] - obs[t]`
     - Distance vector: `obs[t+distance] - obs[t]` (where distance >= way_steps)
   - Filters states with TE >= threshold
   - Prints statistics: min, max, mean, median TE scores

3. **TD-aware Clustering**:
   - Clusters filtered states within `way_steps/2` distance
   - Reduces nodes from thousands to hundreds
   - Computes cluster centers

4. **Build Graph**:
   - Creates nodes from cluster centers
   - Adds edges between nodes within `way_steps` distance
   - Connects disconnected components
   - Much faster than all-pairs proximity check

5. **Extract Subgoals**:
   - Computes shortest paths from sampled nodes
   - Extracts nodes that appear frequently on paths
   - Returns top N subgoals by path frequency

**Output:**

Saves to `output_dir/`:
- `subgoals.json`: List of subgoal dictionaries with embeddings
- `subgoal_assignments.json`: Mapping from subgoal_id to option_idx

**Troubleshooting:**

If you get "0 subgoals extracted":
- Check TE statistics in output
- Try lowering `--te-threshold` (e.g., `-0.5` or `0.0`)
- Try reducing `--way-steps` (e.g., `4`)
- Ensure episodes are terminating successfully

---

### Step 3: Fine-tune with Subgoals

**Purpose:** Fine-tune Option-Critic with intrinsic rewards based on reaching subgoals.

**Command:**

```bash
python scripts/finetune_with_subgoals.py \
    --subgoals-file logs/my_experiment/subgoals/subgoals.json \
    --assignments-file logs/my_experiment/subgoals/subgoal_assignments.json \
    --reward-scale 1.0
```

**Parameters:**

- `--subgoals-file`: Path to subgoals JSON file
- `--assignments-file`: Path to subgoal assignments JSON file
- `--reward-scale`: Scale factor for intrinsic rewards (default: `1.0`)

**What Happens:**

1. Loads subgoals and assignments
2. Creates `SubgoalRewards` module
3. During training, computes intrinsic rewards:
   - When agent reaches a subgoal assigned to current option: `+reward_scale`
   - Otherwise: `0`
4. Adds intrinsic rewards to environment rewards
5. Trains Option-Critic with combined rewards

**Intrinsic Reward Logic:**

```python
def compute_intrinsic_reward(state, option_idx):
    # Check if state is close to any subgoal assigned to this option
    for subgoal_id, assigned_option in assignments.items():
        if assigned_option == option_idx:
            subgoal = subgoals[subgoal_id]
            distance = ||encoder(state) - subgoal['embedding']||
            if distance < threshold:
                return reward_scale
    return 0.0
```

---

## Configuration Options

### Complete Example Config

```yaml
models:
  option_critic:
    # Option-Critic parameters
    n_options: 4
    n_episodes: 1000
    # ... other OC params ...
    
    # Simple Successor Features
    use_simple_sf: true
    sf_d: 256
    lambda_sf: 0.1
    alpha_w: 0.1
    sf_lr_main: 1e-3
    sf_lr_w: 1e-2
    
    # GAS Rollout Collection
    collect_rollouts: true
    rollout_save_dir: "logs/my_experiment/rollouts"
    rollout_save_frequency: 10
```

### Recommended Settings

**For Simple-SF:**
- Start with `lambda_sf=0.1` and `alpha_w=0.1` (small weights)
- Increase if you want stronger SF influence
- Use separate learning rates: `sf_lr_w` typically 10x `sf_lr_main`

**For GAS:**
- `way_steps=8` works well for most environments
- `te_threshold=-1.0` (auto) is recommended
- `num_subgoals=10-20` is usually sufficient
- Lower `te_threshold` if you get 0 subgoals

---

## Architecture Overview

### Shared Components

Both addons use the **Pixel Encoder** (`addons/encoder/pixel_encoder.py`):

```
Observation (104-dim feature vector)
    ↓
Pixel Encoder (MLP mode for feature vectors)
    ↓
z_t (256-dim feature vector)
    ↓
    ├─→ Simple-SF: φ(z), ψ(z,a)
    └─→ GAS: TDR encoder → h_tdr(z)
```

**Encoder Modes:**
- **Feature Vector Mode**: When `feature_vector_size` is provided, uses MLP
- **Grid Mode**: When input is grid-based, uses CNN + embedding

---

## Troubleshooting

### Simple-SF Issues

**Loss not decreasing:**
- Check learning rates (`sf_lr_main`, `sf_lr_w`)
- Try adjusting `lambda_sf` and `alpha_w`
- Ensure encoder is properly initialized

**Training too slow:**
- Reduce `sf_d` dimension
- Use smaller batch sizes

### GAS Issues

**0 subgoals extracted:**
- Check TE statistics in output
- Lower `--te-threshold` (try `-0.5` or `0.0`)
- Reduce `--way-steps` (try `4`)
- Ensure rollouts contain successful episodes

**TDR training too slow:**
- Reduce `--num-epochs` (try `20-30`)
- The code now pre-computes embeddings (should be fast)

**Graph too large:**
- Increase `--te-threshold` to filter more states
- Increase `--way-steps` to reduce clustering
- TD-aware clustering should help (reduces nodes significantly)

**Memory issues:**
- Reduce `--batch-size` for graph construction
- Use fewer rollouts
- Process rollouts in smaller batches

---

## References

- **Simple Successor Features**: [GitHub](https://github.com/raymondchua/simple_successor_features)
- **GAS**: [GitHub](https://github.com/qortmdgh4141/GAS) | [Paper](https://arxiv.org/abs/2506.07744)

---

## Quick Start Checklist

**Simple-SF:**
- [ ] Set `use_simple_sf: true` in config
- [ ] Configure `sf_d`, `lambda_sf`, `alpha_w`
- [ ] Run training: `python main.py experiment=option_critic_example`

**GAS:**
- [ ] Set `collect_rollouts: true` in config
- [ ] Run training to collect rollouts
- [ ] Mine subgoals: `python scripts/mine_subgoals.py ...`
- [ ] (Optional) Fine-tune: `python scripts/finetune_with_subgoals.py ...`

---

For more details, see the individual module documentation in:
- `addons/simple_sf/`
- `addons/gas/`
- `addons/encoder/`
