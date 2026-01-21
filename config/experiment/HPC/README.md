# HPC Experiment Setup

This directory contains scripts for running hyperparameter sweeps on HPC clusters using the LSF (Load Sharing Facility) job scheduler.

## Files Overview

- **`createArguments.py`**: Generates JSON files with different hyperparameter combinations for option_critic and option_critic_nn experiments
- **`createSubmitScripts.py`**: Creates HPC submission scripts (`.sh` files) for each hyperparameter combination
- **`submit_all_scripts.sh`**: Submits all generated submission scripts to the HPC cluster
- **`bashfile.sh`**: Template for a single HPC job submission script

## Usage

### Step 1: Generate Argument Files

First, create the hyperparameter combinations:

```bash
cd config/experiment/HPC
python createArguments.py
```

This will create JSON files in `config/experiment/HPC/arguments/` with different hyperparameter combinations:
- `arguments_oc_0.json` through `arguments_oc_4.json` for option_critic experiments
- `arguments_ocnn_0.json` through `arguments_ocnn_6.json` for option_critic_nn experiments

### Step 2: Generate Submission Scripts

Create the HPC submission scripts:

```bash
python createSubmitScripts.py
```

This will:
- Generate the argument files (if not already created)
- Create submission scripts in `config/experiment/HPC/submit_scripts/`
- Each script contains the appropriate `python main.py` command with hyperparameter overrides

### Step 3: Submit Jobs to HPC

Submit all jobs to the HPC cluster:

```bash
chmod +x submit_all_scripts.sh
./submit_all_scripts.sh
```

Or submit individual scripts:

```bash
bsub < config/experiment/HPC/submit_scripts/submit_HRL_oc_arg_num_0.sh
```

## Customizing Hyperparameters

To modify the hyperparameter combinations, edit `createArguments.py`:

### Option Critic Hyperparameters

- `n_options`: Number of options (2, 4, 8)
- `epsilon`: Initial exploration rate (0.9, 0.95)
- `epsilon_decay`: Decay rate per episode (0.995, 0.998)
- `epsilon_min`: Minimum exploration rate (0.05, 0.1)
- `gamma`: Discount factor (0.95, 0.99)
- `alpha_critic`: Critic learning rate (0.25, 0.5)
- `alpha_theta`: Policy learning rate (0.1, 0.25)
- `alpha_upsilon`: Termination learning rate (0.1, 0.25)
- `temperature`: Temperature for option selection (1e-3, 1e-2)
- `n_episodes`: Number of training episodes (1000, 2000)
- `n_steps`: Maximum steps per episode (1000)

### Option Critic NN Hyperparameters

- `n_options`: Number of options (2, 4)
- `epsilon`: Initial exploration rate (0.9)
- `epsilon_decay`: Decay rate per episode (0.998)
- `epsilon_min`: Minimum exploration rate (0.05)
- `gamma`: Discount factor (0.99)
- `lr`: Learning rate (0.0001, 0.00025, 0.0005)
- `beta_reg`: Beta regularization (0.01, 0.05)
- `entropy_reg`: Entropy regularization (0.01, 0.05)
- `temperature`: Temperature for option selection (1e-2)
- `n_episodes`: Number of training episodes (1000, 2000)
- `n_steps`: Maximum steps per episode (1000)
- `optimizer_name`: Optimizer type ("RMSProp", "Adam")

## HPC Configuration

The submission scripts are configured for DTU HPC with:
- Queue: `gpuv100`
- GPU: 1 GPU in exclusive process mode
- Memory: 16GB RAM
- GPU Memory: 32GB
- Wall time: 2.5 hours (adjustable in `createSubmitScripts.py`)

To modify HPC settings, edit the script template in `createSubmitScripts.py`.

## Checking Job Status

```bash
# View all jobs
bstat

# View specific job
bjobs <job_id>

# Kill a job
bkill <job_id>
```

## Output Files

- Job output: `HRL_train_<model>_<argNum>_<jobId>.out`
- Job errors: `HRL_train_<model>_<argNum>_<jobId>.err`
- Experiment results: Saved in `logs/` directory as configured in the main config
- **Plots**: Automatically generated after training and saved in `logs/<experiment_name>/option_critic/plots/` (or `option_critic_nn/plots/`)

## Automatic Plotting

All submission scripts automatically generate plots after training completes:

- **Option Critic**: Generates all plots (metrics, options, qvalues, sequences) using `--plots all`
- **Option Critic NN**: Generates all plots (metrics, options, sequences, encoder) using `--plots all`
- Plots are saved in a separate `plots/` folder within each experiment directory
- The plotting runs automatically after training/testing completes
- If plotting fails, a warning is printed but the job continues

## Notes

- Make sure your virtual environment is set up on HPC with all required dependencies
- Adjust the project path in `createSubmitScripts.py` if your HPC directory structure differs
- The scripts assume the project is accessible at `/zhome/db/f/168045/HRL` or `~/Documents/HRL` - modify as needed
