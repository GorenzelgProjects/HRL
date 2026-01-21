#!/bin/sh            


### select queue 
#BSUB -q gpuv100

### name of job, output file and err
#BSUB -J HRL_train_oc_3
#BSUB -o HRL_train_oc_3_%J.out
#BSUB -e HRL_train_oc_3_%J.err


### number of cores
#BSUB -n 1

# request cpu
#BSUB -R "rusage[mem=16G]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

# request 32GB of GPU-memory
#BSUB -R "select[gpu32gb]"

### wall time limit - the maximum time the job will run. Currently 2.5 hours. 






#BSUB -W 12:00        


# load the correct scipy module and python

module load python3/3.12.11
module load cuda/11.8


# activate the virtual environment
# NOTE: needs to have been built with the same Python version above!
source .venv/bin/activate

# Change to project directory (adjust path as needed)
cd $LS_SUBCWD || cd /zhome/db/f/168045/HRL 

# Run training and testing
python main.py experiment=option_critic_example environment=four_rooms models=option_critic experiment.levels=[1] experiment.render=false experiment.run_training=true experiment.run_testing=true models.option_critic.n_options=8 models.option_critic.epsilon=0.9 models.option_critic.epsilon_decay=0.998 models.option_critic.epsilon_min=0.05 models.option_critic.gamma=0.99 models.option_critic.alpha_critic=0.5 models.option_critic.alpha_theta=0.25 models.option_critic.alpha_upsilon=0.25 models.option_critic.temperature=0.01 models.option_critic.n_episodes=1000 models.option_critic.n_steps=1000


# Generate plots after training completes
# Find the most recent experiment directory (Hydra creates timestamped dirs like logs/option_critic_T2026-01-21-22-30-45)
# We look for directories matching the experiment name pattern
echo "Looking for experiment directory..."
# Use ls -t to find most recent directory matching the pattern (option_critic_*)
LATEST_EXP_DIR=$(ls -td logs/option_critic_* 2>/dev/null | head -1)

if [ -n "$LATEST_EXP_DIR" ] && [ -d "$LATEST_EXP_DIR" ]; then
    echo "Found experiment directory: $LATEST_EXP_DIR"
    MODEL_DIR="$LATEST_EXP_DIR/option_critic"
    PLOTS_DIR="$MODEL_DIR/plots"
    mkdir -p "$PLOTS_DIR"
    
    # Generate plots for each level
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_1.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_1000_level_1.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 1..."
        python -m model.hrl.option_critic.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 1"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
else
    echo "Warning: Could not find experiment directory in logs/"
fi

