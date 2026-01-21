#!/bin/sh            


### select queue 
#BSUB -q gpuv100

### name of job, output file and err
#BSUB -J HRL_train_ocnn_5
#BSUB -o HRL_train_ocnn_5_%J.out
#BSUB -e HRL_train_ocnn_5_%J.err


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
python main.py experiment=option_critic_nn_example environment=thin_ice models=option_critic_nn experiment.levels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] experiment.render=false experiment.run_training=true experiment.run_testing=true experiment.cuda=true models.option_critic_nn.n_options=2 models.option_critic_nn.epsilon=0.9 models.option_critic_nn.epsilon_decay=0.998 models.option_critic_nn.epsilon_min=0.05 models.option_critic_nn.gamma=0.99 models.option_critic_nn.lr=0.00025 models.option_critic_nn.beta_reg=0.01 models.option_critic_nn.entropy_reg=0.01 models.option_critic_nn.temperature=0.01 models.option_critic_nn.n_episodes=2000 models.option_critic_nn.n_steps=1000 models.option_critic_nn.optimizer_name=RMSProp environment.env_specific_settings.use_image_state_representation=true environment.env_specific_settings.use_coord_state_representation=false


# Generate plots after training completes
# Find the most recent experiment directory (Hydra creates timestamped dirs like logs/option_critic_T2026-01-21-22-30-45)
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
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_1.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_1.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 1..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 1"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_2.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_2.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_2.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 2..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 2"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_3.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_3.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_3.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 3..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 3"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_4.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_4.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_4.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 4..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 4"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_5.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_5.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_5.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 5..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 5"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_6.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_6.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_6.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 6..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 6"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_7.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_7.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_7.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 7..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 7"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_8.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_8.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_8.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 8..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 8"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_9.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_9.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_9.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 9..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 9"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_10.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_10.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_10.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 10..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 10"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_11.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_11.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_11.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 11..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 11"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_12.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_12.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_12.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 12..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 12"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_13.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_13.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_13.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 13..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 13"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_14.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_14.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_14.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 14..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 14"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_15.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_15.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_15.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 15..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 15"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_16.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_16.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_16.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 16..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 16"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_17.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_17.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_17.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 17..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 17"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_18.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_18.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_18.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 18..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 18"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_19.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_19.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_19.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 19..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 19"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_20.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_20.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_20.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 20..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 20"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
    RESULTS_FILE="$MODEL_DIR/results/training_results_level_21.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_2000_level_21.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_2000_level_21.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level 21..."
        python -m model.hrl.option_critic_nn.plotting \
            --results_file "$RESULTS_FILE" \
            --agent_file "$AGENT_FILE" \
            --encoder_file "$ENCODER_FILE" \
            --output_dir "$PLOTS_DIR" \
            --plots all \
            --no-show || echo "Warning: Plotting failed for level 21"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
else
    echo "Warning: Could not find experiment directory in logs/"
fi

