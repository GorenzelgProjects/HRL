import json
import os
import glob
from createArguments import createArguments

def createSubmitScripts():
    """
    Creates HPC submission scripts for all argument files.
    Each script will run an experiment with the hyperparameters from the argument file.
    """
    # First, create the arguments
    createArguments()
    
    # Create directories if they don't exist
    os.makedirs('config/experiment/HPC/submit_scripts', exist_ok=True)
    
    # Find all argument files
    argument_files = sorted(glob.glob('config/experiment/HPC/arguments/arguments_*.json'))
    
    if not argument_files:
        print("No argument files found. Run createArguments() first.")
        return
    
    print(f"Found {len(argument_files)} argument files")
    
    # Create a submission script for each argument file
    for arg_file in argument_files:
        # Read the argument file to get model type
        with open(arg_file, 'r') as f:
            args = json.load(f)
        
        # Extract model type and index from filename
        filename = os.path.basename(arg_file)
        # filename format: arguments_oc_0.json or arguments_ocnn_1.json
        parts = filename.replace('.json', '').split('_')
        model_type = parts[1]  # 'oc' or 'ocnn'
        arg_num = parts[2]  # index
        
        # Determine experiment name based on model type
        if model_type == 'oc':
            experiment_name = 'option_critic_example'
        elif model_type == 'ocnn':
            experiment_name = 'option_critic_nn_example'
        else:
            experiment_name = 'option_critic_example'
        
        # Build the python command with overrides
        overrides = []
        
        # Environment override
        if 'environment' in args:
            overrides.append(f'environment={args["environment"]}')
        
        # Model override
        if 'model' in args:
            if args['model'] == 'option_critic':
                overrides.append('models=option_critic')
            elif args['model'] == 'option_critic_nn':
                overrides.append('models=option_critic_nn')
        
        # Experiment settings
        if 'levels' in args:
            levels_str = ','.join(map(str, args['levels']))
            overrides.append(f'experiment.levels=[{levels_str}]')
        
        if 'render' in args:
            overrides.append(f'experiment.render={str(args["render"]).lower()}')
        
        if 'run_training' in args:
            overrides.append(f'experiment.run_training={str(args["run_training"]).lower()}')
        
        if 'run_testing' in args:
            overrides.append(f'experiment.run_testing={str(args["run_testing"]).lower()}')
        
        if 'cuda' in args:
            overrides.append(f'experiment.cuda={str(args["cuda"]).lower()}')
        
        # Model-specific hyperparameters
        if args['model'] == 'option_critic':
            if 'n_options' in args:
                overrides.append(f'models.option_critic.n_options={args["n_options"]}')
            if 'epsilon' in args:
                overrides.append(f'models.option_critic.epsilon={args["epsilon"]}')
            if 'epsilon_decay' in args:
                overrides.append(f'models.option_critic.epsilon_decay={args["epsilon_decay"]}')
            if 'epsilon_min' in args:
                overrides.append(f'models.option_critic.epsilon_min={args["epsilon_min"]}')
            if 'gamma' in args:
                overrides.append(f'models.option_critic.gamma={args["gamma"]}')
            if 'alpha_critic' in args:
                overrides.append(f'models.option_critic.alpha_critic={args["alpha_critic"]}')
            if 'alpha_theta' in args:
                overrides.append(f'models.option_critic.alpha_theta={args["alpha_theta"]}')
            if 'alpha_upsilon' in args:
                overrides.append(f'models.option_critic.alpha_upsilon={args["alpha_upsilon"]}')
            if 'temperature' in args:
                overrides.append(f'models.option_critic.temperature={args["temperature"]}')
            if 'n_episodes' in args:
                overrides.append(f'models.option_critic.n_episodes={args["n_episodes"]}')
            if 'n_steps' in args:
                overrides.append(f'models.option_critic.n_steps={args["n_steps"]}')
        
        elif args['model'] == 'option_critic_nn':
            if 'n_options' in args:
                overrides.append(f'models.option_critic_nn.n_options={args["n_options"]}')
            if 'epsilon' in args:
                overrides.append(f'models.option_critic_nn.epsilon={args["epsilon"]}')
            if 'epsilon_decay' in args:
                overrides.append(f'models.option_critic_nn.epsilon_decay={args["epsilon_decay"]}')
            if 'epsilon_min' in args:
                overrides.append(f'models.option_critic_nn.epsilon_min={args["epsilon_min"]}')
            if 'gamma' in args:
                overrides.append(f'models.option_critic_nn.gamma={args["gamma"]}')
            if 'lr' in args:
                overrides.append(f'models.option_critic_nn.lr={args["lr"]}')
            if 'beta_reg' in args:
                overrides.append(f'models.option_critic_nn.beta_reg={args["beta_reg"]}')
            if 'entropy_reg' in args:
                overrides.append(f'models.option_critic_nn.entropy_reg={args["entropy_reg"]}')
            if 'temperature' in args:
                overrides.append(f'models.option_critic_nn.temperature={args["temperature"]}')
            if 'n_episodes' in args:
                overrides.append(f'models.option_critic_nn.n_episodes={args["n_episodes"]}')
            if 'n_steps' in args:
                overrides.append(f'models.option_critic_nn.n_steps={args["n_steps"]}')
            if 'optimizer_name' in args:
                overrides.append(f'models.option_critic_nn.optimizer_name={args["optimizer_name"]}')
            
            # Environment-specific settings for option_critic_nn
            if 'use_image_state_representation' in args:
                overrides.append(f'environment.env_specific_settings.use_image_state_representation={str(args["use_image_state_representation"]).lower()}')
            if 'use_coord_state_representation' in args:
                overrides.append(f'environment.env_specific_settings.use_coord_state_representation={str(args["use_coord_state_representation"]).lower()}')
        
        # Build the full command
        override_str = ' '.join(overrides)
        python_cmd = f'python main.py experiment={experiment_name} {override_str}'
        
        # Build plotting commands
        # Get levels from args
        levels = args.get('levels', [1])
        n_episodes = args.get('n_episodes', 1000)
        
        plotting_commands = []
        if args['model'] == 'option_critic':
            # For option_critic, use the plotting script
            # Find experiment directory - Hydra saves to logs/ with timestamp
            # We'll find the most recent directory matching the pattern
            plot_cmd = f'''
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
'''
            for level in levels:
                plot_cmd += f'''    RESULTS_FILE="$MODEL_DIR/results/training_results_level_{level}.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_{n_episodes}_level_{level}.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level {level}..."
        python -m model.hrl.option_critic.plotting \\
            --results_file "$RESULTS_FILE" \\
            --agent_file "$AGENT_FILE" \\
            --output_dir "$PLOTS_DIR" \\
            --plots all \\
            --no-show || echo "Warning: Plotting failed for level {level}"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
'''
            plot_cmd += '''else
    echo "Warning: Could not find experiment directory in logs/"
fi
'''
            plotting_commands.append(plot_cmd)
        
        elif args['model'] == 'option_critic_nn':
            # For option_critic_nn, use the NN plotting script
            plot_cmd = f'''
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
'''
            for level in levels:
                plot_cmd += f'''    RESULTS_FILE="$MODEL_DIR/results/training_results_level_{level}.json"
    AGENT_FILE="$MODEL_DIR/agents/agent_episode_{n_episodes}_level_{level}.json"
    ENCODER_FILE="$MODEL_DIR/agents/agent_episode_{n_episodes}_level_{level}.pt"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Generating plots for level {level}..."
        python -m model.hrl.option_critic_nn.plotting \\
            --results_file "$RESULTS_FILE" \\
            --agent_file "$AGENT_FILE" \\
            --encoder_file "$ENCODER_FILE" \\
            --output_dir "$PLOTS_DIR" \\
            --plots all \\
            --no-show || echo "Warning: Plotting failed for level {level}"
    else
        echo "Warning: Results file not found: $RESULTS_FILE"
    fi
'''
            plot_cmd += '''else
    echo "Warning: Could not find experiment directory in logs/"
fi
'''
            plotting_commands.append(plot_cmd)
        
        plotting_commands_str = '\n'.join(plotting_commands)
        
        # Create the submission script
        script_template = '''#!/bin/sh            


### select queue 
#BSUB -q gpuv100

### name of job, output file and err
#BSUB -J HRL_train_{model}_{argNum}
#BSUB -o HRL_train_{model}_{argNum}_%J.out
#BSUB -e HRL_train_{model}_{argNum}_%J.err


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
{pythonCommand}

{plottingCommands}
'''

        script_content = script_template.format(
            model=model_type,
            argNum=arg_num,
            pythonCommand=python_cmd,
            plottingCommands=plotting_commands_str
        )
        
        script_filename = f'config/experiment/HPC/submit_scripts/submit_HRL_{model_type}_arg_num_{arg_num}.sh'
        with open(script_filename, 'w') as fp:
            fp.write(script_content)
        
        print(f"Created: {script_filename}")
    
    print(f"\nCreated {len(argument_files)} submission scripts")

if __name__ == "__main__":
    createSubmitScripts()
