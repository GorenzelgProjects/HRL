#!/bin/sh

# SET JOB NAME
#BSUB -J HRL_experiment

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 1

# gb memory per core
#BSUB -R "rusage[mem=16G]"

# request 32GB of GPU-memory
#BSUB -R "select[gpu32gb]"

# walltime (adjust as needed)
#BSUB -W 02:30

# output and error files
#BSUB -o config/experiment/HPC/output_%J.out 
#BSUB -e config/experiment/HPC/error_%J.err 

# -- end of LSF options --

# Load modules
module load python3/3.12.11
module load cuda/11.8

# Activate virtual environment
source .venv/bin/activate

# Change to project directory (adjust path as needed)
cd $LS_SUBCWD || cd /work3/s190464/HRL || cd ~/Documents/HRL

# Run experiment (example - adjust as needed)
# For option_critic:
# python main.py experiment=option_critic_example environment=four_rooms

# For option_critic_nn:
# python main.py experiment=option_critic_nn_example environment=thin_ice experiment.cuda=true

wait
