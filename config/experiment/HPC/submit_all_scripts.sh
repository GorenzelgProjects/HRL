#!/bin/sh

# Submit all HPC job scripts
# This script submits all submission scripts created by createSubmitScripts.py

# Change to the directory containing submit scripts
cd config/experiment/HPC/submit_scripts || exit 1

# Submit all option_critic scripts
for script in submit_HRL_oc_arg_num_*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting $script"
        bsub < "$script"
    fi
done

# Submit all option_critic_nn scripts
for script in submit_HRL_ocnn_arg_num_*.sh; do
    if [ -f "$script" ]; then
        echo "Submitting $script"
        bsub < "$script"
    fi
done

echo "All scripts submitted!"