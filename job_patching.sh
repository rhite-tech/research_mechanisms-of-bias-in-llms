#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

# Activate conda environment
source activate $HOME/miniconda3/envs/geometry

# Copy input files to scratch, excluding .git directories
rsync -a --exclude='.git' $HOME/geometry-of-truth /scratch-shared/$USER/
# rsync -a --exclude='.git' $HOME/llama-13b /scratch-shared/$USER/

# Move$HOME/geometry-of-truth /scratch-shared/$USER to correct folder
cd /scratch-shared/$USER/geometry-of-truth

# Define experiment name
EXPERIMENT_NAME="TEST13b_patching"

# Run command
# python bias_patching.py --model llama-13b --device cuda:0 --experiment_name $EXPERIMENT_NAME
python bias_patching.py --model llama-13b --device cuda:0 --experiment_name $EXPERIMENT_NAME

# Copy output directory from scratch to correct folder in home
cp /scratch-shared/tpungas/geometry-of-truth/experimental_outputs/${EXPERIMENT_NAME}.json $HOME/geometry-of-truth/experimental_outputs/

# Create figure from output data
# cd $HOME/geometry-of-truth/
# python patching_nb.py $EXPERIMENT_NAME
