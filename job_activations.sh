#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00

# Activate conda environment
source activate $HOME/miniconda3/envs/geometry

# Copy input files to scratch, excluding .git directories
rsync -a --exclude='.git' $HOME/geometry-of-truth /scratch-shared/$USER/
#rsync -a --exclude='.git' $HOME/llama-13b /scratch-shared/$USER/

# Move$HOME/geometry-of-truth /scratch-shared/$USER to correct folder
cd /scratch-shared/$USER/geometry-of-truth

# Run command
# python generate_acts.py --layers 12 --datasets experiment_intra_stereoset experiment_inter_stereoset experiment_intra_profession_stereoset experiment_intra_religion_stereoset experiment_intra_race_stereoset experiment_inter_profession_stereoset experiment_inter_religion_stereoset experiment_inter_race_stereoset --device remote --model llama-3-8b
python generate_acts.py --layers 12 --datasets experiment_intra_stereoset --device remote --model llama-3-70b

# The results can be accessed at /scratch-shared/tpungas/geometry-of-truth/acts/llama-3-8b/
