#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

# Activate conda environment
source activate $HOME/miniconda3/envs/geometry

# Copy input files to scratch, excluding .git directories
rsync -a --exclude='.git' $HOME/geometry-of-truth /scratch-shared/$USER/
# rsync -a --exclude='.git' $HOME/llama-13b /scratch-shared/$USER/

# Move$HOME/geometry-of-truth /scratch-shared/$USER to correct folder
cd /scratch-shared/$USER/geometry-of-truth

# Define experiment name
EXPERIMENT_NAME="70b_religion_MM"

# Run command
#python interventions.py --model llama-3-70b --device remote --experiment_name ${EXPERIMENT_NAME}_NonePos --train_datasets experiment_intra_religion_stereoset --val_dataset experiment_inter_religion_stereoset --probe LRProbe --intervention none --subset true
#python interventions.py --model llama-3-70b --device remote --experiment_name ${EXPERIMENT_NAME}_NoneNeg --train_datasets experiment_intra_religion_stereoset --val_dataset experiment_inter_religion_stereoset --probe LRProbe --intervention none --subset false
python interventions.py --model llama-3-70b --device remote --experiment_name ${EXPERIMENT_NAME}_AddNeg --train_datasets experiment_intra_religion_stereoset --val_dataset experiment_inter_religion_stereoset --probe MMProbe --intervention add --subset false
python interventions.py --model llama-3-70b --device remote --experiment_name ${EXPERIMENT_NAME}_SubPos --train_datasets experiment_intra_religion_stereoset --val_dataset experiment_inter_religion_stereoset --probe MMProbe --intervention subtract --subset true

# Copy output directory from scratch to correct folder in home
cp /scratch-shared/tpungas/geometry-of-truth/experimental_outputs/${EXPERIMENT_NAME}* $HOME/geometry-of-truth/experimental_outputs/intervention/
