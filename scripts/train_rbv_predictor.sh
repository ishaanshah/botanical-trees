#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=rbv_predictor
#SBATCH --output=rbv_predictor.log

# Discord notifs on start and end
source notify

# Fail on error
set -e

# Copy dataset
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/tree_dataset /ssd_scratch/cvit/ishaanshah/

# Activate Conda environment
source /home2/ishaanshah/anaconda3/bin/activate botanical_trees

# Training script
pushd ~/botanical-trees
wandb on

python rbv_prediction/main.py /ssd_scratch/cvit/ishaanshah/tree_dataset --max_epochs $2 --use_gt --gpus=1 --batch_size=32 --lr $1

popd
