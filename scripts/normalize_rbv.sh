#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=normalize_rbv
#SBATCH --output=normalize_rbv.log

source notify
set -e

# Strip trailing slash
DIR=${1%/}

# Copy dataset
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/$DIR /ssd_scratch/cvit/ishaanshah/

pushd ~/botanical-trees
source /home2/ishaanshah/anaconda3/bin/activate botanical_trees

echo "Normalizing RBVs across species"
python utils/normalize_rbv.py /ssd_scratch/cvit/ishaanshah/$(basename $DIR)

# Copy transformed dataset back to share3
rsync -azh --info=progress2 /ssd_scratch/cvit/ishaanshah/$(basename $DIR) ada.iiit.ac.in:/share3/ishaanshah/
