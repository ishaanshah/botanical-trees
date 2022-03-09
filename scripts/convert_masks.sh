#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=convert_masks
#SBATCH --output=convert_masks.log

source notify
set -e

# Strip trailing slash
DIR=${1%/}

# Copy dataset
rsync -azh --info=progress2 ada.iiit.ac.in:$DIR /ssd_scratch/cvit/ishaanshah/

pushd ~/botanical-trees
source /home2/ishaanshah/anaconda3/bin/activate botanical_trees

echo "Converting masks to grayscale"
python utils/convert_masks.py /ssd_scratch/cvit/ishaanshah/$(basename $DIR)

# Copy transformed dataset back to share3
rsync -azh --info=progress2 /ssd_scratch/cvit/ishaanshah/$(basename $DIR) ada.iiit.ac.in:/share3/ishaanshah/
