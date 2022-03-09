#!/bin/bash
#SBATCH -A research
#SBATCH -n 2
##SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=transform_dataset
#SBATCH --output=transform_dataset.log

# Copy dataset
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/tree_dataset /ssd_scratch/cvit/ishaanshah/
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/cityscapes_dataset/leftImg8bit /ssd_scratch/cvit/ishaanshah/cityscapes_dataset
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/landscape_dataset /ssd_scratch/cvit/ishaanshah/

pushd ~/botanical-trees
source ./venv/bin/activate

echo "Adding background to renders"
python utils/add_background.py /ssd_scratch/cvit/ishaanshah/tree_dataset /ssd_scratch/cvit/ishaanshah/cityscapes_dataset /ssd_scratch/cvit/ishaanshah/landscape_dataset

echo "Converting masks to grayscale"
python utils/convert_masks.py /ssd_scratch/cvit/ishaanshah/tree_dataset

echo "Normalizing RBVs across species"
python utils/normalize_rbv.py /ssd_scratch/cvit/ishaanshah/tree_dataset

# Copy transformed dataset back to share3
rsync -azh --info=progress2 /ssd_scratch/cvit/ishaanshah/tree_dataset ada.iiit.ac.in:/share3/ishaanshah/
