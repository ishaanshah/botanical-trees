#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=segmentation
#SBATCH --output=segmentation/log

pushd ~/botanical-trees
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/pytorch_deeplab_dataset/dataset /ssd_scratch/cvit/ishaanshah/

module load python/3.9.1
source ./venv/bin/activate
wandb on
python main.py /ssd_scratch/cvit/ishaanshah/dataset /ssd_scratch/cvit/ishaanshah/results --epochs 25 --batch_size 2 --num_classes 3 --encoder 'tu-xception41'

rsync -azh --info=progress2 /ssd_scratch/cvit/ishaanshah/results/best_weight.pth ./segmentation/models/xception.pth
popd
