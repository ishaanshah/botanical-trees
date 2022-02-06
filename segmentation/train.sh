#!/bin/bash
##SBATCH -A ishaanshah
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=deeplabv3
#SBATCH --output=output.txt

rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/pytorch_deeplab_dataset/dataset /ssd_scratch/cvit/ishaanshah/

module load python/3.9.1
pushd sources
source ~/Segmentation/venv/bin/activate
wandb on
python main.py /ssd_scratch/cvit/ishaanshah/dataset /ssd_scratch/cvit/ishaanshah/results --epochs 25 --batch_size 2 --num_classes 3 --encoder 'tu-xception41'
popd

rsync -azh --info=progress2 /ssd_scratch/cvit/ishaanshah/results/best_weight.pth ada.iiit.ac.in:/home2/ishaanshah/xception.pth
