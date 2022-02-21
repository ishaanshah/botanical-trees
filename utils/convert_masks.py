""" Converts the RGB segmentation masks to grayscale """
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

def convert_to_grayscale(img):
    gray_img = np.zeros((img.shape[0], img.shape[1]))

    # Trunk
    trunk = cv2.inRange(img, (0, 0, 100), (150, 150, 255))
    gray_img[trunk == 255] = 1
    
    # Leaves
    leaves = cv2.inRange(img, (0, 100, 0), (150, 255, 150))
    gray_img[leaves == 255] = 2

    return gray_img

parser = ArgumentParser()
parser.add_argument("data_root", type=str, help="Path of the directory which contains the dataset")

args = parser.parse_args()

root = Path(args.data_root)
masks = sorted(root.glob("**/mask.png"))
for mask in tqdm(masks):
    img = cv2.cvtColor(cv2.imread(str(mask)), cv2.COLOR_BGR2RGB)
    gray_img = convert_to_grayscale(img)
    cv2.imwrite(str(mask.parent / "gray_mask.png"), gray_img)
