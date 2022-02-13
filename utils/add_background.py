""" Adds background to RGB renders """
from argparse import ArgumentParser
from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm

def get_random_crop(img, crop_height, crop_width):
    h, w = img.shape[:2]
    
    # Upscale if needed
    th, tw = h, w
    if h <= w and h <= crop_height:
        th = crop_height + 10
        tw = int(th / h * w + 1)
    elif w < h and w <= crop_width:
        tw = crop_width + 10
        th = int(tw / w * h + 1)
    img = cv2.resize(img, (tw, th))
    
    # Crop
    max_x = img.shape[1] - crop_width
    max_y = img.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = img[y: y + crop_height, x: x + crop_width]

    return crop

def add_background(img, city_bckg, land_bckg):
    if random.choice([True, False]):
        bckg = cv2.imread(str(random.choice(city_bckg)))
    else:
        bckg = cv2.imread(str(random.choice(land_bckg)))

    alpha = img[:,:,3].astype(float) / 255
    comp = np.zeros((img.shape[0], img.shape[1], 3), dtype=int)
    bckg_c = get_random_crop(bckg, img.shape[0], img.shape[1])
    for i in range(3):
        comp[:,:,i] = alpha * img[:,:,i] + (1 - alpha) * bckg_c[:,:,i]

    return comp

parser = ArgumentParser()
parser.add_argument("data_root", type=str, help="Path of the directory which contains the dataset")
parser.add_argument("cityscapes_root", type=str, help="Path of the directory which contains the cityscapes dataset")
parser.add_argument("landscapes_root", type=str, help="Path of the directory which contains the landscapes dataset")

args = parser.parse_args()

city_bckg = list(Path(args.cityscapes_root).glob("**/*.png"))
land_bckg = list(Path(args.landscapes_root).glob("**/*.jpg"))

root = Path(args.data_root)
rgbs = sorted(root.glob("**/rgb.png"))
for rgb in tqdm(rgbs):
    img = cv2.imread(str(rgb), cv2.IMREAD_UNCHANGED)
    comp = add_background(img, city_bckg, land_bckg)
    cv2.imwrite(str(rgb.parent / "composite.png"), comp)
