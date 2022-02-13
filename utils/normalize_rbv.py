""" Normalizes RBVS across species """

import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("data_root", type=str, help="Path of the directory which contains the dataset")

args = parser.parse_args()

root = Path(args.data_root)
for dir in root.iterdir():
    if not dir.is_dir():
        continue

    print(f"Normalizing for {dir.name}")
    files = sorted(dir.glob("**/info.json"))

    # Read RBVs into a numpy array
    rbvs = np.zeros((len(files), 8, 8))
    for i in range(len(files)):
        with open(files[i]) as f:
            rbvs[i] = np.asarray(json.load(f)['rbv'])

    # Normalize
    rbvs = (rbvs - rbvs.min()) / (rbvs.max() - rbvs.min())

    # Store back normalized RBV
    for i in range(len(files)):
        with open(files[i], "r") as f:
            info = json.load(f)

        info['norm_rbv'] = rbvs[i].tolist()

        with open(files[i], "w") as f:
            json.dump(info, f)
