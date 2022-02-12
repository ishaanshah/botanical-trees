import json
import os
import torch
import glob
import numpy as np
from . import utils
from PIL import Image
from torch.utils.data.dataset import Dataset

class DataLoaderRBV(Dataset):
    def __init__(self, folder_path, mode, use_gt):
        super(DataLoaderRBV, self).__init__()
        folder_path = os.path.join(folder_path, mode)
        self.info_files = glob.glob(os.path.join(folder_path, "Info", "*.*"))
        self.label_files = []
        self.mode = mode
        self.use_gt = use_gt
        for info_file in self.info_files:
            info_filename, _ = os.path.splitext(os.path.basename(info_file))
            label_dir = "Labels" if use_gt else "LabelsPredicted"
            self.label_files.append(
                os.path.join(folder_path, label_dir, f"{info_filename}.png")
            )

    def __getitem__(self, index):
        info_path = self.info_files[index]
        label_path = self.label_files[index]

        with open(info_path) as f:
            rbv8 = np.asarray(json.load(f)["rbv"])

        label = utils.process_label(Image.open(label_path))

        # Generate 1x1, 2x2, 4x4 RBVs from 8x8
        rbv1, rbv2, rbv4 = utils.downsample_rbv(rbv8)
        rbv1 = torch.from_numpy(rbv1).flatten().float()
        rbv2 = torch.from_numpy(rbv2).flatten().float()
        rbv4 = torch.from_numpy(rbv4).flatten().float()
        rbv8 = torch.from_numpy(rbv8).flatten().float()

        return label, rbv1, rbv2, rbv4, rbv8

    def __len__(self):
        return len(self.info_files)
