import json
import torch
import numpy as np
from . import utils
from PIL import Image
from torch.utils.data.dataset import Dataset
from pathlib import Path

class DataLoaderRBV(Dataset):
    def __init__(self, folder_path, use_gt):
        super(DataLoaderRBV, self).__init__()
        self.info_files = list(Path(folder_path).glob("**/info.json"))
        self.label_files = []
        for info_file in self.info_files:
            filename = "gray_mask.png" if use_gt else "predicted_mask.png"
            self.label_files.append(
                info_file.parent / Path(filename)
            )

    def __getitem__(self, index):
        info_path = self.info_files[index]
        label_path = self.label_files[index]

        with open(info_path) as f:
            rbv8 = np.asarray(json.load(f)["norm_rbv"])

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
