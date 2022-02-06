import json
import os
import torch
import glob
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image


class DataLoaderRBV(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode, use_gt=False):
        super(DataLoaderRBV, self).__init__()
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

    def generate_coarse_rbv(self, rbv8):
        """
        Takes an 8x8 RBV and generates 1x1, 2x2 and 4x4 RBVx
        """
        rbvs = []
        for i in range(3):
            size = 2**i
            step = 8 / size
            rbv = np.zeros((size, size))
            for j in range(size):
                for k in range(size):
                    rbv[j, k] = np.max(
                        rbv8[
                            j * step : (j + 1) * step, k * step : (k + 1) * step
                        ]
                    )

            rbvs.append(rbv)

        return rbvs

    def __getitem__(self, index):
        info_path = self.info_files[index]
        label_path = self.label_files[index]

        with open(info_path) as f:
            rbv8 = np.asarray(json.load(f)["rbv"])

        # Create a 3x256x256 tensor to be fed as input to the model
        label = Image.open(label_path)
        label = transforms.Resize(
            (256, 256), interpolation=functional.InterpolationMode.NEAREST
        )(label)
        label_np = np.zeros((256, 256, 3))
        label_np[label == 0, 0] = 1
        label_np[label == 1, 1] = 1
        label_np[label == 2, 2] = 1
        label = functional.to_tensor(label_np)

        # Generate 1x1, 2x2, 4x4 RBVs from 8x8
        rbv1, rbv2, rbv4 = self.generate_coarse_rbv(rbv8)
        rbv1 = torch.from_numpy(rbv1)
        rbv2 = torch.from_numpy(rbv2)
        rbv4 = torch.from_numpy(rbv4)
        rbv8 = torch.from_numpy(rbv8)

        return label, rbv1, rbv2, rbv4, rbv8

    def __len__(self):
        return len(self.info_files)
