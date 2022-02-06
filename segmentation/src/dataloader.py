import os
import torch
import glob
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import functional
from PIL import Image


class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, "Images", "*.*"))
        self.label_files = []
        self.mode = mode
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename}.png"
            self.label_files.append(
                os.path.join(folder_path, "Labels", label_filename_with_ext)
            )

    def transform(self, image, label):
        # Resize
        if self.mode == "val":
            color_transforms = transforms.Compose(
                [
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            # Random horizontal flip
            if random.random() > 0.5:
                image = functional.hflip(image)
                label = functional.hflip(label)

            # Random crop
            # TODO: Check if this works
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=[0.7, 1.0], ratio=[0.8, 1.3])
            image = functional.crop(image, i, j, h, w)
            label = functional.crop(label, i, j, h, w)

            color_transforms = transforms.Compose(
                [
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1)),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        resize = transforms.Resize((512, 512))
        image = resize(image)
        label = resize(label)

        image = functional.to_tensor(image)
        label = functional.to_tensor(label)

        # Normalize back from [0, 1] to [0, 255]
        label *= 255

        # Apply color transforms (only RGB image)
        image = color_transforms(image)

        return image, label

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        # Join label and image
        # image_np = np.asarray(image)
        # label_np = np.asarray(label)
        # image_and_label_np = np.zeros((image_np.shape[0], image_np.shape[1], image_np.shape[2]+1), image_np.dtype)
        # image_and_label_np[:,:,:3] = image_np
        # image_and_label_np[:,:,3] = label_np
        # image_and_label = Image.fromarray(image_and_label_np) 

        # Apply transforms
        # image_and_label = self.transforms(image_and_label)
        # image = image_and_label[:3,:,:]
        # label = image_and_label[3:,:,:].unsqueeze(0)

        image, label = self.transform(image, label)

        #  Convert to int64 and remove second dimension
        label = label.long().squeeze()

        return image, label

    def __len__(self):
        return len(self.img_files)
