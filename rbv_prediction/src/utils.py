import numpy as np
from torchvision import transforms
from torchvision.transforms import functional

def process_label(label):
    """
    Creates a 3x256x256 tenser to be fed as input to the model
    """

    label = transforms.Resize(
        (256, 256), interpolation=functional.InterpolationMode.NEAREST
    )(label)
    label_np = np.zeros((256, 256, 3))
    label_np[label == 0, 0] = 1
    label_np[label == 1, 1] = 1
    label_np[label == 2, 2] = 1
    label = functional.to_tensor(label_np).float()

    return label

def downsample_rbv(rbv8):
    """
    Takes an 8x8 RBV and generates 1x1, 2x2 and 4x4 RBVx
    """
    rbvs = []
    for i in range(3):
        size = 2**i
        step = 8 // size
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