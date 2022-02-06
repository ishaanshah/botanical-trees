import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
import pathlib
import wandb

# Local import
from src.model import initialize_model
from src.dataloader import DataLoaderSegmentation
from src.train import train_model

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def main(
    data_dir,
    dest_dir,
    num_classes,
    batch_size,
    num_epochs,
    encoder,
    atrous_rates,
    activation,
    learning_rate,
    weight
):
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: DataLoaderSegmentation(os.path.join(data_dir, x), x)
        for x in ["train", "val"]
    }
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    print("Initializing Model...")

    # Initialize model
    model = initialize_model(
        encoder, atrous_rates, num_classes, activation
    )

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss(
        weight=(torch.FloatTensor(weight).to(device) if weight else None)
    )

    # Prepare output directory
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Train...")

    # Train and evaluate
    model_state_dict, hist = train_model(
        dataloaders_dict,
        criterion,
        optimizer_ft,
        model,
        device,
        dest_dir,
        num_classes,
        num_epochs=num_epochs,
    )

    print("Save ...")
    torch.save(
        model_state_dict,
        os.path.join(dest_dir, "best_weight.pth"),
    )


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="Specify the dataset directory path, should contain train/Images, train/Labels, val/Images and val/Labels",
    )
    parser.add_argument(
        "dest_dir", help="Specify the  directory where model weights shall be stored."
    )
    parser.add_argument(
        "--num_classes",
        default=3,
        type=int,
        help="Number of classes in the dataset, index 0 for no-label should be included in the count",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch size for training (change depending on how much memory you have)",
    )
    parser.add_argument(
        "--encoder",
        default="xception",
        type=str,
        help="Encoder backbone to use (https://smp.readthedocs.io/en/latest/encoders.html)"
    )
    parser.add_argument(
        "--activation",
        default=None,
        type=str,
        help='Activation function to apply to the output of final convolution network (“sigmoid”, “softmax”, “logsoftmax”, “tanh”, “identity”)'
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
        help="Learning rate to use to train the model"
    )
    parser.add_argument(
        "--atrous_rates",
        action="append",
        default=[6, 12, 18],
        type=int,
        help="Atrous rates to be used in the ASPP module. If this argument is used, then it should be called thrice"
    )
    parser.add_argument(
        "-w",
        action="append",
        type=float,
        help="Add more weight to some classes. If this argument is used, then it should be called as many times as there are classes (see --num_classes)",
    )

    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="Tree Segmentation", entity="ishaanshah")
    wandb.config.update(args)

    # Build weight list
    weight = []
    if args.w:
        for w in args.w:
            weight.append(w)

    main(
        args.data_dir,
        args.dest_dir,
        args.num_classes,
        args.batch_size,
        args.epochs,
        args.encoder,
        args.atrous_rates,
        args.activation,
        args.lr,
        weight
    )

if __name__ == "__main__":
    args_preprocess()
