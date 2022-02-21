import torch
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.model import RBVPredictor
from src.dataloader import DataLoaderRBV
from torch.utils.data import random_split, DataLoader

def main(args):
    # Create logger object
    wandb_logger = WandbLogger(project="RBVPrediction")

    # Create the trainer
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

    # Create the model
    model = RBVPredictor(args)

    # Create dataloaders
    dataset = DataLoaderRBV(args.dataset, args.use_gt)
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()

    # Program specific arguments
    parser.add_argument("dataset", type=str, help="Path to the datset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to train with")
    parser.add_argument("--use_gt", action="store_true", default="./dataset", help="Should the model use ground truth segmentation masks for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Percentage of dataset to use for validation")

    # Model specific arguments
    parser = RBVPredictor.add_model_specific_args(parser)

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # Parse all arguments
    args = parser.parse_args()

    main(args)
