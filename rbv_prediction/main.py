import torch
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.model import RBVPredictor
from src.dataloader import DataLoaderRBV

def main(args):
    # Create logger object
    wandb_logger = WandbLogger(project="RBVPrediction")

    # Create the trainer
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

    # Create the model
    model = RBVPredictor(args)

    # Create dataloaders
    datasets = {
        x: DataLoaderRBV(args.dataset, x, args.use_gt)
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    # Train the model
    trainer.fit(model, dataloaders["train"], dataloaders["val"])

if __name__ == "__main__":
    parser = ArgumentParser()

    # Program specific arguments
    parser.add_argument("--log", action="store_true", help="Log results to WandB")
    parser.add_argument("--dataset", type=str, default="./dataset", help="Path to the datset")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size to train with")
    parser.add_argument("--use_gt", action="store_true", default="./dataset", help="Should the model use ground truth segmentation masks for training")

    # Model specific arguments
    parser = RBVPredictor.add_model_specific_args(parser)

    # Trainer specific arguments
    parser = Trainer.add_argparse_args(parser)

    # Parse all arguments
    args = parser.parse_args()

    main(args)
