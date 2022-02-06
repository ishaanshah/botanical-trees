import torch
from torch import nn

from pytorch_lightning.core.lightning import LightningModule


class RBVPredictor(LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()

        self.lr = lr
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, padding=2, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64, out_channels=64, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=128, out_channels=128, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=128, out_channels=256, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=256, out_channels=256, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=256, out_channels=512, padding=2, kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.base = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
        )

        # Head to predict 1x1 RBV
        self.head1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4, out_features=4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4, out_features=1),
        )
        self.head1_post = nn.Linear(in_features=1, out_features=16)

        # Head to predict 2x2 RBV
        self.head2 = nn.Sequential(
            nn.Linear(in_features=272, out_features=272),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=272, out_features=16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=16, out_features=4),
        )
        self.head2_post = nn.Linear(in_features=4, out_features=32)

        # Head to predict 4x4 RBV
        self.head4 = nn.Sequential(
            nn.Linear(in_features=288, out_features=288),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=288, out_features=64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=16),
        )
        self.head4_post = nn.Linear(in_features=16, out_features=64)

        # Head to predict 8x8 RBV
        self.head8 = nn.Sequential(
            nn.Linear(in_features=320, out_features=320),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=320, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=64),
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.base(x)

        # 1x1 prediction
        y1 = self.head1(x)
        x1 = self.head1_post(y1)

        # 2x2 prediction
        y2 = self.head2(torch.cat((x, x1)))
        x2 = self.head2_post(y2)

        # 4x4 prediction
        y4 = self.head4(torch.cat((x, x2)))
        x4 = self.head4_post(y4)

        # 8x8 prediction
        y8 = self.head8(torch.cat((x, x4)))

        return y1, y2, y4, y8

    def training_step(self, batch):
        x, *y = batch
        yp = self(x)
        criterion = nn.MSELoss()
        losses = []
        for i in range(len(y)):
            losses.append(criterion(y[i], yp[i]))
        loss = sum(losses)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
