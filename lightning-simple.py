#!/usr/bin/env python

from argparse import ArgumentParser
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pytorch_lightning as pl
from model import Net

class PlNet(pl.LightningModule):
    def __init__(self, hparams):
        super(PlNet, self).__init__()

        self.model = Net()
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        data = data.view(data.shape[0], -1)
        output = self.model(data)
        return {'loss': F.nll_loss(output, target)}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.3081, ))
                        ])),
                    batch_size=self.hparams.batch_size, shuffle=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                self.hparams.lr)

def main(hparams):
    trainer = pl.Trainer()
    model = PlNet(hparams)

    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)

    hparams = parser.parse_args()
    main(hparams)
