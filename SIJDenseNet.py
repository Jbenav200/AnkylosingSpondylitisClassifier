import torch
from torch.nn import functional as F
import torchmetrics
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import cv2
import sys
import pandas as pd
from tqdm.notebook import tqdm


class SIJDenseNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.densenet121(pretrained=False)
        self.model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = torch.sigmoid(self.loss_fn(pred, label))

        self.log("Train Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = torch.sigmoid(self.loss_fn(pred, label))

        self.log("Val Loss", loss)
        return loss

    def configure_optimizers(self):
        return [self.optimizer]
