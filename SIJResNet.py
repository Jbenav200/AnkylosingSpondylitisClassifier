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


class SIJResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = self.loss_fn(pred, label)

        self.log("Train Loss", loss)
        self.log("Train Step Acc", self.train_acc(torch.sigmoid(pred), label.int()))
        self.log("Train Step Precision", self.train_precision(self.forward(mri), label.int()))
        self.log("Train Step Recall", self.train_recall(self.forward(mri), label.int()))
        return loss

    def on_train_epoch_end(self):
        self.log("Train Accuracy", self.train_acc.compute())
        self.log("Train Precision", self.train_precision.compute())
        self.log("Train Recall", self.train_recall.compute())

    def validation_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = self.loss_fn(pred, label)

        self.log("Val Loss", loss)
        self.log("Val Step Accuracy", self.val_acc(torch.sigmoid(pred), label.int()))
        self.log("Val Step Precision", self.val_precision(self.forward(mri), label.int()))
        self.log("Val Step Recall", self.val_recall(self.forward(mri), label.int()))
        return loss

    def on_validation_epoch_end(self):
        self.log("Val Accuracy", self.val_acc.compute())
        self.log("Val Precision", self.train_precision.compute())
        self.log("Val Recall", self.train_recall.compute())

    def configure_optimizers(self):
        return [self.optimizer]
