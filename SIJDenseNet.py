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
        self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        self.model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1024, out_features=1, bias=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.accuracy = torchmetrics.classification.Accuracy(task='binary')

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        # self.save_hyperparameters()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = self.loss_fn(pred, label)
        acc = self.accuracy(pred, label)

        self.log("Train Loss", loss)
        self.log("Train Step Accuracy", self.train_acc(torch.sigmoid(pred), label.int()))
        return loss

    def on_train_epoch_end(self):
        self.log("Train Accuracy", self.train_acc.compute())
        self.log("Train Precision", self.train_precision.compute())
        self.log("Train Recall", self.train_recall.compute())
        self.log("Train F1 Score", self.train_f1.compute())

    def validation_step(self, batch, batch_idx):
        mri, label = batch
        pred = self(mri)
        loss = self.loss_fn(pred, label)
        acc = self.accuracy(pred, label)

        self.log("Val Loss", loss)
        self.log("Val Step Accuracy", self.val_acc(torch.sigmoid(pred), label.int()))
        return

    def on_validation_epoch_end(self) -> None:
        self.log("Val Accuracy", self.val_acc.compute())
        self.log("Val Precision", self.val_precision.compute())
        self.log("Val Recall", self.val_recall.compute())
        self.log("Val F1 Score", self.val_f1.compute())

    def configure_optimizers(self):
        return [self.optimizer]
