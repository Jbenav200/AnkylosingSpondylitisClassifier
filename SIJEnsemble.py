import pytorch_lightning as pl
import torch.nn
import torchmetrics.classification

from SIJResNet import SIJResNet
from SIJDenseNet import SIJDenseNet


class SIJEnsemble(pl.LightningModule):
    def __init__(self, modelA=SIJResNet(), modelB=SIJDenseNet(), modelA_callback_path=None, modelB_callback_path=None):
        super(SIJEnsemble, self).__init__()
        self.modelA = modelA.load_from_checkpoint(modelA_callback_path)
        self.modelB = modelB.load_from_checkpoint(modelB_callback_path)
        self.modelA.freeze()
        self.modelB.freeze()

        num_classes = 1
        self.classifier = torch.nn.Linear(2, num_classes)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()

        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()

        # self.save_hyperparameters(ignore=['modelA_params', 'modelB_params'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        x1 = self.modelA(data=x)
        x2 = self.modelB(data=x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.forward(x), y)

        self.log("Train Loss", loss)
        self.log("Train Step Accuracy", self.train_accuracy(self.forward(x), y))
        return loss

    def on_train_epoch_end(self):
        self.log("Train Accuracy", self.train_accuracy.compute())
        self.log("Train Precision", self.train_precision.compute())
        self.log("Train Recall", self.train_recall.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.forward(x), y)

        self.log("Val Loss", loss)
        self.log("Val Step Accuracy", self.val_accuracy(self.forward(x), y))
        return loss

    def on_validation_epoch_end(self):
        self.log("Val Accuracy", self.val_accuracy.compute())
        self.log("Val Precision", self.val_precision.compute())
        self.log("Val Recall", self.val_recall.compute())
