import pytorch_lightning as pl
import torch.nn
import torchmetrics.classification

from SIJResNet import SIJResNet
from SIJDenseNet import SIJDenseNet


class SIJEnsemble(pl.LightningModule):
    def __init__(self, ModelA=SIJResNet(), ModelB=SIJDenseNet()):
        super(SIJEnsemble, self).__init__()
        self.modelA = ModelA
        self.modelB = ModelB
        self.modelA.freeze()
        self.modelB.freeze()


        num_classes = 1
        self.classifier = torch.nn.Linear(2, num_classes)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task='binary')

        # self.save_hyperparameters(ignore=['modelA_params', 'modelB_params'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        x1 = self.modelA(data=x)
        x2 = self.modelB(data=x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.sigmoid(self.loss_fn(self.forward(x), y))
        acc = self.accuracy(self.forward(x), y)

        self.log("Train Loss", loss)
        self.log("Train Accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.sigmoid(self.loss_fn(self.forward(x), y))
        acc = self.accuracy(self.forward(x), y)

        self.log("Val Loss", loss)
        self.log("Val Accuracy", acc)
        return loss

