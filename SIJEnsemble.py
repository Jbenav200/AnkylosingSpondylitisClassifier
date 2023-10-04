import pytorch_lightning as pl
import torch.nn
import torchmetrics.classification

from SIJResNet import SIJResNet
from SIJDenseNet import SIJDenseNet


class SIJEnsemble(pl.LightningModule):
    def __init__(self, modelA_hparams, modelB_hparams,modelA_params=None, modelB_params=None):
        super(SIJEnsemble, self).__init__()
        self.modelA = SIJResNet(**modelA_hparams)
        self.modelB = SIJDenseNet(**modelB_hparams)
        self.modelA.freeze()
        self.modelB.freeze()

        if modelA_params:
            self.modelA.load_state_dict(modelA_params)
        if modelB_params:
            self.modelB.load_state_dict(modelB_params)

        num_classes = 1
        self.classifier = torch.nn.Linear(2 * num_classes, num_classes)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task='binary')

        self.save_hyperparameters(ignore=['modelA_params', 'modelB_params'])

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.sigmoid(self.loss_fn(self.forward(x), y))
        acc = self.accuracy(self.forward(x), y)
        self.log("Train Accuracy", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.sigmoid(self.loss_fn(self.forward(x), y))
        acc = self.accuracy(self.forward(x), y)

        self.log("Test Accuracy", acc)
        return loss
