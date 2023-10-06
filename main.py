from pathlib import Path
from Dataset import SIJDataset
from SIJDenseNet import SIJDenseNet
from SIJResNet import SIJResNet
from SIJEnsemble import SIJEnsemble
from utils import set_train_and_val_transforms, set_model_checkpoints, compare_models
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from dotenv import load_dotenv

load_dotenv()

# Set constants
TRAIN_ROOT_PATH = os.getenv('TRAIN_ROOT_PATH')
TRAIN_SUBJECTS = os.getenv('TRAIN_SUBJECTS')
VAL_ROOT_PATH = os.getenv('VAL_ROOT_PATH')
VAL_SUBJECTS = os.getenv('VAL_SUBJECTS')
LABELS_PATH = os.getenv('LABELS_PATH')
RESNET_LOG_PATH = os.getenv('RESNET_LOG_PATH')
DENSENET_LOG_PATH = os.getenv('DENSENET_LOG_PATH')
ENSEMBLE_LOG_PATH = os.getenv('ENSEMBLE_LOG_PATH')

if __name__ == '__main__':
    print('Program Started.')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_transforms, val_transforms = set_train_and_val_transforms()
    train_dataset = SIJDataset(LABELS_PATH, TRAIN_SUBJECTS, TRAIN_ROOT_PATH, transform=train_transforms)
    val_dataset = SIJDataset(LABELS_PATH, VAL_SUBJECTS, VAL_ROOT_PATH, transform=val_transforms)

    batch_size = 16
    num_workers = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    modelA = SIJResNet()
    modelA.to(device)

    modelB = SIJDenseNet()
    modelB.to(device)

    checkpoint_callbacks = set_model_checkpoints()
    trainerA = pl.Trainer(accelerator='mps', max_epochs=50, logger=TensorBoardLogger(RESNET_LOG_PATH), enable_progress_bar=True, log_every_n_steps=1,
                          callbacks=[checkpoint_callbacks.resnet])
    trainerA.fit(modelA, train_loader, val_loader)
    trainerB = pl.Trainer(accelerator='mps', max_epochs=50, logger=TensorBoardLogger(DENSENET_LOG_PATH), enable_progress_bar=True, log_every_n_steps=1,
                          callbacks=[checkpoint_callbacks.densenet])
    trainerB.fit(modelB, train_loader, val_loader)

    model = SIJEnsemble(modelA, modelB)
    trainer = pl.Trainer(accelerator='mps', max_epochs=50, logger=TensorBoardLogger(ENSEMBLE_LOG_PATH),
                         callbacks=[checkpoint_callbacks.ensemble], log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)

    model_test = SIJEnsemble.load_from_checkpoint(checkpoint_callbacks.ensemble.best_model_path)
    model_test.eval()
