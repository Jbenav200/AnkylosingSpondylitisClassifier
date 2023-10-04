from pathlib import Path
from Dataset import SIJDataset
from SIJDenseNet import SIJDenseNet
from SIJResNet import SIJResNet
from SIJEnsemble import SIJEnsemble
from utils import set_train_and_val_transforms, set_model_checkpoint, compare_models
import torch
import pytorch_lightning as pl


TRAIN_ROOT_PATH = Path('/path/to/train/images')
TRAIN_SUBJECTS = Path('/path/to/train_subjects.npy')
VAL_ROOT_PATH = Path('/path/to/val/images')
VAL_SUBJECTS = Path('/path/to/val_subjects.npy')
LABELS_PATH = Path('/path/to/labels.csv')

if __name__ == '__main__':
    print('Program Started.')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_transforms, val_transforms = set_train_and_val_transforms()
    train_dataset = SIJDataset(LABELS_PATH, TRAIN_SUBJECTS, TRAIN_ROOT_PATH, transform=train_transforms)
    val_dataset = SIJDataset(LABELS_PATH, VAL_SUBJECTS, VAL_ROOT_PATH, transform=val_transforms)

    batch_size = 16
    num_workers = 10

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    modelA = SIJResNet()
    modelA.to(device)

    modelB = SIJDenseNet()
    modelB.to(device)

    trainerA = pl.Trainer(accelerator='mps', max_epochs=50, enable_progress_bar=True, log_every_n_steps=1)
    trainerA.fit(modelA, train_dataloaders=train_loader)
    trainerB = pl.Trainer(accelerator='mps', max_epochs=50, enable_progress_bar=True, log_every_n_steps=1)
    trainerB.fit(modelB, train_dataloaders=train_loader)
    checkpoint_callback = set_model_checkpoint()

    model = SIJEnsemble(modelA.hparams, modelB.hparams, modelA.state_dict(), modelB.state_dict())
    trainer = pl.Trainer(accelerator='mps', max_epochs=50, callbacks=[checkpoint_callback], log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader)

    model_test = SIJEnsemble.load_from_checkpoint(checkpoint_callback.best_model_path, hparams_file=checkpoint_callback.best_model_path.split("checkpoints")[0]+"hparams.yaml")

    compare_models(modelA, model.modelA)
    compare_models(model.modelA, model_test.modelA)
    compare_models(modelB, model.modelB)
    compare_models(model.modelB, model_test.modelB)

