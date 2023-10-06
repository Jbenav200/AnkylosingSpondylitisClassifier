from pathlib import Path
from Dataset import SIJDataset
from SIJDenseNet import SIJDenseNet
from SIJResNet import SIJResNet
from SIJEnsemble import SIJEnsemble
from utils import set_train_and_val_transforms, set_model_checkpoints, train_model, print_model_metrics
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

    train_model(modelA, RESNET_LOG_PATH, checkpoint_callbacks.resnet, train_loader, val_loader)
    train_model(modelB, DENSENET_LOG_PATH, checkpoint_callbacks.densenet, train_loader, val_loader)

    test_model_a = SIJResNet.load_from_checkpoint(checkpoint_callbacks.resnet.best_model_path)
    test_model_a.eval()

    # Print the ResNet18 classifier metrics
    print_model_metrics(test_model_a, 'resnet model', device, val_dataset)

    test_model_b = SIJDenseNet.load_from_checkpoint(checkpoint_callbacks.densenet.best_model.path)
    test_model_b.eval()

    # Print the DenseNet121 classifier metrics
    print_model_metrics(test_model_b, 'densenet model', device, val_dataset)

    model = SIJEnsemble(modelA, modelB)
    train_model(model, ENSEMBLE_LOG_PATH, checkpoint_callbacks.ensemble, train_loader, val_loader)

    model_test = SIJEnsemble.load_from_checkpoint(checkpoint_callbacks.ensemble.best_model_path)
    model_test.eval()

    # print ensemble classifier metrics.
    print_model_metrics(model_test, 'ensemble classifier', device, val_dataset)
