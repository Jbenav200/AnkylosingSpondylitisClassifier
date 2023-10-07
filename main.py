from pathlib import Path
from Dataset import SIJDataset
from SIJDenseNet import SIJDenseNet
from SIJResNet import SIJResNet
from SIJEnsemble import SIJEnsemble
from utils import set_train_and_val_transforms, set_model_checkpoints, train_model, print_model_metrics, ignore_warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
from dotenv import load_dotenv
import sys

load_dotenv()

# Set constants
TRAIN_ROOT_PATH = os.getenv('TRAIN_ROOT_PATH')
TRAIN_SUBJECTS = os.getenv('TRAIN_SUBJECTS')
VAL_ROOT_PATH = os.getenv('VAL_ROOT_PATH')
VAL_SUBJECTS = os.getenv('VAL_SUBJECTS')
TEST_ROOT_PATH = os.getenv('TEST_ROOT_PATH')
TEST_SUBJECTS = os.getenv('TEST_SUBJECTS')
LABELS_PATH = os.getenv('LABELS_PATH')
RESNET_LOG_PATH = os.getenv('RESNET_LOG_PATH')
DENSENET_LOG_PATH = os.getenv('DENSENET_LOG_PATH')
ENSEMBLE_LOG_PATH = os.getenv('ENSEMBLE_LOG_PATH')
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH')
SAVED_ENSEMBLE_PATH = os.getenv('SAVED_ENSEMBLE_PATH')
SAVED_RESNET_PATH = os.getenv('SAVED_RESNET_PATH')
SAVED_DENSENET_PATH = os.getenv('SAVED_DENSENET_PATH')

if __name__ == '__main__':
    print(f"Program started in {sys.argv[1]} mode.")
    # set the device to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # set the data transforms and load the datasets
    train_transforms, val_transforms, test_transforms = set_train_and_val_transforms()
    train_dataset = SIJDataset(LABELS_PATH, TRAIN_SUBJECTS, TRAIN_ROOT_PATH, transform=train_transforms)
    val_dataset = SIJDataset(LABELS_PATH, VAL_SUBJECTS, VAL_ROOT_PATH, transform=val_transforms)
    test_dataset = SIJDataset(LABELS_PATH, TEST_SUBJECTS, TEST_ROOT_PATH, transform=test_transforms)

    # if sys.argv[1] is 'train', start the training process
    if sys.argv[1] == 'train':
        batch_size = 16
        num_workers = 8

        # create the train and validation dataloaders.
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                                 shuffle=False)
        # create an instance of the modified ResNet18 model
        modelA = SIJResNet()
        modelA.to(device)

        # create an instance of the modified DenseNet121 model
        modelB = SIJDenseNet()
        modelB.to(device)

        # set up the callbacks for each model
        resnet_callback, densenet_callback, ensemble_callback = set_model_checkpoints()

        # train the modified ResNet and DenseNet models
        train_model(modelA, RESNET_LOG_PATH, resnet_callback, train_loader, val_loader)
        train_model(modelB, DENSENET_LOG_PATH, densenet_callback, train_loader, val_loader)

        # train the SIJ Ensemble classifier
        model = SIJEnsemble(modelA, modelB, resnet_callback.best_model_path, densenet_callback.best_model_path)
        train_model(model, ENSEMBLE_LOG_PATH, ensemble_callback, train_loader, val_loader)

        # if the second argument passed to the file is 'save' then save each of the models at their best checkpoints
        if sys.argv[2] is not None and sys.argv[2] == 'save':
            resnet = SIJResNet.load_from_checkpoint(resnet_callback.best_model_path)
            resnet.to(device)
            torch.save(resnet, MODEL_SAVE_PATH+'/resnet.pt')
            densenet = SIJDenseNet.load_from_checkpoint(densenet_callback.best_model_path)
            densenet.to(device)
            torch.save(densenet, MODEL_SAVE_PATH+'/densenet.pt')
            ensemble = SIJEnsemble.load_from_checkpoint(ensemble_callback.best_model_path)
            ensemble.to(device)
            torch.save(ensemble, MODEL_SAVE_PATH+'/ensemble.pt')
    # if the first argument passed to the file is results, then analyse the performance of the ensemble classifier
    # and print the results to the terminal.
    elif sys.argv[1] == 'results':
        try:
            # Load the modified ResNet18 model from the best model checkpoint
            test_model_a = SIJResNet.load_from_checkpoint(SAVED_RESNET_PATH)
            # send the model to GPU
            test_model_a.to(device)
            # set the model to evaluate
            test_model_a.eval()

            # Print the ResNet18 classifier metrics
            print_model_metrics(test_model_a, 'resnet model', device, test_dataset)

            # load the modified DenseNet121 model from the best model checkpoint
            test_model_b = SIJDenseNet.load_from_checkpoint(SAVED_DENSENET_PATH)
            # send the model to GPU
            test_model_b.to(device)
            # set the model to evaluate
            test_model_b.eval()

            # Print the DenseNet121 classifier metrics
            print_model_metrics(test_model_b, 'densenet model', device, test_dataset)

            ensemble = torch.load(SAVED_ENSEMBLE_PATH)
            ensemble.eval()
            print_model_metrics(ensemble, 'ensemble classifier', device, test_dataset)
        except FileNotFoundError as e:
            print(e)
    else:
        print('Argument passed was invalid')
