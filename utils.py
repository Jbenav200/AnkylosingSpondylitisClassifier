from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torchmetrics
from tqdm.notebook import tqdm
from pytorch_lightning.loggers import TensorBoardLogger


def set_train_and_val_transforms():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.50, 0.67)
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.50, 0.67)
    ])

    return train_transforms, val_transforms


def set_model_checkpoints():
    ensemble_callback = ModelCheckpoint(
        monitor='Val Loss',
        save_top_k=10,
        mode='min'
    )

    resnet_callback = ModelCheckpoint(
        monitor='Val Loss',
        save_top_k=10,
        mode='min'
    )

    densenet_callback = ModelCheckpoint(
        monitor='Val Loss',
        save_top_k=10,
        mode='min'
    )

    return {'ensemble': ensemble_callback, 'resnet': resnet_callback, 'densenet': densenet_callback}


def compare_models(model1, model2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[0], key_item_2[0]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismatch found at ', key_item_1[0])
            else:
                raise Exception
        if models_differ == 0:
            print('Models match perfectly!')


def print_model_metrics(model, model_name, device, dataset):
    preds = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0]).cuda()
            preds.append(pred)
            labels.append(label)

    preds = torch.tensor(preds)
    labels = torch.tensor(labels).int()

    acc = torchmetrics.Accuracy(task='binary')(preds, labels)
    precision = torchmetrics.Precision(task='binary')(preds, labels)
    recall = torchmetrics.Recall(task='binary')(preds, labels)
    cm = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)(preds, labels)

    print(f"{model_name} Val Accuracy {acc}")
    print(f"{model_name} Val Precision {precision}")
    print(f"{model_name} Val Recall {recall}")
    print(f"Confusion Matrix {cm}")


def train_model(model, log_path, callback, train_loader, val_loader):
    model_trainer = pl.Trainer(accelerator='mps', max_epochs=50, logger=TensorBoardLogger(log_path),
                               enable_progress_bar=True, log_every_n_steps=1,
                               callbacks=[callback])
    model_trainer.fit(model, train_loader, val_loader)
