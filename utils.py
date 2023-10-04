from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


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


def set_model_checkpoint():
    return ModelCheckpoint(
        monitor='Val Loss',
        save_top_k=10,
        mode='min'
    )


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

