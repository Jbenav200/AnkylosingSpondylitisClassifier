from torchvision import transforms


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
