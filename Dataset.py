from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image


class SIJDataset(torch.utils.data.Dataset):
    ''' INIT Function '''

    def __init__(self, path_to_labels, patients, root_path, transform):
        self.labels = pd.read_csv(path_to_labels)
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        data = self.labels[self.labels['image'] == patient]
        label = data['target']
        label = list(label)[0]

        file_path = self.root_path / patient
        image = np.load(f"{file_path}.npy").astype(np.float32)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        image = torch.Tensor(image)
        label = torch.Tensor([label])

        return image, label
