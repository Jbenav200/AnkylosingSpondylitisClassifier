from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image


class SIJDataset(torch.utils.data.Dataset):
    '''
        Init Function
        path_to_labels: path to csv file containing the labels for each image
        patients: the path to the numpy file containing all the patients
        root_path: path to the images to be classified
        transforms: the transforms to be applied to the images
    '''

    def __init__(self, path_to_labels, patients, root_path, transform):
        self.labels = pd.read_csv(path_to_labels)
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    '''
        gets an image for a patient with a given index
        loads the label for the given patient where the image filename is equal the the patient
        the label is the 'target' variable from the csv where the patient is the same as the filename
        the label is then assigned to the 0th index of a list
        the file_path is set to the root_path / patient (filename)
        the image is then loaded from the numpy array in the root_path
        the transforms are applied to the image
        finally, the transformed image and the label are returned
    '''
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
