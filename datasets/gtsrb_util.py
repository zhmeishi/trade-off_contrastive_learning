import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import skimage.data
import scipy.io as sio
import cv2

def image_brightness_normalisation(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    return image

def preprocess_data(X):
    
    for i in range(len(X)):
        X[i,:,:,:] = image_brightness_normalisation(X[i,:,:,:])
   
    return X

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
    