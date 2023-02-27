import torch
import numpy as np
from PIL import Image

class JointDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, transform):
        if len(datasets) > 1:
            # self.data = np.concatenate([dataset.data for dataset in datasets], axis=0)
            self.data = []
            for dataset in datasets:
                # concatenate dataset with different image size
                self.data += [dataset.data[i] for i in range(len(dataset))] 
            offset = 0 
            self.targets = []
            for dataset in datasets:
                self.targets.append(np.array(dataset.targets) + offset)
                offset += len(np.unique(dataset.targets))
            self.targets = np.concatenate(self.targets, axis=0)
        else:
            self.data = datasets[0].data
            self.targets = datasets[0].targets

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        x = self.data[index]
        y = self.targets[index]

        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)

        return x, y