import torch
import numpy as np
import os


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_name, train, transform):
        # do not have transform
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.train = train
        self.single = transform.single
        self.data_path = os.path.join(os.path.dirname(data_dir), f"{dataset_name}")
        if not self.train:
            self.data, self.targets = load_feature(self.data_dir, self.dataset_name, split="test")
        elif self.single:
            self.data, self.targets = load_feature(self.data_dir, self.dataset_name, split="train")
            print("No Data Augmentation")
        else:
            aug_files = os.listdir(self.data_path)
            self.aug_files = [f for f in aug_files if "aug" in f]
            print("Total Available Augmentation number for the dataset:", len(self.aug_files))
            self.schedule = np.random.permutation(np.array(
                [(i,j) for i in range(len(self.aug_files)) for j in range(i+1, len(self.aug_files))]))
            self.step_id = 0
            self.idx1 = -1
            self.idx2 = -1

            self.data, self.targets = self.load_aug_data()
        self.mask = list(range(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.single:
            return torch.Tensor(self.data[index]), self.targets[index]
        else:
            return (torch.Tensor(self.data[index][0]), torch.Tensor(self.data[index][1])), self.targets[index]
            
    def load_aug_data(self):
        idx1, idx2 = self.schedule[self.step_id % len(self.schedule)]
        if idx1 != self.idx1:
            self.idx1 = idx1
            # if idx1 == 0:
            #     self.data1, self.targets1 = load_feature(self.data_dir, self.dataset_name, split=f"train")
            # else:
            self.data1, self.targets1 = load_feature(self.data_dir, self.dataset_name, split=f"aug_{idx1}")
        if idx2 != self.idx2:
            self.idx2 = idx2
            # if idx2 == 0:
            #     self.data2, self.targets2 = load_feature(self.data_dir, self.dataset_name, split=f"train")
            # else:
            self.data2, self.targets2 = load_feature(self.data_dir, self.dataset_name, split=f"aug_{idx2}")

        data = np.concatenate((np.expand_dims(self.data1, axis=1), np.expand_dims(self.data2, axis=1)), axis=1)
        return data, np.array(self.targets1).astype(int)

    def step(self):
        if self.single:
            return
        self.step_id += 1
        self.data, self.targets = self.load_aug_data()
        self.data, self.targets = self.data[self.mask], self.targets[self.mask]
    
    def set_mask(self, mask):
        self.mask = mask
        self.data, self.targets = self.data[self.mask], self.targets[self.mask]
    
    def set_aug_num(self, aug_num):
        if aug_num == 1:
            self.schedule = [(1,1)]
        else:
            self.schedule = np.random.permutation(np.array(
                [(i,j) for i in range(aug_num) for j in range(i+1, aug_num)]))
        self.data, self.targets = self.load_aug_data()

# return a numpy feature
def load_feature(data_dir, dataset_name, split="train"):
    save_folder = os.path.join(os.path.dirname(data_dir), f"{dataset_name}")
    save_path = os.path.join(save_folder, f"{dataset_name}_{split}.pt")
    dic = torch.load(save_path, map_location='cpu')
    features = dic["features"].numpy()
    feature_labels = dic["feature_labels"].numpy()
    return features, feature_labels.astype(int)
