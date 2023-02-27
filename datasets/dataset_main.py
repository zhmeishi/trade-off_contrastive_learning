import torch
import torchvision
import os
import numpy as np
from PIL import Image
from datasets.random_dataset import RandomDataset
from datasets.joint_dataset import JointDataset
from datasets.gtsrb_util import CustomDataset
from datasets.small_imagenet_dataset import SmallImageNet
from datasets.fer2013_dataset import FER2013
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from .feature_dataset import FeatureDataset
import copy

def subsample(dataset, n, get_reg=False):
    if n <= 0 or n >= len(dataset):
        return dataset
    sss = StratifiedShuffleSplit(n_splits=1, test_size=len(dataset)-n, train_size=n, random_state=0)
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=len(np.unique(dataset.targets)), train_size=n, random_state=0)
    train_index, test_index = next(iter(sss.split(np.zeros(len(dataset.targets)), dataset.targets)))
    if hasattr(dataset, 'mask'):
        if get_reg:
            reg_dataset = copy.deepcopy(dataset)
            reg_dataset.set_mask(test_index)
        dataset.set_mask(train_index)
    else:
        if get_reg:
            reg_dataset = copy.deepcopy(dataset)
            reg_dataset.data = reg_dataset.data[test_index]
            reg_dataset.targets = np.array(reg_dataset.targets)[test_index]
        dataset.data = dataset.data[train_index]
        dataset.targets = np.array(dataset.targets)[train_index]
    if get_reg:
        return dataset, reg_dataset
    else:
        return dataset

def to_3d(data):
    data = np.expand_dims(data, axis=3)
    data = np.tile(data, [1, 1, 1, 3])
    return data

def process_imagefolder(dataset):
    dataset.data = []
    dataset.targets = []
    for img_file, y in dataset.imgs:
        with open(img_file, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        dataset.data.append(np.asarray(img))
        dataset.targets.append(y)

    dataset.data = np.array(dataset.data)
    dataset.targets = np.array(dataset.targets)
    return dataset

def get_svhn_dataset(data_dir, transform, train, download):
    dataset = torchvision.datasets.SVHN(data_dir, split='train' if train else 'test', transform=transform, download=download)
    dataset.data = np.transpose(dataset.data, (0, 2, 3, 1))
    dataset.targets = dataset.labels
    return dataset

def get_cinic10_dataset(data_dir, transform, train):
    split='train' if train else 'test'
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "cinic10", split), transform=transform)
    dataset = process_imagefolder(dataset)
    return dataset

def get_facescrub_dataset(data_dir, transform):
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "facescrub"), transform=transform)
    dataset = process_imagefolder(dataset)
    return dataset

def get_intel_dataset(data_dir, transform, train):
    split='train' if train else 'test'
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "intel", split), transform=transform)
    dataset = process_imagefolder(dataset)
    return dataset

def get_imagenet_dataset(data_dir, transform, train):
    split='train' if train else 'val'
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "imagenet", split), transform=transform)
    # dataset.data = np.array(dataset.imgs)
    # dataset.targets = np.array(dataset.targets)
    return dataset

def get_gtsrb_dataset(data_dir, transform, train):
    split='train' if train else 'test'
    loaded = np.load(os.path.join(data_dir, "gtsrb", f'{split}.npz'))
    dataset = CustomDataset(loaded['images'], loaded['labels'], transform=transform)
    return dataset

def get_places365_dataset(data_dir, transform, train, download):
    split='train' if train else 'test'
    loaded = np.load(os.path.join(data_dir, "places365", f'{split}.npz'))
    dataset = CustomDataset(loaded['images'], loaded['labels'], transform=transform)
    return dataset

def get_dataset(dataset, 
                data_dir, 
                transform, 
                train=True, 
                download=False, 
                debug_subset_size=None, 
                max_dataset_size=50000, 
                num_imagenet_classes=1000,
                imagenet_size=50000):

    dataset_name_list = dataset.split('+')
    dataset_list = []
    for dataset_name in dataset_name_list:
        if dataset_name[:7] == "subdata":
            if train==True:
                dic = torch.load(os.path.join(data_dir, "subdata", dataset_name))
                dataset = CustomDataset(dic["data"], dic["targets"], transform)
            else:
                dataset_name = dataset_name.split("_")[1]
                print("Test data:", dataset_name)
        elif "feature" in dataset_name:
            dataset = FeatureDataset(data_dir, dataset_name, train=train, transform=transform)
            split = "training" if train else "test"
            print(f"The size of {dataset_name} {split} dataset is {len(dataset.targets):d}")
            return dataset
        elif dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
            dataset.data = to_3d(dataset.data)
        elif dataset_name == 'fashion_mnist':
            dataset = torchvision.datasets.FashionMNIST(data_dir, train=train, transform=transform, download=download)
            dataset.data = to_3d(dataset.data)
        elif dataset_name == 'qmnist':
            dataset = torchvision.datasets.QMNIST(data_dir, train=train, transform=transform, download=download)
            dataset.data = to_3d(dataset.data)
        elif 'emnist' in dataset_name: # emnist-letters, emnist-digits
            if "letters" in dataset_name:
                dataset = torchvision.datasets.EMNIST(data_dir, split="letters", train=train, transform=transform, download=download)
            elif "digits" in dataset_name:
                dataset = torchvision.datasets.EMNIST(data_dir, split="digits", train=train, transform=transform, download=download)
            elif "balanced" in dataset_name:
                dataset = torchvision.datasets.EMNIST(data_dir, split="balanced", train=train, transform=transform, download=download)
            else:
                raise NotImplementedError
            dataset.data = to_3d(dataset.data)
        elif dataset_name == 'usps':
            dataset = torchvision.datasets.USPS(data_dir, train=train, transform=transform, download=download)
            dataset.data = to_3d(dataset.data)
        elif dataset_name == 'svhn':
            dataset = get_svhn_dataset(data_dir, transform, train, download)
        elif dataset_name == 'stl10':
            dataset = torchvision.datasets.STL10(data_dir, split='unlabeled' if train else 'test', transform=transform, download=download)
        elif dataset_name == 'stl10-eval':
            dataset = torchvision.datasets.STL10(data_dir, split='train' if train else 'test', transform=transform, download=download)
        elif dataset_name == 'small_imagenet': 
            # https://patrykchrabaszcz.github.io/Imagenet32/
            dataset = SmallImageNet(data_dir, train=train, transform=transform)
            mask = dataset.targets < num_imagenet_classes
            dataset.data = dataset.data[mask]
            dataset.targets = dataset.targets[mask]
        elif dataset_name == 'imagenet':
            # https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
            dataset = get_imagenet_dataset(data_dir, transform, train)
            split = "training" if train else "test"
            print(f"The size of {dataset_name} {split} dataset is {len(dataset.targets):d}")
            return dataset
        elif dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
        elif dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
        elif dataset_name == 'places365':
            dataset = get_places365_dataset(data_dir, transform, train, download)
        elif dataset_name == 'celeba':
            dataset = torchvision.datasets.CelebA(data_dir, split='train' if train else 'test', target_type='identity', transform=transform, download=download)
        elif dataset_name == 'random':
            dataset = RandomDataset()
        elif dataset_name == 'cinic10':
            dataset = get_cinic10_dataset(data_dir, transform, train)
        elif dataset_name == 'gtsrb': 
            # https://drive.google.com/file/d/1f37CPYd9YYMHuRk6JM7Oy-nFqUvZLFR2/view?usp=sharing
            dataset = get_gtsrb_dataset(data_dir, transform, train)
        elif dataset_name == 'fer2013': 
            # https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
            # https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing
            dataset = FER2013(data_dir, split='train' if train else 'test', transform=transform)
            dataset.data = to_3d(dataset.data)
        elif dataset_name == 'facescrub': 
            # https://www.kaggle.com/datasets/rajnishe/facescrub-full
            # https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing
            dataset = get_facescrub_dataset(data_dir, transform=transform)
        elif dataset_name == 'intel': 
            # https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download&select=seg_train
            # https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing
            dataset = get_intel_dataset(data_dir, transform, train)
        elif dataset_name == 'sun_32': 
            # http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
            # https://drive.google.com/drive/folders/1f0YDhph4amlXtDRdiMpDLn07ni0SI6Aj?usp=sharing
            data = np.load(os.path.join(data_dir, "SUN397_32", "train.npy"))
            dataset = CustomDataset(data, np.zeros(len(data)), transform)
        else:
            raise NotImplementedError

        if train:
            if "imagenet" in dataset_name:
                dataset = subsample(dataset, n=imagenet_size)
            elif len(dataset.targets) > max_dataset_size:
                dataset = subsample(dataset, n=max_dataset_size)

        split = "training" if train else "test"
        print(f"The size of {dataset_name} {split} dataset is {len(dataset.targets):d}")
        dataset_list.append(dataset)

    return JointDataset(dataset_list, transform=transform)
        
def get_mean_std(dataset_name, image_size = [32,32]):
    transform = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(1.0, 1.0)),transforms.ToTensor()])
    dataset = get_dataset(dataset_name, "./data", transform=transform, train=True, max_dataset_size=-1)
    imgs = [item[0] for item in dataset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print("mean: ", mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print("std:  ", std_r,std_g,std_b)
