import torchvision
from torchvision import transforms
import numpy as np
import os

data_dir = "./../data/"
download = True
save_dir = os.path.join(data_dir, "places365")

transform = transforms.Resize(32)
os.makedirs(save_dir, exist_ok=True)

train_dataset = torchvision.datasets.Places365(data_dir, split='train-standard', small=True, transform=None, download=download)
train_data = []
train_targets = []
cnt = 0
for file, target in train_dataset.imgs:
    image = train_dataset.loader(file)
    image = transform(image)
    train_data.append(np.asarray(image))
    train_targets.append(target)
    cnt += 1
    if cnt == 50000:
      break

train_data = np.array(train_data)
train_targets = np.array(train_targets)
np.save(os.path.join(save_dir, "train"), {"images": train_data, "labels": train_targets})

test_dataset = torchvision.datasets.Places365(data_dir, split='val', small=True, transform=None, download=download)
test_data = []
test_targets = []
cnt = 0
for file, target in test_dataset.imgs:
    image = test_dataset.loader(file)
    image = transform(image)
    test_data.append(np.asarray(image))
    test_targets.append(target)
    cnt += 1
    if cnt == 10000:
      break

test_data = np.array(test_data)
test_targets = np.array(test_targets)
np.save(os.path.join(save_dir, "test"), {"images": test_data, "labels": test_targets})