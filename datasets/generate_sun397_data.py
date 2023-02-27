from torchvision import transforms
import numpy as np
import os
import PIL
from pathlib import Path

# Download from http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
data_dir = "./../data/"
save_dir = os.path.join(data_dir, "SUN397")
save_small_dir = os.path.join(data_dir, "SUN397_32")


transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop((32,32))
    ])
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_small_dir, exist_ok=True)

save_dir = Path(save_dir) 
image_files = list(save_dir.rglob("sun_*.jpg"))

train_data = []
for idx in range(len(image_files)):
    image_file = image_files[idx]
    image = PIL.Image.open(image_file).convert("RGB")
    image = np.asarray(transform(image))
    train_data.append(image)
    if idx % 5000 == 0:
        print(idx, "/", len(image_files))

train_data = np.array(train_data)
np.save(os.path.join(save_small_dir, "train"), train_data)
