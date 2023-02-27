import csv
import pathlib
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image

from torchvision.datasets.vision import VisionDataset
import os

class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._split = split
        super().__init__(root, transform=transform, target_transform=target_transform)

        base_folder = pathlib.Path(self.root) / "fer2013"
        file_name = "icml_face_data.csv"
        data_file = base_folder / file_name
        if not os.path.exists(data_file):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from "
                f"https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge"
            )

        self.data = []
        self.targets = []
        with open(data_file, "r", newline="") as file:
            for row in csv.DictReader(file):
                if split == "train" and "Training" in row[" Usage"]:
                    self.data.append(torch.tensor([int(idx) for idx in row[" pixels"].split()], dtype=torch.uint8).reshape(48, 48).numpy())
                    self.targets.append(int(row["emotion"]))
                elif split == "test" and "Test" in row[" Usage"]:
                    self.data.append(torch.tensor([int(idx) for idx in row[" pixels"].split()], dtype=torch.uint8).reshape(48, 48).numpy())
                    self.targets.append(int(row["emotion"]))
                

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_tensor, target = self.data[idx], self.targets[idx]
        image = Image.fromarray(image_tensor)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


    def extra_repr(self) -> str:
        return f"split={self._split}"