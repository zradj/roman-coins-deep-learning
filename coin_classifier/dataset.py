import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CoinsDataset(Dataset):
    def __init__(self, base_path="CoinsDataset", split="train", transform=None):
        self.base_path = base_path
        self.transform = transform

        images_df = pd.read_csv(
            os.path.join(base_path, "images.txt"),
            sep=" ",
            header=None,
            names=["id", "path"],
        )
        labels_df = pd.read_csv(
            os.path.join(base_path, "labels.txt"),
            sep=" ",
            header=None,
            names=["id", "label"],
        )
        split_df = pd.read_csv(
            os.path.join(base_path, "split.txt"),
            sep=" ",
            header=None,
            names=["id", "split"],
        )

        self.data = pd.merge(images_df, labels_df, on="id")
        self.data = pd.merge(self.data, split_df, on="id")

        split_value = 1 if split == "train" else 0
        self.data = self.data[self.data["split"] == split_value].reset_index(drop=True)
        self.data["label"] = self.data["label"] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.base_path, "images", row["path"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=64, augment=False):
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
