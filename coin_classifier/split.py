import numpy as np
import pandas as pd
from pathlib import Path


def create_stratified_split(base_path="CoinsDataset", test_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)

    images_file = Path(base_path) / "images.txt"
    labels_file = Path(base_path) / "labels.txt"

    if not images_file.exists():
        raise FileNotFoundError(f"Images file not found: {images_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    images_df = pd.read_csv(images_file, sep=" ", header=None, names=["id", "path"])
    labels_df = pd.read_csv(labels_file, sep=" ", header=None, names=["id", "label"])

    data_df = pd.merge(images_df, labels_df, on="id")

    train_data = []
    test_data = []

    for label, group in data_df.groupby("label"):
        class_ids = group["id"].values
        n_test = int(len(class_ids) * test_ratio)

        shuffled = np.random.permutation(class_ids)
        n_train = len(class_ids) - n_test

        for img_id in shuffled[:n_train]:
            train_data.append([img_id, 1])
        for img_id in shuffled[n_train:]:
            test_data.append([img_id, 0])

    all_split_data = sorted(train_data + test_data, key=lambda x: x[0])
    split_df = pd.DataFrame(all_split_data, columns=["id", "split"])
    split_file = Path(base_path) / "split.txt"
    split_df.to_csv(split_file, sep=" ", header=False, index=False)

    train_count = (split_df["split"] == 1).sum()
    test_count = (split_df["split"] == 0).sum()

    print(f"Stratified split created: {train_count} train, {test_count} test")
    return train_count, test_count
