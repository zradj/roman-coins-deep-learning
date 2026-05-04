import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_image_size_distribution(base_path="CoinsDataset"):
    images_file = os.path.join(base_path, "images.txt")
    images_dir = os.path.join(base_path, "images/")

    images_df = pd.read_csv(images_file, sep=" ", header=None, names=["id", "path"])

    sizes = []
    for _, row in images_df.iterrows():
        full_path = os.path.join(images_dir, row["path"])
        try:
            img = mpimg.imread(full_path)
            sizes.append(img.shape[0])
        except Exception:
            pass

    sizes = np.array(sizes)
    mean_size = sizes.mean()
    median_size = np.median(sizes)
    p25, p75 = np.percentile(sizes, [25, 75])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(sizes, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)

    y_max = ax.get_ylim()[1]

    stats = [
        (p25, f"P25 ({p25:.0f}px)", "gray", 0.75),
        (median_size, f"Median ({median_size:.0f}px)", "orange", 0.90),
        (mean_size, f"Mean ({mean_size:.0f}px)", "tomato", 0.82),
        (p75, f"P75 ({p75:.0f}px)", "gray", 0.68),
    ]

    for val, label, color, y_frac in stats:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.4)
        ax.text(
            val + 8, y_max * y_frac, label,
            ha="left", va="center", fontsize=8.5,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    fine_ticks = np.arange(0, 501, 50)
    coarse_ticks = np.arange(750, sizes.max() + 250, 250)
    all_ticks = np.unique(np.concatenate([fine_ticks, coarse_ticks]))
    ax.set_xticks(all_ticks)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax.set_title(
        f"Image Size Distribution   "
        f"(n = {len(sizes):,}  |  unique sizes = {len(set(sizes)):,}  "
        f"|  range = {sizes.min()}–{sizes.max()} px)",
        fontsize=12,
    )
    ax.set_xlabel("Size (px)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.show()


def show_sample_images(base_path="CoinsDataset", n=5):
    images_file = os.path.join(base_path, "images.txt")
    labels_file = os.path.join(base_path, "labels.txt")
    images_dir = os.path.join(base_path, "images/")

    images_df = pd.read_csv(images_file, sep=" ", header=None, names=["id", "path"])
    labels_df = pd.read_csv(labels_file, sep=" ", header=None, names=["id", "label"])
    data_df = pd.merge(images_df, labels_df, on="id")

    plt.figure(figsize=(15, 10))
    for i in range(n):
        sample = data_df.sample(1).iloc[0]
        full_path = os.path.join(images_dir, sample["path"])

        plt.subplot(1, n, i + 1)
        try:
            img = mpimg.imread(full_path)
            plt.imshow(img)
            plt.title(f"ID: {sample['id']}\nLabel: {sample['label']}")
        except Exception:
            plt.title(f"Error loading\nID: {sample['id']}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
