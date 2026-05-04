import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

kan_mixer_path = str(Path(__file__).resolve().parent.parent / "KAN_Mixer")
if kan_mixer_path not in sys.path:
    sys.path.insert(0, kan_mixer_path)

from model import KANMixer
from .config import get_coin_config, get_weights_file_path
from .dataset import CoinsDataset, get_transforms


def build_model(config, device):
    model = KANMixer(
        in_channels=config["in_channels"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        num_classes=config["num_classes"],
        embedding_dim=config["channel_dim"],
        depth=config["depth"],
        token_intermediate_dim=config["token_dim"],
        channel_intermediate_dim=config["channel_dim"],
    ).to(device)
    return model


def get_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            _, predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return (num_correct / num_samples).item()


def train_model(
    base_path="CoinsDataset",
    config=None,
    augment=False,
    device=None,
):
    if config is None:
        config = get_coin_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    os.makedirs(config["model_folder"], exist_ok=True)

    train_transform = get_transforms(config["image_size"], augment=augment)
    test_transform = get_transforms(config["image_size"], augment=False)

    train_dataset = CoinsDataset(base_path, split="train", transform=train_transform)
    test_dataset = CoinsDataset(base_path, split="test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (images, targets) in loop:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config['num_epochs']}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        train_acc = get_accuracy(train_loader, model, device)
        test_acc = get_accuracy(test_loader, model, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(
            f"Epoch [{epoch+1}/{config['num_epochs']}]: "
            f"Loss={epoch_loss:.4f}, "
            f"Train Acc={train_acc*100:.2f}%, "
            f"Test Acc={test_acc*100:.2f}%"
        )

        model_filename = get_weights_file_path(config, f"{epoch+1}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            },
            model_filename,
        )

    return model, train_losses, train_accuracies, test_accuracies
