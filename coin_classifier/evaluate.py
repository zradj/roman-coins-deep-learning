import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from .train import build_model
from .config import get_coin_config


def load_model(model_path, device, config=None):
    if config is None:
        config = get_coin_config()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = build_model(config, device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def evaluate_comprehensive(model, test_loader, device, num_classes=100):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    metrics = {
        "accuracy": (all_predictions == all_labels).mean(),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "f1_micro": f1_score(all_labels, all_predictions, average="micro"),
        "f1_weighted": f1_score(all_labels, all_predictions, average="weighted"),
        "precision_macro": precision_score(all_labels, all_predictions, average="macro"),
        "precision_micro": precision_score(all_labels, all_predictions, average="micro"),
        "recall_macro": recall_score(all_labels, all_predictions, average="macro"),
        "recall_micro": recall_score(all_labels, all_predictions, average="micro"),
        "top_5_accuracy": top_k_accuracy_score(all_labels, all_probabilities, k=5),
        "top_10_accuracy": top_k_accuracy_score(all_labels, all_probabilities, k=10),
        "per_class_f1": f1_score(all_labels, all_predictions, average=None),
        "per_class_precision": precision_score(all_labels, all_predictions, average=None),
        "per_class_recall": recall_score(all_labels, all_predictions, average=None),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
    }

    return metrics, all_predictions, all_labels, all_probabilities


def print_metrics(metrics):
    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Micro):   {metrics['f1_micro']:.4f}")
    print(f"F1 Score (Weighted):{metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"Top-5 Accuracy:     {metrics['top_5_accuracy']:.4f} ({metrics['top_5_accuracy']*100:.2f}%)")
    print(f"Top-10 Accuracy:    {metrics['top_10_accuracy']:.4f} ({metrics['top_10_accuracy']*100:.2f}%)")


def plot_results(metrics, train_losses=None, train_accuracies=None, test_accuracies=None):
    fig = plt.figure(figsize=(20, 16))

    ax1 = plt.subplot(3, 4, 1)
    f1_scores = [metrics["f1_macro"], metrics["f1_micro"], metrics["f1_weighted"]]
    f1_labels = ["Macro", "Micro", "Weighted"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    bars = plt.bar(f1_labels, f1_scores, color=colors, alpha=0.8)
    plt.title("F1 Score Comparison", fontsize=12, fontweight="bold")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)

    ax2 = plt.subplot(3, 4, 2)
    top_k_accs = [metrics["accuracy"], metrics["top_5_accuracy"], metrics["top_10_accuracy"]]
    top_k_labels = ["Top-1", "Top-5", "Top-10"]
    colors = ["#FF9F43", "#10AC84", "#5F27CD"]
    bars = plt.bar(top_k_labels, top_k_accs, color=colors, alpha=0.8)
    plt.title("Top-K Accuracy", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, top_k_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)

    ax3 = plt.subplot(3, 4, (3, 4))
    plt.hist(metrics["per_class_f1"], bins=20, color="skyblue", alpha=0.7, edgecolor="black")
    plt.axvline(metrics["f1_macro"], color="red", linestyle="--", linewidth=2,
                label=f'Macro Avg: {metrics["f1_macro"]:.3f}')
    plt.title("Distribution of Per-Class F1 Scores", fontsize=12, fontweight="bold")
    plt.xlabel("F1 Score")
    plt.ylabel("Number of Classes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 4, (5, 8))
    worst_classes = np.argsort(metrics["per_class_f1"])[:10]
    best_classes = np.argsort(metrics["per_class_f1"])[-10:]
    class_indices = np.concatenate([worst_classes, best_classes])
    f1_values = metrics["per_class_f1"][class_indices]
    class_labels = [f"Class {i+1}" for i in class_indices]
    colors = ["red"] * 10 + ["green"] * 10
    bars = plt.bar(range(len(class_indices)), f1_values, color=colors, alpha=0.7)
    plt.title("Best and Worst Performing Classes (F1 Score)", fontsize=12, fontweight="bold")
    plt.xlabel("Classes")
    plt.ylabel("F1 Score")
    plt.xticks(range(len(class_indices)), class_labels, rotation=90, fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.axvline(9.5, color="black", linestyle="-", alpha=0.5)
    plt.text(4.5, 0.9, "Worst 10", ha="center", fontweight="bold", color="red")
    plt.text(14.5, 0.9, "Best 10", ha="center", fontweight="bold", color="green")

    if train_losses is not None:
        ax5 = plt.subplot(3, 4, 9)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", linewidth=2, marker="o")
        plt.title("Training Loss", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

    if train_accuracies is not None and test_accuracies is not None:
        ax6 = plt.subplot(3, 4, 10)
        epochs = range(1, len(train_accuracies) + 1)
        plt.plot(epochs, [a * 100 for a in train_accuracies], "b-", linewidth=2, marker="o", label="Training")
        plt.plot(epochs, [a * 100 for a in test_accuracies], "r-", linewidth=2, marker="s", label="Test")
        plt.title("Model Accuracy", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_confused_classes(cm, label_index):
    row = pd.Series(cm[label_index])
    print(row[row != 0])
