# Roman Coin Classification with KAN-Mixer

## Project Goal

This project investigates whether a modern MLP-Mixer variant using Kolmogorov-Arnold Networks (KAN-Mixer) can reliably classify Roman Republican coins into 100 reverse-motif categories from raw images alone. The practical motivation is significant: numismatic identification of ancient coins is a slow, expert-dependent task, and a high-accuracy classifier could assist archaeologists, museum curators, and auction houses in cataloguing large coin collections automatically.

---

## KAN-Mixer Library

This project uses the **KAN-Mixer** architecture, which replaces the MLP layers in an MLP-Mixer with Kolmogorov-Arnold Network (KAN) layers. The KAN-Mixer code is included directly in this repository (in [`KAN_Mixer/`](KAN_Mixer/)) rather than cloned as a submodule, because we modified it to support:

- **Checkpoint resumption** — added `resume_epoch` to the config and checkpoint loading logic to `train.py`, enabling training to be interrupted and continued from any saved epoch.
- **Configurable patch size** — the original code hardcoded patch size; our version accepts it as a parameter from the config, which was necessary for scaling to 256×256 input without running out of GPU memory.

The `coin_classifier` package imports the `KANMixer` model class directly from this local `KAN_Mixer/model.py` at runtime.

The directory provides:

- [`KAN_Mixer/model.py`](KAN_Mixer/model.py) — model architecture: `KANMixer`, `KANLinear`, `KAN`, `MixerLayer`, `PatchEmbedding`
- [`KAN_Mixer/config.py`](KAN_Mixer/config.py) — default configuration and weight-path utilities
- [`KAN_Mixer/train.py`](KAN_Mixer/train.py) — standalone training script for MNIST/CIFAR-10 (demo purposes)

The KAN-Mixer architecture is based on:

| Reference | Link |
|-----------|------|
| MLP-Mixer: An All-MLP Architecture for Vision | [arXiv:2105.01601](https://arxiv.org/abs/2105.01601) |
| KAN: Kolmogorov-Arnold Networks | [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| MLP-Mixer Reimplementation (source code) | [GitHub: engichang1467](https://github.com/engichang1467/MLP-Mixer-Reimplementation) |
| Simple KAN (source code) | [GitHub: engichang1467](https://github.com/engichang1467/Simple-KAN) |

---

## Project Structure

```
roman-coins-deep-learning/
├── KAN_Mixer/                  # KAN-Mixer library (model, config, demo training)
│   ├── model.py                # KANMixer architecture
│   ├── config.py               # Default configs and weight-path helper
│   ├── train.py                # Demo training script (MNIST/CIFAR-10)
│   ├── demo.ipynb              # Demo notebook
│   ├── requirements.txt        # Library dependencies
│   └── README.md               # Library documentation and citations
├── coin_classifier/            # Project library (used by the notebook)
│   ├── __init__.py             # Public API — re-exports all modules
│   ├── config.py               # Coin-specific configuration and hyperparameters
│   ├── dataset.py              # CoinsDataset class and transform pipelines
│   ├── split.py                # Stratified train/test split generation
│   ├── train.py                # Model builder, training loop, accuracy
│   ├── evaluate.py             # Comprehensive evaluation, metrics, plotting
│   └── eda.py                  # Exploratory data analysis utilities
├── roman_coins_deep_learning.ipynb  # Main notebook (imports from coin_classifier)
└── README.md
```

The main notebook (`roman_coins_deep_learning.ipynb`) imports all functionality from the `coin_classifier` package rather than defining anything inline — including the training loop itself via `train_model()`. The package can also be used independently:

```python
from coin_classifier import (
    get_coin_config, get_weights_file_path,
    CoinsDataset, get_transforms,
    create_stratified_split,
    build_model, train_model, get_accuracy,
    load_model, evaluate_comprehensive, print_metrics, plot_results, print_confused_classes,
    plot_image_size_distribution, show_sample_images,
)
```

---

## Problem Description and Approach

### The Problem

The dataset used is the **RRCD-Main** (Roman Republican Coin Dataset), introduced by Anwar et al. (2020). It contains approximately 18,000 images of Roman Republican coins spanning 100 classes, where each class corresponds to a distinct reverse motif - depicted subjects such as quadrigas, elephants, griffins, and various deities or symbols. The core classification challenge is that many of these motifs are visually similar: coins from the same era share iconographic conventions, many specimens are worn or corroded, and the available resolution varies dramatically (from 99×99 to 2260×2260 pixels). Distinguishing, say, a quadriga facing left from a quadriga facing right, or identifying a partially erased inscription, is non-trivial even for human experts.

### Our Approach

Rather than using a conventional CNN backbone, we applied **KAN-Mixer**, an architecture that replaces the MLP layers in an MLP-Mixer with Kolmogorov-Arnold Network layers. KAN layers learn activation functions on their edges rather than fixed nonlinearities on nodes, which gives them additional representational flexibility for structured pattern recognition. The hypothesis was that this flexibility would help with the fine-grained, texture-heavy nature of coin imagery.

The end-to-end pipeline was as follows:

1. **Exploratory Data Analysis** — characterised image size distributions, confirmed class balance, and visualised sample images per class.
2. **Preprocessing** — all images were resized to 256×256 pixels and normalised with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). RGB conversion was applied to handle grayscale edge cases.
3. **Data Augmentation** — training images are augmented with random horizontal flips, rotation (±15°), and colour jitter (brightness/contrast ±0.2, saturation ±0.1) to close the train-test generalisation gap.
4. **Stratified Split** — a custom stratified 80/20 split was generated with `random_seed=42`, yielding 14,075 training images and 3,472 test images, each with equal per-class representation.
5. **Model Training** — the KAN-Mixer was trained from scratch with Adam (lr=1e-3), cross-entropy loss, batch size 32, for 20 epochs on a H100 GPU.
6. **Evaluation** — checkpoints were saved after every epoch. The final checkpoint (epoch 20) was used for comprehensive evaluation, including macro/micro/weighted F1, precision, recall, Top-5 and Top-10 accuracy, per-class F1.

---

## Team Members and Contributions

| Name | Contributions |
|------|--------------|
| Ilgar Malikov | _During Phase 1: EDA, image size analysis, dataset structure investigation, stratified split implementation_ |
| Emil Mirzazada | _KAN-Mixer model integration, hyperparameter selection, training loop implementation_ |
| Zaur Rajabov | _Comprehensive evaluation pipeline, per-class F1 analysis, confusion matrix investigation, visualisation code_ |

---

## Dependencies

All development and execution was done on **Google Colab Pro** with a GPU runtime (G4 accelerator). The following packages are required and are either pre-installed in Colab or installable via pip:

```
torch
torchvision
Pillow
pandas
numpy
matplotlib
scikit-learn
tqdm
```

No additional `pip install` steps are needed in a standard Colab Pro environment, since all of these are available by default. The custom KAN-Mixer model code lives in the [`KAN_Mixer/`](KAN_Mixer/) directory of this repository and is imported at runtime (see setup instructions below).

---

## How to Run on Google Colab

### Step 1 - Set up the runtime

Open a new Colab notebook and set the runtime to **GPU** (Runtime -> Change runtime type -> GPU). The notebook was developed and validated on a H100 GPU available on Google Colab Pro (_**`$10 per month`**_). An A100 or T4 will also work; expect different training times. Note that with 256×256 resolution, a GPU with at least 16GB VRAM is recommended.

### Step 2 - Clone the repository

```python
!git clone https://github.com/zradj/roman-coins-deep-learning.git
```

This pulls in the full project structure including the `KAN_Mixer/` library directory and the `coin_classifier/` package.

### Step 3 - Mount Google Drive and download the dataset

Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Download the dataset (~8.7 GB) from the [authors' Google Drive link](https://drive.google.com/file/d/1Bxcoesctd4xKvXMdgbLTCLjQkC7peQPY/view) and place it at `drive/MyDrive/SaveGames/Data/CoinsDataset.zip`. Then run the extraction cell in the notebook, which unzips it to the current working directory.

### Step 4 - (Optional) Download the pretrained model

To skip training and run evaluation directly, download the pretrained checkpoint (~99.6 MB) from [this Google Drive link](https://drive.google.com/file/d/1vrWpiAQRsmYf2ocJFX3Ee0mM354MsuaA/view?usp=sharing) and place it at `./weights_coins/kan_mixer_coins_20.pt`. Set `USE_EXISTING_MODEL = False` and `MODEL_PATH = './weights_coins/kan_mixer_coins_20.pt'` in the evaluation cell.

### Step 5 - Run the notebook top to bottom

Execute all cells in order. Key stages are:

- **EDA cells** — produce image size histogram and sample visualisations (no side effects on data).
- **Split cell** — generates a new `split.txt` inside `CoinsDataset/` with `random_seed=42`. Must be run before the DataLoader cells.
- **Training cell** — calls `train_model()` which trains KAN-Mixer for `num_epochs` epochs with data augmentation, saving a `.pt` checkpoint after each one to `weights_coins/`.
- **Evaluation cell** — uses the trained model (or loads from a checkpoint file), runs inference on the test set, and prints all metrics plus four plots.

### Reproducing the Reported Results

To reproduce the exact numbers in this README:

1. Use `random_seed=42` in the stratified split (this is the default).
1. Use the exact hyperparameters in `get_coin_config()`: `image_size=256`, `patch_size=8`, `channel_dim=256`, `token_dim=128`, `depth=6`, `lr=1e-3`, `batch_size=32`, `num_epochs=20`.
1. Train model by executing all cells starting from text `Start training KAN-Mixer` till `Evaluation` `OR` load checkpoint `kan_mixer_coins_20.pt` (epoch 20) for evaluation.
1. Run `evaluate_comprehensive()` on the test loader.

Minor floating-point differences may appear between GPU types, but accuracy figures should match to within ±0.1%.

---

## Model Architecture Summary

KAN-Mixer operates by dividing each 256×256 input image into non-overlapping 8×8 patches, yielding 1024 patch tokens per image. These tokens are then processed through alternating token-mixing and channel-mixing KAN layers for `depth=6` rounds before a global average pool and linear classifier.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Input size | 256×256 px | Preserves fine coin details (inscriptions, portraits) for distinguishing similar classes |
| Patch size | 8×8 px | Produces 1024 tokens — balances spatial granularity with GPU memory constraints |
| Channel dim | 256 | Sufficient capacity for encoding fine-grained inter-class differences |
| Token dim | 128 | Captures spatial relationships between portrait, inscription, and symbols |
| Depth | 6 | Early layers learn edges/textures; later layers learn coin-type identifiers |
| Optimizer | Adam, lr=1e-3 | Conservative LR prevents overshoot in the 100-class loss surface |
| Batch size | 32 | Reduced to fit 256×256 images with 1024 patches in H100 GPU memory |
| Epochs | 20 | Model converges around epoch 16–17; 20 epochs verifies convergence |

---

## Experimental Results

### Training Progression (Epochs 14–20, resumed from checkpoint 13)

Training was run for 20 epochs total with data augmentation at 256×256 resolution. The table below shows epochs 14–20 (resumed from checkpoint at epoch 13).

| Epoch | Train Loss | Train Accuracy | Test Accuracy |
|-------|-----------|----------------|---------------|
| 14 | 0.3791 | 90.86% | 85.34% |
| 15 | 0.3655 | 91.33% | 85.46% |
| 16 | 0.3194 | 92.16% | 87.01% |
| 17 | 0.2971 | 92.82% | 87.50% |
| 18 | 0.2681 | 91.89% | 86.43% |
| 19 | 0.2299 | 93.68% | 85.37% |
| 20 | 0.2335 | 94.14% | 87.24% |

### Final Evaluation Metrics (Checkpoint: Epoch 20)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 87.24% |
| Top-5 Accuracy | 96.80% |
| Top-10 Accuracy | 98.16% |
| F1 Score (Macro) | 0.860 |
| F1 Score (Micro) | 0.872 |
| F1 Score (Weighted) | 0.872 |
| Precision (Macro) | 0.865 |
| Recall (Macro) | 0.862 |

---

## Analysis of Findings

### What Worked

**The model learns rapidly and generalises well with augmentation.** With data augmentation and 256×256 resolution, the model achieves 87.24% test accuracy while maintaining a healthy train-test gap of only ~7% (94.14% train vs 87.24% test). This is a major improvement over Phase 1's 16% gap (99.4% vs 83.4%), confirming that augmentation effectively regularises the model.

**The majority of classes are classified well.** The per-class F1 distribution is strongly left-skewed: most of the 100 classes sit in the 0.80–1.00 range, and the best-performing classes (Class 23, 40, 96, 85, 46, 69, etc.) achieve F1 scores of 0.95–1.0. The overall macro average of 0.860 is pulled down by a small number of outlier classes rather than reflecting uniform mediocrity.

**Top-K accuracy reveals near-correct predictions.** The 9.6 percentage point jump from Top-1 (87.24%) to Top-5 (96.80%) accuracy means that the correct class is almost always in the model's top candidates. A human expert reviewing the top-5 suggestions would very rarely be looking at the wrong answer entirely.

**Higher resolution improves difficult classes.** Increasing from 64×64 to 256×256 raised Class 60's F1 from ~0.05 to ~0.45, confirming that finer spatial details help disambiguate visually similar coin types.

### What Did Not Work

**Class 15 remains nearly unresolvable.** Despite the resolution increase, Class 15 still achieves only ~0.1 F1. The confusion matrix shows 17 out of 22 Class 15 test samples are misclassified as Class 60, while Class 60 is split evenly (14 correct, 14 confused with Class 15). This bidirectional confusion suggests that these two classes share a reverse motif that is very difficult to distinguish visually — possibly representing a labelling ambiguity in the original dataset.

### Summary

KAN-Mixer achieves a Top-1 accuracy of **87.24%** and a macro F1 of **0.860** on a 100-class Roman coin classification task with data augmentation at 256×256 resolution — a substantial improvement over the Phase 1 baseline of 83.4% / 0.818 at 64×64.

---

## Phase 2 Improvements and Results

The following improvements were implemented in Phase 2 to address the limitations identified in Phase 1:

### 1. Data Augmentation

Training data is augmented to close the train-test generalisation gap (Phase 1: 99.4% train vs 83.4% test). The augmentation pipeline in [`coin_classifier/dataset.py`](coin_classifier/dataset.py) applies:

- **Random horizontal flip** — coins can appear in any orientation after excavation.
- **Random rotation (±15°)** — accounts for slight alignment variation in photographed specimens.
- **Colour jitter** (brightness ±0.2, contrast ±0.2, saturation ±0.1) — simulates variation in lighting conditions and patina colour across different photography setups.

These augmentations are applied only to the training set. The test set uses deterministic resize and normalisation for reproducible evaluation.

**Result:** Train-test gap reduced from ~16% to ~7% (94.14% train vs 87.24% test).

### 2. Higher Input Resolution (256×256)

The image resolution was increased from 64×64 to 256×256 with `patch_size=8` (yielding 1024 patches per image). To fit in GPU memory, `batch_size` was reduced from 128 to 32.

| Resolution | Patch size | Patches | Test Accuracy |
|---|---|---|---|
| 64×64 (Phase 1) | 4×4 | 256 | 83.4% |
| 256×256 (Phase 2) | 8×8 | 1024 | 87.24% |

**Result:** +3.8% absolute improvement in test accuracy. Class 60 F1 improved from ~0.05 to ~0.45.

### 3. Class 15 / Class 60 Confusion Investigation

The bidirectional confusion between Class 15 and Class 60 was partially resolved by the resolution increase. At 256×256, Class 60 improved significantly (F1 ~0.05 → ~0.45), but Class 15 remains problematic (F1 ~0.1) with 17/22 test samples still misclassified as Class 60.

Visual comparison shows these reverse motifs are nearly indistinguishable to the human eye, with the primary difference being finer detail in Class 15 coins.

**Next step:** Consult domain experts to confirm whether the two classes are genuinely distinct or represent a labelling ambiguity in the original dataset.

### 4. Checkpoint Resumption

Training can now be resumed from any saved checkpoint by setting `resume_epoch` in `get_coin_config()`. Set to an integer (e.g. 13) to resume from that epoch's checkpoint, or `None` to train from scratch.
