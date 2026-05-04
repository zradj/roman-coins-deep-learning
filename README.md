# Roman Coin Classification with KAN-Mixer

## Project Goal

This project investigates whether a modern MLP-Mixer variant using Kolmogorov-Arnold Networks (KAN-Mixer) can reliably classify Roman Republican coins into 100 reverse-motif categories from raw images alone. The practical motivation is significant: numismatic identification of ancient coins is a slow, expert-dependent task, and a high-accuracy classifier could assist archaeologists, museum curators, and auction houses in cataloguing large coin collections automatically.

---

## KAN-Mixer Library

This project uses the **KAN-Mixer** architecture, which replaces the MLP layers in an MLP-Mixer with Kolmogorov-Arnold Network (KAN) layers. The library code lives in the [`KAN_Mixer/`](KAN_Mixer/) directory and provides:

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

The main notebook (`roman_coins_deep_learning.ipynb`) imports all functionality from the `coin_classifier` package rather than defining anything inline. The package can also be used independently:

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
2. **Preprocessing** — all images were resized to 64×64 pixels and normalised with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). RGB conversion was applied to handle grayscale edge cases.
3. **Data Augmentation (Phase 2)** — training images are augmented with random horizontal flips, rotation (±15°), and colour jitter (brightness/contrast ±0.2, saturation ±0.1) to close the train-test generalisation gap.
4. **Stratified Split** — a custom stratified 80/20 split was generated with `random_seed=42`, yielding 14,075 training images and 3,472 test images, each with equal per-class representation.
5. **Model Training** — the KAN-Mixer was trained from scratch with Adam (lr=1e-3), cross-entropy loss, batch size 128, 12 epochs on a GPU.
6. **Evaluation** — checkpoints were saved after every epoch. The best checkpoint (epoch 12) was used for final comprehensive evaluation, including macro/micro/weighted F1, precision, recall, Top-5 and Top-10 accuracy, per-class F1.

---

## Team Members and Contributions

| Name | Contributions |
|------|--------------|
| Ilgar Malikov | _EDA, image size analysis, dataset structure investigation, stratified split implementation_ |
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

Open a new Colab notebook and set the runtime to **GPU** (Runtime -> Change runtime type -> GPU). The notebook was developed and validated on a H100 GPU available Google Colab Pro (_**`$10 per month`**_). An A100 or T4 will also work; expect different training times.

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

To skip training and run evaluation directly, download the pretrained checkpoint (~99.6 MB) from [this Google Drive link](https://drive.google.com/file/d/1vrWpiAQRsmYf2ocJFX3Ee0mM354MsuaA/view?usp=sharing) and place it at `./checkpoints/kan_mixer_coins_12.pt`. Set `USE_EXISTING_MODEL = False` and `MODEL_PATH = './checkpoints/kan_mixer_coins_12.pt'` in the evaluation cell.

### Step 5 - Run the notebook top to bottom

Execute all cells in order. Key stages are:

- **EDA cells** — produce image size histogram and sample visualisations (no side effects on data).
- **Split cell** — generates a new `split.txt` inside `CoinsDataset/` with `random_seed=42`. Must be run before the DataLoader cells.
- **Training cell** — trains KAN-Mixer for `num_epochs` epochs with data augmentation, saving a `.pt` checkpoint after each one to `weights_coins/`.
- **Evaluation cell** — loads the checkpoint, runs inference on the test set, and prints all metrics plus four plots.

### Reproducing the Reported Results

To reproduce the exact numbers in this README:

1. Use `random_seed=42` in the stratified split (this is the default).
1. Use the exact hyperparameters in `get_coin_config()`: `image_size=64`, `channel_dim=256`, `token_dim=128`, `depth=6`, `lr=1e-3`, `batch_size=128`.
1. Train model by executing all cells starting from text `Start training KAN-Mixer` till `Evaluation` `OR` load checkpoint `kan_mixer_coins_12.pt` (epoch 12) for evaluation.
1. Run `evaluate_comprehensive()` on the test loader.

Minor floating-point differences may appear between GPU types, but accuracy figures should match to within ±0.1%.

---

## Model Architecture Summary

KAN-Mixer operates by dividing each 64×64 input image into non-overlapping 4×4 patches, yielding 256 patch tokens per image. These tokens are then processed through alternating token-mixing and channel-mixing KAN layers for `depth=6` rounds before a global average pool and linear classifier.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Input size | 64×64 px | Preserves coin details while keeping memory cost manageable |
| Patch size | 4×4 px | Produces 256 tokens - enough spatial resolution for fine features |
| Channel dim | 256 | Sufficient capacity for encoding fine-grained inter-class differences |
| Token dim | 128 | Captures spatial relationships between portrait, inscription, and symbols |
| Depth | 6 | Early layers learn edges/textures; later layers learn coin-type identifiers |
| Optimizer | Adam, lr=1e-3 | Conservative LR prevents overshoot in the 100-class loss surface |
| Batch size | 128 | Fills GPU memory comfortably on a G4 instance |

---

## Experimental Results

### Training Progression

Training was run for 20 epochs; training was interrupted via keyboard interrupt after epoch 12 in the logged run, but all checkpoints were saved. The table below shows the full progression up to the interruption.

| Epoch | Train Loss | Train Accuracy | Test Accuracy |
|-------|-----------|----------------|---------------|
| 1  | 4.3278 | 7.70%  | 7.49%  |
| 2  | 3.8644 | 23.15% | 21.26% |
| 3  | 2.5799 | 58.50% | 51.70% |
| 4  | 1.3986 | 79.89% | 68.87% |
| 5  | 0.8171 | 88.50% | 74.54% |
| 6  | 0.5016 | 94.04% | 77.13% |
| 7  | 0.2928 | 96.99% | 79.23% |
| 8  | 0.1440 | 98.95% | 80.82% |
| 9  | 0.0621 | 99.33% | 82.26% |
| 10 | 0.0327 | 99.35% | 83.47% |
| 11 | 0.0229 | 99.42% | 83.18% |
| 12 | 0.0201 | 99.42% | 83.44% |

### Final Evaluation Metrics (Checkpoint: Epoch 12)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 83.4% |
| Top-5 Accuracy | 95.6% |
| Top-10 Accuracy | 97.6% |
| F1 Score (Macro) | 0.818 |
| F1 Score (Micro) | 0.834 |
| F1 Score (Weighted) | 0.834 |

---

## Analysis of Findings

### What Worked

**The model learns rapidly and converges strongly.** The most striking result in the training curve is the speed of initial learning - test accuracy goes from 7.5% to 74.5% in just five epochs. For a 100-class problem where random chance is 1%, reaching 74% by epoch 5 indicates that KAN-Mixer is successfully extracting discriminative coin features even at 64×64 resolution. The jump is particularly steep between epochs 2 and 3 (21% -> 51%), suggesting the model crosses a threshold in that range where it starts recognising class-level structure rather than just low-level textures.

**The majority of classes are classified well.** The per-class F1 distribution is right-skewed: most of the 100 classes sit in the 0.80-1.00 range, and the best-performing classes (Class 40, 82, 23, 69, etc.) achieve F1 scores of 0.95-0.97. This tells us that the model has genuinely learned to distinguish the visual signatures of most coin types, and the overall macro average of 0.818 is being pulled down by a small number of outlier classes rather than reflecting uniform mediocrity.

**Top-K accuracy reveals near-correct predictions.** The 12.2 percentage point jump from Top-1 (83.4%) to Top-5 (95.6%) accuracy is informative. It means that for roughly one in eight test images the model's first-ranked prediction is wrong, but the correct class is almost always in the model's top 5 candidates. The correct answer is nearby in the model's probability distribution - it is not confused in a random or semantically incoherent way. This property is useful in practice: a human expert reviewing the top-5 suggestions from the classifier would need to make a final selection, but would very rarely be looking at the wrong answer entirely.

### What Did Not Work

**Class 16 and Class 60 are almost entirely misclassified.** The confusion matrix analysis reveals a specific and dramatic failure: of the ~22 test samples belonging to Class 16, approximately 20 are predicted as Class 60, and almost none are predicted correctly (F1 ≈ 0.05). Importantly, Class 60 is also confused with Class 16 in the reverse direction - 22 of Class 60's samples are predicted as Class 16. This is a bidirectional confusion between exactly two classes, which strongly suggests that Classes 16 and 60 share a reverse motif that is visually indistinguishable at 64×64 resolution, or that the two classes were inconsistently labelled in the original dataset. This is not a generalisation failure in the usual sense - it is either a resolution bottleneck or a labelling ambiguity that no classifier can resolve without additional context.

### Summary

KAN-Mixer achieves a Top-1 accuracy of **83.4%** and a macro F1 of **0.818** on a 100-class Roman coin classification task with  no augmentation, and a modest 64×64 input resolution. This is a strong baseline result demonstrating that the architecture can meaningfully learn from numismatic imagery.

---

## Phase 2 Improvements

The following improvements have been implemented in Phase 2 to address the limitations identified above:

### 1. Data Augmentation

Training data is now augmented to close the train-test generalisation gap (99.4% train vs 83.4% test accuracy). The augmentation pipeline in [`coin_classifier/dataset.py`](coin_classifier/dataset.py) applies:

- **Random horizontal flip** — coins can appear in any orientation after excavation.
- **Random rotation (±15°)** — accounts for slight alignment variation in photographed specimens.
- **Colour jitter** (brightness ±0.2, contrast ±0.2, saturation ±0.1) — simulates variation in lighting conditions and patina colour across different photography setups.

These augmentations are applied only to the training set. The test set uses deterministic resize and normalisation for reproducible evaluation. To enable augmentation, pass `augment=True` to `get_transforms()` or `train_model()`.

### 2. Configurable Input Resolution

The image resolution is now a configurable parameter in [`coin_classifier/config.py`](coin_classifier/config.py) via `image_size`, and both the dataset transforms and model architecture adapt automatically. This enables comparison across:

| Resolution | Patches (4×4) | Expected Effect |
|---|---|---|
| 64×64 (baseline) | 256 | Fastest training; sufficient for most classes |
| 128×128 | 1024 | Recovers fine inscription detail; should help ambiguous classes |
| 256×256 | 4096 | Maximum detail; high memory cost, likely diminishing returns |

To experiment with a different resolution, update `image_size` in `get_coin_config()` and retrain.

### 3. Class 16 / Class 60 Confusion Investigation

The bidirectional confusion between Class 16 and Class 60 (F1 ≈ 0.05 for both) was investigated. Visual comparison of samples from both classes shows that their reverse motifs are nearly indistinguishable to the human eye, with the primary difference being finer detail in Class 16 coins — possibly reflecting improvements in coin production tools from that era. At 64×64 resolution, these fine details are lost.

The recommended next steps for resolving this are:

- **Increase resolution to 128×128 or 256×256** to recover the distinguishing fine details.
- **Consult domain experts** to confirm whether the two classes are genuinely distinct or represent a labelling ambiguity in the original dataset.
