# Gemini Project Documentation

This document provides a comprehensive overview of the miTCGR project, designed to facilitate understanding and interaction with an LLM assistant.

## Important Special Instruction

Always update this file (`GEMINI.md`), whenever you implement any changes or create files. This will help LLMs to be updated with current version.

## 1. Project Overview

The project, "miTCGR," is a deep learning application designed to predict the interaction between microRNA (miRNA) and messenger RNA (mRNA) sequences. It treats the prediction task as a binary classification problem: does an interaction occur (1) or not (0)?

The core methodology involves a unique feature engineering step where RNA sequences are converted into 2D graphical representations called **Frequency Chaos Game Representations (FCGRs)**. These FCGR "images" are then processed by a dual-branch Convolutional Neural Network (CNN) architecture to learn spatial patterns indicative of molecular interactions.

## 2. Core Workflow

The entire pipeline is orchestrated from the `main.py` script.

1.  **Configuration**: All parameters and hyperparameters are centralized in `config/settings.py`. This is the primary file for controlling experiments.
2.  **Data Loading**: `main.py` loads the interaction data from the CSV file specified in the config. The primary dataset for evaluation is `data/miraw.csv`, while `data/deepmirtar.csv` is a smaller subset used for quickly testing the pipeline.
3.  **Data Splitting**: The dataset is split into training (70%), validation (15%), and testing (15%) sets.
4.  **Dataset Preparation**: `utils/DatasetLoader.py` handles the data preparation. For each miRNA-mRNA pair:
    *   The RNA sequences are retrieved.
    *   The sequences are converted into FCGR matrices using `utils/FCGR.py`. The resolution of the matrix depends on the `k_mer` size defined in the configuration (e.g., k=6 results in a 64x64 matrix).
    *   The FCGR matrices and corresponding labels are converted into PyTorch tensors.
5.  **Model Training**:
    *   The `core.train.Trainer` class manages the entire training and validation process.
    *   It uses the `InteractionModel` from `core/model.py`.
    *   The training loop includes optimization (AdamW), loss calculation (configurable for `BCEWithLogitsLoss` or `CrossEntropyLoss`), learning rate scheduling (`ReduceLROnPlateau`), and gradient clipping.
    *   **Early Stopping**: `utils/EarlyStopping.py` is used to monitor a validation metric (`Val_Accuracy` or `Val_Loss`) and halt training if no improvement is observed, preventing overfitting.
6.  **Evaluation & Visualization**:
    *   After training, the model's performance is evaluated using a comprehensive set of metrics calculated by `utils/metrics.py`.
    *   `utils/visuals.py` generates and saves plots for the training/validation history (loss and accuracy) and the confusion matrix of the best-performing model.

## 3. Model Architecture

The primary model is the `InteractionModel` defined in `core/model.py`.

*   **Dual-Branch CNN**: The model consists of two parallel, independent CNN branches: one for the mRNA's FCGR image and one for the miRNA's FCGR image.
*   **K-mer Specific Models**: The architecture of the CNN branches (`ModelK3` through `ModelK6`) is chosen based on the `k_mer` value in the configuration, ensuring the network is adapted to the input image size. Each branch uses a series of convolutional, pooling, and inception blocks to extract hierarchical features.
*   **Feature Fusion**: The feature maps from both CNN branches are flattened, concatenated, and then fed into a deep, fully-connected network (classifier).
*   **Output**: The final layer produces the classification output.

**Note on other model files:**
*   `core/crossmodelattention.py`: Contains a more advanced `InteractionModel` that uses a spatial cross-attention mechanism instead of simple concatenation. This is not currently used by `main.py` but represents an alternative architecture.
*   `core/model_init.py`: Appears to be another, possibly earlier, version of the model architecture.

## 4. Key Files and Directories

```
miTCGR/
├── config/
│   └── settings.py: Central configuration for hyperparameters, paths, etc.
├── core/
│   ├── model.py: Defines the main InteractionModel and CNN backbones.
│   ├── train.py: Contains the Trainer class for the training/validation loop.
│   ├── crossmodelattention.py: (Alternative) Defines an InteractionModel with cross-attention.
│   └── model_init.py: (Alternative) An earlier version of the model architecture.
├── data/
│   ├── miraw.csv: The main, larger dataset for model training and evaluation.
│   └── deepmirtar.csv: A smaller dataset for quick pipeline testing.
├── utils/
│   ├── DatasetLoader.py: Prepares data and converts sequences to FCGR matrices.
│   ├── FCGR.py: Implements the FCGR algorithm.
│   ├── metrics.py: Calculates various performance metrics.
│   ├── visuals.py: Generates plots for results.
│   └── EarlyStopping.py: Implements early stopping logic.
├── main.py: Main entry point to run the entire pipeline.
├── training.log: Log file for training runs.
└── output/: Default directory for saved models, plots, and logs.
```

## 5. How to Run

To run the project, execute the main script from the root directory:

```bash
python main.py
```

The script will handle all steps from data loading to saving the results based on the settings in `config/settings.py`.
