import logging
import os
import random
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

from utils.EarlyStopping import EarlyStopping
from utils.metrics import DetailedMetrics
from utils.visuals import Plotter

# --- Reusable Code from your project ---

# Set up logging (reused from main.py)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gcn_training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Seed setting function (reused from main.py)
def set_seed(seed: int = 123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- New GCN-specific Components ---


def get_kmer_vocab(k: int):
    """Creates a vocabulary and mapping for all possible k-mers."""
    vocab = ["".join(p) for p in product("ACGT", repeat=k)]
    kmer_to_int = {kmer: i for i, kmer in enumerate(vocab)}
    return vocab, kmer_to_int


def sequence_to_kmer_graph(
    sequence: str, k: int, kmer_to_int: dict, overlap_threshold: int = 1
) -> Data:
    """
    Converts an RNA sequence into a graph of k-mers with edges based on adjacency and overlap.
    """
    kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    if not kmers:
        return Data(
            x=torch.empty(0, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    # Map k-mers to integer indices
    node_indices = [kmer_to_int.get(kmer, -1) for kmer in kmers]
    # Filter out any k-mers that might not be in the vocabulary (e.g., containing 'N')
    valid_nodes = [
        (i, node_idx) for i, node_idx in enumerate(node_indices) if node_idx != -1
    ]
    if not valid_nodes:
        return Data(
            x=torch.empty(0, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    original_indices, node_features = zip(*valid_nodes)
    x = torch.tensor(node_features, dtype=torch.long)
    
    # Map original sequence index to new node index
    original_to_new_idx = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices)}

    edge_list = []
    # Add edges for adjacent k-mers
    for i in range(len(original_indices) - 1):
        u, v = original_indices[i], original_indices[i+1]
        if abs(u - v) == 1: # Ensure they were originally adjacent
            edge_list.append([original_to_new_idx[u], original_to_new_idx[v]])

    # Add edges for overlapping k-mers
    for i in range(len(kmers)):
        for j in range(i + 1, len(kmers)):
            # Check for overlap between k-mer strings
            overlap = 0
            for offset in range(1, k):
                if kmers[i].endswith(kmers[j][: k - offset]) or kmers[j].endswith(
                    kmers[i][: k - offset]
                ):
                    overlap = k - offset
                    break
            if overlap >= overlap_threshold:
                if i in original_to_new_idx and j in original_to_new_idx:
                    edge_list.append([original_to_new_idx[i], original_to_new_idx[j]])

    if not edge_list:
        # If no edges, create a graph with nodes but no connections
        return Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))

    # Remove duplicate edges and create tensor
    edge_index = torch.tensor(list(set(map(tuple, edge_list))), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class GCNDataset(Dataset):
    """Dataset class for loading RNA pairs as graphs."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        k: int,
        kmer_to_int: dict,
        m_rna_col: str,
        mi_rna_col: str,
        class_col: str,
    ):
        self.dataset = dataset
        self.k = k
        self.kmer_to_int = kmer_to_int
        self.m_rna_col = m_rna_col
        self.mi_rna_col = mi_rna_col
        self.class_col = class_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        mrna_seq = str(row[self.m_rna_col]).replace("U", "T")
        mirna_seq = str(row[self.mi_rna_col]).replace("U", "T")
        label = int(row[self.class_col])

        mrna_graph = sequence_to_kmer_graph(mrna_seq, self.k, self.kmer_to_int)
        mirna_graph = sequence_to_kmer_graph(mirna_seq, self.k, self.kmer_to_int)

        return mrna_graph, mirna_graph, torch.tensor([label], dtype=torch.float)


def custom_collate_gcn(batch):
    """Collate function to batch multiple graph pairs."""
    mrna_graphs, mirna_graphs, labels = zip(*batch)
    valid_indices = [
        i
        for i, (g1, g2) in enumerate(zip(mrna_graphs, mirna_graphs))
        if g1.num_nodes > 0 and g2.num_nodes > 0
    ]

    if not valid_indices:
        return None, None, None

    mrna_batch = Batch.from_data_list([mrna_graphs[i] for i in valid_indices])
    mirna_batch = Batch.from_data_list([mirna_graphs[i] for i in valid_indices])
    labels_tensor = torch.stack([labels[i] for i in valid_indices], dim=0)

    return mrna_batch, mirna_batch, labels_tensor


# --- GCN Model Definition ---


class KmerGCNEncoder(nn.Module):
    """Encodes a single RNA sequence (represented as a graph of k-mers) into a single feature vector."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout_rate: float
    ):
        super(KmerGCNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch):
        x = self.embedding(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class KmerGCNInteractionModel(nn.Module):
    """A Siamese-style GCN model to predict miRNA-mRNA interaction."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout_rate: float
    ):
        super(KmerGCNInteractionModel, self).__init__()
        self.gcn_encoder = KmerGCNEncoder(
            vocab_size, embedding_dim, hidden_dim, dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, mrna_batch: Batch, mirna_batch: Batch) -> torch.Tensor:
        mrna_embedding = self.gcn_encoder(
            mrna_batch.x, mrna_batch.edge_index, mrna_batch.batch
        )
        mirna_embedding = self.gcn_encoder(
            mirna_batch.x, mirna_batch.edge_index, mirna_batch.batch
        )
        combined_embedding = torch.cat([mrna_embedding, mirna_embedding], dim=1)
        return self.classifier(combined_embedding)


# --- Main Training and Evaluation Script ---


def main():
    # --- Configuration ---
    SEED = 123
    K_MER = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 128
    DROPOUT_RATE = 0.5
    EARLY_STOPPING_PATIENCE = 10
    DATA_PATH = "data/deepmirtar.csv"
    M_RNA_COL = "Target Site"
    MI_RNA_COL = "miRNA_seq"
    CLASS_COL = "label"
    SAVE_DIR = "gcn_results"

    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading and preparing data...")
    vocab, kmer_to_int = get_kmer_vocab(K_MER)
    dataset_df = pd.read_csv(DATA_PATH)
    train_df, val_test_df = train_test_split(
        dataset_df, test_size=0.3, random_state=SEED, stratify=dataset_df[CLASS_COL]
    )
    val_df, _ = train_test_split(
        val_test_df,
        test_size=0.5,
        random_state=SEED,
        stratify=val_test_df[CLASS_COL],
    )

    train_dataset = GCNDataset(
        train_df, K_MER, kmer_to_int, M_RNA_COL, MI_RNA_COL, CLASS_COL
    )
    val_dataset = GCNDataset(
        val_df, K_MER, kmer_to_int, M_RNA_COL, MI_RNA_COL, CLASS_COL
    )

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_gcn,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_gcn,
    )
    logger.info("Data preparation complete.")

    # --- Model Initialization ---
    model = KmerGCNInteractionModel(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters."
    )
    logger.info(
        f"Starting training for up to {EPOCHS} epochs with patience {EARLY_STOPPING_PATIENCE}..."
    )

    # --- Training Loop ---
    history = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_preds = {"y_true": [], "y_pred": [], "y_prob": []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for mrna_batch, mirna_batch, labels in train_pbar:
            if mrna_batch is None: continue
            mrna_batch, mirna_batch, labels = (mrna_batch.to(device), mirna_batch.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(mrna_batch, mirna_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item())
            total_samples += labels.size(0)
            train_pbar.set_postfix(loss=f"{(total_loss / total_samples):.4f}", acc=f"{(total_correct / total_samples):.4f}")
        
        history["train_loss"].append(total_loss / total_samples)
        history["train_acc"].append(total_correct / total_samples)

        # --- Validation Loop ---
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        epoch_preds = {"y_true": [], "y_pred": [], "y_prob": []}
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for mrna_batch, mirna_batch, labels in val_pbar:
                if mrna_batch is None: continue
                mrna_batch, mirna_batch, labels = (mrna_batch.to(device), mirna_batch.to(device), labels.to(device))
                outputs = model(mrna_batch, mirna_batch)
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                total_val_loss += loss.item() * labels.size(0)
                total_val_correct += ((preds == labels).sum().item())
                total_val_samples += labels.size(0)
                epoch_preds["y_true"].extend(labels.cpu().numpy().flatten())
                epoch_preds["y_pred"].extend(preds.cpu().numpy().flatten())
                epoch_preds["y_prob"].extend(probs.cpu().numpy().flatten())
                val_pbar.set_postfix(loss=f"{(total_val_loss / total_val_samples):.4f}", acc=f"{(total_val_correct / total_val_samples):.4f}")

        val_accuracy = total_val_correct / total_val_samples
        history["val_loss"].append(total_val_loss / total_val_samples)
        history["val_acc"].append(val_accuracy)

        logger.info(f"Epoch {epoch + 1} Summary | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {val_accuracy:.4f}")

        if early_stopper.best_score == -float('inf') or val_accuracy > early_stopper.best_score:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "gcn_best_model.pth"))
            logger.info(f"Validation accuracy improved. Saved new best model. (Val Acc: {val_accuracy:.4f})")
            best_val_preds = epoch_preds

        if early_stopper(val_accuracy):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    
    # --- Post-Training Analysis ---
    logger.info("Training finished. Generating plots...")
    plotter = Plotter(
        train_losses=history["train_loss"],
        val_losses=history["val_loss"],
        train_accuracies=history["train_acc"],
        val_accuracies=history["val_acc"],
        save_dir=SAVE_DIR,
    )
    plotter.plot_training()
    logger.info(f"Training history plot saved to {os.path.join(SAVE_DIR, 'training_history.png')}")

    if best_val_preds["y_true"]:
        cm = confusion_matrix(best_val_preds["y_true"], best_val_preds["y_pred"])
        plotter.plot_confusion_matrix(cm)
        logger.info(f"Confusion matrix plot saved to {os.path.join(SAVE_DIR, 'confusion_matrix.png')}")

        metrics_calculator = DetailedMetrics(
            y_true=best_val_preds["y_true"],
            y_pred=best_val_preds["y_pred"],
            y_prob=best_val_preds["y_prob"],
        )
        metrics = metrics_calculator._calculate_all_metrices()
        logger.info(f"Metrics for best model: {metrics}")


if __name__ == "__main__":
    main()
