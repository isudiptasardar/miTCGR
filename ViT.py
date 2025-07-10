import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Literal, Tuple
from utils.FCGR import FCGR
from tqdm import tqdm

# --- Dataset Loader ---
class DatasetLoader(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        dataset_type: Literal['Train', 'Test', 'Validation'],
        k_mer: int,
        m_rna_col_name: str,
        mi_rna_col_name: str,
        class_col_name: str
    ):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.k_mer = k_mer
        self.m_rna_col_name = m_rna_col_name
        self.mi_rna_col_name = mi_rna_col_name
        self.class_col_name = class_col_name

        logger.info(f'Loaded {self.dataset_type} dataset with {len(self.dataset)} samples. Processing {self.k_mer}-mer...')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset.iloc[index]
        m_rna_seq = str(row[self.m_rna_col_name]).replace('U', 'T')
        mi_rna_seq = str(row[self.mi_rna_col_name]).replace('U', 'T')
        label = int(row[self.class_col_name])

        m_rna_fcgr = FCGR(sequence=m_rna_seq, k=self.k_mer).generate_fcgr()
        mi_rna_fcgr = FCGR(sequence=mi_rna_seq, k=self.k_mer).generate_fcgr()

        x_mrna = torch.tensor(m_rna_fcgr, dtype=torch.float32).unsqueeze(0)
        x_mirna = torch.tensor(mi_rna_fcgr, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        return x_mrna, x_mirna, y

# Custom collate to batch separate inputs
def custom_collate_fn(batch):
    x_mrna_batch = torch.stack([item[0] for item in batch], dim=0)
    x_mirna_batch = torch.stack([item[1] for item in batch], dim=0)
    y_batch = torch.stack([item[2] for item in batch], dim=0)
    return x_mrna_batch, x_mirna_batch, y_batch

# --- Patch Embeddings via Conv2d ---
class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels=1, H, W)
        x = self.proj(x)                 # (batch, embed_dim, H/ps, W/ps)
        x = x.flatten(2)                 # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)            # (batch, num_patches, embed_dim)
        return x

# --- Vision Transformer Model ---
class VisionTransformerBinary(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image must be divisible by patch size."

        num_patches = (img_size // patch_size) ** 2
        total_tokens = 2 * num_patches + 1

        self.patch_embed_mrna = PatchEmbeddings(1, patch_size, embed_dim)
        self.patch_embed_mirna = PatchEmbeddings(1, patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))

        # batch_first for convenience
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, x_mrna: torch.Tensor, x_mirna: torch.Tensor) -> torch.Tensor:
        t_m = self.patch_embed_mrna(x_mrna)
        t_i = self.patch_embed_mirna(x_mirna)
        tokens = torch.cat([t_m, t_i], dim=1)  # (B, 2N, E)

        B, _, E = tokens.shape
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.pos_embed  # (B, 2N+1, E)

        enc = self.transformer(tokens)            # (B, 2N+1, E)
        cls_out = enc[:, 0]                       # (B, E)
        logits = self.mlp_head(cls_out)           # (B, 2)
        return logits

# --- Training Loop with Progress Bar ---
def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x_mrna, x_mirna, y in bar:
            x_mrna, x_mirna, y = x_mrna.to(device), x_mirna.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_mrna, x_mirna)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")

# --- Load CONFIG & Run ---
if __name__ == '__main__':
    # --- Logger Setup ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler('vit_logger.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    from config.settings import CONFIG
    logger.info("Loaded configuration settings.")
    df = pd.read_csv(CONFIG['raw_data_path'])
    logger.info(f"Loaded raw data from {CONFIG['raw_data_path']}.")

    dataset = DatasetLoader(
        dataset=df,
        dataset_type='Train',
        k_mer=CONFIG['k_mer'],
        m_rna_col_name=CONFIG['m_rna_col_name'],
        mi_rna_col_name=CONFIG['mi_rna_col_name'],
        class_col_name=CONFIG['class_col_name']
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    logger.info("Created DataLoader.")

    model = VisionTransformerBinary(
        img_size=64,
        patch_size=CONFIG.get('patch_size', 8),
        embed_dim=CONFIG.get('embed_dim', 128),
        depth=CONFIG.get('depth', 6),
        num_heads=CONFIG.get('num_heads', 8),
        mlp_dim=CONFIG.get('mlp_dim', 256),
        dropout=CONFIG.get('dropout', 0.1)
    )
    logger.info("Vision Transformer model created.")

    train(
        model,
        loader,
        epochs=CONFIG['total_epochs'],
        lr=CONFIG.get('learning_rate', 1e-3)
    )
