import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from utils.FCGR import FCGR
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Configure logger
logging.basicConfig(
    filename='vit_logger.log', filemode='w', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Dataset ---
class FCGRDataset(Dataset):
    def __init__(self, csv_path: str, k_mer: int):
        self.df = pd.read_csv(csv_path)
        self.k = k_mer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mirna = row['mature_miRNA_Transcript'].replace('U', 'T')
        mrna = row['mRNA_Site_Transcript'].replace('U', 'T')
        f1 = FCGR(sequence=mirna, k=self.k).generate_fcgr()
        f2 = FCGR(sequence=mrna, k=self.k).generate_fcgr()
        x = torch.stack([torch.FloatTensor(f1), torch.FloatTensor(f2)], dim=0)
        y = torch.tensor(int(row['validation']), dtype=torch.long)
        return x, y

# --- Vision Transformer Model ---
class ConvStem(nn.Module):
    def __init__(self, in_ch=2, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, img_size):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        hdim = dim // heads
        self.scale = hdim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FCGRHybridViT(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_ch=2,
                 num_classes=2, dim=128, depth=4, heads=4, mlp_ratio=2.0,
                 dropout=0.1):
        super().__init__()
        self.stem = ConvStem(in_ch, out_ch=32)
        self.patch_embed = PatchEmbedding(in_ch=32, embed_dim=dim,
                                          patch_size=patch_size, img_size=img_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# --- Training Loop ---
def train_model():
    dataset = FCGRDataset("data/miraw.csv", k_mer=6)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    logger.info(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCGRHybridViT().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    class EarlyStopping:
        def __init__(self, patience=10):
            self.best_score = None
            self.counter = 0
            self.patience = patience
            self.early_stop = False

        def step(self, score):
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    stopper = EarlyStopping(patience=10)

    for epoch in range(1, 101):
        model.train()
        total_loss, correct = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / len(train_set)
        train_acc = correct / len(train_set)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * y.size(0)
                val_correct += (out.argmax(dim=1) == y).sum().item()

        val_loss /= len(val_set)
        val_acc = val_correct / len(val_set)

        logger.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Epoch {epoch} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)
        stopper.step(val_acc)
        if stopper.early_stop:
            logger.info("Early stopping triggered.")
            break

if __name__ == '__main__':
    train_model()
