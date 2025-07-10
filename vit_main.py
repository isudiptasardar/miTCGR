import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Assuming the script is run from the root directory
from config.settings import CONFIG
from utils.DatasetLoader import DatasetLoader, custom_collate_fn
from core.train import Trainer
from utils.visuals import Plotter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PatchEmbedding(nn.Module):
    """
    Converts a 2D image into a 1D sequence of patch embeddings.
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # A convolutional layer to project the image patches into a flat embedding vector.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # Input x: (batch_size, in_channels, img_size, img_size)
        # Output proj(x): (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = self.proj(x)
        
        # Flatten the spatial dimensions
        # Output: (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        
        # Transpose to get the sequence format required by Transformers
        # Output: (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer model for dual-branch image classification.
    """
    def __init__(self, 
                 img_size=64, 
                 patch_size=8, 
                 in_channels=1, 
                 embed_dim=768, 
                 depth=12, 
                 n_heads=12, 
                 mlp_ratio=4., 
                 dropout=0.1,
                 n_classes=1):
        super().__init__()
        
        # --- Patch Embedding Branches ---
        # Create separate patch embedding layers for miRNA and mRNA
        self.patch_embed_mirna = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.patch_embed_mrna = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Total number of patches from both branches
        self.n_patches = self.patch_embed_mirna.n_patches + self.patch_embed_mrna.n_patches

        # --- CLS Token and Positional Embedding ---
        # A learnable token that will be prepended to the sequence of patches.
        # Its final hidden state will be used for classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings for the CLS token + all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True, # Important: expects (batch, seq, feature)
            norm_first=True   # Pre-LN is more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # --- Classification Head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x_mrna, x_mirna):
        # 1. Get patch embeddings for both inputs
        # (batch_size, n_patches, embed_dim)
        mrna_patches = self.patch_embed_mrna(x_mrna)
        mirna_patches = self.patch_embed_mirna(x_mirna)
        
        # 2. Concatenate patch sequences
        # (batch_size, self.n_patches, embed_dim)
        x = torch.cat((mrna_patches, mirna_patches), dim=1)
        
        # 3. Prepend CLS token
        # (batch_size, self.n_patches + 1, embed_dim)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 5. Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 6. Get the CLS token output for classification
        # The first token in the sequence is the CLS token
        cls_output = x[:, 0]
        
        # 7. Pass through classification head
        cls_output = self.norm(cls_output)
        output = self.head(cls_output)
        
        # For BCEWithLogitsLoss, we need a single logit output
        return output.squeeze(-1)

def set_seed(seed: int = 123):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(seed: int = 123):
    """
    Main function to set up and run the ViT training pipeline.
    """
    logging.info("--- Starting ViT Model Training ---")
    
    # Set seed for reproducibility
    set_seed(seed)

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    logging.info("Loading and splitting dataset...")
    dataset_df = pd.read_csv(CONFIG['raw_data_path'])
    
    # training -> 70%, validation -> 15%, testing -> 15%
    train_df, val_test_df = train_test_split(dataset_df, test_size=0.3, random_state=seed, stratify=dataset_df[CONFIG['class_col_name']], shuffle=True)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=seed, stratify=val_test_df[CONFIG['class_col_name']], shuffle=True)

    train_dataset = DatasetLoader(
        dataset=train_df, 
        dataset_type='Train', 
        k_mer=CONFIG['k_mer'],
        m_rna_col_name=CONFIG['m_rna_col_name'],
        mi_rna_col_name=CONFIG['mi_rna_col_name'],
        class_col_name=CONFIG['class_col_name']
    )
    val_dataset = DatasetLoader(
        dataset=val_df, 
        dataset_type='Validation', 
        k_mer=CONFIG['k_mer'],
        m_rna_col_name=CONFIG['m_rna_col_name'],
        mi_rna_col_name=CONFIG['mi_rna_col_name'],
        class_col_name=CONFIG['class_col_name']
    )
    # test_dataset is not used in the training loop, but we create it for completeness
    test_dataset = DatasetLoader(
        dataset=test_df, 
        dataset_type='Test', 
        k_mer=CONFIG['k_mer'],
        m_rna_col_name=CONFIG['m_rna_col_name'],
        mi_rna_col_name=CONFIG['mi_rna_col_name'],
        class_col_name=CONFIG['class_col_name']
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    logging.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # --- Model Initialization ---
    logging.info("Initializing Vision Transformer model...")
    # ViT Hyperparameters (can be tuned)
    vit_params = {
        'img_size': 2**CONFIG['k_mer'], # 64 for 6-mer
        'patch_size': 8,
        'embed_dim': 256,  # Reduced from 768 for faster training
        'depth': 6,        # Reduced from 12
        'n_heads': 8,      # Reduced from 12
        'mlp_ratio': 4.,
        'dropout': 0.1,
        'n_classes': 1
    }
    model = VisionTransformer(**vit_params)
    model.to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Training Components ---
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

    # --- Trainer Initialization ---
    logging.info("Initializing Trainer...")
    save_dir = os.path.join(CONFIG['save_dir'], 'vit_results', f"k{CONFIG['k_mer']}_b{CONFIG['batch_size']}")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=CONFIG['total_epochs'],
        save_dir=save_dir,
        early_stopping_metric='Val_Accuracy',
        early_stopping_patience=CONFIG['early_stopping_patience'],
        early_stopping_delta=CONFIG['early_stopping_delta']
    )

    # --- Start Training ---
    train_losses, val_losses, train_accs, val_accs, best_val_acc, best_val_loss, best_metrics = trainer.train()

    # --- Plotting Results ---
    logging.info("Training finished. Plotting results...")
    plotter = Plotter(save_dir=save_dir)
    plotter.plot_loss(train_losses, val_losses)
    plotter.plot_accuracy(train_accs, val_accs)
    if best_metrics:
        plotter.plot_confusion_matrix(best_metrics['confusion_matrix'], class_names=['Negative', 'Positive'])
        plotter.plot_roc_curve(best_metrics['fpr'], best_metrics['tpr'], best_metrics['roc_auc'])
    
    logging.info(f"--- ViT Model Training Complete ---")
    logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logging.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logging.info(f"Find plots and model in: {save_dir}")


if __name__ == "__main__":
    main()
