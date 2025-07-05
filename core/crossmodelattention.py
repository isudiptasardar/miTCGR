"""
This file defines the model architectures, including the CNN backbone,
a spatial cross-attention mechanism, and the final interaction model.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialCrossAttention(nn.Module):
    """
    Performs spatial cross-attention between two modalities (e.g., mRNA and miRNA).
    It takes spatially-rich feature maps from CNNs, reshapes them into sequences,
    and applies multi-head attention. This allows the model to learn which parts
    of one modality's feature map are most relevant to the other.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super(SpatialCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (batch_size, seq_len_q, embed_dim) - The sequence from the target modality.
            key:   (batch_size, seq_len_kv, embed_dim) - The sequence from the source modality.
            value: (batch_size, seq_len_kv, embed_dim) - The sequence from the source modality.
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)
        
        # Store residual connection
        residual = query
        
        # Project and reshape for multi-head attention
        Q = self.query_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V are now (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        # (B, H, S_q, D_h) @ (B, H, D_h, S_kv) -> (B, H, S_q, S_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (B, H, S_q, S_kv) @ (B, H, S_kv, D_h) -> (B, H, S_q, D_h)
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project back to original embedding dimension
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attended_values)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output

class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int], dropout_rate: float = 0.1):
        super(InceptionBlock, self).__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(p=dropout_rate)
            ) for kernel_size in kernel_sizes
        ])
    
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        concatenated_out = torch.cat(tensors=outputs, dim=1)
        return concatenated_out

class ModelK6(nn.Module):
    """
    This model has been designed for 64*64 FCGR input.
    It acts as a CNN backbone to extract spatial features, which are then
    passed to the attention mechanism.
    K=6
    """
    def __init__(self, in_channels: int, out_channels: int, inception_kernel_sizes: list[int], dropout_rate: float):
        super(ModelK6, self).__init__()
        
        logging.info(f"ModelK6 initialized with inception kernel sizes {inception_kernel_sizes} and dropout_rate {dropout_rate}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.inception_block = InceptionBlock(
            in_channels=out_channels, out_channels=64, kernel_sizes=inception_kernel_sizes, dropout_rate=0.1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(4,4))
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)           # output: (B, 32, 64, 64)
        x = self.inception_block(x) # output: (B, 192, 64, 64)
        x = self.pool1(x)           # output: (B, 192, 32, 32)
        x = self.conv2(x)           # output: (B, 128, 32, 32)
        x = self.pool2(x)           # output: (B, 128, 16, 16)
        x = self.conv3(x)           # output: (B, 64, 16, 16)
        x = self.pool3(x)           # output: (B, 64, 4, 4)
        x = self.dropout(x)
        # The output is a spatial feature map, NOT flattened.
        return x

class InteractionModel(nn.Module):
    def __init__(self, dropout_rate: float, k: int, use_cross_attention: bool = True):
        super(InteractionModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.k = k
        self.use_cross_attention = use_cross_attention

        # Load model according to the k_mer provided
        match self.k:
            case 6:
                self.m_rna_model = ModelK6(in_channels=1, out_channels=32, inception_kernel_sizes=[1,5,9], dropout_rate=dropout_rate)
                self.mi_rna_model = ModelK6(in_channels=1, out_channels=32, inception_kernel_sizes=[1,3,5], dropout_rate=dropout_rate)
            case _:
                logging.error(f"Invalid k_mer provided: {self.k}. It should be between 3-9")
        
        cnn_output_channels = 64
        cnn_output_size = 4 * 4
        
        if self.use_cross_attention:
            self.mrna_to_mirna_attention = SpatialCrossAttention(
                embed_dim=cnn_output_channels, num_heads=8, dropout_rate=dropout_rate
            )
            self.mirna_to_mrna_attention = SpatialCrossAttention(
                embed_dim=cnn_output_channels, num_heads=8, dropout_rate=dropout_rate
            )
            
            # Feature fusion layer for the combined attended features
            fused_feature_dim = (cnn_output_channels * cnn_output_size) * 2
            self.feature_fusion = nn.Sequential(
                nn.Linear(in_features=fused_feature_dim, out_features=1024),
                nn.BatchNorm1d(num_features=1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            )
            classifier_input_size = 1024
        else:
            # Original concatenation approach if attention is disabled
            classifier_input_size = (cnn_output_channels * cnn_output_size) * 2
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=classifier_input_size, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.6),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.3),

            nn.Linear(in_features=32, out_features=2)
        )

        self._initialize_weights()
    
    def forward(self, x_m_rna, x_mi_rna):
        # Extract spatial feature maps from both modalities
        m_rna_features = self.m_rna_model(x_m_rna)    # (B, C, H, W) -> (B, 64, 4, 4)
        mi_rna_features = self.mi_rna_model(x_mi_rna)  # (B, C, H, W) -> (B, 64, 4, 4)
        
        if self.use_cross_attention:
            batch_size, channels, height, width = m_rna_features.shape
            seq_len = height * width
            
            # Reshape from (B, C, H, W) to (B, S, E) where S=H*W, E=C
            m_rna_seq = m_rna_features.view(batch_size, channels, seq_len).permute(0, 2, 1)
            mi_rna_seq = mi_rna_features.view(batch_size, channels, seq_len).permute(0, 2, 1)
            
            # Apply bidirectional cross-modal attention
            # Get mRNA-aware representation of miRNA
            attended_mirna = self.mirna_to_mrna_attention(query=mi_rna_seq, key=m_rna_seq, value=m_rna_seq)
            
            # Get miRNA-aware representation of mRNA
            attended_mrna = self.mrna_to_mirna_attention(query=m_rna_seq, key=mi_rna_seq, value=mi_rna_seq)
            
            # Flatten the attended sequences for the classifier
            attended_mirna_flat = attended_mirna.flatten(start_dim=1)
            attended_mrna_flat = attended_mrna.flatten(start_dim=1)
            
            # Combine attended features and pass through fusion layer
            combined_features = torch.cat(tensors=(attended_mrna_flat, attended_mirna_flat), dim=1)
            combined_features = self.feature_fusion(combined_features)
        else:
            # Original concatenation approach
            m_rna_flat = m_rna_features.flatten(start_dim=1)
            mi_rna_flat = mi_rna_features.flatten(start_dim=1)
            combined_features = torch.cat(tensors=(m_rna_flat, mi_rna_flat), dim=1)

        output = self.fc(combined_features)
        return output
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(tensor=module.bias, val=0)
            
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(tensor=module.weight)
                if module.bias is not None:
                    nn.init.constant_(tensor=module.bias, val=0)
            
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(tensor=module.weight, val=1)
                nn.init.constant_(tensor=module.bias, val=0)
