import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_heads: int = 8, dropout_rate: float = 0.1):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query_features, key_value_features):
        """
        Args:
            query_features: (batch_size, feature_dim) - features from one modality
            key_value_features: (batch_size, feature_dim) - features from other modality
        """
        batch_size = query_features.size(0)
        
        # Store residual connection
        residual = query_features
        
        # Project to Q, K, V
        Q = self.query_proj(query_features)  # (batch_size, hidden_dim)
        K = self.key_proj(key_value_features)  # (batch_size, hidden_dim)
        V = self.value_proj(key_value_features)  # (batch_size, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # (batch_size, num_heads, head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)  # (batch_size, num_heads, head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)  # (batch_size, num_heads, head_dim)
        
        # Compute attention scores
        scores = torch.einsum('bqd,bkd->bqk', Q, K) / math.sqrt(self.head_dim)  # (batch_size, num_heads, num_heads)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.einsum('bqk,bkd->bqd', attention_weights, V)  # (batch_size, num_heads, head_dim)
        
        # Concatenate heads
        attended_values = attended_values.view(batch_size, self.hidden_dim)  # (batch_size, hidden_dim)
        
        # Final projection
        output = self.out_proj(attended_values)  # (batch_size, feature_dim)
        
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
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            ) for kernel_size in kernel_sizes
        ])
    
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        concatenated_out = torch.cat(tensors=outputs, dim=1)
        return concatenated_out

class ModelK6(nn.Module):
    """
    This model has been designed for 64*64 FCGR input
    K=6
    """

    def __init__(self, in_channels: int, out_channels: int, inception_kernel_sizes: list[int], dropout_rate: float):
        super(ModelK6, self).__init__()
        
        logging.info(f"ModelK6 initialized with inception kernel sizes {inception_kernel_sizes} and dropout_rate {dropout_rate}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.inception_block = InceptionBlock(
            in_channels=out_channels,
            out_channels=64,
            kernel_sizes=inception_kernel_sizes,
            dropout_rate=0.1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(4,4)) # Try with AdaptiveAveragePool

        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)           # output: 32*64*64
        x = self.inception_block(x) # output: 192*64*64
        x = self.pool1(x)           # output: 192*32*32
        x = self.conv2(x)           # output: 128*32*32
        x = self.pool2(x)           # output: 128*16*16
        x = self.conv3(x)           # output: 64*16*16
        x = self.pool3(x)           # output: 64*4*4
        x = self.dropout(x)
        x = torch.flatten(input=x, start_dim=1)
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
                self.m_rna_model = ModelK6(in_channels=1,
                                           out_channels=32,
                                           inception_kernel_sizes=[1,5,9],
                                           dropout_rate=dropout_rate)
                self.mi_rna_model = ModelK6(in_channels=1,
                                            out_channels=32,
                                            inception_kernel_sizes=[1,3,5],
                                            dropout_rate=dropout_rate)
            case _:
                logging.error(f"Invalid k_mer provided: {self.k}. It should be between 3-9")
        
        # Cross-modal attention modules
        if self.use_cross_attention:
            feature_dim = 64*4*4  # 1024
            self.mrna_to_mirna_attention = CrossModalAttention(
                feature_dim=feature_dim,
                hidden_dim=256,
                num_heads=8,
                dropout_rate=dropout_rate
            )
            self.mirna_to_mrna_attention = CrossModalAttention(
                feature_dim=feature_dim,
                hidden_dim=256,
                num_heads=8,
                dropout_rate=dropout_rate
            )
            
            # Feature fusion layer
            self.feature_fusion = nn.Sequential(
                nn.Linear(in_features=feature_dim*2, out_features=1024),
                nn.BatchNorm1d(num_features=1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            )
            
            # Adjusted classifier input size
            classifier_input_size = 1024
        else:
            classifier_input_size = 64*4*4*2  # Original concatenation size
        
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
        # Extract features from both modalities
        m_rna_features = self.m_rna_model(x_m_rna)    # (batch_size, 1024)
        mi_rna_features = self.mi_rna_model(x_mi_rna)  # (batch_size, 1024)
        
        if self.use_cross_attention:
            # Apply cross-modal attention
            # mRNA features attended by miRNA features
            attended_mrna = self.mrna_to_mirna_attention(m_rna_features, mi_rna_features)
            
            # miRNA features attended by mRNA features  
            attended_mirna = self.mirna_to_mrna_attention(mi_rna_features, m_rna_features)
            
            # Combine attended features
            combined_features = torch.cat(tensors=(attended_mrna, attended_mirna), dim=1)
            
            # Feature fusion
            combined_features = self.feature_fusion(combined_features)
        else:
            # Original concatenation approach
            combined_features = torch.cat(tensors=(m_rna_features, mi_rna_features), dim=1)

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