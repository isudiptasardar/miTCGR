import torch
import torch.nn as nn
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int]):
        super(InceptionBlock, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # nn.Dropout2d(p=0.1)  # Spatial dropout for regularization
                nn.Dropout2d(p=0.2)
            ) for kernel_size in kernel_sizes
        ])
    
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        concatenated_out = torch.cat(outputs, dim=1)
        return concatenated_out

class mRNAModel(nn.Module):
    """
    mRNA feature extraction branch for 64x64 input images
    Uses progressive feature extraction with proper regularization
    """
    def __init__(self):
        super(mRNAModel, self).__init__()
        
        # Initial convolution: 64x64x1 -> 64x64x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.1)  # Light dropout for regularization
            nn.Dropout(p=0.2)
        )
        
        # Inception block: 64x64x32 -> 64x64x192 (64*3 channels)
        self.incblock = InceptionBlock(in_channels=32, out_channels=64, kernel_sizes=[3, 5, 7])
        
        # First pooling: 64x64 -> 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution block: 32x32x192 -> 32x32x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64*3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p=0.15)  # Slightly higher dropout as we go deeper
            nn.Dropout2d(p=0.2)
        )
        
        # Third convolution block: 32x32x128 -> 32x32x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p=0.15)
            nn.Dropout2d(p=0.2)
        )
        
        # Second pooling: 32x32 -> 16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final convolution: 16x16x64 -> 16x16x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)  # Higher dropout before flattening
        )
        
        # Third pooling: 16x16 -> 8x8
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling to reduce overfitting and parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # 8x8 -> 4x4
        
        # Feature dimension: 32 * 4 * 4 = 512
    
    def forward(self, x):
        x = self.conv1(x)           # 64x64x32
        x = self.incblock(x)        # 64x64x192
        x = self.pool1(x)           # 32x32x192
        x = self.conv2(x)           # 32x32x128
        x = self.conv3(x)           # 32x32x64
        x = self.pool2(x)           # 16x16x64
        x = self.conv4(x)           # 16x16x32
        x = self.pool3(x)           # 8x8x32
        x = self.global_avg_pool(x) # 4x4x32
        x = torch.flatten(x, start_dim=1)  # 512 features
        return x

class miRNAModel(nn.Module):
    """
    miRNA feature extraction branch for 64x64 input images
    Similar architecture to mRNA but with different inception kernel sizes
    for capturing different sequence patterns
    """
    def __init__(self):
        super(miRNAModel, self).__init__()
        
        # Initial convolution: 64x64x1 -> 64x64x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.1)
            nn.Dropout2d(p=0.2)
        )
        
        # Inception block: 64x64x32 -> 64x64x192 (64*3 channels)
        # Smaller kernels for miRNA as they are shorter sequences
        self.incblock = InceptionBlock(in_channels=32, out_channels=64, kernel_sizes=[1, 3, 5])
        
        # First pooling: 64x64 -> 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolution block: 32x32x192 -> 32x32x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64*3, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(p=0.15)
            nn.Dropout2d(p=0.2)
        )
        
        # Third convolution block: 32x32x128 -> 32x32x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(p=0.15)
            nn.Dropout2d(p=0.2)
        )
        
        # Second pooling: 32x32 -> 16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Final convolution: 16x16x64 -> 16x16x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        
        # Third pooling: 16x16 -> 8x8
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))  # 8x8 -> 4x4
        
        # Feature dimension: 32 * 4 * 4 = 512

    def forward(self, x):
        x = self.conv1(x)           # 64x64x32
        x = self.incblock(x)        # 64x64x192
        x = self.pool1(x)           # 32x32x192
        x = self.conv2(x)           # 32x32x128
        x = self.conv3(x)           # 32x32x64
        x = self.pool2(x)           # 16x16x64
        x = self.conv4(x)           # 16x16x32
        x = self.pool3(x)           # 8x8x32
        x = self.global_avg_pool(x) # 4x4x32
        x = torch.flatten(x, start_dim=1)  # 512 features
        return x

class InteractionModel(nn.Module):
    """
    Complete mRNA-miRNA interaction prediction model
    Combines features from both branches and uses a deep feedforward network
    with extensive regularization to prevent overfitting
    """
    def __init__(self, dropout_rate: float = 0.3):
        super(InteractionModel, self).__init__()
        
        # Feature extraction branches
        self.mrnaModel = mRNAModel()
        self.mirnaModel = miRNAModel()
        
        # Combined feature size: 512 + 512 = 1024
        combined_features = 512 + 512
        
        # Deep feedforward network with progressive dimension reduction
        self.fusion_layers = nn.Sequential(
            # First hidden layer: 1024 -> 512
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout for regularization
            
            # Second hidden layer: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            # Third hidden layer: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            # Fourth hidden layer: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.5),  # Reduced dropout near output
            
            # Output layer: 64 -> 2 (binary classification)
            nn.Linear(64, 2)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def forward(self, x_mrna, x_mirna):
        # Extract features from both branches
        x_mrna_features = self.mrnaModel(x_mrna)    # 512 features
        x_mirna_features = self.mirnaModel(x_mirna) # 512 features
        
        # Concatenate features
        combined_features = torch.cat((x_mrna_features, x_mirna_features), dim=1)  # 1024 features
        
        # Pass through fusion network
        output = self.fusion_layers(combined_features)
        return output
    
    def _initialize_weights(self):
        """Initialize weights using appropriate methods for different layer types"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                # Standard initialization for batch norm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)