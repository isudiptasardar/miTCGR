import logging
import torch
import torch.nn as nn

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
class ModelK3(nn.Module):

    """
    This model has been designed for 8*8 FCGR input
    K=3
    """

    def __init__(self, in_channels: int, out_channels: int, inception_kernel_sizes: list[int], dropout_rate: float):
        super(ModelK3, self).__init__()

        logging.info(f"ModelK3 initialized with inception kernel sizes {inception_kernel_sizes} and dropout_rate {dropout_rate}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.inception_block = InceptionBlock(
            in_channels=out_channels,
            out_channels=64,
            kernel_sizes=inception_kernel_sizes,
            dropout_rate=0.1
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )
    
    def forward(self, x):
        x = self.conv1(x)           # output: 32*8*8
        x = self.inception_block(x) # output: 192*8*8
        x = self.conv2(x)           # output: 128*8*8
        x = self.pool(x)            # output: 128*4*4
        x = self.conv3(x)           # output: 64*4*4
        x = torch.flatten(input=x, start_dim=1)
        return x

class ModelK4(nn.Module):
    """
    This model has been designed for 16*16 FCGR input
    K=4
    """

    def __init__(self, in_channels: int, out_channels: int, inception_kernel_sizes: list[int], dropout_rate: float):
        super(ModelK4, self).__init__()
        logging.info(f"ModelK4 initialized with inception kernel sizes {inception_kernel_sizes} and dropout_rate {dropout_rate}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        x = self.conv1(x)           # output: 32*16*16
        x = self.inception_block(x) # output: 192*16*16
        x = self.pool1(x)           # output: 192*8*8
        x = self.conv2(x)           # output: 128*8*8
        x = self.pool2(x)           # output: 128*4*4
        x = self.conv3(x)           # output: 64*4*4
        x = torch.flatten(input=x, start_dim=1)
        return x

class ModelK5(nn.Module):
    """
    This model has been designed for 32*32 FCGR input
    K=5
    """

    def __init__(self, in_channels: int, out_channels: int, inception_kernel_sizes: list[int], dropout_rate: float):
        super(ModelK5, self).__init__()
        logging.info(f"ModelK5 initialized with inception kernel sizes {inception_kernel_sizes} and dropout_rate {dropout_rate}")

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)           # output: 32*32*32
        x = self.inception_block(x) # output: 192*32*32
        x = self.pool1(x)           # output: 192*16*16
        x = self.conv2(x)           # output: 128*16*16
        x = self.pool2(x)           # output: 128*8*8
        x = self.conv3(x)           # output: 64*8*8
        x = self.pool3(x)           # output: 64*4*4
        x = torch.flatten(input=x, start_dim=1)
        return x
    
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
            nn.ReLU()
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
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
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate)
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
    def __init__(self, dropout_rate: float, k: int, useBCEWithLogits: bool = False):
        super(InteractionModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.k = k
        self.useBCEWithLogits = useBCEWithLogits

        # Load model according to the k_mer provided
        match self.k:
            case 3:
                self.m_rna_model = ModelK3(in_channels=1,
                                           out_channels=32,
                                           inception_kernel_sizes=[1,5,9],
                                           dropout_rate=dropout_rate)
                self.mi_rna_model = ModelK3(in_channels=1,
                                            out_channels=32,
                                            inception_kernel_sizes=[1,3,5],
                                            dropout_rate=dropout_rate)
            case 4:
                self.m_rna_model = ModelK4(in_channels=1,
                                           out_channels=32,
                                           inception_kernel_sizes=[1,5,9],
                                           dropout_rate=dropout_rate)
                self.mi_rna_model = ModelK4(in_channels=1,
                                            out_channels=32,
                                            inception_kernel_sizes=[1,3,5],
                                            dropout_rate=dropout_rate)
            case 5:
                self.m_rna_model = ModelK5(in_channels=1,
                                           out_channels=32,
                                           inception_kernel_sizes=[1,5,9],
                                           dropout_rate=dropout_rate)
                self.mi_rna_model = ModelK5(in_channels=1,
                                            out_channels=32,
                                            inception_kernel_sizes=[1,3,5],
                                            dropout_rate=dropout_rate)
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
        
        self.fc = nn.Sequential(
            # Assuming that each branch will output 64*4*4
            nn.Linear(in_features=64*4*4*2, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.6),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate * 0.3), # Lower dropout rate before final prediction
        )
        self.output = nn.Linear(in_features=32, out_features=1) if self.useBCEWithLogits else nn.Linear(in_features=32, out_features=2)

        self._initialize_weights()
    
    def forward(self, x_m_rna, x_mi_rna):
        m_rna_features = self.m_rna_model(x_m_rna)
        mi_rna_features = self.mi_rna_model(x_mi_rna)

        combined_features = torch.cat(tensors=(m_rna_features, mi_rna_features), dim=1)

        output = self.fc(combined_features)
        final_output = self.output(output)
        return final_output
    
    
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