import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, filter_size, downsample=3):
        """
        Args:
            in_channels: Number of input channels
            filter_size: Number of output channels for each path
        """
        super(ResBlock, self).__init__()
        
        # Path 1: 1x1 conv + BatchNorm + activation
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=1, padding=0),
            nn.BatchNorm2d(filter_size),
            nn.ReLU()
        )
        
        # Path 2: 3x3 conv + BatchNorm + activation (twice)
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=downsample, stride=downsample)
    
    def forward(self, x):
        # Pass input through both paths
        out1 = self.path1(x)
        out2 = self.path2(x)
        
        # Element-wise addition
        skip = out1 + out2
        out = self.maxpool(skip)
        return skip, out

class CLSNet(nn.Module):
    def __init__(self, input_dim: int = 1, filter_sizes: list[int] = [64, 128, 256, 512], fc_dims: list[int] = [256, 128], dropout: float = 0.2):
        super(CLSNet, self).__init__()
        
        self.skip_path_1 = ResBlock(input_dim, filter_sizes[0])
        self.skip_path_2 = ResBlock(filter_sizes[0], filter_sizes[1])
        self.skip_path_3 = ResBlock(filter_sizes[1], filter_sizes[2])
        self.skip_path_4 = ResBlock(sum(filter_sizes[:3]), filter_sizes[3], downsample=4)
        
        flatten_dim = filter_sizes[-1] * 13 * 13
        fc_dims = [flatten_dim] + fc_dims

        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.fc_layers.append(self.dropout)
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.BatchNorm1d(fc_dims[i + 1]),
                nn.ReLU()
            ))
        
        # Output layer
        self.fc_layers.append(nn.Linear(fc_dims[-1], 2))

    def forward(self, x):
        # Path 1
        skip1, x1 = self.skip_path_1(x)      # Apply SkipBlock
        skip2, x2 = self.skip_path_2(skip1)     # Apply SkipBlock to pooled x1
        skip3, x3 = self.skip_path_3(skip2)     # Apply SkipBlock to pooled x2

        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Path 4
        _, x4 = self.skip_path_4(x)
        

        x = x4.view(x4.size(0), -1)
        
        # FC layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x

class AECLSNet(nn.Module):
    def __init__(self, input_dim: int = 1, filter_sizes: list[int] = [64, 128, 256, 512], fc_dims: list[int] = [256, 128], dropout: float = 0.2):
        super(AECLSNet, self).__init__()
        
        self.skip_path_1 = ResBlock(input_dim, filter_sizes[0])
        self.skip_path_2 = ResBlock(filter_sizes[0], filter_sizes[1])
        self.skip_path_3 = ResBlock(filter_sizes[1], filter_sizes[2])
        self.skip_path_4 = ResBlock(sum(filter_sizes[:3]), filter_sizes[3], downsample=4)
        
        flatten_dim = filter_sizes[-1] * 13 * 13
        fc_dims = [flatten_dim] + fc_dims
        
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.fc_layers.append(self.dropout)
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.BatchNorm1d(fc_dims[i + 1]),
                nn.ReLU()
            ))
        
        # Output layer
        self.fc_layers.append(nn.Linear(fc_dims[-1], 2))

        self.decoder = Decoder(filter_sizes[0:3], input_dim)
    
    def forward(self, x):
        # Path 1
        skip1, x1 = self.skip_path_1(x)      # Apply SkipBlock
        skip2, x2 = self.skip_path_2(skip1)     # Apply SkipBlock to pooled x1
        _, x3 = self.skip_path_3(skip2)     # Apply SkipBlock to pooled x2
        decoded = self.decoder(x1, x2, x3)

        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Path 4
        _, x4 = self.skip_path_4(x)
        

        x = x4.view(x4.size(0), -1)
        
        # FC layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return decoded, x

class Decoder(nn.Module):
    def __init__(self, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(sum(hidden_dims), output_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], kernel_size=3, stride=3)
        self.up2 = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[1], kernel_size=3, stride=3)
        self.up3 = nn.ConvTranspose2d(hidden_dims[2], hidden_dims[2], kernel_size=3, stride=3)
    
    def forward(self, x1, x2, x3):
        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x3 = self.up3(x3)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.decoder(x)

if __name__ == "__main__":
    model = AECLSNet()
    x = torch.randn(4, 1, 166, 166)
    output, decoded = model(x)
    print(decoded.shape)
    print(output.shape)