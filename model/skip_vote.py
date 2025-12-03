import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SkipVoteNet(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dims: list[int] = [64, 128, 256, 512], fc_dims: list[int] = [256, 128]):
        super(SkipVoteNet, self).__init__()
        
        # Path 1: input -> skip1 -> pool -> x1
        self.skip_path_1 = nn.Sequential(
            SkipBlock(input_dim, hidden_dims[0]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Path 2: skip1 -> skip2 -> pool -> x2
        self.skip_path_2 = nn.Sequential(
            SkipBlock(hidden_dims[0], hidden_dims[1]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Path 3: skip2 -> skip3 -> pool -> x3
        self.skip_path_3 = nn.Sequential(
            SkipBlock(hidden_dims[1], hidden_dims[2]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pools to align spatial dimensions before concatenation
        self.align_pool_1 = nn.AdaptiveAvgPool2d((7, 7))  # Align x1 to 7x7
        self.align_pool_2 = nn.AdaptiveAvgPool2d((7, 7))  # Align x2 to 7x7
        self.align_pool_3 = nn.AdaptiveAvgPool2d((7, 7))  # Align x3 to 7x7
        
        # Path 4: concatenated features -> skip4 -> pool
        self.skip_path_4 = nn.Sequential(
            SkipBlock(sum(hidden_dims[:3]), hidden_dims[3]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Final adaptive pool before FC layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        flatten_dim = hidden_dims[3] * 4 * 4
        fc_dims = [flatten_dim] + fc_dims
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.BatchNorm1d(fc_dims[i + 1]),
                nn.ReLU()
            ))
        
        # Output layer
        self.fc_layers.append(nn.Linear(fc_dims[-1], 1))
    
    def forward(self, x):
        # Path 1
        skip1 = self.skip_path_1[0](x)      # Apply SkipBlock
        x1 = self.skip_path_1[1:](skip1)    # Apply pooling
        
        # Path 2
        skip2 = self.skip_path_2[0](x1)     # Apply SkipBlock to pooled x1
        x2 = self.skip_path_2[1:](skip2)    # Apply pooling
        
        # Path 3
        skip3 = self.skip_path_3[0](x2)     # Apply SkipBlock to pooled x2
        x3 = self.skip_path_3[1:](skip3)    # Apply pooling
        
        # Align spatial dimensions before concatenation
        x1_aligned = self.align_pool_1(x1)
        x2_aligned = self.align_pool_2(x2)
        x3_aligned = self.align_pool_3(x3)
        
        # Concatenate along channel dimension
        x = torch.cat([x1_aligned, x2_aligned, x3_aligned], dim=1)
        
        # Path 4
        x4 = self.skip_path_4(x)
        
        # Final pooling and flatten
        x = self.adaptive_pool(x4)
        x = x.view(x.size(0), -1)
        
        # FC layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channels, filter_size):
        """
        Args:
            in_channels: Number of input channels
            filter_size: Number of output channels for each path
        """
        super(SkipBlock, self).__init__()
        
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
    
    def forward(self, x):
        # Pass input through both paths
        out1 = self.path1(x)
        out2 = self.path2(x)
        
        # Element-wise addition
        out = out1 + out2
        return out
# class SkipVoteNet(nn.Module):
#     def __init__(self, input_dim: int = 1, hidden_dims: list[int] = [64, 128, 256, 512], fc_dims: list[int] = [256, 128]):
#         super(SkipVoteNet, self).__init__()

#         self.skip_path_1 = nn.Sequential(
#             SkipBlock(input_dim, hidden_dims[0]),
#             nn.MaxPool2d(kernel_size=3)
#         )

#         self.skip_path_2 = nn.Sequential(
#             SkipBlock(hidden_dims[0], hidden_dims[1]),
#             nn.MaxPool2d(kernel_size=3)
#         )

#         self.skip_path_3 = nn.Sequential(
#             SkipBlock(hidden_dims[1], hidden_dims[2]),
#             nn.MaxPool2d(kernel_size=3)
#         )

#         self.skip_path_4 = nn.Sequential(
#             SkipBlock(sum(hidden_dims[:3]), hidden_dims[3]),
#             nn.MaxPool2d(kernel_size=4)
#         )

#         # Ensure a fixed spatial size before the FC layers regardless of input HxW
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((13, 13))

#         flatten_dim = hidden_dims[3] * 13 * 13
#         fc_dims = [flatten_dim] + fc_dims

#         self.fc_layers = nn.ModuleList()
#         for i in range(len(fc_dims) - 1):
#             self.fc_layers.append(nn.Sequential(
#                 nn.Linear(fc_dims[i], fc_dims[i + 1]),
#                 nn.Tanh()
#             ))
        
#         self.fc_layers.append(nn.Sequential(
#             nn.Linear(fc_dims[-1], 1),
#         ))


#     def forward(self, x):
#         skip1 = self.skip_path_1[0](x)
#         x1 = self.skip_path_1[1:](skip1)
        
#         skip2 = self.skip_path_2[0](skip1)
#         x2 = self.skip_path_2[1:](skip2)
        
#         x3 = self.skip_path_3(skip2)
        
#         x = torch.cat([x1, x2, x3], dim=1)

#         x4 = self.skip_path_4(x)

#         # Adapt to fixed 13x13 to match the declared flatten_dim
#         x = self.adaptive_pool(x4)
#         x = x.view(x.size(0), -1) # flatten
#         for fc_layer in self.fc_layers:
#             x = fc_layer(x)

#         return x

# class SkipBlock(nn.Module):
#     def __init__(self, in_channels, filter_size):
#         """
#         Args:
#             in_channels: Number of input channels
#             filter_size: Number of output channels for each path
#         """
#         super(SkipBlock, self).__init__()
        
#         # Path 1: 1x1 conv + tanh
#         self.path1 = nn.Sequential(
#             nn.Conv2d(in_channels, filter_size, kernel_size=1, padding=0),
#             nn.Tanh()
#         )
        
#         # Path 2: 3x3 conv + tanh (twice)
#         self.path2 = nn.Sequential(
#             nn.Conv2d(in_channels, filter_size, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
#             nn.Tanh()
#         )
    
#     def forward(self, x):
#         # Pass input through both paths
#         out1 = self.path1(x)
#         out2 = self.path2(x)
        
#         # Element-wise addition
#         out = out1 + out2

#         return out

if __name__ == "__main__":
    from torchview import draw_graph
    x = torch.randn(4, 1, 166, 166)
    skip_vote_net = SkipVoteNet(1, [64, 128, 256, 512])
    draw_graph(skip_vote_net, input_size=x.shape, save_graph=True, filename="skip_vote", expand_nested=True)