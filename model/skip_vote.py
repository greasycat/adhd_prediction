import torch
import torch.nn as nn

class SkipVoteNet(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dims: list[int] = [64, 128, 256, 512], fc_dims: list[int] = [256, 128]):
        super(SkipVoteNet, self).__init__()

        self.skip_path_1 = nn.Sequential(
            SkipBlock(input_dim, hidden_dims[0]),
            nn.MaxPool2d(kernel_size=3)
        )

        self.skip_path_2 = nn.Sequential(
            SkipBlock(hidden_dims[0], hidden_dims[1]),
            nn.MaxPool2d(kernel_size=3)
        )

        self.skip_path_3 = nn.Sequential(
            SkipBlock(hidden_dims[1], hidden_dims[2]),
            nn.MaxPool2d(kernel_size=3)
        )

        self.skip_path_4 = nn.Sequential(
            SkipBlock(sum(hidden_dims[:3]), hidden_dims[3]),
            nn.MaxPool2d(kernel_size=4)
        )

        # Ensure a fixed spatial size before the FC layers regardless of input HxW
        self.adaptive_pool = nn.AdaptiveAvgPool2d((13, 13))

        flatten_dim = hidden_dims[3] * 13 * 13
        fc_dims = [flatten_dim] + fc_dims

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_dims) - 1):
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.Tanh()
            ))
        
        self.fc_layers.append(nn.Sequential(
            nn.Linear(fc_dims[-1], 1),
        ))


    def forward(self, x):
        skip1 = self.skip_path_1[0](x)
        x1 = self.skip_path_1[1:](skip1)
        
        skip2 = self.skip_path_2[0](skip1)
        x2 = self.skip_path_2[1:](skip2)
        
        x3 = self.skip_path_3(skip2)
        
        x = torch.cat([x1, x2, x3], dim=1)

        x4 = self.skip_path_4(x)

        # Adapt to fixed 13x13 to match the declared flatten_dim
        x = self.adaptive_pool(x4)
        x = x.view(x.size(0), -1) # flatten
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
        
        # Path 1: 1x1 conv + tanh
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=1, padding=0),
            nn.Tanh()
        )
        
        # Path 2: 3x3 conv + tanh (twice)
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(filter_size, filter_size, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Pass input through both paths
        out1 = self.path1(x)
        out2 = self.path2(x)
        
        # Element-wise addition
        out = out1 + out2

        return out

if __name__ == "__main__":
    from torchview import draw_graph
    x = torch.randn(4, 1, 166, 166)
    skip_vote_net = SkipVoteNet(1, [64, 128, 256, 512])
    draw_graph(skip_vote_net, input_size=x.shape, save_graph=True, filename="skip_vote", expand_nested=True)