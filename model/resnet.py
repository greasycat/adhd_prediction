import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for fully-connected layers."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Shortcut connection
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = self.shortcut(x)
        out = self.block(x)
        out = out + residual
        return out


class ResNetClassifier(nn.Module):
    """Residual Network for binary classification.
    
    Accepts input of shape (BATCHSIZE, 90) and outputs binary prediction.
    """
    
    def __init__(
        self,
        input_dim: int = 90,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        num_classes: int = 1,
    ):
        """
        Args:
            input_dim: Number of input features (default: 90)
            hidden_dims: List of hidden layer dimensions (default: [256, 128, 64])
            dropout: Dropout probability (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
            num_classes: Number of output classes (1 for binary) (default: 1)
        """
        super().__init__()
        
        # Input projection
        layers = []
        prev_dim = input_dim
        
        # Build residual blocks
        for hidden_dim in hidden_dims:
            layers.append(
                ResidualBlock(
                    in_features=prev_dim,
                    out_features=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm
                )
            )
            prev_dim = hidden_dim
        
        self.residual_layers = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (BATCHSIZE, 90)
        
        Returns:
            Output tensor of shape (BATCHSIZE, 1) for binary classification
        """
        # Apply residual blocks
        x = self.residual_layers(x)
        
        # Classification
        output = self.classifier(x)
        
        return output


def main():
    """Simple forward test for ResNet model."""
    print("=" * 60)
    print("ResNet Model Forward Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 4
    input_dim = 90
    
    # Create dummy input: (batch_size, 90)
    x = torch.randn(batch_size, input_dim)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    
    # Test ResNetClassifier
    print("\n" + "-" * 60)
    print("Testing ResNetClassifier")
    print("-" * 60)
    model = ResNetClassifier(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        probs = torch.sigmoid(output)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("Binary probabilities (sigmoid):")
    for i, prob in enumerate(probs.squeeze()):
        print(f"  Sample {i+1}: {prob.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Forward test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

