import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :] # type: ignore
        return x


class TransformerClassifier(nn.Module):
    """Transformer model for binary classification.
    
    Accepts input of shape (BATCHSIZE, 90) and outputs binary prediction.
    """
    
    def __init__(
        self,
        input_dim: int = 90,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        """
        Args:
            input_dim: Number of input features (default: 90)
            d_model: Dimension of the model/embeddings (default: 128)
            nhead: Number of attention heads (default: 8)
            num_layers: Number of transformer encoder layers (default: 3)
            dim_feedforward: Dimension of feedforward network (default: 512)
            dropout: Dropout probability (default: 0.1)
            num_classes: Number of output classes (1 for binary) (default: 1)
        """
        super().__init__()
        
        # Input projection: maps input_dim to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # (batch, seq, features) format
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global average pooling over sequence dimension
        # After transformer: (batch, 1, d_model) -> (batch, d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        # Project input to d_model: (batch_size, 90) -> (batch_size, d_model)
        x = self.input_projection(x)
        
        # Add sequence dimension: (batch_size, d_model) -> (batch_size, 1, d_model)
        # Treat each sample as a single token sequence
        x = x.unsqueeze(1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder: (batch_size, 1, d_model) -> (batch_size, 1, d_model)
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension: (batch_size, 1, d_model) -> (batch_size, d_model)
        x = x.squeeze(1)
        
        # Classification: (batch_size, d_model) -> (batch_size, 1)
        output = self.classifier(x)
        
        return output


# Alternative implementation that treats each feature as a separate token
class TransformerClassifierMultiToken(nn.Module):
    """Transformer model treating each feature as a separate token.
    
    Accepts input of shape (BATCHSIZE, 90) where each feature becomes a token.
    """
    
    def __init__(
        self,
        input_dim: int = 90,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        """
        Args:
            input_dim: Number of input features (default: 90)
            d_model: Dimension of the model/embeddings (default: 128)
            nhead: Number of attention heads (default: 8)
            num_layers: Number of transformer encoder layers (default: 3)
            dim_feedforward: Dimension of feedforward network (default: 512)
            dropout: Dropout probability (default: 0.1)
            num_classes: Number of output classes (1 for binary) (default: 1)
        """
        super().__init__()
        
        # Input projection: maps each feature to d_model
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
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
        # Reshape: (batch_size, 90) -> (batch_size, 90, 1)
        x = x.unsqueeze(-1)
        
        # Project each feature: (batch_size, 90, 1) -> (batch_size, 90, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder: (batch_size, 90, d_model) -> (batch_size, 90, d_model)
        x = self.transformer_encoder(x)
        
        # Global average pooling: (batch_size, 90, d_model) -> (batch_size, d_model)
        x = x.mean(dim=1)
        
        # Classification: (batch_size, d_model) -> (batch_size, 1)
        output = self.classifier(x)
        
        return output


def main():
    """Simple forward test for transformer models."""
    print("=" * 60)
    print("Transformer Model Forward Test")
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
    
    # Test TransformerClassifier
    print("\n" + "-" * 60)
    print("Testing TransformerClassifier (single token version)")
    print("-" * 60)
    model1 = TransformerClassifier(input_dim=input_dim)
    print(f"Model created with {sum(p.numel() for p in model1.parameters())} parameters")
    
    model1.eval()
    with torch.no_grad():
        output1 = model1(x)
        probs1 = torch.sigmoid(output1)
    
    print(f"Output shape: {output1.shape}")
    print(f"Output range: [{output1.min().item():.4f}, {output1.max().item():.4f}]")
    print("Binary probabilities (sigmoid):")
    for i, prob in enumerate(probs1.squeeze()):
        print(f"  Sample {i+1}: {prob.item():.4f}")
    
    # Test TransformerClassifierMultiToken
    print("\n" + "-" * 60)
    print("Testing TransformerClassifierMultiToken (multi-token version)")
    print("-" * 60)
    model2 = TransformerClassifierMultiToken(input_dim=input_dim)
    print(f"Model created with {sum(p.numel() for p in model2.parameters())} parameters")
    
    model2.eval()
    with torch.no_grad():
        output2 = model2(x)
        probs2 = torch.sigmoid(output2)
    
    print(f"Output shape: {output2.shape}")
    print(f"Output range: [{output2.min().item():.4f}, {output2.max().item():.4f}]")
    print("Binary probabilities (sigmoid):")
    for i, prob in enumerate(probs2.squeeze()):
        print(f"  Sample {i+1}: {prob.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Forward test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
