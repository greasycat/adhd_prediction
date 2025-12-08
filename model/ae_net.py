import torch
from torch import nn

class AE_Net(nn.Module):
    def __init__(self, input_dim, ae_dim=50, res_dim=30):
        super(AE_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ae_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(ae_dim, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(ae_dim, res_dim),
            Classifier(res_dim),
            nn.Linear(res_dim, 2),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        reconstructed = self.decoder(encoded)
        
        logits = self.classifier(encoded)
        
        return reconstructed, logits

class Classifier(nn.Module):
    def __init__(self, res_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            ResBlock(res_dim),
            ResBlock(res_dim),
            ResBlock(res_dim),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, res_dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(res_dim, res_dim),
            nn.BatchNorm1d(res_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(res_dim, res_dim),
            nn.BatchNorm1d(res_dim),
            nn.Dropout(0.2),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block(x) + x
        x = self.relu(x)
        return x

if __name__ == "__main__":
    x = torch.randn(4, 90)
    model = AE_Net(input_dim=90, ae_dim=30, res_dim=20)
    reconstructed, logits = model(x)
    print(reconstructed.shape)
    print(logits.shape)