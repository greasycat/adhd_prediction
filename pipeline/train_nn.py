import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pipeline import SubjectDataset
from model.transformer import TransformerClassifierMultiToken
from model.resnet import ResNetClassifier


class FCDataset(Dataset):
    """PyTorch Dataset for Functional Connectivity data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Labels array of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels


def train_model(
    model: nn.Module,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    model_save_path: str
):
    """Train any neural network model on NYU train set, validate on NYU validation set, and test on NEURO.
    
    Args:
        model: PyTorch model to train
        model_name: Name of the model for display
        X_train: Training features (NYU split)
        y_train: Training labels (NYU split)
        X_val: Validation features (NYU split)
        y_val: Validation labels (NYU split)
        X_test: Test features (NEURO)
        y_test: Test labels (NEURO)
        config: Configuration dictionary
        model_save_path: Path to save the trained model
    """
    # Training config
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    learning_rate = training_config.get("learning_rate", 0.001)
    num_epochs = training_config.get("num_epochs", 100)
    weight_decay = training_config.get("weight_decay", 0.0001)
    
    # Create datasets and dataloaders
    train_ds = FCDataset(X_train, y_train)
    val_ds = FCDataset(X_val, y_val)
    test_ds = FCDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    prev_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nTraining {model_name}...")
    print(f"Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples | Test set: {len(X_test)} samples")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            prev_lr = current_lr
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': model_name,
            }, model_save_path)
        
        # Print progress every epoch
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | Best Val={best_val_acc:.4f}@{best_epoch}")
    
    # Load best model for final evaluation on test set
    print("\nEvaluating on test set (NEURO)...")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(labels, predictions, target_names=['Control', 'ADHD']))


def create_model(model_name: str, input_dim: int, config: dict) -> nn.Module:
    """Create a model instance based on the model name.
    
    Args:
        model_name: Name of the model to create
        input_dim: Input dimension
        config: Configuration dictionary
    
    Returns:
        PyTorch model instance
    """
    model_config = config.get("model", {})
    dropout = model_config.get("dropout", 0.1)
    
    if model_name == "Transformer":
        d_model = model_config.get("d_model", 128)
        nhead = model_config.get("nhead", 8)
        num_layers = model_config.get("num_layers", 3)
        dim_feedforward = model_config.get("dim_feedforward", 512)
        
        return TransformerClassifierMultiToken(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=1
        )
    
    elif model_name == "ResNet":
        hidden_dims = model_config.get("hidden_dims", [256, 128, 64])
        use_batch_norm = model_config.get("use_batch_norm", True)
        
        return ResNetClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            num_classes=1
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models() -> list[tuple[str, str]]:
    """Get list of available models.
    
    Returns:
        List of (model_name, description) tuples
    """
    return [
        ("Transformer", "Transformer with multi-token attention"),
        ("ResNet", "Residual Network with fully-connected layers"),
    ]


def display_model_menu(available_models: list[tuple[str, str]]):
    """Display the model selection menu."""
    print("\n" + "-" * 60)
    print("Available Models:")
    print("-" * 60)
    for i, (name, description) in enumerate(available_models, 1):
        print(f"{i}) {name}: {description}")
    print(f"{len(available_models) + 1}) Quit")
    print("-" * 60)


def train_test_nn(config: dict):
    """Main function to run training with model selection menu."""
    
    # Get paths from config
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    
    feature_config = config["feature"]
    site_to_train = feature_config.get("site_to_train", "NYU")
    selected_features_path = f"{preprocessed_dir}/{site_to_train}_feature_support.npy"
    
    # Verify selected features file exists
    if not Path(selected_features_path).exists():
        raise FileNotFoundError(
            f"Selected features file not found: {selected_features_path}\n"
            f"Please run RFE feature selection first."
        )
    
    # Create NYU dataset (will be split into train/validation)
    nyu_dataset = SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/NYU_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    )
    
    # Create NEURO dataset (used only for final testing)
    test_dataset = SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/NEURO_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    )
    
    # Get NYU data and split into train/validation
    X_nyu = nyu_dataset.get_fc_array(flatten=True, selected_features=selected_features_path)
    y_nyu = np.array(nyu_dataset.get_labels(binary=True), dtype=np.float32)
    
    # Split NYU dataset into train and validation (80/20 split)
    training_config = config.get("training", {})
    val_split = training_config.get("val_split", 0.2)
    random_state = training_config.get("random_state", 42)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_nyu, y_nyu,
        test_size=val_split,
        random_state=random_state,
        stratify=y_nyu
    )
    
    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    
    # Get test data (NEURO)
    X_test = test_dataset.get_fc_array(flatten=True, selected_features=selected_features_path)
    y_test = np.array(test_dataset.get_labels(binary=True), dtype=np.float32)
    
    input_dim = X_train.shape[1]
    
    print("NYU dataset split:")
    print(f"  Train: {len(X_train)} samples ({np.bincount(y_train.astype(int))})")
    print(f"  Validation: {len(X_val)} samples ({np.bincount(y_val.astype(int))})")
    print(f"NEURO test dataset: {len(X_test)} samples ({np.bincount(y_test.astype(int))})")
    
    available_models = get_available_models()
    
    while True:
        display_model_menu(available_models)
        choice = input("\nSelect a model to train (enter number): ").strip()
        
        if choice == str(len(available_models) + 1) or choice.lower() in ['q', 'quit']:
            print("Exiting...")
            break
        
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(available_models):
                print(f"Invalid choice. Please enter a number between 1 and {len(available_models) + 1}.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        model_name, _ = available_models[choice_idx]
        
        model = create_model(model_name, input_dim, config)
        
        model_save_path = f"model/{model_name.lower()}_best.pt"
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        train_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            config=config,
            model_save_path=model_save_path
        )
        
        print(f"\nModel saved to: {model_save_path}")
        
        continue_choice = input("\nTrain another model? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
