# Author: R Jin, Y Yu
# Last Modified Date: 2025-12-11
# Description: This file contains the code for training and testing the neural network models (Other than AECLSNet).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import SubjectDataset
from model.aecls import CLSNet
from utils.label_extractor import SITES


class FCDataset(Dataset):
    """PyTorch Dataset for Functional Connectivity data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X) # type: ignore
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y) # type: ignore

        self.X = X
        self.y = y

        if len(self.X) != len(self.y):
            raise ValueError(f"X and y must have the same length. X: {len(self.X)}, y: {len(self.y)}")
    
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
        outputs = model(X)
        loss = criterion(outputs, y)
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
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
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
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
    # Training config
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    learning_rate = training_config.get("learning_rate", 0.001)
    num_epochs = training_config.get("num_epochs", 100)
    weight_decay = training_config.get("weight_decay", 0.0001)
    log_dir = training_config.get("log_dir", "logs")
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create datasets and dataloaders
    train_ds = FCDataset(X_train, y_train)
    val_ds = FCDataset(X_val, y_val)
    test_ds = FCDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
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
        history['train_loss'].append(train_loss)   
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    # save history to csv
    pd.DataFrame(history).to_csv(f"{log_dir}/{model_name}_history.csv", index=False)

    # Load best model for final evaluation on test set
    print("\nEvaluating on test set..")
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
    
    if model_name == "CLSNet":
        return CLSNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models() -> list[tuple[str, str, dict[str, bool]]]:
    """Get list of available models.
    
    Returns:
        List of (model_name, description, config) tuples
    """
    return [
        ("CLSNet", "CLSNet", {"use_rfe": False, "flatten": False, "use_segments": True}),
    ]


def display_model_menu(available_models: list[tuple[str, str, dict[str, bool]]]):
    """Display the model selection menu."""
    print("\n" + "-" * 60)
    print("Available Models:")
    print("-" * 60)
    for i, (name, description, config) in enumerate(available_models, 1):
        print(f"{i}) {name}: {description} | use_rfe: {config['use_rfe']} | flatten: {config['flatten']} | use_segments: {config['use_segments']}")
    print(f"{len(available_models) + 1}) Quit")
    print("-" * 60)

def merge_test_datasets(test_datasets: list[SubjectDataset], selected_features = None, flatten: bool = True):
    X_test = []
    y_test = []
    for test_dataset in test_datasets:
        X_test.append(test_dataset.get_fc_array(flatten=flatten, selected_features=selected_features))
        y_test.append(np.array(test_dataset.get_labels(binary=True), dtype=np.float32))
    return np.concatenate(X_test, axis=0), np.concatenate(y_test, axis=0)

def merge_test_datasets_segments(test_datasets: list[SubjectDataset], n_segments: int = 7):
    X_test = []
    y_test = []
    for test_dataset in test_datasets:
        X, y = test_dataset.get_fc_segments_and_labels(n_segments=n_segments) # type: ignore
        print(X.shape)
        X_test.append(X)
        y_test.append(y)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    return X_test, y_test

def prepare_data(
    config: dict,
    use_rfe: bool, use_segments: bool = False,
    flatten: bool = True,
    use_phenotypes: bool = False,
    regions_to_remove: list[int] = [115],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    site_to_train = config["feature"].get("site_to_train", "NYU")

    nyu_dataset = SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site_to_train}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    )

    test_datasets = [SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    ) for site in SITES.keys() if site != site_to_train] # Pick all sites except the one to train

    if use_rfe: # If RFE is used, use the selected features
        selected_features_path = f"{preprocessed_dir}/{site_to_train}_feature_support.npy"
        if not Path(selected_features_path).exists():
            raise FileNotFoundError(
                f"Selected features file not found: {selected_features_path}\n"
                f"Please run RFE feature selection first."
            )
        X_train = nyu_dataset.get_fc_array(flatten=True, selected_features=selected_features_path)
        y_train = np.array(nyu_dataset.get_labels(binary=True), dtype=np.float32)
        X_test, y_test = merge_test_datasets(test_datasets, selected_features_path, flatten=True)
    elif use_segments:
        X_train, y_train = nyu_dataset.get_fc_segments_and_labels(n_segments=7) # type: ignore
        X_test, y_test = merge_test_datasets_segments(test_datasets, n_segments=7)
    else: # If RFE is not used, use all features
        X_train = nyu_dataset.get_fc_array(flatten=flatten)
        y_train = np.array(nyu_dataset.get_labels(binary=True), dtype=np.float32)
        X_test, y_test = merge_test_datasets(test_datasets, flatten=flatten)


    # Optionally concatenate phenotype information (Gender, Age, FD, IQ, ...)
    # Only supported for flattened feature representations.
    if use_phenotypes and flatten:
        # These column names must match those produced in `extract_label`
        phenotype_columns = [
            "Gender",
            "Age",
            "MeanFD",
            "MaxFD",
            "VerbalIQ",
            "PerformanceIQ",
        ]
        X_train_pheno = nyu_dataset.get_phenotype_array(phenotype_columns)
        X_test_pheno_list = [
            ds.get_phenotype_array(phenotype_columns) for ds in test_datasets
        ]
        X_test_pheno = np.concatenate(X_test_pheno_list, axis=0)

        # Concatenate along feature dimension
        X_train = np.concatenate([X_train, X_train_pheno], axis=1)
        X_test = np.concatenate([X_test, X_test_pheno], axis=1)

    # Train / validation split on NYU (subject-level, stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # Feature scaling (fit on training split only to avoid data leakage)
    if flatten:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        input_dim = X_train.shape[1] # type: ignore
    else:
        # expand dimensions to add channel dimension for 2D CNN
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        input_dim = X_train.shape[2:] # type: ignore

    return X_train, X_val, X_test, y_train, y_val, y_test, input_dim # type: ignore


def train_test_nn(config: dict):
    
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
        
        model_name, _, model_config = available_models[choice_idx]

        X_train, X_val, X_test, y_train, y_val, y_test, input_dim = prepare_data(config, use_rfe=model_config["use_rfe"], use_segments=model_config["use_segments"], flatten=model_config["flatten"])

        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"input_dim: {input_dim}")

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
