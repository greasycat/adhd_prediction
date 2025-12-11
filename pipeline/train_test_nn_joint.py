# Author: R Jin, Y Yu
# Last Modified Date: 2025-12-11
# Description: This file contains the code for training and testing the joint model (AECLSNet).
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from pipeline.train_test_nn import FCDataset
from model.aecls import AECLSNet
from pipeline import SubjectDataset
from utils.label_extractor import SITES


# Train and test the joint model (AECLSNet)
def train_test_nn_joint(config: dict):
    metrics = {
        "val_accuracy": [],
        "val_f1_score": [],
        "val_precision": [],
        "val_recall": [],
        "val_roc_auc_score": [],
        "test_accuracy": [],
        "test_f1_score": [],
        "test_precision": [],
        "test_recall": [],
        "test_roc_auc_score": [],
    }
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(prepare_data(config)):
        model = AECLSNet()
        train_model(model, "AECLSNet", X_train, y_train, X_val, y_val, config, f"model/joint_loss_{i}.pt", verbose=True, num_epochs=30) # type: ignore
        val_metrics, test_metrics = test_model(model, X_val, y_val, X_test, y_test, f"model/joint_loss_{i}.pt")
        metrics["val_accuracy"].append(val_metrics["accuracy"])
        metrics["val_f1_score"].append(val_metrics["f1_score"])
        metrics["val_precision"].append(val_metrics["precision"])
        metrics["val_recall"].append(val_metrics["recall"])
        metrics["val_roc_auc_score"].append(val_metrics["roc_auc_score"])
        metrics["test_accuracy"].append(test_metrics["accuracy"])
        metrics["test_f1_score"].append(test_metrics["f1_score"])
        metrics["test_precision"].append(test_metrics["precision"])
        metrics["test_recall"].append(test_metrics["recall"])
        metrics["test_roc_auc_score"].append(test_metrics["roc_auc_score"])
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("joint_loss_metrics.csv", index=False)
    print(metrics_df.describe())
    return metrics_df

# Test on new sites for the joint model (AECLSNet)
def train_test_nn_joint_new_sites(config: dict):
    test_on_new_sites(config)
    return

# Ablation study for the joint model (AECLSNet without reconstruction loss)
def train_test_nn_joint_ablation_study(config: dict):
    ablation_study(config)
    return

def ablation_study(config: dict):
    metrics = {
        "val_accuracy": [],
        "val_f1_score": [],
        "val_precision": [],
        "val_recall": [],
        "val_roc_auc_score": [],
        "test_accuracy": [],
        "test_f1_score": [],
        "test_precision": [],
        "test_recall": [],
        "test_roc_auc_score": [],
    }
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(prepare_data(config)):
        model = AECLSNet()
        train_model(model, "CLSNet", X_train, y_train, X_val, y_val, config, f"model/cls_loss_{i}.pt", verbose=True, num_epochs=30, alpha=1) # type: ignore
        val_metrics, test_metrics = test_model(model, X_val, y_val, X_test, y_test, f"model/cls_loss_{i}.pt")
        metrics["val_accuracy"].append(val_metrics["accuracy"])
        metrics["val_f1_score"].append(val_metrics["f1_score"])
        metrics["val_precision"].append(val_metrics["precision"])
        metrics["val_recall"].append(val_metrics["recall"])
        metrics["val_roc_auc_score"].append(val_metrics["roc_auc_score"])
        metrics["test_accuracy"].append(test_metrics["accuracy"])
        metrics["test_f1_score"].append(test_metrics["f1_score"])
        metrics["test_precision"].append(test_metrics["precision"])
        metrics["test_recall"].append(test_metrics["recall"])
        metrics["test_roc_auc_score"].append(test_metrics["roc_auc_score"])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("ablation_study_metrics.csv", index=False)
    print(metrics_df.describe())
    return metrics_df

def test_on_new_sites(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test, y_test = prepare_testing_datasets(config)

    metrics = {
        "test_accuracy": [],
        "test_f1_score": [],
        "test_precision": [],
        "test_recall": [],
        "test_roc_auc_score": [],
    }

    for i in range(5):
        model = AECLSNet()
        model.load_state_dict(torch.load(f"model/joint_loss_{i}.pt")['model_state_dict'])
        model = model.to(device)
        _, test_metrics = test_model(model, X_test, y_test, X_test, y_test, f"model/joint_loss_{i}.pt")
        metrics["test_accuracy"].append(test_metrics["accuracy"])
        metrics["test_f1_score"].append(test_metrics["f1_score"])
        metrics["test_precision"].append(test_metrics["precision"])
        metrics["test_recall"].append(test_metrics["recall"])
        metrics["test_roc_auc_score"].append(test_metrics["roc_auc_score"])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("new_sites_metrics.csv", index=False)
    print(metrics_df.describe())
    return metrics_df

# Prepare the data for training and testing the joint model (AECLSNet)
def prepare_data(config: dict):
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    site_to_train = config["feature"].get("site_to_train", "NYU")
    regions_to_remove = config["feature"].get("regions_to_remove", [115])

    nyu_dataset = SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site_to_train}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    )

    X, y = nyu_dataset.get_fc_segments_and_labels(n_segments=7, binary=True) # type: ignore
    if regions_to_remove is not None:
        mask = ~np.isin(np.arange(X.shape[2]), regions_to_remove)
        # Apply mask to both connectivity dimensions (axes 2 and 3)
        X = X[:, :, mask][:, :, :, mask]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        X_train_val, X_test = X[train_index], X[test_index]
        y_train_val, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

        yield X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_testing_datasets(config: dict) -> tuple[np.ndarray, np.ndarray]:
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    site_to_train = config["feature"].get("site_to_train", "NYU")
    regions_to_remove = config["feature"].get("regions_to_remove", [115])

    test_datasets = [SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    ) for site in SITES.keys() if site != site_to_train] # Pick all sites except the one to train

    X_test = []
    y_test = []

    for test_dataset in test_datasets:
        X_test.append(test_dataset.get_fc_array(flatten=False))
        y_test.append(np.array(test_dataset.get_labels(binary=True), dtype=np.float32))

    X_test = np.concatenate(X_test, axis=0)
    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.concatenate(y_test, axis=0)

    if regions_to_remove is not None:
        mask = ~np.isin(np.arange(X_test.shape[2]), regions_to_remove)  # type: ignore
        X_test = X_test[:, :, mask][:, :, :, mask] # type: ignore

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_test, y_test

def train_model(
    model: nn.Module,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    model_save_path: str,
    num_epochs: int = 100,
    alpha: float = 0.3,
    verbose: bool = False
):
    # Training config
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    learning_rate = training_config.get("learning_rate", 0.0001)
    weight_decay = training_config.get("weight_decay", 0.0001)
    
    # Create datasets and dataloaders
    train_ds = FCDataset(X_train, y_train)
    val_ds = FCDataset(X_val, y_val)
    
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
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=10
    # )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    prev_lr = optimizer.param_groups[0]['lr']
    
    if verbose:
        print(f"\nTraining {model_name}...")
        print(f"Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_recon_loss, train_cls_loss = train_epoch(model, train_loader, optimizer, device, alpha=alpha)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device, alpha=alpha)
        
        # Update learning rate based on validation loss
        # scheduler.step(val_metrics['loss'])
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            prev_lr = current_lr
        
        # Save best model based on validation accuracy
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'model_name': model_name,
            }, model_save_path)
        
        # Print progress every epoch
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f}"
              f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} | Best Val={best_val_acc:.4f}@{best_epoch}")
    

def test_model(model, X_val, y_val, X_test, y_test, model_save_path):
    val_metrics = {}
    test_metrics = {}
    # Load best model for final evaluation on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    val_ds = FCDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    val_results = evaluate(model, val_loader, device, return_predictions=True)
    val_predictions = val_results['predictions']
    val_labels = val_results['labels']

    val_metrics['accuracy'] = accuracy_score(val_labels, val_predictions)
    val_metrics['f1_score'] = f1_score(val_labels, val_predictions)
    val_metrics['precision'] = precision_score(val_labels, val_predictions)
    val_metrics['recall'] = recall_score(val_labels, val_predictions)
    val_metrics['roc_auc_score'] = roc_auc_score(val_labels, val_predictions)


    test_ds = FCDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    test_results = evaluate(model, test_loader, device, return_predictions=True)
    test_predictions = test_results['predictions']
    test_labels = test_results['labels']
    
    test_metrics['accuracy'] = accuracy_score(test_labels, test_predictions)
    test_metrics['f1_score'] = f1_score(test_labels, test_predictions)
    test_metrics['precision'] = precision_score(test_labels, test_predictions)
    test_metrics['recall'] = recall_score(test_labels, test_predictions)
    test_metrics['roc_auc_score'] = roc_auc_score(test_labels, test_predictions)

    return val_metrics, test_metrics

def train_epoch(model, dataloader, optimizer, device, alpha=0.5):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_cls_loss = 0
    
    reconstruction_criterion = nn.MSELoss()  # or nn.L1Loss()
    classification_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, logits = model(data)
        
        # Compute losses
        recon_loss = reconstruction_criterion(reconstructed, data)

        cls_loss = classification_criterion(logits, labels)
        
        # Combined loss
        loss = (1-alpha) * recon_loss + alpha * cls_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_cls_loss += cls_loss.item()
    
    return total_loss / len(dataloader), total_recon_loss / len(dataloader), total_cls_loss / len(dataloader)

def evaluate(model, dataloader, device, alpha=0.5, return_predictions=False):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_cls_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        all_labels = []
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            reconstructed, logits = model(data)
            
            # Compute losses
            recon_loss = reconstruction_criterion(reconstructed, data)
            cls_loss = classification_criterion(logits, labels)
            loss = (1-alpha) * recon_loss + alpha * cls_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Classification metrics
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_predictions.append(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    metrics = {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'cls_loss': avg_cls_loss,
        'accuracy': accuracy,
    }
    
    if return_predictions:
        metrics['predictions'] = all_predictions
        metrics['labels'] = all_labels
    
    return metrics