import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from copy import deepcopy
from tqdm import tqdm


from pipeline.train_test_nn import FCDataset, prepare_data
from model.ae_net import AE_Net

def train_test_nn_joint(config: dict):
    X_train, X_val, X_test, y_train, y_val, y_test, input_dim = prepare_data(config, use_rfe=True, use_segments=False, flatten=True)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"input_dim: {input_dim}")

    # model = AE_Net(input_dim=input_dim)
    # train_model(model, "AE_Net", X_train, y_train, X_val, y_val, X_test, y_test, config, "model/ae_net.pt", verbose=True)
    # reconstruct_test(model, X_train, y_train, X_val, y_val, "model/ae_net.pt")
    train_model_bh(X_train, y_train, X_val, y_val)
    pass

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
    model_save_path: str,
    verbose: bool = False
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
    
    if verbose:
        print(f"\nTraining {model_name}...")
        print(f"Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples | Test set: {len(X_test)} samples")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_recon_loss, train_cls_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])
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
    
    # Load best model for final evaluation on test set
    print("\nEvaluating on test set..")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, return_predictions=True)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(test_metrics['labels'], test_metrics['predictions'], target_names=['Control', 'ADHD']))

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
        loss = recon_loss + alpha * cls_loss
        
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
    all_labels = []
    all_logits = []
    
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            reconstructed, logits = model(data)
            
            # Compute losses
            recon_loss = reconstruction_criterion(reconstructed, data)
            cls_loss = classification_criterion(logits, labels)
            loss = recon_loss + alpha * cls_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Classification metrics
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
    
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
        metrics['predictions'] = np.array(all_predictions)
        metrics['labels'] = np.array(all_labels)
        metrics['logits'] = np.array(all_logits)
    
    return metrics

def reconstruct_test(model, X_train, y_train, X_test, y_test, model_save_path: str):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_reconstructed = []
    dataloader = DataLoader(FCDataset(X_train, y_train), batch_size=1, shuffle=False)
    with torch.no_grad():
        for X, y in dataloader:
            reconstructed, _ = model(X)
            all_reconstructed.append(reconstructed.cpu().numpy())
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)

    dataloader = DataLoader(FCDataset(X_test, y_test), batch_size=1, shuffle=False)
    ae_prediction = []
    bh_prediction = []
    with torch.no_grad():
        for X, y in dataloader:
            reconstructed, logits = model(X)
            ae_prediction.append(np.argmax(logits.cpu().numpy(), axis=1))
            bh_prediction.append(binary_hypothesis_test(all_reconstructed, y_train, reconstructed.cpu().numpy()))
    bh_prediction = np.array(bh_prediction)
    ae_prediction = np.array(ae_prediction)
    print(f"Accuracy: {accuracy_score(y_test, bh_prediction)}")
    print(f"Classification Report: {classification_report(y_test, bh_prediction)}")
    print(f"Accuracy: {accuracy_score(y_test, ae_prediction)}")
    print(f"Classification Report: {classification_report(y_test, ae_prediction)}")

def binary_hypothesis_test(all_reconstructed, labels, X):
    all_data = np.concatenate([all_reconstructed, X], axis=0)

    h0_labels = np.append(labels, 0)
    h1_labels = np.append(labels, 1)

    inter_var_h0 = compute_inter_class_variance(all_data, h0_labels)
    intra_var_h0 = compute_intra_class_variance(all_data, h0_labels)
    ratio_h0 = intra_var_h0 / inter_var_h0

    inter_var_h1 = compute_inter_class_variance(all_data, h1_labels)
    intra_var_h1 = compute_intra_class_variance(all_data, h1_labels)
    ratio_h1 = intra_var_h1 / inter_var_h1

    print(f"Ratio H0: {ratio_h0}, Ratio H1: {ratio_h1}")

    if ratio_h1 < ratio_h0:
        return 1
    else:
        return 0


def compute_inter_class_variance(features, classes):
    # Compute global mean across all samples
    global_mean = np.mean(features, axis=0)  # shape: (n_features,)
    
    unique_classes = np.unique(classes)
    
    class_means = []
    class_counts = []
    
    for cls in unique_classes:
        class_mask = classes == cls
        class_samples = features[class_mask]
        
        # Compute mean for this class
        class_mean = np.mean(class_samples, axis=0)
        class_means.append(class_mean)
        class_counts.append(np.sum(class_mask))
    
    class_means = np.array(class_means)  # shape: (n_classes, n_features)
    class_counts = np.array(class_counts)  # shape: (n_classes,)
    
    inter_class_var = 0
    for i, cls_mean in enumerate(class_means):
        diff = cls_mean - global_mean
        inter_class_var += class_counts[i] * np.sum(diff ** 2)
    
    # Normalize by total number of samples
    inter_class_var /= len(features)
    
    return inter_class_var


def compute_intra_class_variance(features, classes):
    unique_classes = np.unique(classes)
    intra_class_var = 0
    
    for cls in unique_classes:
        # Get samples belonging to this class
        class_mask = classes == cls
        class_samples = features[class_mask]
        
        # Compute mean for this class
        class_mean = np.mean(class_samples, axis=0)
        
        # Compute sum of squared differences (vectorized)
        diff = class_samples - class_mean
        intra_class_var += np.sum(diff ** 2)
    
    # Normalize by total number of samples
    intra_class_var /= len(features)
    
    return intra_class_var


def train_model_bh(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h0_model = AE_Net(90)
    h1_model = deepcopy(h0_model)

    h0_model = h0_model.to(device)
    h1_model = h1_model.to(device)

    h0_optimizer = optim.Adam(
        h0_model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    h1_optimizer = optim.Adam(
        h1_model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    bh_predictions = []


    for x_test, _ in tqdm(zip(X_test, y_test), total=len(X_test)):

        X_train_bh = np.concatenate([x_test.reshape(1, -1), X_train], axis=0)
        y_train_h0 = np.append(0, y_train)
        y_train_h1 = np.append(1, y_train)


        train_loader_h0 = DataLoader(FCDataset(X_train_bh, y_train_h0), batch_size=4, shuffle=True, num_workers=0, drop_last=True)
        train_loader_h1 = DataLoader(FCDataset(X_train_bh, y_train_h1), batch_size=4, shuffle=True, num_workers=0, drop_last=True)

        for _ in range(50):
            train_epoch(h0_model, train_loader_h0, h0_optimizer, device)
            train_epoch(h1_model, train_loader_h1, h1_optimizer, device)


        train_loader_h0 = DataLoader(FCDataset(X_train_bh, y_train_h0), batch_size=1, shuffle=False, num_workers=0)
        train_loader_h1 = DataLoader(FCDataset(X_train_bh, y_train_h1), batch_size=1, shuffle=False, num_workers=0)
        h0_model.eval()
        h1_model.eval()
        
        h0_X = []
        h1_X = []
        with torch.no_grad():
            for X, y in train_loader_h0:
                X = X.to(device)
                reconstructed, _ = h0_model(X)
                h0_X.append(reconstructed.cpu().numpy())
            for X, y in train_loader_h1:
                X = X.to(device)
                reconstructed, _ = h1_model(X)
                h1_X.append(reconstructed.cpu().numpy())
        h0_X = np.concatenate(h0_X, axis=0)
        h1_X = np.concatenate(h1_X, axis=0)

        ratio_h0 = compute_inter_class_variance(h0_X, y_train_h0) / compute_intra_class_variance(h0_X, y_train_h0)
        ratio_h1 = compute_inter_class_variance(h1_X, y_train_h1) / compute_intra_class_variance(h1_X, y_train_h1)

        if ratio_h1 < ratio_h0:
            bh_predictions.append(1)
        else:
            bh_predictions.append(0)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, bh_predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report: {classification_report(y_test, bh_predictions)}")
    return accuracy
