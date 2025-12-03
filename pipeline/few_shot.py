import copy
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.modules import BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from pipeline.train_test_nn import FCDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

from model.skip_vote import SkipVoteNet
from pipeline import SubjectDataset
from utils.label_extractor import SITES

class FewShotDataset(Dataset):
    """PyTorch Dataset for Functional Connectivity data."""
    
    def __init__(self, datasets, k_shot: int, n_query: int, device: torch.device):
        self.datasets = datasets
        self.k_shot = k_shot
        self.n_query = n_query
        self.device = device
        self.Xs = []
        self.ys = []
        for dataset in self.datasets:
            X, y = dataset.get_fc_segments_and_labels(n_segments=7)
            y = np.expand_dims(y, axis=1)
            self.Xs.append(X)
            self.ys.append(y)
        pass

    def sample(self, idx):
        X = self.Xs[idx]
        y = self.ys[idx]
        X_main, X_support, y_main, y_support = train_test_split(X, y, test_size=self.k_shot * 2, stratify=y) # times 2 because we need k shots for 2 classes
        _, X_query, _, y_query = train_test_split(X_main, y_main, test_size=self.n_query * 2, stratify=y_main)
        X_support = torch.FloatTensor(X_support).to(self.device)
        y_support = torch.FloatTensor(y_support).to(self.device)
        X_query = torch.FloatTensor(X_query).to(self.device)
        y_query = torch.FloatTensor(y_query).to(self.device)
        return X_support, y_support, X_query, y_query
    
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):
        return self.sample(idx)

def prepare_few_shot_data(config: dict):
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    site_to_train = config["feature"].get("site_to_train", "NYU")
    k_shot = config["few_shot"].get("k_shot", 5)
    n_query = config["few_shot"].get("n_query", 15)
    site_to_test = config["few_shot"].get("site_to_test", "NEURO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    ) for site in SITES.keys() if site != site_to_train and site != site_to_test]

    few_shot_dataset = FewShotDataset(datasets, k_shot, n_query, device)
    return few_shot_dataset

def split_test_data(config: dict):
    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")
    site_to_test = config["few_shot"].get("site_to_test", "NEURO")
    n_query = config["few_shot"].get("n_query", 15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = SubjectDataset(
        image_dir=image_dir,
        label_path=f"{preprocessed_dir}/{site_to_test}_labels.csv",
        fc_dir=f"{preprocessed_dir}/fc"
    )
    X, y = test_dataset.get_fc_segments_and_labels(n_segments=7)
    y = np.expand_dims(y, axis=1)
    n_samples = X.shape[0]
    print("Number of samples:", n_samples)
    k_fold = StratifiedKFold(n_splits= (n_samples // (n_query * 2)), shuffle=True, random_state=42)
    for train_index, test_index in k_fold.split(X, y):
        X_adapt, y_adapt = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_adapt = torch.FloatTensor(X_adapt).to(device)
        y_adapt = torch.FloatTensor(y_adapt).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        yield X_adapt, y_adapt, X_test, y_test


def maml_train(model, few_shot_dataset, meta_lr=0.001, inner_lr=0.01, 
               inner_steps=5, num_epochs=1000):
    meta_optimizer = Adam(model.parameters(), lr=meta_lr)
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        meta_loss = 0.0

        # Resample every epoch
        dataloader = DataLoader(few_shot_dataset, batch_size=None, shuffle=False, num_workers=0)
        
        for X_support, y_support, X_query, y_query in dataloader:

            # Clone model to create task-specific parameters
            task_model = clone_model(model)
            task_optimizer = SGD(task_model.parameters(), lr=inner_lr)
            
            # Perform K gradient steps on support set
            for step in range(inner_steps):
                support_predictions = task_model(X_support.unsqueeze(1))
                support_loss = loss_fn(support_predictions, y_support)
                
                task_optimizer.zero_grad()
                support_loss.backward()
                task_optimizer.step()
            
            # Evaluate adapted model on query set
            query_predictions = task_model(X_query.unsqueeze(1))
            query_loss = loss_fn(query_predictions, y_query)
            
            meta_loss += query_loss
        
        meta_loss = meta_loss / len(dataloader)
        
        meta_optimizer.zero_grad()
        meta_loss.backward() # type: ignore (meta_loss is a float)
        meta_optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Meta-Loss: {meta_loss.item():.4f}") # type: ignore (meta_loss is a float)
    
    return model  # Returns model with optimized initial parameters

def clone_model(model):
    """Create a copy of model with same parameters but separate computation graph"""
    cloned = copy.deepcopy(model)
    return cloned


def adapt_to_new_site(model, new_site_data, new_site_labels, 
                      inner_lr=0.01, inner_steps=10):
    adapted_model = clone_model(model)
    optimizer = SGD(adapted_model.parameters(), lr=inner_lr)
    loss_fn = BCEWithLogitsLoss()
    
    # Fine-tune on new site data
    for step in range(inner_steps):
        predictions = adapted_model(new_site_data.unsqueeze(1))
        loss = loss_fn(predictions, new_site_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Adaptation step {step}, Loss: {loss.item():.4f}")
    
    return adapted_model

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X.unsqueeze(1))
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels


def evaluate_model(config, model, device, adapt=True):
    total_accuracy = 0.0
    count = 0
    for X_adapt, y_adapt, X_test, y_test in split_test_data(config):
        count += 1
        if adapt:
            model = adapt_to_new_site(model, X_adapt, y_adapt)
        test_ds = FCDataset(X_test, y_test) # type: ignore
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
        test_loss, test_acc, predictions, labels = evaluate(model, test_loader, BCEWithLogitsLoss(), device)
        total_accuracy += test_acc
        # print("\nClassification Report (Test Set):")
        # print(classification_report(labels, predictions, target_names=['Control', 'ADHD']))

    print(f"Total Accuracy: {total_accuracy / count:.4f}")

def train_few_shot_model(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    few_shot_dataset = prepare_few_shot_data(config)

    model = SkipVoteNet()
    model = model.to(device)

    model.load_state_dict(torch.load("model/skipvotenet_best.pt")['model_state_dict'])

    print("Evaluating model before training...")
    evaluate_model(config, model, device, adapt=False)

    if os.path.exists("model/few_shot_model.pt"):
        # model.load_state_dict(torch.load("model/few_shot_model.pt"))
        pass
    else:
        model = maml_train(
            model=model,
            few_shot_dataset=few_shot_dataset,
            meta_lr=0.001,
            inner_lr=0.01,
            inner_steps=5,
            num_epochs=100
        )
        # save model
        torch.save(model.state_dict(), "model/few_shot_model.pt")

    print("Evaluating model after training...")
    evaluate_model(config, model, device)

    
