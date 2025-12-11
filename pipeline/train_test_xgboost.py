import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score

from pipeline import SubjectDataset
from model.transformer import TransformerClassifierMultiToken
from model.resnet import ResNetClassifier
from model.skip_vote import SkipVoteNet
from utils.label_extractor import SITES

from pipeline.train_test_nn import prepare_data
from xgboost import XGBClassifier


def get_available_models() -> list[tuple[str, str, dict[str, bool]]]:
    """Get list of available models.
    
    Returns:
        List of (model_name, description, config) tuples
    """
    return [
        (
            "XGBoost Classifier", 
            "Transformer with multi-token attention", 
            {
                'use_rfe':False,
                'flatten':True,
                'use_segments':False,
            }
        )]


def display_model_menu(available_models: list[tuple[str, str, dict[str, bool]]]):
    """Display the model selection menu."""
    print("\n" + "-" * 60)
    print("Available Models:")
    print("-" * 60)
    for i, (name, description, config) in enumerate(available_models, 1):
        print(f"{i}) {name}: {description} | use_rfe: {config['use_rfe']} | flatten: {config['flatten']} | use_segments: {config['use_segments']}")
    print(f"{len(available_models) + 1}) Quit")
    print("-" * 60)


def train_model(
    model: XGBClassifier, 
    model_name: str, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    config: dict, 
    model_save_path: str):
    """Train an XGBoost model on the training data, validate on the validation data, and test on the test data."""
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    y_pred = model.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, y_pred)}")
    print(f"Validation balanced accuracy: {balanced_accuracy_score(y_val, y_pred)}")
    print(f"Validation F1 score: {f1_score(y_val, y_pred)}")
    print(f"Validation ROC AUC score: {roc_auc_score(y_val, y_pred)}")


def train_test_xgboost(config: dict):
    
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

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=500,
            learning_rate=0.001,
            early_stopping_rounds=100,
            max_depth=20,
            min_child_weight=5)
            
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
