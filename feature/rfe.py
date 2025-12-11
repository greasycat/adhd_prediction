# Author: R Jin, Y Yu
# Last Modified Date: 2025-12-11
# Description: This file contains the code for reducing the features using SVM RFE.
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from pipeline.subject import SubjectDataset

# Reduce Feature Elimination with SVM
def svm_rfe(X, y, n_features=90):
    svc = SVC(kernel="linear")
    selector = RFE(estimator=svc, n_features_to_select=n_features, step=1)
    selector = selector.fit(X, y)
    return selector.support_, selector.ranking_

def display_menu():
    print("-"*10 + "RFE MENU" + "-"*10)
    print("1) Reduce Feature Elimination with SVM")
    print("q) Quit")
    return input("Enter your choice: ")

def rfe(config: dict):
    feature_config = config["feature"]
    site_to_train = feature_config["site_to_train"]
    n_features = feature_config["rfe_n_features"]

    subject_dataset = SubjectDataset(site_to_train)

    X = subject_dataset.get_fc_array(flatten=True)
    y = subject_dataset.get_labels(binary=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    while True:
        choice = display_menu()
        if choice == "q":
            break
        elif choice == "1":
            print("Running SVM RFE... may take 15 minutes or longer to complete")
            support, ranking = svm_rfe(X_train, y_train, n_features=n_features)
            support_indices = np.where(support)[0]
            np.save(f"data/processed/{site_to_train}_feature_support.npy", support_indices)
            np.save(f"data/processed/{site_to_train}_feature_ranking.npy", ranking)
            print(f"Saved feature support and ranking to data/processed/{site_to_train}_feature_support.npy and data/processed/{site_to_train}_feature_ranking.npy")

