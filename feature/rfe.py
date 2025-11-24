# Reduce Feature Elimination with SVM
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

from pipeline.subject import SubjectDataset

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
    site_to_use = feature_config["site_to_use"]
    n_features = feature_config["rfe_n_features"]

    subject_dataset = SubjectDataset(site_to_use)

    X = subject_dataset.get_fc_array(flatten=True)
    y = subject_dataset.get_labels(binary=True)

    while True:
        choice = display_menu()
        if choice == "q":
            break
        elif choice == "1":
            print("Running SVM RFE... may take 15 minutes or longer to complete")
            support, ranking = svm_rfe(X, y, n_features=n_features)
            support_indices = np.where(support)[0]
            np.save(f"data/processed/{site_to_use}_feature_support.npy", support_indices)
            np.save(f"data/processed/{site_to_use}_feature_ranking.npy", ranking)
            print(f"Saved feature support and ranking to data/processed/{site_to_use}_feature_support.npy and data/processed/{site_to_use}_feature_ranking.npy")

