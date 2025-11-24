import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from pipeline.subject import SubjectDataset

def print_scores(scores):
    print(scores['test_score'])
    # mean
    mean_score = np.mean(scores['test_score'])
    print(f"Mean score: {mean_score}")
    return mean_score

def cross_validate_model(model_name, model, X, y, verbose=True):
    scores = cross_validate(model, X, y, cv=5, return_estimator=True)
    best_index = np.argmax(scores["test_score"])
    if verbose:
        print(f"{model_name} - Best accuracy: {scores["test_score"][best_index]}", end="\t")
        print(f"{model_name} - Average accuracy: {np.mean(scores["test_score"])}")

    best_estimator = scores["estimator"][np.argmax(scores["test_score"])]
    return best_estimator

def random_forest_classifier():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def svm_classifier(kernel="linear"):
    return SVC(kernel=kernel)

def knn_classifier(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def ridge_classifier(alpha=0.1):
    return RidgeClassifier(alpha=alpha)


def build_menu():
    return [
        ("Random Forest Classifier", random_forest_classifier()),
        ("SVM Classifier (Linear)", svm_classifier(kernel="linear")),
        ("SVM Classifier (RBF)", svm_classifier(kernel="rbf")),
        ("KNN Classifier", knn_classifier(n_neighbors=5)),
        ("Ridge Classifier", ridge_classifier(alpha=0.1)),
    ]

def train_and_evaluate_model(subject_dataset: SubjectDataset, selected_features: np.ndarray):
    X = subject_dataset.get_fc_array(flatten=True, selected_features=selected_features)
    y = subject_dataset.get_labels(binary=True)

    best_estimators = []
    for model_name, model in build_menu():
        best_estimator = cross_validate_model(model_name, model, X, y)
        best_estimators.append((model_name, best_estimator))

    return best_estimators
    

def test_model(subject_dataset: SubjectDataset, best_estimators, selected_features: str):
    X = subject_dataset.get_fc_array(flatten=True, selected_features=selected_features)
    y = subject_dataset.get_labels(binary=True)

    for model_name, best_estimator in best_estimators:
        print("-"*10 + f"Testing {model_name}" + "-"*10)
        predictions = best_estimator.predict(X)
        print(f"Accuracy: {accuracy_score(y, predictions)}")
        print(classification_report(y, predictions))

def train_test_classics(config: dict):

    dataset_config = config["dataset"]
    image_dir = dataset_config.get("raw_dir", "data/raw")
    preprocessed_dir = dataset_config.get("preprocessed_dir", "data/processed")

    feature_config = config["feature"]
    site_to_train = feature_config.get("site_to_train", "NYU")


    selected_features = preprocessed_dir + f"/{site_to_train}_feature_support.npy"

    # Train on NYU dataset
    NYU_dataset = SubjectDataset(image_dir=image_dir, label_path=preprocessed_dir + "/NYU_labels.csv", fc_dir=preprocessed_dir + "/fc")
    best_estimators = train_and_evaluate_model(NYU_dataset, selected_features)
    
    # Test on NEURO dataset
    NEURO_dataset = SubjectDataset(image_dir=image_dir, label_path=preprocessed_dir + "/NEURO_labels.csv", fc_dir=preprocessed_dir + "/fc")
    test_model(NEURO_dataset, best_estimators, selected_features)

