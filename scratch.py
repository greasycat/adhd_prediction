from utils.subject import SubjectDataset

subject = SubjectDataset()
from nilearn.datasets import fetch_atlas_aal
aal = fetch_atlas_aal(version="3v2")

print(
    "AAL atlas nifti image (3D) "
    f" is located at: {aal['maps']}"
)

# %%


# %%
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

# ConnectivityMeasure from Nilearn uses simple 'correlation' to compute
# connectivity matrices for all subjects in a list
def compute_connectivity(img):
    connectome_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )

    masker = NiftiLabelsMasker(
        labels_img=aal["maps"],  # Both hemispheres
        lookup_lut=aal["labels"],
        standardize="zscore_sample",
        memory="nilearn_cache",
        n_jobs=24,
    )

    ts = masker.fit_transform(img)
    connectome = connectome_measure.fit_transform([ts])
    return connectome[0]


# %%
import numpy as np

from nilearn import plotting

connectome1 = compute_connectivity(subject.get_images()[0])
np.fill_diagonal(connectome1, 0)
plotting.plot_matrix(
    connectome1,
    figure=(10, 8),
    labels=aal.labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Confounds",
    reorder=True,
)

# %%
from tqdm import tqdm
from pathlib import Path

fc_dir = Path("data/processed/fc")
if not fc_dir.exists():
    fc_dir.mkdir(parents=True)

for id, img, mask in tqdm(subject.enumerate_subjects()):
    fc_path = fc_dir / f"{id}.npy"
    if fc_path.exists():
        print(f"Skipping {id} because it already exists")
        continue
    connectome = compute_connectivity(img)
    np.fill_diagonal(connectome, 0)
    np.save(fc_dir / f"{id}.npy", connectome)

# %%
corr = subject.get_fc_array()

# get upper triangular part
upper_tri = np.triu_indices(corr.shape[1], k=1)
corr_upper = corr[:, upper_tri[0], upper_tri[1]]

print(corr_upper.shape)

# %%
labels = subject.get_labels()
labels_merged = [0 if x == 0 else 1 for x in labels]
print(labels_merged)

# %%
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

svc = SVC(kernel="linear")
selector = RFE(estimator=svc, n_features_to_select=90, step=1)
selector = selector.fit(corr_upper, labels_merged)
print(selector.support_)
print(selector.ranking_)

# %%
# get the selected features
selected_features = np.where(selector.support_)[0]
print(selected_features)

# get the selected features from the corr_upper
selected_corr_upper = corr_upper[:, selected_features]
print(selected_corr_upper.shape)

# save the selected features
np.save("data/processed/fc/selected_corr_upper.npy", selected_corr_upper)


# %%
# create a rbf kernel and run classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

train_corr_upper, test_corr_upper, train_labels, test_labels = train_test_split(
    selected_corr_upper, labels_merged, test_size=0.2, random_state=42
)

svc = SVC(kernel="rbf")
svc.fit(train_corr_upper, train_labels)

predictions = svc.predict(test_corr_upper)

from sklearn.metrics import accuracy_score, classification_report

print(f"Accuracy: {accuracy_score(test_labels, predictions)}")
print(classification_report(test_labels, predictions))


# %%
# 5-fold cross validation
from sklearn.model_selection import cross_validate


def run_cv(model, X, y):
    scores = cross_validate(model, X, y, cv=5, return_estimator=True)

    print(scores['test_score'])
    # mean
    mean_score = np.mean(scores['test_score'])
    print(f"Mean score: {mean_score}")
    return scores

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = run_cv(rf, selected_corr_upper, labels_merged)


svc = SVC(kernel="rbf")
svc_scores = run_cv(svc, selected_corr_upper, labels_merged)



# %%
print(scores['test_score'])
# find the best estimator
best_estimator = scores["estimator"][np.argmax(scores["test_score"])]
print(best_estimator)


# %%
# find the top 10important features
importances = best_estimator.feature_importances_

print(len(importances))

# get the indices of the top 10 features
top_10_indices = np.argsort(importances)[-10:]

# get the top 10 features
top_10_features = selected_corr_upper[:, top_10_indices]

print(top_10_features.shape)

def flatten_idx_to_pair(flat_idx, upper_tri):
    """Convert flattened index to original (row, col) pair"""
    row = upper_tri[0][flat_idx]
    col = upper_tri[1][flat_idx]
    return row, col

index_pairs = [flatten_idx_to_pair(idx, upper_tri) for idx in top_10_indices]
print(index_pairs)




