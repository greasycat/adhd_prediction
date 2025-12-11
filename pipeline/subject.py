from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np

class SubjectDataset:
    def __init__(self, image_dir="data/raw", label_path="data/processed/NYU_labels.csv", fc_dir="data/processed/fc"):
        self.image_dir = image_dir
        self.label_path = label_path
        self.fc_dir = fc_dir
        self.label_df = pd.read_csv(label_path)

    def __str__(self):
        return f"SubjectDataset(image_dir={self.image_dir}, label_path={self.label_path})"

    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.label_df)

    def get_mask_path(self, id):
        path = f"{self.image_dir}/{id}/{id}_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        if not Path(path).exists():
            raise FileNotFoundError(f"Mask file not found: {path}")
        return path

    def get_image_path(self, id):
        path = f"{self.image_dir}/{id}/{id}_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        if not Path(path).exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return path
    
    def get_label(self, id, binary=False):
        label = self.label_df[self.label_df["id"] == id]["label"].values[0] # type: ignore
        if binary:
            return 1 if label > 0 else 0
        return label
    
    def get_labels(self, binary=False):
        if binary:
            return self.label_df["label"].clip(0, 1).tolist()

        return self.label_df["label"].tolist()

    def get_phenotype_array(self, columns=None):
        """
        Get phenotypic features.
        """
        if columns is None:
            # Use all columns except the identifier and label as phenotypes
            columns = [
                c
                for c in self.label_df.columns
                if c not in ["id", "label"]
            ]
        return self.label_df[columns].to_numpy(dtype=np.float32)
    
    def get_all_ids(self):
        return self.label_df["id"].tolist()
    
    def get_images(self):
        return [self.get_image_path(id) for id in self.get_all_ids()]
    
    def enumerate_subjects(self):
        for id in self.get_all_ids():
            yield id, self.get_image_path(id), self.get_mask_path(id)
    
    def get_fc(self, id):
        fc_path = Path(self.fc_dir) / "full" / f"{id}.npy"
        if not fc_path.exists():
            raise FileNotFoundError(f"FC file not found, please run `uv run utils/feature_extract.py` to extract the FC: {fc_path}")
        return np.load(fc_path)
    
    def get_fc_segments(self, id, n_segments: int = 7):
        fc_segments = []
        for i in range(n_segments):
            fc_path = Path(self.fc_dir) / f"n_segments_{n_segments}" / f"{id}_{i}.npy"
            if not fc_path.exists():
                raise FileNotFoundError(f"FC file not found, please run `uv run utils/feature_extract.py` to extract the FC: {fc_path}")
            fc_segments.append(np.load(fc_path, allow_pickle=True))
        return fc_segments
    
    def get_fc_array(self, flatten=False, selected_features=None):
        corr = np.stack([self.get_fc(id) for id in self.get_all_ids()])

        if flatten:
            # Get upper triangular part of the correlation arrays
            upper_tri = np.triu_indices(corr.shape[1], k=1)
            # Flatten the upper triangular part
            corr = corr[:, upper_tri[0], upper_tri[1]]

            
            if selected_features is not None:
                selected_features = np.load(selected_features)
                corr = corr[:, selected_features]
        return corr
    
    def get_fc_segments_and_labels(self, n_segments: int = 7, binary=True, with_id=False, batch_dim=True):
        all_fc_segments = []
        all_labels = []
        all_id = []
        for id in self.get_all_ids():
            fc_segments = self.get_fc_segments(id, n_segments)
            label = self.get_label(id, binary=binary)
            labels = [label] * n_segments
            all_fc_segments.extend(fc_segments)
            all_labels.extend(labels)
            all_id.extend([id] * n_segments)

        all_fc_segments = np.stack(all_fc_segments)
        if batch_dim:
            all_fc_segments = np.expand_dims(all_fc_segments, axis=1)
        all_labels = np.array(all_labels)
        all_id = np.array(all_id)
        if with_id:
            return all_fc_segments, all_labels, all_id
        else:
            return all_fc_segments, all_labels
    
    def get_stratifed_ids(self):
        ids = self.get_all_ids()
        labels = np.array(self.get_labels(binary=True))
        return train_test_split(ids, labels, test_size=0.2, random_state=42, stratify=labels)
    
    def get_fc_segments_and_labels_by_ids(self, ids, n_segments: int = 7, binary=True):
        X = []
        y = []
        for id in ids:
            fc_segments = self.get_fc_segments(id, n_segments)
            label = self.get_label(id, binary=binary) 
            X.append(fc_segments)
            y.append([label] * n_segments)
        X = np.concatenate(X, axis=0)
        X = np.expand_dims(X, axis=1)
        y = np.concatenate(y, axis=0)
        return X, y
    
    # For few-shot learning
    # Get array of one random fc segments and label pair from each subject
    def get_subject_random_fc_segments_and_labels(self, n_segments: int = 7, random_state=42):
        X = []
        y = []
        # generate a random index for each subject
        random_indices = np.random.RandomState(random_state).choice(n_segments, size=len(self.get_all_ids()))
        for i, id in enumerate(self.get_all_ids()):
            fc_segments = self.get_fc_segments(id, n_segments)
            random_index = random_indices[i]
            X.append(fc_segments[random_index])
            y.append(self.get_label(id))
        return np.array(X), np.array(y)

    
    def get_train_val_test_split(self, selected_features=None, test_size=0.2, random_state=42, flatten=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = self.get_fc_array(flatten=flatten)
        X = np.expand_dims(X, axis=1)
        y = np.array(self.get_labels(binary=True), dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        if selected_features is not None:
            selected_features = np.load(selected_features)
            X_train = X_train[:, selected_features] # type: ignore
            X_test = X_test[:, selected_features] # type: ignore
        return X_train, X_test, y_train, y_test # type: ignore
    
    def k_fold_split(self, k=5, random_state=42, flatten=True):
        X = self.get_fc_array(flatten=flatten)
        X = np.expand_dims(X, axis=1)
        y = np.array(self.get_labels(binary=True))
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_index, test_index in kf.split(X, y):
            yield X[train_index], X[test_index], y[train_index], y[test_index]
    
if __name__ == "__main__":
    subject = SubjectDataset(image_dir="data/raw", label_path="data/processed/NYU_labels.csv")
    train_ids, test_ids, train_labels, test_labels = subject.get_stratifed_ids()
    # value count of train_labels and test_labels
    print(np.unique(train_labels, return_counts=True))
    print(np.unique(test_labels, return_counts=True))
    X_train, y_train = subject.get_fc_segments_and_labels_by_ids(train_ids, n_segments=7, binary=True)
    print(X_train.shape, y_train.shape)