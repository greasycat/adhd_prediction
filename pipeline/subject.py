from pathlib import Path
from sklearn.model_selection import train_test_split
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
    
    def get_label(self, id):
        return self.label_df[self.label_df["id"] == id]["label"].values[0] # type: ignore
    
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
        fc_path = Path(self.fc_dir) / f"{id}.npy"
        if not fc_path.exists():
            raise FileNotFoundError(f"FC file not found, please run `uv run utils/feature_extract.py` to extract the FC: {fc_path}")
        return np.load(fc_path)
    
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
    
    def get_train_val_test_split(self, selected_features=None, test_size=0.2, random_state=42):
        X = self.get_fc_array(flatten=True)
        y = np.array(self.get_labels(binary=True), dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        if selected_features is not None:
            selected_features = np.load(selected_features)
            X_train = X_train[:, selected_features] # type: ignore
            X_test = X_test[:, selected_features] # type: ignore
        return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    subject = SubjectDataset(image_dir="data/raw", label_path="data/processed/NYU_labels.csv")
    print(subject.get_mask_path("sub-0010001"))
    print(subject.get_image_path("sub-0010001"))
    print(subject.get_label("sub-0010001"))