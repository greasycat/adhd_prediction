from pathlib import Path
import pandas as pd

class SubjectDataset:
    def __init__(self, image_dir="data/raw", label_path="data/processed/NYU_labels.csv"):
        self.image_dir = image_dir
        self.label_path = label_path
        self.label_df = pd.read_csv(label_path)

    def __str__(self):
        return f"SubjectDataset(image_dir={self.image_dir}, label_path={self.label_path})"

    def __repr__(self):
        return self.__str__()

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
        return self.label_df[self.label_df["id"] == id]["label"].values[0]
    

if __name__ == "__main__":
    subject = SubjectDataset(image_dir="data/raw", label_path="data/processed/NYU_labels.csv")
    print(subject.get_mask_path("sub-0010001"))
    print(subject.get_image_path("sub-0010001"))
    print(subject.get_label("sub-0010001"))