# Downsize the FC to 90 ROIs
# removing the 26 cerebellar regions from the 116 regions in AAL atlases version 2    

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal

from utils.label_extractor import SITES
from pipeline.subject import SubjectDataset

class FC:
    def __init__(self, config: dict):

        self.dataset_config = config["dataset"]
        self.data_dir = self.dataset_config.get("data_dir", "data")
        self.processed_dir = self.dataset_config.get("processed_dir", "data/processed")
        self.raw_dir = self.dataset_config.get("raw_dir", "data/raw")
        self.fc_dir = self.dataset_config.get("fc_dir", "data/processed/fc")

        # Ensure output directories exist
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.fc_dir).parent.mkdir(parents=True, exist_ok=True)

        self.aal = fetch_atlas_aal(version="SPM12") # Get the AAL atlas, version 2, 116 regions in total

        # write map labels to a csv file
        labels = [str(l) for l in self.aal["labels"]] # type: ignore
        pd.DataFrame(labels, columns=["label"]).to_csv(self.processed_dir + "/aal_labels.csv", index=False)

        # Build indices to keep 90 cortical/subcortical regions (remove cerebellum/vermis)
        # AAL cerebellar regions typically contain "Cerebel" and "Vermis" in their names
        self.keep_indices = np.array(
            [
                i
                for i, name in enumerate(labels)
                if ("Cerebel" not in name and "Vermis" not in name and name.lower() != "background")
            ],
            dtype=int,
        )
        # Expect 90 remaining regions from original 116
        if len(self.keep_indices) != 90:
            print(f"Warning: expected 90 kept regions, got {len(self.keep_indices)}")

        # Save the 90-region label list for reference
        kept_labels = [labels[i] for i in self.keep_indices]
        pd.DataFrame(kept_labels, columns=["label"]).to_csv(self.processed_dir + "/aal90_labels.csv", index=False)

        self.datasets = {}
        for site, _ in SITES.items():
            self.datasets[site] = SubjectDataset(image_dir=self.raw_dir, label_path=self.processed_dir + f"/{site}_labels.csv", fc_dir=self.fc_dir)

    def _compute_connectivity(self, img):
        connectome_measure = ConnectivityMeasure(
            kind="correlation",
            standardize="zscore_sample", # type: ignore
        )

        masker = NiftiLabelsMasker(
            labels_img=self.aal["maps"], 
            lookup_lut=self.aal["labels"],
            standardize="zscore_sample", # type: ignore
            memory="nilearn_cache",
            n_jobs=24,
        )

        ts = masker.fit_transform(img)
        # Downselect to 90 non-cerebellar/vermian regions
        ts = ts[:, self.keep_indices]
        connectome = connectome_measure.fit_transform([ts])
        return connectome[0]
    
    def compute_all_connectivity(self, site: str):
        subject_dataset: SubjectDataset = self.datasets[site]

        fc_dir = Path(self.processed_dir) / "fc"
        if not fc_dir.exists():
            fc_dir.mkdir(parents=True)
        
        progress = tqdm(total=len(subject_dataset), desc=f"Computing FC for {site}")

        for id, img, _ in subject_dataset.enumerate_subjects():
            progress.update(1)
            fc_path = fc_dir / f"{id}.npy"

            # if fc_path.exists():
            #     progress.set_description(f"Skipping {id} because it already exists")
            #     continue

            connectome = self._compute_connectivity(img)
            np.fill_diagonal(connectome, 0)
            # get upper triangular part
            np.save(fc_path, connectome)
            print(f"Shape of connectome: {connectome.shape}")
        
        progress.close()

def display_menu():
    print("-"*10 + "FC MENU" + "-"*10)
    for site, id in SITES.items():
        print(f"{id}) {site}")
    print("all) Compute FC for all sites")
    print("q) Quit")
    return input("Enter your choice: ")
    
def compute_fc(config: dict):
    fc = FC(config)
    site_swap = dict(zip(SITES.values(), SITES.keys()))
    while True:
        choice = display_menu()
        if choice == "q":
            break
        elif choice == "all":
            for site in site_swap.keys():
                fc.compute_all_connectivity(site_swap[int(site)])
            return
        else:
            choice = int(choice)
            if choice not in site_swap.keys():
                print("Invalid choice")
                continue
            fc.compute_all_connectivity(site_swap[choice])


if __name__ == "__main__":
    config = {
        "dataset": {
            "data_dir": "data",
            "processed_dir": "data/processed",
            "raw_dir": "data/raw",
            "fc_dir": "data/processed/fc",
        }
    }
    compute_fc(config)