from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal

from utils.label_extractor import SITES
from pipeline.subject import SubjectDataset

def split_ts(x, n_segments: int = 7):
    t = x.shape[0]
    segment_size = 2 * t // (n_segments + 1)
    step = segment_size // 2  # 50% overlap

    segments = [x[i*step : i*step + segment_size] for i in range(n_segments)]
    return segments

class FC:
    def __init__(self, config: dict):

        self.dataset_config = config["dataset"]
        self.data_dir = self.dataset_config.get("data_dir", "data")
        self.processed_dir = self.dataset_config.get("processed_dir", "data/processed")
        self.raw_dir = self.dataset_config.get("raw_dir", "data/raw")
        self.fc_dir = self.dataset_config.get("fc_dir", "data/processed/fc")

        self.aal = fetch_atlas_aal(version="3v2") # Get the AAL atlas

        # write map labels to a csv file
        pd.DataFrame(self.aal["labels"], columns=["label"]).to_csv(self.processed_dir + "/aal_labels.csv", index=False) # type: ignore

        self.datasets = {}
        for site, _ in SITES.items():
            self.datasets[site] = SubjectDataset(image_dir=self.raw_dir, label_path=self.processed_dir + f"/{site}_labels.csv", fc_dir=self.fc_dir)

    def _compute_connectivity(self, img, n_segments: int = 7):
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
        # split the ts into n_segments+1 segments and attach consecutive 2 to form n_segments
        segments = split_ts(ts, n_segments)
        connectome = connectome_measure.fit_transform(segments)
        return np.squeeze(connectome)
    
    def compute_all_connectivity(self, site: str, n_segments: int = 7):
        subject_dataset: SubjectDataset = self.datasets[site]

        fc_dir = Path(self.processed_dir) / "fc" / f"n_segments_{n_segments}"
        if not fc_dir.exists():
            fc_dir.mkdir(parents=True)
        
        progress = tqdm(total=len(subject_dataset), desc=f"Computing FC for {site}")

        for id, img, _ in subject_dataset.enumerate_subjects():
            progress.update(1)

            # if fc_path.exists():
            #     progress.set_description(f"Skipping {id} because it already exists")
            #     continue
            connectome = self._compute_connectivity(img)
            for i, c in enumerate(connectome):
                fc_path = fc_dir / f"{id}_{i}.npy"
                np.fill_diagonal(c, 0)
                np.save(fc_path, c)
        
        progress.close()

def display_menu():
    print("-"*10 + "FC MENU" + "-"*10)
    for site, id in SITES.items():
        print(f"{id}) {site}")
    print("all) Compute FC for all sites")
    print("1) Compute FC for NYU dataset with 7 segmentations")
    print("q) Quit")
    return input("Enter your choice: ")
    
def compute_fc(config: dict):
    fc = FC(config)
    site_swap = dict(zip(SITES.values(), SITES.keys()))
    while True:
        choice = display_menu()
        if choice == "q":
            break
        elif choice == "1":
            fc.compute_all_connectivity("NYU", 7)
        elif choice == "all":
            for site in site_swap.keys():
                fc.compute_all_connectivity(site_swap[int(site)])
        else:
            choice = int(choice)
            if choice not in site_swap.keys():
                print("Invalid choice")
                continue
            fc.compute_all_connectivity(site_swap[choice])