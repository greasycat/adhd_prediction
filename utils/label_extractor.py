import tomllib
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Some sites are ignored because they are not available in the fMRIprep dataset due to artifacts
SITES = {
    "PEKING": 1,
    # "BROWN": 2, 
    # "KKI": 3,
    "NEURO": 4,
    "NYU": 5,
    "OHSU": 6,
    # "PITT": 7,
}

def merge_phenotypic_data(downloaded_id, output_dir="data", processed_dir="data/processed"):
    phenotypic_files = [x for x in Path(output_dir).glob("*.csv")]

    final_df = pd.DataFrame()
    for x in phenotypic_files:
        print(x)
        df = pd.read_csv(x)
        if str(x) == output_dir + "/Peking_1_TestRelease_phenotypic.csv":

            # rename ID to "ScanDir ID"
            df = df.rename(columns={"ID": "ScanDir ID"})
        final_df = pd.concat([final_df, df])

    final_df.iloc[:, 0] = final_df.iloc[:, 0].apply(lambda x: "sub-" + str(x).zfill(7))
    final_df = final_df[final_df["ScanDir ID"].isin(downloaded_id)]
    final_df.to_csv(processed_dir + "/all_phenotypic.csv", index=False)
    return final_df

def plot_label_distribution(df,plot_dir="data/plots", site="NYU"):

    if not Path(plot_dir).exists():
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    sns.countplot(x="DX", data=df)
    plt.title(f"{site} label distribution")
    plt.savefig(plot_dir + f"/{site}_label_distribution.png")
    plt.close()

def extract_label(df, output_dir="data", site="NYU", site_id=5, plot_dir="data/plots"):
    df = df[df["Site"] == site_id]
    df[["ScanDir ID", "DX"]].rename(columns={"ScanDir ID": "id", "DX": "label"}).to_csv(output_dir + f"/{site}_labels.csv", index=False) # type: ignore
    plot_label_distribution(df, plot_dir, site)

def extract_labels(config: dict):
    dataset_config = config["dataset"]

    data_dir = dataset_config.get("data_dir", "data")
    phenotypes_dir = dataset_config.get("phenotypes_dir", "data/phenotypes")
    output_dir = dataset_config.get("preprocessed_dir", "data/processed")
    plot_dir = dataset_config.get("plot_dir", "data/plots")

    downloaded_id = pd.read_csv(data_dir + "/downloaded.txt", header=None)[0].tolist()
    final_df = merge_phenotypic_data(downloaded_id, phenotypes_dir, output_dir)
    if not Path(plot_dir).exists():
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for site, site_id in SITES.items():
        extract_label(final_df,output_dir, site, site_id=site_id, plot_dir=plot_dir)

if __name__ == "__main__":
    with open("config.toml", "rb") as toml_file:
        config = tomllib.load(toml_file)
        extract_labels(config)
