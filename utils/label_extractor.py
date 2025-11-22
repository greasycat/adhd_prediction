import argparse
import pandas as pd
from pathlib import Path

def merge_phenotypic_data(output_dir="data", processed_dir="data/processed"):
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
    final_df.to_csv(processed_dir + "/all_phenotypic.csv", index=False)
    return final_df

def plot_label_distribution(df, downloaded_id, output_dir="data", site="NYU"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.countplot(x="DX", data=df[df["ScanDir ID"].isin(downloaded_id)])
    plt.savefig(output_dir + f"/{site}_label_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    downloaded_id = pd.read_csv(args.data_dir + "/downloaded.txt", header=None)[0].tolist()
    final_df = merge_phenotypic_data(args.data_dir, args.output_dir)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    NYU_df = final_df[final_df["Site"] == 5]
    NYU_df[NYU_df["ScanDir ID"].isin(downloaded_id)][["ScanDir ID", "DX"]].rename(columns={"ScanDir ID": "id", "DX": "label"}).to_csv(args.output_dir + "/NYU_labels.csv", index=False)
    if args.plot:
        plot_label_distribution(NYU_df, downloaded_id)



