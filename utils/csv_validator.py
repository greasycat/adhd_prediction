# This file checks if the preprocessed subject ids matches the ones in the datset phenotype data.
import pandas as pd
from pathlib import Path

SUB_ID_LEN = 7


def validate_subject_ids(csv_dir, ids_to_check, prefix="sub-"):
    csv_dir = list(Path("data").glob("*.csv"))
    print(f"Found {len(csv_dir)} CSV files.")

    for csv in csv_dir:
        df = pd.read_csv(csv)
        id_col = df.iloc[:, 0].apply(lambda x: prefix + str(x).zfill(SUB_ID_LEN))

        missing_id_num = 0
        for id in id_col:
            if id not in ids_to_check:
                missing_id_num += 1

        print("Missing IDs in {}: {}/{}".format(csv.name, missing_id_num, len(id_col)))


def main():
    validate_subject_ids("data", ["sub-0010002", "sub-0000002"])


if __name__ == "__main__":
    main()
