# This file use aws api
import boto3
import tomllib
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import sys
from tqdm import tqdm
import shutil

from csv_validator import validate_subject_ids


class S3Browser:
    def __init__(self, uri, region="us-east-1"):
        bucket_name, prefix = self.parse_s3_uri(uri)

        self.bucket_name = bucket_name
        self.client = boto3.client(
            "s3", region_name=region, config=Config(signature_version=UNSIGNED)
        )

        if prefix and not prefix.endswith("/"):
            raise ValueError("Prefix must end with '/'")
        self.prefix = prefix
        self.prefix_len = len(prefix)

    def download(self, file, destination="data", overwrite_path=False):
        file_path = Path(file)
        if overwrite_path:
            file_path = file_path.name  # only keep the file name
        local_path = Path(destination) / file_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(self.bucket_name, self.prefix + file, str(local_path))

    def list_s3(self, folder):
        if folder != "" and not folder.endswith("/"):
            raise ValueError("Folder must end with '/' or be empty (root)")

        files = []
        folders = []
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.prefix + folder, Delimiter="/"
            )

            # Extract folder names from CommonPrefixes (which is folder in s3 terms)
            for prefix_info in response.get("CommonPrefixes", []):
                folder_name = prefix_info["Prefix"]
                folders.append(folder_name[self.prefix_len :])

            # These are files (excluding folders)
            for content in response.get("Contents", []):
                file_name = content["Key"]
                files.append(file_name[self.prefix_len :])

            return files, folders

        except Exception as e:
            raise ValueError(f"Error accessing S3 bucket: {e}")

    @staticmethod
    def parse_s3_uri(s3_uri):
        """
        Parse S3 URI to extract bucket name and prefix
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError("S3 URI must start with 's3://'")

        # Remove s3:// prefix
        path = s3_uri[5:]

        # Split into bucket and prefix
        parts = path.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Ensure prefix ends with / if it's not empty
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        return bucket_name, prefix


def pretty_print_list(items, indent=2):
    print("[")
    for x in items:
        print("-" * indent + f" {x},")
    print("]")


def download_descriptions(description_uri, destination="data"):
    s3 = S3Browser(description_uri)
    files, _ = s3.list_s3("")
    print(f"Downloading {len(files)} description files...")
    pretty_print_list(files, indent=2)
    action = input("Proceed to download? [y]es/[s]kip/[n]o: ")
    if action == "n":
        sys.exit("Download cancelled by user.")

    elif action == "y":
        for x in files:
            s3.download(x, destination=destination)


MNI_TAG = "_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-"


def download_preprocessed_data(preprocessed_uri, destination="data"):
    downloaded_ids = []
    if Path("data/downloaded.txt").exists():
        with open("data/downloaded.txt", "r") as f:
            downloaded_ids = f.read().splitlines()

    s3 = S3Browser(preprocessed_uri)
    _, subject_dirs = s3.list_s3("")
    progress = tqdm(total=len(subject_dirs), desc="Subjects found")
    with open("data/downloaded.txt", "a") as f:
        for dir in subject_dirs:
            if not dir.startswith("sub"):
                continue

            id = dir.strip("/")
            if id in downloaded_ids:
                print(f"{id} already downloaded, skipping.")
                progress.update(1)
                continue
            mask_file = f"{id}/ses-1/func/{id}{MNI_TAG}brain_mask.nii.gz"
            image_file = f"{id}/ses-1/func/{id}{MNI_TAG}preproc_bold.nii.gz"
            dest = f"data/{id}"

            if Path(dest).exists():
                shutil.rmtree(dest)

            try:
                progress.set_description(f"Downloading {id} to data/{id}")
                s3.download(mask_file, destination=f"data/{id}", overwrite_path=True)
                s3.download(image_file, destination=f"data/{id}", overwrite_path=True)
                f.write(f"{id}\n")
                f.flush()
                progress.update(1)
            except Exception as e:
                print(f"Error downloading data for subject {id}: {e}")
                progress.update(1)
                continue

    #        -MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    # sub-0010001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

    validate_subject_ids("data", [dir.strip("/") for dir in subject_dirs])


def main():
    with open("config.toml", "rb") as toml_file:
        toml_config = tomllib.load(toml_file)
        dataset_config = toml_config["dataset"]

    preprocessed_uri = dataset_config.get("preprocessed_uri", None)
    if preprocessed_uri is None:
        raise ValueError("S3 URI not found in config.toml under [dataset] section.")

    description_uri = dataset_config.get("description_uri", None)
    if description_uri is None:
        raise ValueError("S3 URI not found in config.toml under [dataset] section.")

    download_descriptions(description_uri, destination="data")
    download_preprocessed_data(preprocessed_uri, destination="data")


if __name__ == "__main__":
    main()
