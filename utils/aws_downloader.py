# This file use aws api
import boto3
import tomllib
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import sys
from tqdm import tqdm
import shutil
import argparse

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


def download_preprocessed_data(preprocessed_uri, image_destination="data/raw", info_destination="data"):

    if not Path(image_destination).exists():
        Path(image_destination).mkdir(parents=True, exist_ok=True)

    downloaded_ids = []
    if Path(info_destination + "/downloaded.txt").exists():
        with open(info_destination + "/downloaded.txt", "r") as f1:
            downloaded_ids = f1.read().splitlines()

    s3 = S3Browser(preprocessed_uri)
    _, subject_dirs = s3.list_s3("")
    progress = tqdm(total=len(subject_dirs), desc="Subjects found")
    with open(info_destination + "/downloaded.txt", "a") as f1, open(info_destination + "/failed.txt", "a") as f2:
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
            dest = f"{image_destination}/{id}"

            if Path(dest).exists():
                shutil.rmtree(dest)

            try:
                progress.set_description(f"Downloading {id} to {image_destination}/{id}")
                s3.download(mask_file, destination=dest, overwrite_path=True)
                s3.download(image_file, destination=dest, overwrite_path=True)
                f1.write(f"{id}\n")
                f1.flush()
                progress.update(1)
            except Exception as e:
                shutil.rmtree(dest)
                print(f"Error downloading data for subject {id}: {e}")
                f2.write(f"{id}\n")
                f2.flush()
                progress.update(1)
                continue


def ls_folder(s3, folder):
    files, folders = s3.list_s3(folder)
    for f in files + folders:
        print(" - " + f)


def check_failed_downloads(preprocessed_uri):
    failed_ids = []
    with open("data/failed.txt", "r") as f:
        failed_ids = f.read().splitlines()

    s3 = S3Browser(preprocessed_uri)
    for id in failed_ids:
        print(f"\n\nChecking {id}:")
        ls_folder(s3, f"{id}/")
        ls_folder(s3, f"{id}/ses-1/")
        ls_folder(s3, f"{id}/ses-1/func/")
        s = input("q to quit, anything else to continue: ")
        if s.lower() == "q":
            break

def interactive_browser(s3: S3Browser):
    path_stack = [""]
    while True:
        files, folders = s3.list_s3(path_stack[-1])
        for f in files:
            print(f" - {f}")
        for i, f in enumerate(folders):
            print(f" {i+1}. {f}")

        s = input("Enter a number to select a file or folder, 'b' to go back, or 'q' to quit: ")
        if s.lower() == "q":
            break
        elif s.lower() == "b" and len(path_stack) > 1:
            print(f"Going back to {path_stack[-2]}")
            path_stack.pop()
            continue
        elif s.isdigit():
            s = int(s)
            if s > 0 and s <= len(folders):
                path_stack.append(folders[s-1])
            else:
                print("Invalid input")
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true")
    args = parser.parse_args()


    with open("config.toml", "rb") as toml_file:
        toml_config = tomllib.load(toml_file)
        dataset_config = toml_config["dataset"]

    preprocessed_uri = dataset_config.get("preprocessed_uri", None)
    if preprocessed_uri is None:
        raise ValueError("S3 URI not found in config.toml under [dataset] section.")

    description_uri = dataset_config.get("description_uri", None)
    if description_uri is None:
        raise ValueError("S3 URI not found in config.toml under [dataset] section.")

    if args.interactive:
        interactive_browser(S3Browser(preprocessed_uri))
        return

    download_descriptions(description_uri, destination="data")
    download_preprocessed_data(preprocessed_uri, image_destination="data/raw", info_destination="data")


if __name__ == "__main__":
    main()
