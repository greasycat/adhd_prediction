# Environment setup
The project uses `uv` as python package management, follow the [official guide](https://docs.astral.sh/uv/getting-started/installation/) to install it.

```bash
# Then, download all the required packages
uv sync
```

# Prepare the dataset

The dataset is downloaded from ADHD200 public AWS S3 bucket

The script `utils/aws_downloader` downloads all the available preprocessed images and phenotypic table

```bash
uv run utils/aws_downloader.py
```

> The script also provideds a `-i` flag to browse the S3 bucket interactively

The script `utils/label_extractor.py` cleans the phenotypic table and generate lables for later training (default location: `data/processed`)

```bash
uv run utils/label_extractor.py
```

Currently, it only generate the NYU label

# Extract functional connectivity correlation matrix

The notebook `feature_extract.ipynb` notebook contain steps to extract FC
