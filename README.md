# Environment setup
The project uses `uv` as python package management, follow the [official guide](https://docs.astral.sh/uv/getting-started/installation/) to install it.

```bash
# Then, download all the required packages
uv sync
```
# Run the interactive command line
Run the following to start interactive command line with default configuration

You can select a list of operation to perform

```bash
uv run main.py
```

The following table explain functionality of each option
| Option | Description | Notes |
|--------|-------------|-------|
| 1 | Download the dataset | |
| 2 | Extract & visualize labels | |
| 3 | Compute FC | |
| 4 | Reduce Feature Elimination | Not reported in the paper |
| 5 | Train and test classic models | Not reported in the paper |
| 6 | Train and test neural network models | Not reported in the paper |
| 7 | Train and test joint model | |
| 8 | Test on new sites | |
| 9 | Run ablation study | |
| q | Quit | |

A recommended sequence of option to reproduce the results is as follows:
- (1): Download all labels and data
- (2): Extract cleaned label from the raw data
- (3): Choose 167 AAL and Compute FC for NYU dataset with 7 segmentation
- (7): Train and test the AECLS model (Reconstruction loss + Classification Loss)
- (8): Test AECLS model on new site (All other sites except NYU)
- (9): Train and test CLS model (No reconstruction loss)