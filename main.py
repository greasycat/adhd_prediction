import tomllib
import sys
from utils import download_dataset, extract_labels
from feature.fc import compute_fc
from feature.rfe import rfe
from model.classic_models import train_test_classics
from pipeline.train_test_nn import train_test_nn
from pipeline.few_shot import train_few_shot_model

def build_menu(config: dict):
    return [
        ("Browse the dataset", lambda: download_dataset(config, interactive=True)),
        ("Download the dataset", lambda: download_dataset(config)),
        ("Extract & visualize labels", lambda: extract_labels(config)),
        ("Compute FC", lambda: compute_fc(config)),
        ("Reduce Feature Elimination", lambda: rfe(config)),
        ("Train and test classic models", lambda: train_test_classics(config)),
        ("Train and test neural network models", lambda: train_test_nn(config)),
        ("Train few-shot learning model", lambda: train_few_shot_model(config)),
    ]

def display_menu(menu: list):
    print("-"*10 + "MENU" + "-"*10)

    for i, (label, _) in enumerate(menu):
        print(f"{i}) {label}")
    
    print("q) Quit")

    choice = input("Enter your choice: ")
    if choice == "q":
        sys.exit(0)

    if choice not in [str(i) for i in range(len(menu))]:
        return None

    return int(choice)

def main():
    with open("config.toml", "rb") as toml_file:
        config = tomllib.load(toml_file)
        menu = build_menu(config)

        while True:
            choice = display_menu(menu)
            if choice is None:
                continue
            func = menu[choice][1]
            func()


if __name__ == "__main__":
    main()
