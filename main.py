import tomllib
import sys
import importlib
from utils import download_dataset, extract_labels
from feature.rfe import rfe
from model.classic_models import train_test_classics
from pipeline.train_test_nn import train_test_nn
from pipeline.train_test_nn_joint import ablation_study, test_on_new_sites, train_test_nn_joint

def run_compute_fc(config: dict):
    feature_cfg = config.get("feature", {})
    default_variant = str(feature_cfg.get("fc_variant", "fc90")).lower()
    print("\n" + "-" * 40)
    print("Select FC variant to compute:")
    print("1) FC90 (AAL90, 90 ROIs, unsegmented)")
    print("2) 167 AAL (segmented)")
    print("Enter to use default:", default_variant)
    print("-" * 40)
    choice = input("Choice [1/2/default]: ").strip().lower()
    if choice in ("1", "fc90", "aal90", "90"):
        module_name = "feature.fc_90"
    elif choice in ("2", "full", "aal", "all", "full_aal"):
        module_name = "feature.fc"
    else:
        module_name = "feature.fc_90" if default_variant in ("fc90", "aal90", "90") else "feature.fc"
    compute_fc = importlib.import_module(module_name).compute_fc
    return compute_fc(config)

def build_menu(config: dict):
    return [
        ("Browse the dataset", lambda: download_dataset(config, interactive=True)),
        ("Download the dataset", lambda: download_dataset(config)),
        ("Extract & visualize labels", lambda: extract_labels(config)),
        ("Compute FC", lambda: run_compute_fc(config)),
        ("Reduce Feature Elimination (Not reported in the paper)", lambda: rfe(config)),
        ("Train and test classic models (Not reported in the paper)", lambda: train_test_classics(config)),
        ("Train and test neural network models (Not reported in the paper)", lambda: train_test_nn(config)),
        ("Train and test joint model", lambda: train_test_nn_joint(config)),
        ("Test on new sites", lambda: test_on_new_sites(config)),
        ("Run ablation study", lambda: ablation_study(config)),
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
