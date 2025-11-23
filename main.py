import tomllib
import sys
from utils import download_dataset, extract_labels

def build_menu(config: dict):
    return [
        ("Browse the dataset", lambda: download_dataset(config, interactive=True)),
        ("Download the dataset", lambda: download_dataset(config)),
        ("Extract & visualize labels", lambda: extract_labels(config)),
    ]

def display_menu(menu: list):
    print("-"*10 + "MENU" + "-"*10)

    for i, (label, _) in enumerate(menu):
        print(f"{i}) {label}")
    
    print("q) Quit")

    choice = input("Enter your choice: ")
    if choice == "q":
        sys.exit("Exiting...")

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
