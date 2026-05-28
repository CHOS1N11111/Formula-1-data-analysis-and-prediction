import kagglehub
import shutil
from pathlib import Path


DATASETS = [
    "davidcochran/formula-1-race-data-sqlite",
    "prathamsharma123/formula-1-fantasy-2021",
    "tusharsingh1411/formula1-data-1950-2022",
]


def directory_has_files(path):
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def download_dataset(dataset):
    cache_path = Path(kagglehub.dataset_download(dataset))

    if not directory_has_files(cache_path):
        print(f"Local Kaggle cache is empty for {dataset}. Downloading again...")
        cache_path = Path(kagglehub.dataset_download(dataset, force_download=True))

    if not directory_has_files(cache_path):
        raise RuntimeError(f"Dataset download finished, but no files were found: {cache_path}")

    target_path = Path(__file__).resolve().parent / dataset.split("/")[-1]

    shutil.copytree(cache_path, target_path, dirs_exist_ok=True)

    print("Path to dataset files:", target_path)


def main():
    for dataset in DATASETS:
        download_dataset(dataset)


if __name__ == "__main__":
    main()
