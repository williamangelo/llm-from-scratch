import os
import argparse
from datasets import load_dataset


DATASET_CONFIGS = {
    "gutenberg": {
        "name": "sedthh/gutenberg_english",
        "split": "train",
        "description": "Project Gutenberg English books dataset"
    }
}


def download_dataset(dataset_key: str, output_dir: str = None):
    """
    Download a dataset from HuggingFace and save to disk.
    
    Args:
       dataset_key: Key identifying the dataset (e.g., 'gutenberg')
        output_dir: Directory where the dataset will be saved. If None, uses data/raw/{dataset_key}
    """
    if dataset_key not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available datasets: {available}")
    
    config = DATASET_CONFIGS[dataset_key]
    
    if output_dir is None:
        output_dir = f"data/raw/{dataset_key}"
    
    print(f"Downloading {config['description']}: {config['name']}")
    print(f"This may take a while depending on your connection speed...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset(config["name"], split=config["split"])
    
    print(f"\nDownloaded {len(dataset)} items")
    print(f"Dataset fields: {dataset.column_names}")
    
    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print(f"\n✓ Dataset successfully saved to {os.path.abspath(output_dir)}")
    
    if dataset_key == "gutenberg":
        print(f"\nYou can now process it with:")
        print(f"  python scripts/clean_gutenberg_dataset.py --data_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--gutenberg",
        action="store_true",
        help="Download the Gutenberg dataset"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to download (e.g., 'gutenberg')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/raw/{dataset_name})"
    )
    
    args = parser.parse_args()
    
    if args.gutenberg:
        dataset_key = "gutenberg"
    elif args.dataset:
        dataset_key = args.dataset
    else:
        parser.error("Please specify a dataset using --gutenberg or --dataset")
    
    download_dataset(dataset_key, args.output)


if __name__ == "__main__":
    main()

