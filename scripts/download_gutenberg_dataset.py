import os
from datasets import load_dataset


def download_gutenberg_dataset(output_dir="data/gutenberg/raw", dataset_name="sedthh/gutenberg_english"):
    """
    Download the Project Gutenberg dataset from HuggingFace and save to disk.
    
    Args:
        output_dir: Directory where the dataset will be saved
        dataset_name: HuggingFace dataset identifier
    """
    print(f"Downloading Project Gutenberg dataset: {dataset_name}")
    print(f"This may take a while depending on your connection speed...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    # Note: The sedthh/gutenberg_english dataset is a popular English subset of Project Gutenberg
    dataset = load_dataset(dataset_name, split="train")
    
    print(f"\nDownloaded {len(dataset)} books")
    print(f"Dataset fields: {dataset.column_names}")
    
    # Save to disk in Arrow format
    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print(f"\n✓ Dataset successfully saved to {os.path.abspath(output_dir)}")
    print(f"\nYou can now process it with:")
    print(f"  python prepare_gutenberg_dataset.py --data_dir {output_dir}")


if __name__ == "__main__":
    download_gutenberg_dataset()

