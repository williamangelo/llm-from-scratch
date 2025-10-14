
import argparse
import os
import re
from tqdm import tqdm
from datasets import load_from_disk


# Compile regex patterns once at module level for performance
START_PATTERNS = [
    re.compile(r"\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\*START\*.*?PROJECT GUTENBERG", re.IGNORECASE | re.DOTALL),
    re.compile(r"START OF THE PROJECT GUTENBERG EBOOK", re.IGNORECASE | re.DOTALL),
]

END_PATTERNS = [
    re.compile(r"\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\*END\*.*?PROJECT GUTENBERG", re.IGNORECASE | re.DOTALL),
    re.compile(r"END OF THE PROJECT GUTENBERG EBOOK", re.IGNORECASE | re.DOTALL),
    re.compile(r"End of (the )?Project Gutenberg", re.IGNORECASE | re.DOTALL),
]

WHITESPACE_PATTERN = re.compile(r'\n\s*\n+')


def strip_headers(text):
    """Remove Project Gutenberg header and footer boilerplate text."""
    # Find the start of the actual content
    start_pos = 0
    for pattern in START_PATTERNS:
        match = pattern.search(text)
        if match:
            start_pos = match.end()
            break

    # Find the end of the actual content
    end_pos = len(text)
    for pattern in END_PATTERNS:
        match = pattern.search(text, start_pos)
        if match:
            end_pos = match.start()
            break

    # Extract the content between markers
    cleaned_text = text[start_pos:end_pos].strip()

    return cleaned_text


def is_english(text, threshold=0.9, sample_size=10000):
    """
    Check if text is primarily English based on ASCII character ratio.
    Uses sampling for large texts to improve performance.
    """
    if len(text) == 0:
        return False

    # Sample text for large documents to improve performance
    if len(text) > sample_size:
        # Sample from beginning, middle, and end
        sample_size_third = sample_size // 3
        samples = [
            text[:sample_size_third],
            text[len(text)//2 - sample_size_third//2:len(text)//2 + sample_size_third//2],
            text[-sample_size_third:]
        ]
        text = ''.join(samples)

    # Use bytes to avoid character iteration
    text_bytes = text.encode('utf-8', errors='ignore')
    ascii_chars = sum(1 for b in text_bytes if b < 128)

    return ascii_chars / len(text_bytes) > threshold


def write_batch_streaming(file_handle, content, separator, is_first_batch):
    """
    Write content to an already open file handle.
    Returns the number of bytes written.
    """
    bytes_written = 0

    for i, text in enumerate(content):
        # Add separator before content (except for very first content)
        if not (is_first_batch and i == 0):
            sep_bytes = file_handle.write(separator)
            bytes_written += sep_bytes

        text_bytes = file_handle.write(text)
        bytes_written += text_bytes

    return bytes_written


def process_gutenberg_dataset(input_dir, output_dir, max_size_mb=500, separator="<|endoftext|>",
                             buffer_size=8192*1024):
    """
    Process Project Gutenberg dataset from HuggingFace datasets format.

    Args:
        input_dir: Directory containing the Arrow format dataset
        output_dir: Directory where cleaned files will be saved
        max_size_mb: Maximum size in MB for each output file
        separator: Token to use between books
        buffer_size: Buffer size for file writing (default 8MB)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    print(f"Loading dataset from {input_dir}...")
    dataset = load_from_disk(input_dir)
    print(f"Loaded {len(dataset)} books.")

    # Pre-calculate separator size once
    separator_bytes = separator.encode('utf-8')
    separator_size = len(separator_bytes)
    max_size_bytes = max_size_mb * 1024 * 1024

    current_content = []
    current_size = 0
    file_counter = 1
    books_processed = 0
    books_skipped = 0
    books_oversized = 0
    total_size_written = 0

    # Open first file with buffering
    output_file = os.path.join(output_dir, f"combined_{file_counter}.txt")
    current_file = open(output_file, "w", encoding="utf-8", buffering=buffer_size)
    is_first_in_file = True

    for example in tqdm(dataset, desc="Processing books"):
        book_id = example['id']
        text = example['text']

        # Check if primarily English (with sampling for performance)
        if not is_english(text):
            tqdm.write(f"Skipping book {book_id} - not primarily English text")
            books_skipped += 1
            continue

        # Strip Gutenberg boilerplate
        cleaned_text = strip_headers(text)

        # Skip if cleaning removed everything
        if len(cleaned_text) < 100:
            tqdm.write(f"Skipping book {book_id} - too short after cleaning")
            books_skipped += 1
            continue

        # Normalize whitespace using pre-compiled pattern
        cleaned_text = WHITESPACE_PATTERN.sub('\n\n', cleaned_text)

        # Calculate size once and reuse
        text_bytes = cleaned_text.encode('utf-8')
        text_size = len(text_bytes)

        # Warn if single book exceeds max size
        if text_size > max_size_bytes:
            tqdm.write(f"Warning: Book {book_id} ({text_size / 1024 / 1024:.2f} MB) exceeds max_size_mb ({max_size_mb} MB)")
            books_oversized += 1

        # Calculate size including separator (if not first book in batch)
        size_with_separator = text_size
        if current_content:
            size_with_separator += separator_size

        # Check if we need to start a new file
        if current_content and (current_size + size_with_separator > max_size_bytes):
            # Write accumulated content
            bytes_written = write_batch_streaming(current_file, current_content, separator, is_first_in_file)
            total_size_written += bytes_written

            # Close current file
            current_file.close()
            actual_size = os.path.getsize(output_file)
            tqdm.write(f"Wrote {output_file} ({actual_size / 1024 / 1024:.2f} MB)")

            # Open new file
            file_counter += 1
            output_file = os.path.join(output_dir, f"combined_{file_counter}.txt")
            current_file = open(output_file, "w", encoding="utf-8", buffering=buffer_size)
            is_first_in_file = True

            current_content = [cleaned_text]
            current_size = text_size
        else:
            current_content.append(cleaned_text)
            current_size += size_with_separator

        # Flush batch periodically to avoid excessive memory usage
        if len(current_content) >= 100:  # Write every 100 books
            bytes_written = write_batch_streaming(current_file, current_content, separator, is_first_in_file)
            is_first_in_file = False
            current_content = []
            # Don't reset current_size, keep accumulating

        books_processed += 1

    # Write any remaining content
    if current_content:
        bytes_written = write_batch_streaming(current_file, current_content, separator, is_first_in_file)
        total_size_written += bytes_written

    # Close final file
    current_file.close()
    actual_size = os.path.getsize(output_file)
    print(f"Wrote {output_file} ({actual_size / 1024 / 1024:.2f} MB)")

    print(f"\nProcessing complete:")
    print(f"  Books processed: {books_processed}")
    print(f"  Books skipped: {books_skipped}")
    print(f"  Books exceeding max size: {books_oversized}")
    print(f"  Output files: {file_counter}")
    print(f"  Total size written: {total_size_written / 1024 / 1024:.2f} MB")
    print(f"  Average file size: {total_size_written / file_counter / 1024 / 1024:.2f} MB")
    print(f"  Output directory: {os.path.abspath(output_dir)}")

    return file_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Project Gutenberg dataset: strip boilerplate and combine books"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gutenberg/raw",
        help="Directory containing the Arrow format dataset"
    )
    parser.add_argument(
        "--max_size_mb",
        type=int,
        default=1,
        help="Maximum file size for each combined output file in megabytes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/gutenberg/clean",
        help="Directory where the cleaned data will be saved"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=8192*1024,
        help="Buffer size for file writing in bytes (default: 8MB)"
    )

    args = parser.parse_args()

    process_gutenberg_dataset(
        input_dir=args.data_dir,
        output_dir=args.output_dir,
        max_size_mb=args.max_size_mb,
        buffer_size=args.buffer_size
    )
