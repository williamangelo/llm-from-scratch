import argparse
import json
import logging
import os
import re
import sys
from tqdm import tqdm
from datasets import load_from_disk


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

# Pre-compile sets for faster membership checks
SENTENCE_END_CHARS = frozenset('.!?')
CONTINUATION_PUNCT = frozenset(',;:')
QUOTE_CHARS = frozenset('"\'')

WHITESPACE_PATTERN = re.compile(r'\n\s*\n+')
MULTIPLE_SPACES_PATTERN = re.compile(r' {2,}')
MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n{3,}')


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid interfering with progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def normalize_whitespace(text):
    """
    Normalize whitespace in text for LLM training.
    
    Gutenberg texts often have line breaks in the middle of sentences for formatting.
    This function:
    - Merges lines that are clearly part of the same sentence (conservative approach)
    - Removes excessive blank lines
    - Preserves paragraph breaks (double newlines after sentence endings)
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    
    # Process lines to merge continuations and remove single blank lines
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()  # Cache strip() result
        
        # Skip empty lines for now (we'll handle them separately)
        if not line_stripped:
            # Count consecutive empty lines
            empty_count = 0
            while i < len(lines) and not lines[i].strip():
                empty_count += 1
                i += 1
            
            # If we have 2+ empty lines, preserve as paragraph break
            if empty_count >= 2 and result_lines:
                last_result_stripped = result_lines[-1].strip()  # Cache strip() result
                if last_result_stripped:
                    # Check if previous line ends with sentence punctuation
                    prev_line = result_lines[-1].rstrip()
                    if prev_line and prev_line[-1] in SENTENCE_END_CHARS:
                        result_lines.append('')
                    # Otherwise, just add one blank line max
                    elif empty_count >= 2:
                        result_lines.append('')
            # Single empty line: check if it's a real paragraph break
            elif empty_count == 1 and result_lines:
                prev_line = result_lines[-1].rstrip()
                # Only keep if previous line ends with sentence punctuation
                if prev_line and prev_line[-1] in SENTENCE_END_CHARS:
                    result_lines.append('')
            # Otherwise skip single empty lines (formatting artifacts)
            continue
        
        # Non-empty line: check if it should be merged with previous
        if result_lines:
            last_result_stripped = result_lines[-1].strip()  # Cache strip() result
            if last_result_stripped:
                prev_line = result_lines[-1].rstrip()
            
            # Conservative merging: only merge if clearly a continuation
            # 1. Previous line doesn't end with sentence punctuation (.!?)
            # 2. Previous line ends with lowercase letter, comma, semicolon, or colon
            # 3. Current line starts with lowercase (not a new sentence)
            should_merge = False
            if prev_line:
                last_char = prev_line[-1]
                # Check if previous line ends with continuation punctuation or lowercase
                if last_char in CONTINUATION_PUNCT or (last_char.isalpha() and last_char.islower()):
                    # Use cached stripped line (already computed above)
                    if line_stripped:
                        first_char = line_stripped[0]
                        # Only merge if current line starts with lowercase (continuation)
                        if first_char.islower():
                            should_merge = True
                        # Also merge if current line starts with a quote or parenthesis (likely continuation)
                        elif first_char in QUOTE_CHARS:
                            should_merge = True
            
            if should_merge:
                # Merge with space
                result_lines[-1] = prev_line + ' ' + line
            else:
                # New sentence/paragraph
                result_lines.append(line)
        else:
            # First line or after empty line
            result_lines.append(line)
        
        i += 1
    
    text = '\n'.join(result_lines)
    
    # Final cleanup: normalize multiple spaces (using pre-compiled pattern)
    text = MULTIPLE_SPACES_PATTERN.sub(' ', text)
    
    # Ensure paragraph breaks are double newlines (using pre-compiled pattern)
    text = MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
    
    return text.strip()


def strip_headers(text):
    """Remove Project Gutenberg header and footer boilerplate text."""
    start_pos = 0
    for pattern in START_PATTERNS:
        match = pattern.search(text)
        if match:
            start_pos = match.end()
            break

    end_pos = len(text)
    for pattern in END_PATTERNS:
        match = pattern.search(text, start_pos)
        if match:
            end_pos = match.start()
            break

    cleaned_text = text[start_pos:end_pos].strip()
    return cleaned_text


def is_english(text, threshold=0.9, sample_size=10000):
    """
    Check if text is primarily English based on ASCII character ratio.
    Uses sampling for large texts to improve performance.
    """
    if len(text) == 0:
        return False

    if len(text) > sample_size:
        sample_size_third = sample_size // 3
        samples = [
            text[:sample_size_third],
            text[len(text)//2 - sample_size_third//2:len(text)//2 + sample_size_third//2],
            text[-sample_size_third:]
        ]
        text = ''.join(samples)

    # Encode once and count ASCII bytes efficiently
    text_bytes = text.encode('utf-8', errors='ignore')
    if len(text_bytes) == 0:
        return False
    
    # Use bytes.count() with a generator for better performance on large texts
    ascii_chars = sum(b < 128 for b in text_bytes)
    ratio = ascii_chars / len(text_bytes)

    return ratio > threshold


def write_batch_streaming(file_handle, content, separator_str, separator_size, is_first_batch):
    """
    Write content to an already open file handle (text mode).
    Returns the number of bytes written (calculated for tracking).
    
    Args:
        file_handle: Open file handle to write to (text mode)
        content: List of text strings to write
        separator_str: Separator string (with newline) to write
        separator_size: Pre-calculated byte size of separator for tracking
        is_first_batch: Whether this is the first batch in the file
    """
    bytes_written = 0

    for i, text in enumerate(content):
        if not (is_first_batch and i == 0):
            # Write string to text file handle
            file_handle.write(separator_str)
            bytes_written += separator_size

        # Write string to text file handle (encoding handled by file)
        file_handle.write(text)
        # Calculate bytes for tracking purposes
        bytes_written += len(text.encode('utf-8'))

    return bytes_written


def process_gutenberg_dataset(input_dir, output_dir, max_size_mb, separator, buffer_size, min_doc_length, max_doc_length, debug=False):
    """
    Process Project Gutenberg dataset from HuggingFace datasets format.

    Args:
        input_dir: Directory containing the Arrow format dataset
        output_dir: Directory where cleaned files will be saved
        max_size_mb: Maximum size in MB for each output file
        separator: Token to use between books
        buffer_size: Buffer size for file writing
        min_doc_length: Minimum document length in characters after cleaning
        max_doc_length: Maximum document length in characters after cleaning (None for no limit)
        debug: Enable debug logging
    """
    # Configure logging to use tqdm.write() to avoid interfering with progress bar
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create handler that uses tqdm.write()
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading dataset from {input_dir}...")
    
    dataset = load_from_disk(input_dir)
    logger.info(f"Loaded {len(dataset)} books.")
    logger.debug(f"Dataset columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")

    # Pre-calculate separator string and size for performance
    separator_with_newline = separator + '\n'
    separator_size = len(separator_with_newline.encode('utf-8'))
    max_size_bytes = max_size_mb * 1024 * 1024
    
    logger.debug(f"Separator size: {separator_size} bytes, Max file size: {max_size_bytes / 1024 / 1024:.2f} MB")

    current_content = []
    current_size = 0
    file_counter = 1
    books_processed = 0
    books_skipped = 0
    books_oversized = 0
    books_too_short = 0
    books_too_long = 0
    books_not_english = 0
    total_size_written = 0
    current_file_size = 0

    output_file = os.path.join(output_dir, f"combined_{file_counter}.txt")
    current_file = open(output_file, "w", encoding="utf-8", buffering=buffer_size)
    is_first_in_file = True

    for idx, example in enumerate(tqdm(dataset, desc="Processing books")):
        text = example.get('TEXT') or example.get('text', '')
        
        book_id = f"book_{idx}"
        if 'METADATA' in example and example['METADATA']:
            try:
                metadata = json.loads(example['METADATA']) if isinstance(example['METADATA'], str) else example['METADATA']
                if isinstance(metadata, dict):
                    book_id = str(metadata.get('text_id', metadata.get('title', book_id)))
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        if not is_english(text):
            logger.debug(f"Skipping book {book_id} - not primarily English text")
            books_not_english += 1
            books_skipped += 1
            continue

        cleaned_text = strip_headers(text)
        cleaned_text = normalize_whitespace(cleaned_text)
        
        # Check document length after normalization
        doc_length = len(cleaned_text)
        
        if doc_length < min_doc_length:
            logger.debug(f"Skipping book {book_id} - too short after cleaning ({doc_length} chars, min: {min_doc_length})")
            books_too_short += 1
            books_skipped += 1
            continue
        
        if max_doc_length is not None and doc_length > max_doc_length:
            logger.debug(f"Skipping book {book_id} - too long after cleaning ({doc_length} chars, max: {max_doc_length})")
            books_too_long += 1
            books_skipped += 1
            continue

        # Encode once and reuse
        text_bytes = cleaned_text.encode('utf-8')
        text_size = len(text_bytes)

        # Note: If a single document exceeds max_size_bytes, it will still be written
        # to ensure document integrity (documents are never split). The file may exceed
        # the limit, but this is acceptable for training.
        if text_size > max_size_bytes:
            logger.debug(f"Book {book_id} ({text_size / 1024 / 1024:.2f} MB) exceeds max_size_mb ({max_size_mb} MB)")
            books_oversized += 1

        # Calculate size with separator if needed
        # IMPORTANT: We check BEFORE adding to ensure documents are never split between files.
        # This is critical for training - split documents would create invalid training samples.
        size_with_separator = text_size + (separator_size if current_content else 0)
        total_size_if_added = current_file_size + current_size + size_with_separator

        if total_size_if_added > max_size_bytes:
            # File would exceed limit - write current batch and start new file
            # Note: The new document is NOT in current_content, so it will go to the new file
            if current_content:
                bytes_written = write_batch_streaming(current_file, current_content, separator_with_newline, separator_size, is_first_in_file)
                total_size_written += bytes_written
                current_file_size += bytes_written

                current_file.flush()
                current_file.close()
                actual_size = os.path.getsize(output_file)
                logger.debug(f"File {file_counter} closed: {actual_size / 1024 / 1024:.2f} MB, {len(current_content)} documents")

                file_counter += 1
                output_file = os.path.join(output_dir, f"combined_{file_counter}.txt")
                current_file = open(output_file, "w", encoding="utf-8", buffering=buffer_size)
                is_first_in_file = True
                current_file_size = 0

            # Add document to new file (document is atomic - never split)
            current_content = [cleaned_text]
            current_size = text_size
        else:
            # Document fits in current file
            current_content.append(cleaned_text)
            current_size += size_with_separator

        # Batch write when buffer reaches threshold
        if len(current_content) >= 100:
            bytes_written = write_batch_streaming(current_file, current_content, separator_with_newline, separator_size, is_first_in_file)
            total_size_written += bytes_written
            current_file_size += bytes_written
            current_file.flush()
            is_first_in_file = False
            current_content = []
            current_size = 0

        books_processed += 1

    # Write remaining content
    if current_content:
        bytes_written = write_batch_streaming(current_file, current_content, separator_with_newline, separator_size, is_first_in_file)
        total_size_written += bytes_written
        current_file_size += bytes_written

    current_file.flush()
    current_file.close()
    actual_size = os.path.getsize(output_file)
    logger.debug(f"Final file {file_counter} closed: {actual_size / 1024 / 1024:.2f} MB")
    print(f"Wrote {output_file} ({actual_size / 1024 / 1024:.2f} MB)")

    # Summary statistics
    print(f"\nProcessing complete:")
    print(f"  Books processed: {books_processed}")
    print(f"  Books skipped: {books_skipped}")
    print(f"    - Too short (< {min_doc_length} chars): {books_too_short}")
    if max_doc_length is not None:
        print(f"    - Too long (> {max_doc_length} chars): {books_too_long}")
    print(f"    - Not primarily English: {books_not_english}")
    print(f"  Books exceeding max size: {books_oversized}")
    print(f"  Output files: {file_counter}")
    print(f"  Total size written: {total_size_written / 1024 / 1024:.2f} MB")
    if file_counter > 0:
        print(f"  Average file size: {total_size_written / file_counter / 1024 / 1024:.2f} MB")
    print(f"  Output directory: {os.path.abspath(output_dir)}")

    return file_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Project Gutenberg dataset: strip boilerplate and combine books"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/gutenberg",
        help="Directory containing the Arrow format dataset"
    )
    parser.add_argument(
        "--max_size_mb",
        type=int,
        default=50,
        help="Maximum file size for each combined output file in megabytes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/clean/gutenberg",
        help="Directory where the cleaned data will be saved"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=8192*1024*8,
        help="Buffer size for file writing in bytes (default: 64MB)"
    )
    parser.add_argument(
        "--min_doc_length",
        type=int,
        default=500,
        help="Minimum document length in characters after cleaning (default: 500)"
    )
    parser.add_argument(
        "--max_doc_length",
        type=int,
        default=None,
        help="Maximum document length in characters after cleaning (default: None, no limit)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    process_gutenberg_dataset(
        input_dir=args.data_dir,
        output_dir=args.output_dir,
        max_size_mb=args.max_size_mb,
        separator="<|endoftext|>",
        buffer_size=args.buffer_size,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        debug=args.debug
    )

