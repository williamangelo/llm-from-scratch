"""
Optimized GPT Model Training Script

This script recreates the training loop from ch05.ipynb but uses the optimized
GPT model from models.gpt instead of the from_scratch implementation.

Model optimizations:
- MultiHeadAttention with scaled dot product and Flash Attention
- PyTorch's native GELU activation
- PyTorch's native LayerNorm

Training optimizations:
- Multi-worker data loading for faster data preprocessing
- Early stopping based on validation loss
- Progress bars and concise logging with tqdm
- Automatic device selection (CUDA > MPS > CPU)
"""

import argparse
import glob
import logging
import os

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from models.gpt import GPT


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid interfering with progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)



# File I/O
FILE_READ_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
PROGRESS_THRESHOLD = 50 * 1024 * 1024  # 50MB
TOKENIZATION_PROGRESS_THRESHOLD = 10 * 1024 * 1024  # 10MB

# Memory safety thresholds
MEMORY_MULTIPLIER = 3.0  # Peak memory is ~3x raw text size during tokenization
MEMORY_SAFETY_MARGIN = 0.8  # Use at most 80% of available memory


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

class GPTDatasetV2(Dataset):
    """Optimized GPT Dataset with lazy tokenization and on-the-fly slicing.
    
    Performance improvements over GPTDatasetV1:
    - Tokenizes once and stores as a single contiguous tensor (not individual chunks)
    - Slices on-the-fly in __getitem__ (avoids creating thousands of tensors upfront)
    - ~10-100x faster initialization for large datasets
    - Lower memory overhead
    
    Memory optimization:
    - Use from_tokens() to avoid duplicating text data during train/val split
    """
    
    def __init__(self, txt, tokenizer, max_length, stride, show_progress=True):
        """Initialize dataset by tokenizing text once.
        
        Args:
            txt: Raw text string to tokenize
            tokenizer: Tokenizer (e.g., tiktoken)
            max_length: Context window size
            stride: Stride for sliding window (controls overlap)
            show_progress: Whether to show progress bar during tokenization
        """
        # Tokenize with progress indication for large texts
        text_size_bytes = len(txt.encode('utf-8'))
        text_size_mb = text_size_bytes / (1024 * 1024)
        if show_progress and text_size_bytes > TOKENIZATION_PROGRESS_THRESHOLD:
            logger.info(f"Tokenizing text ({text_size_mb:.1f} MB)...")
        
        token_ids_list = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        if show_progress and text_size_bytes > TOKENIZATION_PROGRESS_THRESHOLD:
            logger.info(f"Tokenization complete: {len(token_ids_list):,} tokens")
        
        # Convert to tensor
        self.token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        
        # Calculate number of samples using stride
        # Each sample needs max_length + 1 tokens (input + target)
        self.num_samples = max(0, (len(self.token_ids) - max_length - 1) // stride + 1)
    
    @classmethod
    def from_tokens(cls, token_ids, max_length, stride):
        """Create dataset from pre-tokenized tensor (memory efficient).
        
        Args:
            token_ids: Pre-tokenized tensor of token IDs
            max_length: Context window size
            stride: Stride for sliding window
        """
        instance = cls.__new__(cls)
        instance.token_ids = token_ids
        instance.max_length = max_length
        instance.stride = stride
        instance.num_samples = max(0, (len(token_ids) - max_length - 1) // stride + 1)
        return instance
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample by slicing the token tensor on-the-fly."""
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length
        
        # Input: tokens[start:end]
        # Target: tokens[start+1:end+1] (shifted by 1)
        input_chunk = self.token_ids[start_idx:end_idx]
        target_chunk = self.token_ids[start_idx + 1:end_idx + 1]
        
        return input_chunk, target_chunk


class StreamingGPTDataset(IterableDataset):
    """Memory-efficient streaming dataset that loads and tokenizes files lazily.
    
    Designed for large datasets (10GB+) that don't fit in memory.
    Files are processed one at a time, tokenized, and samples yielded on-the-fly.
    
    Memory usage: Only one file's tokens are held in memory at a time.
    For a 16GB dataset split into 300 files (~53MB each), peak memory is ~100-150MB
    for token data instead of 16GB+.
    """
    
    def __init__(self, file_paths, tokenizer, max_length, stride, 
                 train_ratio=0.85, is_validation=False):
        """
        Args:
            file_paths: List of paths to .txt files
            tokenizer: Tokenizer (e.g., tiktoken)
            max_length: Context window size
            stride: Stride for sliding window
            train_ratio: Fraction of each file to use for training (rest for validation)
            is_validation: If True, use last (1-train_ratio) of each file
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.is_validation = is_validation
        self._estimated_samples = None
    
    def _tokenize_file(self, file_path):
        """Load and tokenize a single file, returning tensor of token IDs."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        del text
        return torch.tensor(token_ids, dtype=torch.long)
    
    def _generate_samples_from_tokens(self, token_ids):
        """Generate (input, target) samples from a token tensor."""
        split_idx = int(self.train_ratio * len(token_ids))
        if self.is_validation:
            token_ids = token_ids[split_idx:]
        else:
            token_ids = token_ids[:split_idx]
        
        num_samples = max(0, (len(token_ids) - self.max_length - 1) // self.stride + 1)
        
        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.max_length
            yield token_ids[start_idx:end_idx], token_ids[start_idx + 1:end_idx + 1]
    
    def __iter__(self):
        """Iterate through all files, yielding samples lazily."""
        for file_path in self.file_paths:
            token_ids = self._tokenize_file(file_path)
            yield from self._generate_samples_from_tokens(token_ids)
            del token_ids
    
    def estimate_total_samples(self):
        """Estimate total samples by sampling a few files (for progress bars)."""
        if self._estimated_samples is not None:
            return self._estimated_samples
        
        if len(self.file_paths) == 0:
            return 0
        
        sample_files = self.file_paths[:min(3, len(self.file_paths))]
        total_samples = 0
        
        for file_path in sample_files:
            file_size = os.path.getsize(file_path)
            # Rough estimate: ~4 bytes per character, ~4 chars per token on average
            estimated_tokens = file_size // 4
            if self.is_validation:
                effective_len = int((1 - self.train_ratio) * estimated_tokens)
            else:
                effective_len = int(self.train_ratio * estimated_tokens)
            samples = max(0, (effective_len - self.max_length - 1) // self.stride + 1)
            total_samples += samples
        
        avg_samples = total_samples / len(sample_files)
        self._estimated_samples = int(avg_samples * len(self.file_paths))
        return self._estimated_samples


def get_available_memory():
    """Get available system memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    
    # Fallback for Linux: read from /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError):
        pass
    
    # Fallback for macOS: use vm_stat
    import platform
    if platform.system() == 'Darwin':
        try:
            import subprocess
            # Get page size
            page_size = int(subprocess.check_output(['sysctl', '-n', 'hw.pagesize']).strip())
            # Get vm_stat output
            vm_stat = subprocess.check_output(['vm_stat']).decode('utf-8')
            
            stats = {}
            for line in vm_stat.split('\n'):
                if ':' in line:
                    key, value = line.split(':')
                    # Remove trailing period and whitespace
                    value = value.strip().rstrip('.')
                    try:
                        stats[key.strip()] = int(value)
                    except ValueError:
                        pass
            
            # Available = free + inactive (pages that can be reclaimed)
            free_pages = stats.get('Pages free', 0)
            inactive_pages = stats.get('Pages inactive', 0)
            available = (free_pages + inactive_pages) * page_size
            return available
        except (subprocess.SubprocessError, ValueError, KeyError):
            pass
    
    return None


def estimate_memory_required(data_size_bytes):
    """Estimate peak memory required for loading and tokenizing data.
    
    Returns:
        Estimated peak memory in bytes
    """
    # Peak memory during tokenization:
    # - Raw text: data_size_bytes
    # - Token IDs list (Python list): ~2x text size (list overhead + int objects)
    # - Final tensor: ~0.5x text size (int64, but fewer tokens than chars)
    # During tokenization, all three exist simultaneously
    return int(data_size_bytes * MEMORY_MULTIPLIER)


def check_memory_for_dataset(data_path, num_files=None, streaming=False):
    """Check if dataset will fit in memory and warn if not.
    
    Args:
        data_path: Path to data directory or file
        num_files: Optional limit on number of files
        streaming: Whether streaming mode is enabled
    
    Returns:
        Tuple of (data_size_bytes, should_use_streaming, warning_message)
    """
    if os.path.isfile(data_path):
        data_size = os.path.getsize(data_path)
        file_count = 1
    else:
        txt_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))
        if num_files is not None:
            txt_files = txt_files[:num_files]
        data_size = sum(os.path.getsize(f) for f in txt_files)
        file_count = len(txt_files)
    
    available_memory = get_available_memory()
    estimated_peak = estimate_memory_required(data_size)
    
    data_size_gb = data_size / (1024**3)
    estimated_peak_gb = estimated_peak / (1024**3)
    
    warning_msg = None
    should_use_streaming = False
    
    if available_memory is not None:
        available_gb = available_memory / (1024**3)
        safe_limit = available_memory * MEMORY_SAFETY_MARGIN
        
        if not streaming and estimated_peak > safe_limit:
            should_use_streaming = True
            warning_msg = (
                f"WARNING: Dataset ({data_size_gb:.1f} GB, {file_count} files) may exceed available memory!\n"
                f"  Estimated peak memory: {estimated_peak_gb:.1f} GB\n"
                f"  Available memory: {available_gb:.1f} GB\n"
                f"  Recommendation: Use --streaming flag to reduce memory usage"
            )
        elif not streaming and estimated_peak > safe_limit * 0.5:
            warning_msg = (
                f"NOTE: Dataset ({data_size_gb:.1f} GB) will use significant memory (~{estimated_peak_gb:.1f} GB peak).\n"
                f"  Consider using --streaming if you experience memory issues."
            )
    else:
        # Can't determine available memory, use heuristic based on dataset size
        if not streaming and data_size > 2 * 1024**3:  # > 2GB
            should_use_streaming = True
            warning_msg = (
                f"WARNING: Large dataset detected ({data_size_gb:.1f} GB, {file_count} files).\n"
                f"  Estimated peak memory: {estimated_peak_gb:.1f} GB\n"
                f"  Could not determine available system memory.\n"
                f"  Recommendation: Use --streaming flag to reduce memory usage"
            )
    
    return data_size, should_use_streaming, warning_msg


def get_txt_files(data_path, num_files=None):
    """Get list of .txt files from a directory or single file path.
    
    Args:
        data_path: Path to directory containing .txt files or path to a single .txt file
        num_files: Optional number of files to return (returns all if None)
    
    Returns:
        List of file paths
    """
    if os.path.isfile(data_path):
        return [data_path]
    
    txt_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_path}")
    
    if num_files is not None:
        txt_files = txt_files[:num_files]
    
    return txt_files


def load_text_data_from_dir(data_path, num_files=None, chunk_size=None):
    """Load text data from a directory or single file with memory-efficient chunked reading.

    Args:
        data_path: Path to directory containing .txt files or path to a single .txt file
        num_files: Optional number of files to read (reads all if None, ignored for single files)
        chunk_size: Size of chunks to read at a time in bytes (default: FILE_READ_CHUNK_SIZE)

    Returns:
        Combined text data from all files
    """
    if chunk_size is None:
        chunk_size = FILE_READ_CHUNK_SIZE
    
    # Check if path is a file or directory
    if os.path.isfile(data_path):
        # Single file case - read in chunks to avoid loading entire large file into memory
        file_size = os.path.getsize(data_path)
        if file_size > LARGE_FILE_THRESHOLD:
            logger.info(f"Loading large file ({file_size / (1024*1024):.1f} MB): {os.path.basename(data_path)}")
        else:
            logger.info(f"Loading file: {os.path.basename(data_path)}")
        
        chunks = []
        with open(data_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        return "".join(chunks)

    # Directory case - get all .txt files in the directory
    txt_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))

    if not txt_files:
        raise ValueError(f"No .txt files found in {data_path}")

    # Limit number of files if specified
    if num_files is not None:
        txt_files = txt_files[:num_files]

    logger.info(f"Loading {len(txt_files)} file(s) from {data_path}")

    # Read and combine all files with chunked reading for large files
    combined_text = []
    total_size = sum(os.path.getsize(f) for f in txt_files)
    show_progress = total_size > PROGRESS_THRESHOLD
    
    if show_progress:
        pbar = tqdm(txt_files, desc="Reading files", unit="file", leave=False)
    else:
        pbar = txt_files
    
    for file_path in pbar:
        if show_progress:
            pbar.set_postfix(file=os.path.basename(file_path)[:30])
        
        chunks = []
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        combined_text.append("".join(chunks))

    # Join with newlines to separate different files
    return "\n\n".join(combined_text)


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, tokenizer):
    """Create training and validation dataloaders using optimized GPTDatasetV2.
    
    Memory optimization: Tokenize once, then split to avoid duplicating large text strings.
    """
    # Tokenize full dataset once (more memory efficient than tokenizing splits separately)
    logger.info(f"Tokenizing full dataset ({len(text_data):,} characters)...")
    token_ids_list = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
    logger.info(f"Tokenization complete: {len(token_ids_list):,} tokens")
    
    # Free the text data immediately after tokenization
    del text_data
    
    # Convert to tensor
    token_ids = torch.tensor(token_ids_list, dtype=torch.long)
    del token_ids_list  # Free the list
    
    # Split tokenized data (avoids duplicating large text strings)
    split_idx = int(train_ratio * len(token_ids))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]
    del token_ids  # Free the full tensor
    
    # Create datasets from pre-tokenized data
    train_dataset = GPTDatasetV2.from_tokens(train_tokens, max_length, stride)
    val_dataset = GPTDatasetV2.from_tokens(val_tokens, max_length, stride)
    
    logger.info(f"Datasets: {len(train_dataset):,} train, {len(val_dataset):,} val samples")

    # Create dataloaders
    num_workers = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def create_streaming_dataloaders(file_paths, batch_size, max_length, stride, tokenizer, train_ratio=0.85):
    """Create streaming dataloaders for large datasets that don't fit in memory.
    
    Each file is loaded, tokenized, and processed one at a time.
    Train/val split is done per-file (first train_ratio for train, rest for val).
    
    Memory usage: Only one file's worth of tokens in memory at a time.
    """
    logger.info(f"Creating streaming dataloaders for {len(file_paths)} files")
    
    train_dataset = StreamingGPTDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        train_ratio=train_ratio,
        is_validation=False
    )
    
    val_dataset = StreamingGPTDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        train_ratio=train_ratio,
        is_validation=True
    )
    
    train_samples = train_dataset.estimate_total_samples()
    val_samples = val_dataset.estimate_total_samples()
    logger.info(f"Estimated samples: ~{train_samples:,} train, ~{val_samples:,} val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# ============================================================================
# Loss Calculation Functions
# ============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches):
    """Calculate average loss over a dataloader."""
    total_loss = 0.0
    batches_processed = 0
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        batches_processed += 1

    if batches_processed == 0:
        return float("nan")
    return total_loss / batches_processed


# ============================================================================
# Training Functions
# ============================================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, patience,
                gradient_accumulation_steps, scaler):
    """Train model with early stopping, gradient accumulation, and mixed precision.
    
    Args:
        patience: Number of eval steps without improvement before early stopping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        scaler: GradScaler for mixed precision training (None to disable)
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen, lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    steps_without_improvement = 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            # Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{(loss.item() * gradient_accumulation_steps):.3f}', 'lr': f'{current_lr:.2e}'})

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                lrs.append(current_lr)
                
                logger.info(f"Step {global_step:05d} | Train: {train_loss:.3f} | "
                          f"Val: {val_loss:.3f} | LR: {current_lr:.2e}")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    logger.info(f"New best val loss: {best_val_loss:.3f}")
                else:
                    steps_without_improvement += 1
                    if steps_without_improvement >= patience:
                        logger.info(f"Early stopping after {patience} steps without improvement")
                        return train_losses, val_losses, track_tokens_seen, lrs

    return train_losses, val_losses, track_tokens_seen, lrs


# ============================================================================
# Model Saving Functions
# ============================================================================

def save_model(model, config, model_name, save_dir="data/models"):
    """Save trained model to disk.
    
    Args:
        model: Trained GPT model
        config: Model configuration dictionary
        model_name: Name identifier for the model (e.g., 'gpt2-small')
        save_dir: Directory to save model files (default: 'data/models')
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with model name
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    
    # Save model state dict and configuration
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_name': model_name
    }
    
    torch.save(checkpoint, model_path)
    logger.info(f"Model saved: {model_path}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train GPT model on text data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/the_verdict.txt",
        help="Directory containing training text files (.txt) or path to a single .txt file (default: data/the_verdict.txt)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Number of files to read from data_dir (optional, reads all files if not specified)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xlarge"],
        help="GPT model size to train (default: gpt2-small)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0004,
        help="Learning rate (default: 0.0004)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in eval steps (default: 5)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100,
        help="Evaluate model every N training steps (default: 100)"
    )
    parser.add_argument(
        "--eval-iter",
        type=int,
        default=10,
        help="Number of batches to use for evaluation during training (default: 10)"
    )
    parser.add_argument(
        "--initial-eval-batches",
        type=int,
        default=50,
        help="Number of batches to use for initial evaluation (default: 50)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile for better performance (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision (fp16) training to reduce memory usage (requires CUDA)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (allows larger effective batch size with less memory)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets (loads files lazily, reduces memory usage)"
    )

    args = parser.parse_args()

    # Configure logging to use tqdm.write() to avoid interfering with progress bar
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

    # Set random seed for reproducibility
    torch.manual_seed(123)

    # Base configuration
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Model-specific configurations
    MODEL_CONFIGS = {
        "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},      # 124M params
        "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},    # 355M params
        "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},     # 774M params
        "gpt2-xlarge": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},    # 1558M params
    }

    # Create final configuration by combining base and model-specific settings
    GPT_CONFIG = BASE_CONFIG.copy()
    GPT_CONFIG.update(MODEL_CONFIGS[args.model])

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Check memory requirements before loading data
    data_size, should_use_streaming, memory_warning = check_memory_for_dataset(
        args.data_dir, args.num_files, args.streaming
    )
    
    if memory_warning:
        logger.warning(memory_warning)
    
    if should_use_streaming and not args.streaming:
        logger.warning("Automatically enabling --streaming mode due to memory constraints")
        args.streaming = True

    # Create dataloaders (streaming or standard mode)
    if args.streaming:
        file_paths = get_txt_files(args.data_dir, args.num_files)
        total_size = sum(os.path.getsize(f) for f in file_paths)
        logger.info(f"Streaming mode: {len(file_paths)} files, {total_size / (1024**3):.2f} GB total")
        
        train_loader, val_loader = create_streaming_dataloaders(
            file_paths=file_paths,
            batch_size=args.batch_size,
            max_length=GPT_CONFIG["context_length"],
            stride=GPT_CONFIG["context_length"],
            tokenizer=tokenizer,
            train_ratio=0.85
        )
    else:
        text_data = load_text_data_from_dir(args.data_dir, args.num_files)
        logger.info(f"Loaded {len(text_data):,} characters")
        
        train_loader, val_loader = create_dataloaders(
            text_data=text_data,
            train_ratio=0.85,
            batch_size=args.batch_size,
            max_length=GPT_CONFIG["context_length"],
            stride=GPT_CONFIG["context_length"],
            tokenizer=tokenizer
        )

    # Initialize model
    model = GPT(GPT_CONFIG)
    
    # Calculate and display model info
    total_params = sum(p.numel() for p in model.parameters())
    
    # Format parameters in M/B notation
    def format_params(params):
        if params >= 1e9:
            return f"{params/1e9:.1f}B"
        else:
            return f"{params/1e6:.0f}M"
    
    logger.info(f"Model: {args.model} ({format_params(total_params)} params, "
                f"{GPT_CONFIG['emb_dim']}d, {GPT_CONFIG['n_layers']}L, {GPT_CONFIG['n_heads']}H)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Device: {device}")
    model.to(device)
    
    # Setup mixed precision training if requested
    scaler = None
    if args.mixed_precision:
        if device.type == "cuda":
            scaler = torch.amp.GradScaler('cuda')
            logger.info("Mixed precision (fp16) enabled")
        else:
            logger.info("Mixed precision requested but not available on this device (requires CUDA)")
    
    # Compile model if requested (PyTorch 2.0+ optimization)
    if args.compile:
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Log memory optimization settings
    if args.gradient_accumulation_steps > 1:
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps} steps "
                   f"(effective batch size: {args.batch_size * args.gradient_accumulation_steps})")

    # Initial evaluation
    if args.streaming:
        initial_eval_batches = args.initial_eval_batches
    else:
        initial_eval_batches = min(args.initial_eval_batches, len(train_loader), len(val_loader))
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=initial_eval_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=initial_eval_batches)
    logger.info(f"Initial loss - Train: {train_loss:.3f}, Val: {val_loss:.3f}")

    # Train model
    logger.info("Starting training")
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scaler=scaler
    )

    # Save the trained model
    save_model(model, GPT_CONFIG, args.model)
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
