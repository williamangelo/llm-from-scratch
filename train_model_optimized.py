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
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.gpt import GPT


# ============================================================================
# Logging Setup
# ============================================================================

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid interfering with progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Module-level logger (logging.getLogger() returns a singleton, thread-safe)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# File I/O constants
# For 50MB files, 4MB chunks provide good balance between memory efficiency and I/O performance
FILE_READ_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB - optimal for ~50MB files
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB - threshold for "large file" logging
PROGRESS_THRESHOLD = 50 * 1024 * 1024  # 50MB - threshold for showing progress bars
TOKENIZATION_PROGRESS_THRESHOLD = 10 * 1024 * 1024  # 10MB - threshold for tokenization progress


# ============================================================================
# Optimized Dataset Implementation
# ============================================================================

class GPTDatasetV2(Dataset):
    """Optimized GPT Dataset with lazy tokenization and on-the-fly slicing.
    
    Performance improvements over GPTDatasetV1:
    - Tokenizes once and stores as a single contiguous tensor (not individual chunks)
    - Slices on-the-fly in __getitem__ (avoids creating thousands of tensors upfront)
    - ~10-100x faster initialization for large datasets
    - Lower memory overhead
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


# ============================================================================
# Text Generation Functions
# ============================================================================

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Numerical stability: subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop if end-of-sequence token is encountered
        if idx_next == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================================
# Data Loading Functions
# ============================================================================


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
    """Create training and validation dataloaders using optimized GPTDatasetV2."""
    # Split data
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Create datasets using optimized GPTDatasetV2
    train_dataset = GPTDatasetV2(train_data, tokenizer, max_length, stride, show_progress=True)
    val_dataset = GPTDatasetV2(val_data, tokenizer, max_length, stride, show_progress=True)
    
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


# ============================================================================
# Loss Calculation Functions
# ============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over a dataloader."""
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    total_loss = 0.0  # Accumulate on CPU as Python float
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()  # Convert to Python float immediately

    return total_loss / num_batches


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


def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    """Generate and print a sample text from the model."""
    model.eval()
    # Access context_size from the positional embedding layer
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    logger.info(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, patience=5):
    """Train model with early stopping.
    
    Args:
        patience: Number of eval steps without improvement before early stopping
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
        
        for input_batch, target_batch in pbar:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'lr': f'{current_lr:.2e}'})

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

        # Generate sample text after each epoch
        logger.info(f"Epoch {epoch+1} sample:")
        generate_and_print_sample(model, tokenizer, device, start_context)

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
        "--data_dir",
        type=str,
        default="data/the_verdict.txt",
        help="Directory containing training text files (.txt) or path to a single .txt file (default: data/the_verdict.txt)"
    )
    parser.add_argument(
        "--num_files",
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
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
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
        "--eval_freq",
        type=int,
        default=100,
        help="Evaluate model every N training steps (default: 100)"
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=10,
        help="Number of batches to use for evaluation during training (default: 10)"
    )
    parser.add_argument(
        "--initial_eval_batches",
        type=int,
        default=50,
        help="Number of batches to use for initial evaluation (default: 50)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile for better performance (requires PyTorch 2.0+)"
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

    # Load text data
    text_data = load_text_data_from_dir(args.data_dir, args.num_files)

    logger.info(f"Loaded {len(text_data):,} characters")

    # Create dataloaders
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
    
    # Compile model if requested (PyTorch 2.0+ optimization)
    if args.compile:
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # Initial evaluation
    initial_eval_batches = min(args.initial_eval_batches, len(train_loader), len(val_loader))
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=initial_eval_batches)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=initial_eval_batches)
    logger.info(f"Initial loss - Train: {train_loss:.3f}, Val: {val_loss:.3f}")

    # Train model
    logger.info("Starting training")
    train_losses, val_losses, tokens_seen, lrs = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        patience=args.patience
    )

    # Save the trained model
    save_model(model, GPT_CONFIG, args.model)
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
