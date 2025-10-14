"""
Optimized GPT Model Training Script

This script recreates the training loop from ch05.ipynb but uses the optimized
GPT model from models.gpt instead of the from_scratch implementation.

The optimized model uses:
- MultiHeadAttention with scaled dot product and Flash Attention
- PyTorch's native GELU activation
- PyTorch's native LayerNorm
"""

import argparse
import glob
import os
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from models.gpt import GPT
from models.from_scratch import GPTDatasetV1


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


def load_text_data_from_dir(data_path, num_files=None):
    """Load text data from a directory or single file.

    Args:
        data_path: Path to directory containing .txt files or path to a single .txt file
        num_files: Optional number of files to read (reads all if None, ignored for single files)

    Returns:
        Combined text data from all files
    """
    # Check if path is a file or directory
    if os.path.isfile(data_path):
        # Single file case
        print(f"Loading single file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            return f.read()

    # Directory case - get all .txt files in the directory
    txt_files = sorted(glob.glob(os.path.join(data_path, "*.txt")))

    if not txt_files:
        raise ValueError(f"No .txt files found in {data_path}")

    # Limit number of files if specified
    if num_files is not None:
        txt_files = txt_files[:num_files]

    print(f"Loading {len(txt_files)} file(s) from {data_path}")

    # Read and combine all files
    combined_text = []
    for file_path in txt_files:
        print(f"  Reading: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8") as f:
            combined_text.append(f.read())

    # Join with newlines to separate different files
    return "\n\n".join(combined_text)


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, tokenizer):
    """Create training and validation dataloaders using from_scratch implementation."""
    # Split data
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Create datasets using GPTDatasetV1 from from_scratch
    train_dataset = GPTDatasetV1(train_data, tokenizer, max_length, stride)
    val_dataset = GPTDatasetV1(val_data, tokenizer, max_length, stride)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over a dataloader."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

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
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


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
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0004,
        help="Learning rate (default: 0.0004)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(123)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load text data
    text_data = load_text_data_from_dir(args.data_dir, args.num_files)

    print(f"Total characters: {len(text_data):,}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        text_data=text_data,
        train_ratio=0.85,
        batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        tokenizer=tokenizer
    )

    # Initialize model
    model = GPT(GPT_CONFIG_124M)
    # model = torch.compile(model)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # Calculate initial losses
    print("\nCalculating initial losses...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Initial training loss: {train_loss:.3f}")
    print(f"Initial validation loss: {val_loss:.3f}")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # Generate sample text with different strategies
    print("\nGenerating sample text with different strategies:")

    print("\n1. Greedy sampling (temperature=0.0):")
    token_ids = generate_text(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.0
    )
    print(token_ids_to_text(token_ids, tokenizer))

    print("\n2. Temperature sampling (temperature=1.4, top_k=25):")
    token_ids = generate_text(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=1.4,
        top_k=25
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
