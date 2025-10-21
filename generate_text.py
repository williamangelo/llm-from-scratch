"""
Simple script to load a trained GPT model and generate text.

Usage:
    python generate_text.py
    python generate_text.py --prompt "Hello world" --max_tokens 100 --temperature 1.0
"""

import argparse
import torch
import tiktoken
from models.gpt import GPT


def generate_text_from_model(model_path, prompt, max_tokens=50, temperature=0.0, top_k=None):
    """Load a saved model and generate text.
    
    Args:
        model_path: Path to saved model checkpoint
        prompt: Starting text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k sampling (None = disabled)
    
    Returns:
        Generated text string
    """
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}\n")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    print(f"Prompt: \"{prompt}\"")
    print(f"Generating {max_tokens} tokens...\n")
    
    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get predictions
            idx_cond = encoded_tensor[:, -config["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val, 
                    torch.tensor(float("-inf")).to(logits.device), 
                    logits
                )
            
            # Sample next token
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            encoded_tensor = torch.cat((encoded_tensor, idx_next), dim=1)
    
    # Decode
    generated_text = tokenizer.decode(encoded_tensor.squeeze(0).tolist())
    
    print("=" * 70)
    print(generated_text)
    print("=" * 70)
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/models/gpt2-small.pth",
        help="Path to saved model checkpoint (default: data/models/gpt2-small.pth)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Every effort moves you",
        help="Starting prompt text (default: 'Every effort moves you')"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Number of tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0.0=greedy, higher=more random (default: 0.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter (default: None)"
    )
    
    args = parser.parse_args()
    
    generate_text_from_model(
        model_path=args.model_path,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

