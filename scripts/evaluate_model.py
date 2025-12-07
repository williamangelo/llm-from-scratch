"""
Comprehensive model evaluation script for text generation.

This script loads a trained GPT model and evaluates it with various prompts,
token lengths, and sampling strategies.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model data/models/custom_model.pth
"""

import argparse
import os
import sys
import torch
import tiktoken
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.gpt import GPT


# Curated prompts for evaluation - covering different styles and domains
EVALUATION_PROMPTS = [
    "Every effort moves you",
    "Once upon a time",
    "The future of artificial intelligence will",
    "In a shocking discovery, scientists found that",
    "The secret to happiness is",
]


def load_model(model_path, device):
    """Load a trained GPT model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint file
        device: Torch device to load model on
    
    Returns:
        Tuple of (model, config, tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    return model, config, tokenizer


def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_tokens, 
    temperature=1.0, 
    top_k=None, 
    top_p=None,
    repetition_penalty=1.0,
    config=None,
    device=None
):
    """Generate text from a prompt with various sampling strategies.
    
    Args:
        model: GPT model instance
        tokenizer: Tiktoken tokenizer
        prompt: Starting text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_k: Top-k sampling (None = disabled)
        top_p: Top-p (nucleus) sampling (None = disabled)
        repetition_penalty: Penalty for repetition (>1.0 reduces repetition)
        config: Model configuration dict
        device: Torch device
    
    Returns:
        Generated text string
    """
    if device is None:
        device = next(model.parameters()).device
    
    if config is None:
        raise ValueError("config must be provided")
    
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = encoded_tensor[:, -config["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens[-50:]):  # Look at last 50 tokens
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits
                )
            
            # Apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))
            
            # Sample next token
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(idx_next.item())
            encoded_tensor = torch.cat((encoded_tensor, idx_next), dim=1)
    
    # Decode full sequence
    full_sequence = encoded_tensor.squeeze(0).tolist()
    generated_text = tokenizer.decode(full_sequence)
    
    return generated_text


def print_section(title):
    """Print a compact section header."""
    print(f"\n{title}")
    print("-" * len(title))


def evaluate_model(model_path):
    """Run comprehensive evaluation on a trained model.
    
    Args:
        model_path: Path to model checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.mps.is_available() 
                         else "cpu")
    
    print(f"Loading {model_path} on {device}")
    model, config, tokenizer = load_model(model_path, device)
    print("Model loaded\n")
    
    token_lengths = [15, 50, 100]
    temperatures = [0.0, 0.7, 1.2, 1.5]
    
    # Test 1: Basic generation with different token lengths
    print_section("Test 1: Token Length Variation")
    prompt = EVALUATION_PROMPTS[0]
    
    for length in token_lengths:
        print(f"\n[{length} tokens, temp=0.7] {prompt}")
        text = generate_text(
            model, tokenizer, prompt, length,
            temperature=0.7, top_p=0.9,
            config=config, device=device
        )
        print(text)
    
    # Test 2: Temperature comparison
    print_section("Test 2: Temperature Comparison")
    prompt = EVALUATION_PROMPTS[1]
    
    for temp in temperatures:
        print(f"\n[temp={temp}] {prompt}")
        text = generate_text(
            model, tokenizer, prompt, token_lengths[1],
            temperature=temp, top_p=0.9,
            config=config, device=device
        )
        print(text)
    
    # Test 3: All prompts with high temperature
    print_section("Test 3: All Prompts (temp=1.5)")
    high_temp = 1.5
    
    for prompt in EVALUATION_PROMPTS:
        print(f"\n{prompt}")
        text = generate_text(
            model, tokenizer, prompt, token_lengths[1],
            temperature=high_temp, top_p=0.9, repetition_penalty=1.1,
            config=config, device=device
        )
        print(text)
    
    # Test 4: Sampling strategy comparison
    print_section("Test 4: Sampling Strategies")
    prompt = EVALUATION_PROMPTS[2]
    
    strategies = [
        ("Greedy", {"temperature": 0.0}),
        ("Top-k=40", {"temperature": 0.8, "top_k": 40}),
        ("Top-p=0.9", {"temperature": 0.8, "top_p": 0.9}),
        ("Top-p + RepPen", {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.2}),
    ]
    
    for strategy_name, params in strategies:
        print(f"\n[{strategy_name}] {prompt}")
        text = generate_text(
            model, tokenizer, prompt, token_lengths[1],
            config=config, device=device, **params
        )
        print(text)
    
    # Test 5: Multiple samples (variance test)
    print_section("Test 5: Generation Variance")
    prompt = EVALUATION_PROMPTS[3]
    
    for sample_num in range(1, 4):
        print(f"\n[Sample {sample_num}] {prompt}")
        text = generate_text(
            model, tokenizer, prompt, token_lengths[0],
            temperature=1.2, top_p=0.9,
            config=config, device=device
        )
        print(text)
    
    # Test 6: Long generation test
    print_section("Test 6: Long Generation")
    prompt = EVALUATION_PROMPTS[4]
    print(f"{prompt}\n")
    
    text = generate_text(
        model, tokenizer, prompt, token_lengths[-1],
        temperature=0.8, top_p=0.9, repetition_penalty=1.15,
        config=config, device=device
    )
    print(text)
    
    print("\nEvaluation complete")


def find_available_models(models_dir="data/models"):
    """Find all available model files in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    model_files = list(models_path.glob("*.pth"))
    return sorted(model_files)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GPT model with comprehensive text generation tests"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/gpt2-small.pth",
        help="Path to model checkpoint (default: data/models/gpt2-small.pth)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in data/models/ and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        models = find_available_models()
        if models:
            print("Available models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        else:
            print("No model files found in data/models/")
        return
    
    # Normalize model path
    if not os.path.isabs(args.model):
        model_path = os.path.join(os.getcwd(), args.model)
    else:
        model_path = args.model
    
    try:
        evaluate_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable models:")
        models = find_available_models()
        if models:
            for model in models:
                print(f"  - {model}")
        else:
            print("  No models found in data/models/")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()

