from pathlib import Path
import torch
import time
from safetensors.torch import load_file

from llms_from_scratch.ch05 import generate
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, load_weights_into_qwen, Qwen3Tokenizer


def load_qwen3_model_and_tokenizer(
    model_path: str = "models/Qwen3-0.6B",
    use_instruct_model: bool = False,
    use_reasoning_model: bool = True
):
    """
    Load Qwen3 model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        use_instruct_model: Whether to use instruct model
        use_reasoning_model: Whether to use reasoning model
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    model_file = Path(model_path, "model.safetensors")
    
    print("Loading model...")
    model = Qwen3Model(QWEN_CONFIG_06_B)
    weights_dict = load_file(model_file)
    load_weights_into_qwen(model, QWEN_CONFIG_06_B, weights_dict)
    
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    if use_reasoning_model:
        tok_filename = "tokenizer.json"    
    else:
        tok_filename = "tokenizer-base.json"   
    
    print("Loading tokenizer...")
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=Path(model_path, "tokenizer.json"),
        repo_id=model_path,
        apply_chat_template=use_reasoning_model,
        add_generation_prompt=use_reasoning_model,
        add_thinking=not use_instruct_model
    )
    
    return model, tokenizer, device


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str = "Give me a short introduction to large language models.",
    max_new_tokens: int = 150,
    top_k: int = 1,
    temperature: float = 0.0,
    seed: int = 123
):
    """
    Generate text using the loaded model.
    
    Args:
        model: The loaded Qwen3 model
        tokenizer: The loaded tokenizer
        device: The device to run inference on
        prompt: Input prompt text
        max_new_tokens: Maximum number of new tokens to generate
        top_k: Top-k sampling parameter
        temperature: Temperature for sampling
        seed: Random seed for reproducibility
    
    Returns:
        str: Generated text
    """
    input_token_ids = tokenizer.encode(prompt)
    print(f"Input prompt: {prompt}")
    print(f"Input token count: {len(input_token_ids)}")
    
    torch.manual_seed(seed)
    
    # Use a more reasonable context size - use actual input length + some buffer
    # Instead of the full 40960, which is too large for CPU
    actual_context_size = min(len(input_token_ids) + 150, 2048)  # Reasonable size for CPU
    print(f"Using context size: {actual_context_size} (max: {QWEN_CONFIG_06_B['context_length']})")
    
    start = time.time()
    print("\nStarting generation...")
    
    output_token_ids = generate(
        model=model,
        idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
        max_new_tokens=max_new_tokens,
        context_size=actual_context_size,
        top_k=top_k,
        temperature=temperature
    )
    
    total_time = time.time() - start
    print(f"Time: {total_time:.2f} sec")
    print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")
    
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
    
    output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())
    
    print("\n\nOutput text:\n\n", output_text + "...")
    
    return output_text


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer, device = load_qwen3_model_and_tokenizer()
    
    # Generate text
    generate_text(model, tokenizer, device)
