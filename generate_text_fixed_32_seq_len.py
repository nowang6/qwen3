from llms_from_scratch.ch05 import generate_one_token
import time
from pathlib import Path
import torch
from safetensors.torch import load_file

from llms_from_scratch.qwen3_fixed_32_seq_len import Qwen3Model, QWEN_CONFIG_06_B_FIXED_32, load_weights_into_qwen, Qwen3Tokenizer

model_path= "models/Qwen3-0.6B"

model_file = Path(model_path,"model.safetensors")

print("Loading model with fixed sequence length of 32...")
model = Qwen3Model(QWEN_CONFIG_06_B_FIXED_32)
weights_dict = load_file(model_file)
load_weights_into_qwen(model, QWEN_CONFIG_06_B_FIXED_32, weights_dict)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")
model.to(device);
model.eval()  # Set to evaluation mode


USE_INSTRUCT_MODEL = False
USE_REASONING_MODEL = True


if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

print("Loading tokenizer...")
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path(model_path,"tokenizer.json"),
    repo_id=model_path,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)

prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
print(f"Input prompt: {prompt}")
print(f"Input token count: {len(input_token_ids)}")

# Check if input exceeds fixed sequence length
if len(input_token_ids) > 32:
    print(f"Warning: Input length ({len(input_token_ids)}) exceeds fixed sequence length of 32. Truncating to 32 tokens.")
    input_token_ids = input_token_ids[:32]
    print(f"Truncated input token count: {len(input_token_ids)}")

torch.manual_seed(123)

# Use fixed context size of 32
actual_context_size = 32  # Fixed sequence length
print(f"Using fixed context size: {actual_context_size} (max: {QWEN_CONFIG_06_B_FIXED_32['context_length']})")

start = time.time()
print("\nStarting generation...")
print("Generated text: ", end="", flush=True)

# Initialize with input tokens
idx = torch.tensor(input_token_ids, device=device).unsqueeze(0)
max_new_tokens = 5
eos_id = None  # Can be set to stop token ID if needed

# Generate tokens one by one
for i in range(max_new_tokens):
    # Check if current sequence length exceeds fixed limit
    if idx.shape[1] >= 32:
        print(f"\nReached maximum sequence length of 32 tokens. Stopping generation.")
        break

    next_token = generate_one_token(
        model=model,
        idx=idx,
        context_size=actual_context_size,
        top_k=1,
        temperature=0.,
        eos_id=eos_id
    )

    # Check for EOS token if specified
    if eos_id is not None and next_token.item() == eos_id:
        break

    # Decode and print the new token immediately
    token_text = tokenizer.decode([next_token.item()])
    print(token_text, end="", flush=True)

    # Append the new token to the sequence
    idx = torch.cat((idx, next_token), dim=1)

print()  # New line after generation

output_token_ids = idx

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")