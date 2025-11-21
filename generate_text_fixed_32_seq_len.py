from llms_from_scratch.ch05 import generate_one_token
import time
from pathlib import Path
import torch
from safetensors.torch import load_file

from llms_from_scratch.qwen3_fixed_32_seq_len import Qwen3Model, QWEN_CONFIG_06_B_FIXED_32, load_weights_into_qwen, Qwen3Tokenizer

def pad_or_truncate_to_32_tokens(token_ids, pad_token_id):
    """严格将token序列调整为32个token，不足补padding，超过截断"""
    if len(token_ids) > 32:
        # 超过32个token，截断并警告
        print(f"Warning: Input length ({len(token_ids)}) exceeds fixed sequence length of 32. Truncating to 32 tokens.")
        return token_ids[:32]
    elif len(token_ids) < 32:
        # 不足32个token，补padding
        padding_length = 32 - len(token_ids)
        padded_tokens = token_ids + [pad_token_id] * padding_length
        print(f"Info: Input length ({len(token_ids)}) less than 32. Padding with {padding_length} pad tokens.")
        return padded_tokens
    else:
        # 正好32个token
        return token_ids

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

# 严格格式化为32个token
input_token_ids = pad_or_truncate_to_32_tokens(input_token_ids, tokenizer.pad_token_id)
print(f"Final input token count: {len(input_token_ids)} (strictly 32)")

torch.manual_seed(123)

# Use fixed context size of 32
actual_context_size = 32  # Fixed sequence length
print(f"Using fixed context size: {actual_context_size} (max: {QWEN_CONFIG_06_B_FIXED_32['context_length']})")

start = time.time()
print("\nStarting generation...")
print("Generated text: ", end="", flush=True)

# 初始化输入，确保严格32个token
idx = torch.tensor(input_token_ids, device=device).unsqueeze(0)
max_new_tokens = 5
eos_id = None  # Can be set to stop token ID if needed

generated_tokens = []

print("\nStarting generation...")
print("Generated text: ", end="", flush=True)

# 生成新token，每次保持32个token的固定长度
for i in range(max_new_tokens):
    # 确保输入是严格的32个token长度
    current_seq_len = idx.shape[1]
    if current_seq_len > 32:
        # 如果超过32，取最后32个token
        idx = idx[:, -32:]
    elif current_seq_len < 32:
        # 如果不足32，补padding
        padding_needed = 32 - current_seq_len
        pad_tokens = torch.full((idx.shape[0], padding_needed), tokenizer.pad_token_id, device=device)
        idx = torch.cat([pad_tokens, idx], dim=1)

    # 确保输入严格为32长度
    assert idx.shape[1] == 32, f"Input must be exactly 32 tokens, got {idx.shape[1]}"

    next_token = generate_one_token(
        model=model,
        idx=idx,
        context_size=32,  # 固定32长度
        top_k=1,
        temperature=0.,
        eos_id=eos_id
    )

    # 检查EOS token
    if eos_id is not None and next_token.item() == eos_id:
        break

    # 解码并打印新token
    token_text = tokenizer.decode([next_token.item()])
    print(token_text, end="", flush=True)

    generated_tokens.append(next_token.item())

    # 更新输入序列：移除第一个token，添加新token，保持32长度
    idx = torch.cat([idx[:, 1:], next_token], dim=1)

print()  # New line after generation

# 构建完整的输出序列（原始输入 + 生成的token）
original_input_length = len(input_token_ids)  # 原始的32个token
all_tokens = input_token_ids + generated_tokens

# 过滤掉padding token，只显示实际内容
actual_tokens = [token for token in all_tokens if token != tokenizer.pad_token_id]
output_token_ids = torch.tensor([actual_tokens], device=device)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(actual_tokens)

print("\n\nOutput text:\n\n", output_text + "...")