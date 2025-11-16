from llms_from_scratch.ch05 import generate
import time


from pathlib import Path
import torch
from safetensors.torch import load_file

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, load_weights_into_qwen

model_path= "models/Qwen3-0.6B"

model_file = Path(model_path,"model.safetensors")

model = Qwen3Model(QWEN_CONFIG_06_B)
weights_dict = load_file(model_file)
load_weights_into_qwen(model, QWEN_CONFIG_06_B, weights_dict)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);

from llms_from_scratch.qwen3 import Qwen3Tokenizer

USE_INSTRUCT_MODEL = False
USE_REASONING_MODEL = True


if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path(model_path,"tokenizer.json"),
    repo_id=model_path,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)

prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)


torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
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