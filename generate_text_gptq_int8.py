from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Local model path
model_path = "models/Qwen3-0.6B-GPTQ-Int8"

# Load the tokenizer and the model
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# Print model dtype
model_dtype = next(model.parameters()).dtype
print(f"Model dtype: {model_dtype}")

# Prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Conduct text completion
print("Generating text...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("\n" + "="*80)
print("Thinking content:")
print("="*80)
print(thinking_content)
print("\n" + "="*80)
print("Content:")
print("="*80)
print(content)
print("="*80)

