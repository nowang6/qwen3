from pathlib import Path
import torch
from safetensors.torch import load_file
import onnx
import onnxslim

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B, load_weights_into_qwen, Qwen3Tokenizer

model_path = "models/Qwen3-0.6B"

# Load model (same as generate_text.py)
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

# Load tokenizer to create example input
print("Loading tokenizer...")
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=Path(model_path, "tokenizer.json"),
    repo_id=model_path,
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True
)

# Prepare example input (similar to generate_text.py)
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
print(f"Input prompt: {prompt}")
print(f"Input token count: {len(input_token_ids)}")

# Create input tensor
example_input = torch.tensor(input_token_ids, device=device).unsqueeze(0)
print(f"Example input shape: {example_input.shape}")

# Export to ONNX
onnx_model_path = "output/qwen3_0.6b_orig.onnx"
print(f"\nExporting to ONNX format...")

input_names = ["input_ids"]
output_names = ["logits"]

# Define dynamic axes for variable sequence length
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "seq_length"},
    "logits": {0: "batch_size", 1: "seq_length"},
}

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(example_input,),
        f=onnx_model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        opset_version=14,
        export_params=True,
        verbose=False
    )

print(f"ONNX model exported to: {onnx_model_path}")

# Verify the exported model
try:
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
except Exception as e:
    print(f"Warning: ONNX model verification failed: {e}")

# Optimize model with onnxslim
print(f"\nOptimizing model with onnxslim...")
slimmed_model = onnxslim.slim(onnx_model)
slimmed_model_path = "output/qwen3_0.6b.onnx"
onnx.save(slimmed_model, slimmed_model_path)
print(f"Slimmed ONNX model saved to: {slimmed_model_path}")

# Verify the slimmed model
try:
    onnx.checker.check_model(slimmed_model)
    print("Slimmed ONNX model verification passed!")
except Exception as e:
    print(f"Warning: Slimmed ONNX model verification failed: {e}")

