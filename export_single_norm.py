from pathlib import Path
import torch
from safetensors.torch import load_file

from llms_from_scratch.qwen3_fixed_32_seq_len import (
    Qwen3Model,
    QWEN_CONFIG_06_B_FIXED_32,
    load_weights_into_qwen,
    RMSNorm
)

model_path = "models/Qwen3-0.6B"
output_path = "output"
model_file = Path(model_path, "model.safetensors")

print("Loading model with fixed sequence length of 32...")
model = Qwen3Model(QWEN_CONFIG_06_B_FIXED_32)
weights_dict = load_file(model_file)
load_weights_into_qwen(model, QWEN_CONFIG_06_B_FIXED_32, weights_dict)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")
model.to(device)
model.eval()  # Set to evaluation mode

# Export first transformer block's norm1 to ONNX format
print("Exporting first transformer block's norm1 to ONNX format...")
try:
    # Get the norm1 layer from the first transformer block
    first_block = model.trf_blocks[0]
    norm1 = first_block.norm1
    norm1.eval()

    # Define dummy input: embeddings [batch_size, 32, emb_dim]
    # norm1 only needs the input embeddings, shape: [batch_size, seq_len, emb_dim]
    dummy_input = torch.randn(1, 32, QWEN_CONFIG_06_B_FIXED_32['emb_dim'], device=device, dtype=torch.float32)
    
    # Define ONNX export path
    onnx_path = Path(output_path, "qwen3_0.6b_transformer_block_0_norm1.onnx")

    # Export to ONNX with only 1 input: (x)
    torch.onnx.export(
        norm1,
        (dummy_input,),  # Single input argument (tuple format required)
        str(onnx_path),
        input_names=['embeddings'],
        output_names=['normalized_embeddings'],
        dynamic_axes={
            'embeddings': {0: 'batch_size'},
            'normalized_embeddings': {0: 'batch_size'}
            # seq_len (32) and emb_dim are fixed, so no dynamic axes needed for them
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"Transformer block 0's norm1 successfully exported to ONNX format: {onnx_path}")
    print(f"ONNX model file size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    print("Note: The exported ONNX model requires 1 input: embeddings [batch_size, 32, emb_dim]")
    print(f"Output shape: [batch_size, 32, {QWEN_CONFIG_06_B_FIXED_32['emb_dim']}]")
except Exception as e:
    print(f"Error exporting to ONNX: {e}")
    import traceback
    traceback.print_exc()

